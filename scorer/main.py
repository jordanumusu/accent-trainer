from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import whisper
import librosa
import torch
import numpy as np
import base64
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
import subprocess
import tempfile
import os

from audio_io import decode_audio_bytes, b64_wav_from_tensor, validate_audio_length, TARGET_SR

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Load models once at startup
whisper_model = None
reference_bank = {}  # word -> {"ipa": str, "audio": np.ndarray}
calibration_stats = {}  # lang -> {"mean": float, "std": float}

@dataclass
class PhoneScore:
    phone: str
    score: float
    feedback: str

@dataclass
class ProsodyResult:
    score: float
    stress_ok: bool
    timing_ratio: float

class PronunciationScorer:
    def __init__(self):
        self.simple_g2p = {
            "en": {"hello": "həˈloʊ", "world": "wɜrld"},
            "fr": {"bonjour": "bɔ̃ʒur", "merci": "mɛrsi"},
            "es": {"hola": "ola", "gracias": "graθjas"}
        }
        
        # Simple phone similarity lookup
        self.phone_similarities = {
            ("ɔ̃", "ɔ"): 0.7,  # nasal vs oral
            ("ʁ", "r"): 0.8,   # uvular vs alveolar r
            ("θ", "s"): 0.5,   # common L2 substitution
            ("ð", "z"): 0.5,
        }

    def text_to_ipa(self, text: str, lang: str) -> str:
        """Convert text to IPA. Fallback to simple lookup for demo."""
        text_clean = text.lower().strip()
        
        if lang in self.simple_g2p and text_clean in self.simple_g2p[lang]:
            return self.simple_g2p[lang][text_clean]
        
        # In real implementation: use phonemizer/gruut
        # For demo: return grapheme approximation
        return text_clean
    
    def asr_with_confidence(self, audio: torch.Tensor) -> Tuple[str, float]:
        """Get ASR hypothesis and word-level confidence."""
        try:
            # Whisper accepts tensors directly
            result = whisper_model.transcribe(
                audio.float().cpu(), 
                word_timestamps=True,
                condition_on_previous_text=False
            )
            
            if not result["segments"]:
                return "", 0.0
                
            # Get first word and its confidence
            words = result["segments"][0].get("words", [])
            if not words:
                return result["text"].strip(), 0.5  # fallback confidence
                
            text = words[0]["word"].strip()
            confidence = words[0].get("probability", 0.5)
            return text, confidence
            
        except Exception as e:
            logging.error(f"ASR failed: {e}")
            return "", 0.0

    def phone_similarity(self, target: str, hypothesis: str) -> float:
        """Calculate similarity between two phones."""
        if target == hypothesis:
            return 1.0
            
        # Check similarity lookup
        pair = (target, hypothesis)
        if pair in self.phone_similarities:
            return self.phone_similarities[pair]
        if (hypothesis, target) in self.phone_similarities:
            return self.phone_similarities[(hypothesis, target)]
            
        # Default mismatch penalty
        return 0.3

    def simple_ipa_align(self, target: str, hypothesis: str) -> List[Tuple[str, Optional[str]]]:
        """Simple character-level alignment for IPA strings."""
        # For demo: just pair up characters, pad with None if lengths differ
        alignment = []
        max_len = max(len(target), len(hypothesis))
        
        for i in range(max_len):
            t_char = target[i] if i < len(target) else None
            h_char = hypothesis[i] if i < len(hypothesis) else None
            
            if t_char:
                alignment.append((t_char, h_char))
                
        return alignment

    def score_phones(self, target_ipa: str, hypothesis_ipa: str, asr_confidence: float) -> List[PhoneScore]:
        """Score individual phones."""
        alignment = self.simple_ipa_align(target_ipa, hypothesis_ipa)
        scores = []
        
        for target_phone, hyp_phone in alignment:
            if not target_phone:
                continue
                
            # Calculate phone score
            edit_score = self.phone_similarity(target_phone, hyp_phone) if hyp_phone else 0.1
            conf_score = asr_confidence  # Simplified: use word confidence for all phones
            
            phone_score = (edit_score ** 0.6) * (conf_score ** 0.4)
            phone_score = max(0.0, min(1.0, phone_score))  # clamp to [0,1]
            
            # Generate feedback
            feedback = "good" if phone_score > 0.8 else "needs work"
            if hyp_phone and target_phone != hyp_phone:
                feedback = f"pronounced as /{hyp_phone}/"
                
            scores.append(PhoneScore(
                phone=target_phone,
                score=round(phone_score, 2),
                feedback=feedback
            ))
            
        return scores

    def extract_basic_prosody(self, audio: torch.Tensor, sr: int = TARGET_SR) -> Dict:
        """Extract basic prosodic features."""
        # Convert to numpy only for librosa
        audio_np = audio.float().cpu().numpy()
        
        # Fundamental frequency
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_np, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        
        # Energy/RMS
        rms = librosa.feature.rms(y=audio_np)[0]
        
        # Duration
        duration = audio.numel() / sr
        
        return {
            "f0_mean": float(np.nanmean(f0)) if not np.all(np.isnan(f0)) else 0.0,
            "energy_mean": float(np.mean(rms)),
            "duration": duration
        }

    def compare_prosody(self, user_features: Dict, ref_features: Dict) -> ProsodyResult:
        """Compare user prosody to reference."""
        # Duration ratio
        timing_ratio = user_features["duration"] / max(ref_features["duration"], 0.1)
        dur_score = np.exp(-abs(np.log(timing_ratio))) if timing_ratio > 0 else 0.0
        
        # F0 similarity (simplified)
        f0_diff = abs(user_features["f0_mean"] - ref_features["f0_mean"])
        f0_score = max(0.0, 1.0 - f0_diff / 100.0)  # normalize by 100Hz
        
        # Energy similarity
        energy_diff = abs(user_features["energy_mean"] - ref_features["energy_mean"])
        energy_score = max(0.0, 1.0 - energy_diff / 0.1)
        
        # Combined prosody score
        prosody_score = 0.5 * dur_score + 0.3 * f0_score + 0.2 * energy_score
        
        return ProsodyResult(
            score=round(prosody_score, 2),
            stress_ok=0.8 <= timing_ratio <= 1.2,  # simplified stress check
            timing_ratio=round(timing_ratio, 2)
        )

    def calibrate_score(self, raw_score: float, lang: str) -> float:
        """Apply calibration based on collected data."""
        if lang not in calibration_stats:
            return raw_score  # no calibration data
            
        stats = calibration_stats[lang]
        z_score = (raw_score - stats["mean"]) / max(stats["std"], 0.1)
        
        # Sigmoid to [0,1]
        return 1.0 / (1.0 + np.exp(-z_score))

    def generate_tip(self, phone_scores: List[PhoneScore], prosody: ProsodyResult) -> str:
        """Generate a helpful tip based on scores."""
        # Find lowest scoring phone
        worst_phone = min(phone_scores, key=lambda p: p.score) if phone_scores else None
        
        if prosody.timing_ratio < 0.8:
            return "Try speaking more slowly and clearly."
        elif prosody.timing_ratio > 1.2:
            return "Try to maintain a steadier pace."
        elif worst_phone and worst_phone.score < 0.6:
            return f"Focus on the /{worst_phone.phone}/ sound - {worst_phone.feedback}."
        else:
            return "Good pronunciation! Keep practicing."

# Create scorer instance at module level
scorer = PronunciationScorer()

def generate_espeak_reference(text: str, lang: str) -> torch.Tensor:
    """
    Generate reference audio using espeak-ng and return as mono float32 tensor.
    """
    voice_map = {
        "en": "en-us",    # American English
        "fr": "fr",       # French  
        "es": "es",       # Spanish
        "de": "de",       # German (for future)
    }
    voice = voice_map.get(lang, "en")
    
    with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:  # delete=True by default
        try:
            subprocess.run([
                'espeak-ng',
                f'-v{voice}',
                '-s', '120',         # slower for clarity
                '-a', '200',         # amplitude
                '-w', tmp.name,      # write output wav to temp file
                text
            ], check=True)  # removed capture_output since we don't use it
            
            # Read before file is deleted
            tmp.seek(0)
            with open(tmp.name, 'rb') as f:
                return decode_audio_bytes(f.read(), sr=TARGET_SR)
                
        except FileNotFoundError:
            logging.error("espeak-ng not found. Please install espeak-ng.")
            return torch.zeros(TARGET_SR)
        except subprocess.CalledProcessError as e:
            logging.warning(f"espeak-ng failed for '{text}': {e}")
            return torch.zeros(TARGET_SR)

@app.on_event("startup")
async def startup_event():
    global whisper_model, reference_bank, calibration_stats
    
    # Load Whisper model
    whisper_model = whisper.load_model("base")
    
    # Demo words with target IPA
    demo_words = {
        ("hello", "en"): "həˈloʊ",
        ("world", "en"): "wɜrld", 
        ("test", "en"): "tɛst",
        ("bonjour", "fr"): "bɔ̃ʒur",
        ("merci", "fr"): "mɛrsi",
        ("hola", "es"): "ola",
        ("gracias", "es"): "graθjas"
    }
    
    # Generate reference audio bank
    reference_bank = {}
    for (word, lang), ipa in demo_words.items():
        reference_bank[(word, lang)] = {
            "ipa": ipa,
            "waveform": generate_espeak_reference(word, lang)
        }
        logging.info(f"Generated reference for '{word}' in {lang}")
    
    # Load calibration stats
    calibration_stats = {
        "en": {"mean": 0.6, "std": 0.2},
        "fr": {"mean": 0.55, "std": 0.25},
        "es": {"mean": 0.58, "std": 0.22}
    }

@app.post("/eval")
async def score_pronunciation(
    audio: UploadFile = File(...),
    text: str = Form(...),
    lang: str = Form(...),
    verbose: bool = Form(False)
):
    try:
        # Normalize and validate inputs
        text = text.lower().strip()
        lang = lang.lower().strip()
        
        # Validate language support
        supported_langs = {"en", "fr", "es"}
        if lang not in supported_langs:
            raise HTTPException(400, f"Language '{lang}' not supported. Use: {', '.join(supported_langs)}")
        
        logging.info(f"Uploaded: filename={audio.filename!r} content_type={audio.content_type!r}")

        # Load and validate audio using clean audio_io
        audio_bytes = await audio.read()
        waveform = decode_audio_bytes(audio_bytes, sr=TARGET_SR)
        validate_audio_length(waveform)
        
        # Check if we have reference for this word
        ref_key = (text, lang)
        if ref_key not in reference_bank:
            available_words = [word for word, l in reference_bank.keys() if l == lang]
            raise HTTPException(400, f"Word '{text}' not available in {lang}. Try: {', '.join(available_words)}")
        
        reference = reference_bank[ref_key]
        ref_waveform = reference["waveform"]
        
        # Get target IPA
        target_ipa = scorer.text_to_ipa(text, lang)
        
        # ASR + confidence (pass tensor, convert inside method)
        asr_text, asr_confidence = scorer.asr_with_confidence(waveform)
        hypothesis_ipa = scorer.text_to_ipa(asr_text, lang)
        
        # Score phones
        phone_scores = scorer.score_phones(target_ipa, hypothesis_ipa, asr_confidence)
        
        # Prosody analysis (pass tensor, convert inside method)
        user_prosody = scorer.extract_basic_prosody(waveform, sr=TARGET_SR)
        ref_prosody = scorer.extract_basic_prosody(ref_waveform, sr=TARGET_SR)
        prosody_result = scorer.compare_prosody(user_prosody, ref_prosody)
        
        # Overall score
        segment_score = float(torch.tensor([p.score for p in phone_scores]).mean()) if phone_scores else 0.0
        raw_overall = 0.95 * segment_score + 0.05 * prosody_result.score # prosody with TTS is not great
        overall_score = scorer.calibrate_score(raw_overall, lang)
        
        # Generate response
        response = {
            "score": round(overall_score, 2),
            "targetIPA": target_ipa,
            "hypothesisIPA": hypothesis_ipa,
            "phones": [{"p": p.phone, "score": p.score, "feedback": p.feedback} for p in phone_scores],
            "prosody": {
                "score": prosody_result.score,
                "stress_ok": prosody_result.stress_ok,
                "timing_ratio": prosody_result.timing_ratio
            },
            "tip": scorer.generate_tip(phone_scores, prosody_result),
            "playback": {
                "user": b64_wav_from_tensor(waveform, TARGET_SR),
                "reference": b64_wav_from_tensor(ref_waveform, TARGET_SR)
            }
        }
        
        if verbose:
            response["debug"] = {
                "asr_text": asr_text,
                "asr_confidence": asr_confidence,
                "user_prosody": user_prosody,
                "ref_prosody": ref_prosody
            }
        
        return response
        
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        raise HTTPException(500, "Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)