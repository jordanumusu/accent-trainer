# main.py
import io
import tempfile
import os
import subprocess
from typing import List, Optional
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
import torch
import torchaudio
from espeakng import ESpeakNG

# Try to import soundfile as backup
try:
    import soundfile as sf
    import numpy as np
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

app = FastAPI(title="Accent Trainer – Wav2Vec2 (greedy) + eSpeak NG")

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().eval()
labels: List[str] = bundle.get_labels()
SAMPLE_RATE = bundle.sample_rate

BLANK_TOKENS = {"<blank>", "<pad>", ""}

esng = ESpeakNG()  

# Map common ISO codes to eSpeak NG voices
LANG_MAP = {
    "en": "en-us",  # Default English to US
    "en-us": "en-us",
    "en-gb": "en-gb",
    "fr": "fr-fr",
    "es": "es-es",
    "de": "de-de",
    "pt": "pt-pt",
    "pt-br": "pt-br",
    "it": "it-it",
    "nl": "nl-nl",
}

def map_lang(lang: str) -> str:
    lang = (lang or "en-us").lower().strip()
    # If exact match exists, use it
    if lang in LANG_MAP:
        return LANG_MAP[lang]
    # Try base language (e.g., "en" from "en-us")
    base_lang = lang.split('-')[0]
    if base_lang in LANG_MAP:
        return LANG_MAP[base_lang]
    # Default to US English
    return "en-us"

# ----------------- Audio conversion with ffmpeg -----------------
def convert_audio_to_wav(input_bytes: bytes, input_ext: str = '.webm') -> bytes:
    """
    Convert any audio format to WAV using ffmpeg
    Returns WAV bytes
    """
    input_file = None
    output_file = None
    
    try:
        # Create temp input file
        with tempfile.NamedTemporaryFile(suffix=input_ext, delete=False) as input_file:
            input_file.write(input_bytes)
            input_path = input_file.name
        
        # Create temp output file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as output_file:
            output_path = output_file.name
        
        # Convert using ffmpeg
        # -i: input file
        # -ar: sample rate (16000 is good for speech)
        # -ac: audio channels (1 = mono)
        # -f: format (wav)
        # -y: overwrite output
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ar', '16000',
            '-ac', '1',
            '-f', 'wav',
            '-y',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"ffmpeg failed: {result.stderr}")
        
        # Read the converted WAV
        with open(output_path, 'rb') as f:
            wav_bytes = f.read()
        
        return wav_bytes
        
    finally:
        # Clean up temp files
        for f in [input_path, output_path]:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass

# ----------------- Greedy CTC decode -----------------
def greedy_decode(emissions: torch.Tensor) -> str:
    """
    emissions: (batch, time, num_labels)
    returns: decoded string (spaces represented by '|' in labels -> real spaces)
    """
    # Take argmax over label dimension
    indices = torch.argmax(emissions, dim=-1)[0].tolist()  # (time,)
    prev_idx = None
    out: List[str] = []
    
    for idx in indices:
        # Skip if same as previous (collapse repeats)
        if idx == prev_idx:
            continue
            
        tok = labels[idx]
        # Skip blank tokens
        if tok not in BLANK_TOKENS:
            out.append(tok)
        
        prev_idx = idx
    
    # Join and clean up
    result = "".join(out)
    # Replace pipe with space (W2V2 uses | for space)
    result = result.replace("|", " ")
    # Remove any stray dashes that aren't part of words
    result = result.replace("-", "")
    # Clean up whitespace
    result = " ".join(result.split())
    
    return result.strip().lower()  # Lowercase for consistency

# ----------------- Edit distance on IPA characters -----------------
def levenshtein_chars(a: str, b: str) -> int:
    # operate on character lists (strip spaces to avoid penalizing word spacing)
    a_chars = [c for c in a if not c.isspace()]
    b_chars = [c for c in b if not c.isspace()]
    n, m = len(a_chars), len(b_chars)
    if n == 0: return m
    if m == 0: return n
    dp = [list(range(m + 1))] + [[i] + [0]*m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        ai = a_chars[i - 1]
        row = dp[i]
        prev_row = dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b_chars[j - 1] else 1
            row[j] = min(
                prev_row[j] + 1,      # deletion
                row[j - 1] + 1,       # insertion
                prev_row[j - 1] + cost  # substitution
            )
    return dp[n][m]

# ----------------- /eval endpoint -----------------
@app.post("/eval")
async def eval_pronunciation(
    audio: UploadFile = File(...),
    text: str = Form(...),
    lang: str = Form("en-us"),
):
    # Read upload
    audio_bytes = await audio.read()
    
    # Get file extension from filename
    ext = '.webm'  # Default to webm
    if audio.filename:
        file_ext = os.path.splitext(audio.filename)[1].lower()
        if file_ext:
            ext = file_ext
    
    # Convert to WAV if not already WAV
    if ext != '.wav':
        try:
            wav_bytes = convert_audio_to_wav(audio_bytes, ext)
        except Exception as e:
            return JSONResponse(
                {"error": f"Failed to convert audio from {ext} to WAV: {e}. Make sure ffmpeg is installed."},
                status_code=400
            )
    else:
        wav_bytes = audio_bytes
    
    # Load the WAV audio - try multiple approaches
    temp_wav = None
    try:
        # First write to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_wav.write(wav_bytes)
            temp_wav_path = temp_wav.name
        
        # Try loading with explicit backend
        try:
            # Try with soundfile backend (most reliable for WAV)
            waveform, sr = torchaudio.load(temp_wav_path, backend="soundfile")
        except:
            try:
                # Fallback to sox backend
                waveform, sr = torchaudio.load(temp_wav_path, backend="sox_io")
            except:
                # Last resort - try without specifying backend but with format
                waveform, sr = torchaudio.load(temp_wav_path, format="wav")
                
    except Exception as e:
        # If torchaudio completely fails, try using scipy/soundfile directly
        try:
            import soundfile as sf
            import numpy as np
            
            waveform_np, sr = sf.read(io.BytesIO(wav_bytes))
            # Convert to torch tensor and add batch dimension if needed
            waveform = torch.from_numpy(waveform_np).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 2 and waveform.shape[1] > waveform.shape[0]:
                waveform = waveform.transpose(0, 1)
        except Exception as sf_error:
            return JSONResponse(
                {"error": f"Failed to load audio with any method. torchaudio: {e}, soundfile: {sf_error}. Check if soundfile is installed: pip install soundfile"},
                status_code=400
            )
    finally:
        if temp_wav and os.path.exists(temp_wav_path):
            try:
                os.unlink(temp_wav_path)
            except:
                pass

    # Mono + resample to model rate
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    # ASR → greedy CTC
    with torch.inference_mode():
        emissions, _ = model(waveform)  # (1, T, num_labels)
    hypothesis = greedy_decode(emissions)

    # eSpeak NG: target + hypothesis to IPA
    voice = map_lang(lang)
    
    # Process target text to IPA
    try:
        # Set the voice first
        esng.voice = voice
        # Clean the target text
        clean_text = text.strip()
        if clean_text:
            # g2p() with ipa=2 returns IPA transcription
            target_ipa = esng.g2p(clean_text, ipa=2)
            # Clean up IPA output
            target_ipa = target_ipa.strip()
    except Exception as e:
        # Try with default US English voice
        try:
            esng.voice = "en-us"
            target_ipa = esng.g2p(clean_text, ipa=2)
            target_ipa = target_ipa.strip()
        except Exception as last_e:
            target_ipa = f"[IPA conversion failed: {str(e)[:50]}]"

    # Process hypothesis to IPA
    if hypothesis and hypothesis.strip():
        # Clean hypothesis - remove any leading dashes or special chars
        clean_hypothesis = hypothesis.strip()
        # Remove any stray dashes or special characters that might confuse espeak
        clean_hypothesis = ''.join(c for c in clean_hypothesis if c.isalnum() or c.isspace())
        clean_hypothesis = ' '.join(clean_hypothesis.split())  # normalize whitespace
        
        if clean_hypothesis:
            try:
                esng.voice = voice
                hyp_ipa = esng.g2p(clean_hypothesis, ipa=2)
                hyp_ipa = hyp_ipa.strip()
            except:
                try:
                    esng.voice = "en-us"
                    hyp_ipa = esng.g2p(clean_hypothesis, ipa=2)
                    hyp_ipa = hyp_ipa.strip()
                except:
                    hyp_ipa = ""
        else:
            hyp_ipa = ""
    else:
        hyp_ipa = ""

    # Score over IPA characters (space-insensitive)
    dist = levenshtein_chars(target_ipa, hyp_ipa)
    tgt_len = max(len([c for c in target_ipa if not c.isspace()]), 1)
    score = max(0.0, 1.0 - dist / tgt_len)

    return JSONResponse({
        "target": text,
        "hypothesis": hypothesis,
        "target_ipa": target_ipa,
        "hypothesis_ipa": hyp_ipa,
        "score": round(score, 4),
        "meta": {
            "sr_in": sr,
            "sr_model": SAMPLE_RATE,
            "decoder": "greedy_ctc",
            "ipa_scoring": "levenshtein_char_space_insensitive",
            "audio_format": ext
        }
    })

# Health check endpoint
@app.get("/health")
async def health_check():
    # Check if ffmpeg is available
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        ffmpeg_available = result.returncode == 0
    except:
        ffmpeg_available = False
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "ffmpeg_available": ffmpeg_available
    }

# To run locally:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)