import base64
# main.py
import io
import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from numpy.typing import NDArray
from phonemizer import phonemize
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# =========================================================
# Logging
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("pronounce-api")

# =========================================================
# Model + DSP constants
# =========================================================
SR = 16000
FRAME_LENGTH = 1024
HOP_LENGTH = 256
MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"

# Load once at import
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.eval()

# =========================================================
# Language normalization
# =========================================================
# Accept flexible inputs and map to espeak/phonemizer voices
_LANG_ALIASES = {
    "fr": "fr-fr",
    "fr-fr": "fr-fr",
    "fr_fr": "fr-fr",
    "fr-france": "fr-fr",
    "en": "en-us",
    "en-us": "en-us",
    "en_us": "en-us",
    "en-us-generic": "en-us",
    "es": "es",
    "es-es": "es",
    "es_es": "es",
    "spanish": "es",
    "de": "de",
    "de-de": "de",
    "de_de": "de",
    "german": "de",
    "it": "it",
    "it-it": "it",
    "it_it": "it",
    "italian": "it",
}

SUPPORTED_LANGS = {"fr-fr", "en-us", "es", "de", "it"}


def normalize_lang(lang: str) -> str:
    if not lang:
        return ""
    key = lang.strip().lower()
    return _LANG_ALIASES.get(key, key)


def espeak_voice(lang: str) -> str:
    """Map normalized language to espeak-ng voice."""
    return {"fr-fr": "fr-fr", "en-us": "en-us", "es": "es", "de": "de", "it": "it"}.get(lang, lang)


# =========================================================
# Audio utils
# =========================================================
def ffmpeg_to_wav16k_mono(data: bytes) -> bytes:
    """Decode arbitrary input (webm/opus/etc.) -> WAV mono @ 16k."""
    try:
        p = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-f",
                "wav",
                "-ac",
                "1",
                "-ar",
                str(SR),
                "pipe:1",
            ],
            input=data,
            stdout=subprocess.PIPE,
            check=True,
        )
        return p.stdout
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail="Invalid or corrupted audio file") from e


def synth_espeak_wav(text: str, lang: str, rate: int = SR) -> bytes:
    """Synthesize a canonical reference with espeak-ng and resample to 16k mono."""
    try:
        with tempfile.TemporaryDirectory() as td:
            wref = os.path.join(td, "ref.wav")
            subprocess.run(
                ["espeak-ng", "-v", espeak_voice(lang), "-s", "150", "-w", wref, text],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            p = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    wref,
                    "-f",
                    "wav",
                    "-ac",
                    "1",
                    "-ar",
                    str(rate),
                    "pipe:1",
                ],
                check=True,
                stdout=subprocess.PIPE,
            )
            return p.stdout
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail="TTS synthesis failed") from e


# =========================================================
# Prosody features
# =========================================================
def extract_f0_energy(y: NDArray[np.float32], sr: int = SR) -> Dict[str, float]:
    """Return basic prosody stats: pitch mean/range, slope, terminal delta, loudness, duration."""
    f0, _, _ = librosa.pyin(y, fmin=60, fmax=450, sr=sr, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    f0 = np.array(f0, dtype=np.float32)

    # Fill unvoiced frames by linear interp of voiced frames to stabilize stats
    t = np.arange(len(f0))
    voiced = ~np.isnan(f0)
    if voiced.any():
        f0[~voiced] = np.interp(t[~voiced], t[voiced], f0[voiced])
    else:
        f0[:] = 0.0

    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP_LENGTH)

    f0_nonzero = f0[f0 > 0]
    f0_mean = float(np.mean(f0_nonzero)) if f0_nonzero.size else 0.0
    f0_p05 = float(np.percentile(f0_nonzero, 5)) if f0_nonzero.size else 0.0
    f0_p95 = float(np.percentile(f0_nonzero, 95)) if f0_nonzero.size else 0.0
    f0_range = float(f0_p95 - f0_p05)

    def hz_to_st(x: NDArray[np.float32]) -> NDArray[np.float32]:
        return 12 * np.log2(np.maximum(x, 1e-5) / 55.0)

    st = hz_to_st(f0)
    slope = float(np.polyfit(times, st, 1)[0]) if len(times) >= 2 else 0.0
    end_st = float(st[-5:].mean()) if len(st) >= 5 else float(st[-1])
    start_st = float(st[:5].mean()) if len(st) >= 5 else float(st[0])
    end_delta_st = end_st - start_st

    return {
        "f0_mean_hz": f0_mean,
        "f0_range_hz": f0_range,
        "f0_slope_st_per_s": slope,
        "end_delta_st": end_delta_st,
        "rms_mean": float(np.mean(rms)),
        "dur_s": float(len(y) / sr),
    }


def compare_prosody(user: Dict[str, float], ref: Dict[str, float], lang: str) -> Dict[str, Any]:
    """Heuristic prosody comparison -> small set of actionable tips."""
    d_range = user["f0_range_hz"] - ref["f0_range_hz"]
    d_mean = user["f0_mean_hz"] - ref["f0_mean_hz"]
    d_slope = user["f0_slope_st_per_s"] - ref["f0_slope_st_per_s"]
    d_end = user["end_delta_st"] - ref["end_delta_st"]

    tips: List[str] = []

    # Expressiveness
    if user["f0_range_hz"] < max(40.0, 0.6 * ref["f0_range_hz"]):
        tips.append("Use more pitch movement—lift on the stressed syllable.")
    if user["f0_range_hz"] > 1.6 * ref["f0_range_hz"]:
        tips.append("Range is very wide; smooth large jumps for natural flow.")

    # Terminal contour heuristics by language
    if lang.startswith("en"):
        if user["end_delta_st"] < -0.5:
            tips.append("Avoid a steep final drop—finish with a gentler fall.")
        if user["end_delta_st"] > 0.8:
            tips.append("Rising tail sounds interrogative; use a slight fall for statements.")
    elif lang.startswith("fr"):
        if user["end_delta_st"] > 0.8:
            tips.append("French statements rarely end with a strong rise—keep it flat/slightly falling.")
        if abs(d_mean) > 40:
            tips.append("Keep overall pitch closer to the reference—avoid speaking too high/low.")
    elif lang == "es":
        if user["end_delta_st"] > 1.0:
            tips.append("Spanish statements end with gentle fall—avoid strong rises.")
        if user["f0_range_hz"] < 60:
            tips.append("Spanish uses more pitch movement—add expressiveness.")
    elif lang == "de":
        if user["end_delta_st"] < -1.0:
            tips.append("German final fall is less steep—ease into the ending.")
        if user["f0_range_hz"] > 120:
            tips.append("German pitch range is narrower—reduce dramatic swings.")
    elif lang == "it":
        if user["end_delta_st"] > 0.5:
            tips.append("Italian statements fall more clearly—avoid rising endings.")
        if user["f0_range_hz"] < 80:
            tips.append("Italian is melodic—use more pitch variation.")

    # Loudness + timing
    if user["rms_mean"] < 0.6 * ref["rms_mean"]:
        tips.append("Project more—steady breath support helps tone stability.")
    if abs(user["dur_s"] - ref["dur_s"]) > 0.4 * ref["dur_s"]:
        tips.append("Match timing—mirror the reference pace and vowel length.")

    return {
        "delta": {"range_hz": d_range, "mean_hz": d_mean, "slope_st_per_s": d_slope, "end_delta_st": d_end},
        "tips": tips[:2] or ["Hum the reference melody, then speak it with the same contour."],
    }


def xlsr_accent_distance(y_user: NDArray[np.float32], y_ref: NDArray[np.float32]) -> float:
    """Coarse accent-ish distance via cosine( mean-pooled last hidden state ). 0≈close, higher=farther."""
    with torch.no_grad():
        iu = processor(y_user, sampling_rate=SR, return_tensors="pt")
        ir = processor(y_ref, sampling_rate=SR, return_tensors="pt")
        hu = model(**iu, output_hidden_states=True).hidden_states[-1].mean(dim=1)
        hr = model(**ir, output_hidden_states=True).hidden_states[-1].mean(dim=1)
        sim = F.cosine_similarity(hu, hr).item()
    return float(1.0 - sim)


# =========================================================
# Phones / decoding
# =========================================================
def phonemize_target(text: str, lang: str) -> str:
    """Phonemize canonical target using phonemizer (espeak backend)."""
    return phonemize(
        text,
        language=espeak_voice(lang),
        backend="espeak",
        strip=True,
        preserve_punctuation=False,
        with_stress=False,
    )


def argmax_decode(logits: torch.Tensor, vocab: Dict[str, int]) -> str:
    """Simple CTC argmax decode with repeat + blank removal, returns espeak-ish phone string."""
    ids = torch.argmax(logits, dim=-1).cpu().numpy()[0]
    id2tok = {v: k for k, v in vocab.items()}
    out: List[str] = []
    prev = None
    blank = processor.tokenizer.pad_token_id
    for i in ids:
        if i == blank or i == prev:
            prev = i
            continue
        out.append(id2tok.get(i, ""))
        prev = i
    return "".join(out).replace("|", "").strip()


def align(a: str, b: str) -> List[Tuple[str, str]]:
    """LCS backtrace alignment over characters (quick phoneme proxy for demo)."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    bt = [[None] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
                bt[i + 1][j + 1] = "D"
            elif dp[i][j + 1] >= dp[i + 1][j]:
                dp[i + 1][j + 1] = dp[i][j + 1]
                bt[i + 1][j + 1] = "U"
            else:
                dp[i + 1][j + 1] = dp[i + 1][j]
                bt[i + 1][j + 1] = "L"
    i, j = m, n
    pairs: List[Tuple[str, str]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bt[i][j] == "D":
            pairs.append((a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or bt[i][j] == "L"):
            pairs.append(("-", b[j - 1]))
            j -= 1
        else:
            pairs.append((a[i - 1], "-"))
            i -= 1
    pairs.reverse()
    return pairs


def per_phone_scores(target: str, hyp: str) -> Tuple[float, List[Dict[str, Any]]]:
    """Toy per-phone scores (1 for match, 0 otherwise)."""
    pairs = align(target, hyp)
    scores: List[Dict[str, Any]] = []
    for t, h in pairs:
        if t == "-" or h == "-":
            s = 0.0
        else:
            s = 1.0 if t == h else 0.0
        scores.append({"target": t, "hyp": h, "score": s})
    real = [p for p in scores if p["target"] != "-"]
    avg = (sum(p["score"] for p in real) / max(1, len(real))) if real else 0.0
    return avg, scores


# Common confusions -> tiny rule tips
TIPS = {
    ("ʁ", "r"): "French /ʁ/ is uvular; relax tongue tip, constrict near the uvula.",
    ("y", "u"): "French /y/ = rounded /i/; say /i/ then round lips tightly.",
    ("θ", "s"): "For /θ/, place tongue lightly between teeth and blow.",
    ("ð", "z"): "For /ð/, tongue between teeth, keep voicing.",
    ("ɪ", "i"): "Shorter, laxer /ɪ/; don't tense like /i/.",
    ("æ", "e"): "Open jaw more for /æ/; tongue front and low.",
    ("ɾ", "r"): "Spanish /ɾ/ is a quick tap; touch roof of mouth lightly.",
    ("x", "h"): "Spanish /x/ is deeper than /h/; constrict at the back.",
    ("ʃ", "s"): "German /ʃ/ rounds lips slightly; not as sharp as /s/.",
    ("ʊ", "u"): "German /ʊ/ is shorter; don't hold as long as /u/.",
    ("ʎ", "l"): "Italian /ʎ/ palatalizes; tongue touches hard palate.",
    ("ts", "s"): "Italian /ts/ is affricate; start with /t/ then /s/.",
}


def pick_tip(target: str, hyp: str) -> str:
    for (t, h), tip in TIPS.items():
        if t in target and h in hyp:
            return tip
    return "Slow down. Hold the vowel steady, then shape the consonant cleanly."


# =========================================================
# FastAPI
# =========================================================
app = FastAPI(title="Pronunciation Eval API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/eval")
async def eval_endpoint(audio: UploadFile, text: str = Form(...), lang: str = Form(...)) -> Dict[str, Any]:
    # Validate inputs
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text parameter cannot be empty")

    lang_norm = normalize_lang(lang)
    if lang_norm not in SUPPORTED_LANGS:
        raise HTTPException(
            status_code=400,
            detail=f"Language must be one of: {sorted(list(SUPPORTED_LANGS))}"
        )

    if not audio or not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")

    # Read & decode audio
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio file")

    wav = ffmpeg_to_wav16k_mono(raw)

    # Load into numpy
    try:
        y, sr = sf.read(io.BytesIO(wav), dtype="float32")
        if y.ndim > 1:
            y = y[:, 0]
        if len(y) == 0:
            raise ValueError("no samples")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode audio: {e}") from e

    # Reference TTS (for prosody + accent distance)
    ref_wav = synth_espeak_wav(text, lang_norm)
    y_ref, _ = sf.read(io.BytesIO(ref_wav), dtype="float32")
    if y_ref.ndim > 1:
        y_ref = y_ref[:, 0]

    # Prosody features & comparison
    user_pros = extract_f0_energy(y, sr=SR)
    ref_pros = extract_f0_energy(y_ref, sr=SR)
    prosody_cmp = compare_prosody(user_pros, ref_pros, lang=lang_norm)

    # Accent-ish distance (optional)
    try:
        accent_dist = xlsr_accent_distance(y, y_ref)
    except Exception:
        accent_dist = None

    # Model inference -> phone hypothesis
    try:
        with torch.no_grad():
            inputs = processor(y, sampling_rate=SR, return_tensors="pt")
            logits = model(**inputs).logits
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise HTTPException(status_code=507, detail="Audio too large for processing") from e
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}") from e

    # Decode + scoring
    try:
        hyp = argmax_decode(logits, processor.tokenizer.get_vocab())
        tgt = phonemize_target(text, lang_norm)
        avg, phones = per_phone_scores(tgt, hyp)
        tip = pick_tip(tgt, hyp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}") from e

    return {
        "avg": avg,  # 0..1
        "targetIPA": tgt,
        "hypothesisIPA": hyp,
        "perPhone": phones,
        "tip": tip,
        "prosody": {
            "user": user_pros,
            "ref": ref_pros,
            "compare": prosody_cmp,
        },
        "accentDistance": accent_dist,
        # New: base64 audio for playback in the UI
        "refAudioWavBase64": base64.b64encode(ref_wav).decode("ascii"),
        "userAudioWavBase64": base64.b64encode(wav).decode("ascii"),
    }