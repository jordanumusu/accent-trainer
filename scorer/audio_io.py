# audio_io.py
import io
import os
import base64
import torch
import torchaudio
from fastapi import HTTPException
import tempfile
import subprocess
import shutil

TARGET_SR = 16000

def _load_with_torchaudio(path: str, sr: int) -> torch.Tensor:
    waveform, orig_sr = torchaudio.load(path)  # may fail for webm/opus on some builds
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
    return waveform.squeeze(0)

def _load_with_ffmpeg(path: str, sr: int) -> torch.Tensor:
    """
    Decode any format via ffmpeg -> s16le PCM, then to torch tensor.
    Requires `ffmpeg` binary in the container.
    """
    if not shutil.which("ffmpeg"):
        raise HTTPException(400, "Invalid audio format and ffmpeg not available to transcode it.")
    # -i input -ac 1 mono -ar sr sample rate -f s16le raw PCM to stdout
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", path, "-ac", "1", "-ar", str(sr), "-f", "s16le", "pipe:1"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or len(proc.stdout) == 0:
        err = proc.stderr.decode(errors="ignore")[:400]
        raise HTTPException(400, f"Invalid audio format - browser audio should work: {err}")
    pcm = torch.frombuffer(proc.stdout, dtype=torch.int16).to(torch.float32) / 32768.0
    return pcm

def decode_audio_bytes(audio_bytes: bytes, sr: int = TARGET_SR) -> torch.Tensor:
    """
    Decode audio (wav, mp3, m4a, webm, ogg, etc.) to mono float32 in [-1, 1].
    Returns shape: (time,) tensor.
    """
    # Write to a temp file with no assumption about extension
    # ffmpeg auto-detects container, so suffix isn't critical.
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        path = tmp.name

    try:
        # First, try torchaudio (good for WAV/MP3 on many builds)
        try:
            return _load_with_torchaudio(path, sr)
        except Exception:
            # Fallback to ffmpeg (handles WebM/Opus etc.)
            return _load_with_ffmpeg(path, sr)
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass

def wav_bytes_from_tensor(waveform: torch.Tensor, sr: int = TARGET_SR) -> bytes:
    """
    Convert mono float32 tensor in [-1, 1] to WAV bytes.
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # (1, time)
    buf = io.BytesIO()
    # torchaudio.save expects CPU tensor
    torchaudio.save(buf, waveform.cpu(), sr, format="wav")
    return buf.getvalue()

def b64_wav_from_tensor(waveform: torch.Tensor, sr: int = TARGET_SR) -> str:
    """
    Base64-encode a WAV representation of the given tensor.
    """
    return base64.b64encode(wav_bytes_from_tensor(waveform, sr)).decode()

def validate_audio_length(waveform: torch.Tensor, min_seconds: float = 0.25, max_seconds: float = 3.0) -> None:
    """
    Validate audio length and raise HTTPException if invalid.
    """
    duration = waveform.numel() / TARGET_SR
    if duration < min_seconds:
        raise HTTPException(400, f"Audio too short ({duration:.2f}s). Please record at least {min_seconds}s.")
    if duration > max_seconds:
        raise HTTPException(400, f"Audio too long ({duration:.2f}s). Please record a single word under {max_seconds}s.")
