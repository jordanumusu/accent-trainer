# audio_io.py
# Utilities for decoding arbitrary audio bytes into a mono torch.Tensor at TARGET_SR,
# validating duration, and exporting/wrapping as 16‑bit PCM WAV (base64).

from __future__ import annotations

import io
import os
import base64
import wave
import math
import tempfile
import subprocess
from typing import Optional, Tuple

import numpy as np
import torch

try:
    import soundfile as sf  # libsndfile backend; handles WAV/FLAC/OGG, some MP3s
    _HAS_SF = True
except Exception:
    _HAS_SF = False

# ---------- Config ----------
TARGET_SR: int = 16_000
MIN_DURATION_S: float = 0.18     # leniency for clipped utterances
MAX_DURATION_S: float = 6.5      # keep demo snappy; protects against huge uploads
MAX_ABS_CLIP: float = 0.9995     # hard clip threshold when writing WAV


# ---------- Core decode ----------
def decode_audio_bytes(data: bytes, sr: int = TARGET_SR) -> torch.Tensor:
    """
    Decode arbitrary audio bytes into a mono float32 tensor at sample rate `sr`.
    Tries (1) soundfile (libsndfile) from bytes; if that fails, (2) ffmpeg fallback.

    Returns:
        1-D torch.Tensor [num_samples] float32 in range [-1, 1] (not guaranteed normalized).
    """
    # 1) Try soundfile from memory (if available)
    if _HAS_SF:
        try:
            with sf.SoundFile(io.BytesIO(data)) as f:
                audio = f.read(dtype="float32", always_2d=True)  # shape [frames, channels]
                in_sr = f.samplerate
            x = _postprocess_to_mono(audio, in_sr, sr)
            return torch.from_numpy(x)
        except Exception:
            # fall through to ffmpeg
            pass

    # 2) ffmpeg fallback for any container/codec (e.g., MKV, MP4, AAC, AMR, etc.)
    # We stream raw s16le PCM out of ffmpeg and then up/downsample if needed.
    x, out_sr = _ffmpeg_decode_to_pcm(data)
    if x is None:
        # Give one last try: write temp file and let soundfile open it by path
        if _HAS_SF:
            try:
                with tempfile.NamedTemporaryFile(suffix=".bin", delete=True) as tmp:
                    tmp.write(data)
                    tmp.flush()
                    with sf.SoundFile(tmp.name) as f:
                        audio = f.read(dtype="float32", always_2d=True)
                        in_sr = f.samplerate
                    x = _postprocess_to_mono(audio, in_sr, sr)
                    return torch.from_numpy(x)
            except Exception:
                pass
        # If all decoders fail, return empty tensor (caller handles as "silence")
        return torch.zeros(0, dtype=torch.float32)

    # x from ffmpeg is mono float32 at out_sr (we asked -ac 1, -ar TARGET_SR)
    if out_sr != sr:
        x = _resample_np(x, out_sr, sr)
    return torch.from_numpy(x)


def _ffmpeg_decode_to_pcm(data: bytes) -> Tuple[Optional[np.ndarray], int]:
    """
    Use ffmpeg to decode arbitrary input bytes to mono s16le at TARGET_SR, streamed via stdout.
    Returns (float32 numpy array in [-1, 1], sample_rate) or (None, 0) on failure.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".blob", delete=True) as tmp_in:
            tmp_in.write(data)
            tmp_in.flush()

            cmd = [
                "ffmpeg",
                "-v", "error",
                "-i", tmp_in.name,
                "-ac", "1",               # mono
                "-ar", str(TARGET_SR),    # target sample rate
                "-f", "s16le",            # raw PCM
                "-"                        # stdout
            ]
            proc = subprocess.run(cmd, capture_output=True, check=False)
            if proc.returncode != 0 or len(proc.stdout) == 0:
                return None, 0

            pcm = np.frombuffer(proc.stdout, dtype=np.int16)
            if pcm.size == 0:
                return None, 0
            audio = (pcm.astype(np.float32) / 32768.0).reshape(-1)
            return audio, TARGET_SR
    except FileNotFoundError:
        # ffmpeg not installed
        return None, 0
    except Exception:
        return None, 0


# ---------- Validation ----------
def validate_audio_length(waveform: torch.Tensor, min_s: float = MIN_DURATION_S, max_s: float = MAX_DURATION_S, sr: int = TARGET_SR) -> None:
    """
    Raise ValueError if audio length is outside [min_s, max_s].
    """
    if waveform.numel() == 0:
        raise ValueError("Empty audio.")
    dur = float(waveform.numel()) / float(sr)
    if dur < min_s:
        raise ValueError(f"Audio too short ({dur:.2f}s). Please record a bit longer.")
    if dur > max_s:
        raise ValueError(f"Audio too long ({dur:.2f}s). Please keep it under {max_s:.1f}s.")


# ---------- Export: WAV (base64) ----------
def b64_wav_from_tensor(audio: torch.Tensor, sr: int = TARGET_SR) -> bytes:
    """
    Serialize a mono float32 tensor [-1,1] to 16‑bit PCM WAV bytes and return base64-encoded bytes.
    Returns a bytes object (base64 ascii). Caller can base64.b64decode(...) to get raw WAV bytes.
    """
    x = audio.detach().cpu().numpy().astype(np.float32).reshape(-1)
    if x.size == 0:
        # Return a valid, tiny WAV header with no samples
        return _empty_wav_b64(sr)
    # Hard clip & convert to int16
    x = np.clip(x, -MAX_ABS_CLIP, MAX_ABS_CLIP)
    pcm16 = (x * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    raw_bytes = buf.getvalue()
    return base64.b64encode(raw_bytes)


def _empty_wav_b64(sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"")
    return base64.b64encode(buf.getvalue())


# ---------- Small DSP helpers ----------
def _postprocess_to_mono(audio_2d: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    """
    audio_2d: float32 array [frames, channels]
    - Mix to mono
    - Resample if needed
    - Ensure contiguous float32 [-1,1]
    """
    if audio_2d.ndim != 2 or audio_2d.shape[1] < 1:
        return np.zeros(0, dtype=np.float32)
    # Mix down
    if audio_2d.shape[1] > 1:
        x = np.mean(audio_2d, axis=1, dtype=np.float32)
    else:
        x = audio_2d[:, 0].astype(np.float32, copy=False)

    # Resample if needed
    if in_sr != out_sr and x.size > 0:
        x = _resample_np(x, in_sr, out_sr)

    return x.astype(np.float32, copy=False).reshape(-1)


def _resample_np(x: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    """
    Lightweight bandlimited resampler using polyphase (scipy-free).
    Falls back to naive linear if polyphase isn’t possible.
    """
    if in_sr == out_sr or x.size == 0:
        return x.astype(np.float32, copy=False)

    try:
        # Polyphase via rational approximation (no scipy dependency)
        from math import gcd
        g = gcd(in_sr, out_sr)
        up = out_sr // g
        down = in_sr // g

        # Upsample by zero-stuffing then low-pass FIR filter; then decimate.
        # To keep this dependency-free and fast for short demo clips, we use
        # a simple Kaiser-windowed low-pass design with small taps.
        num_taps = 32 * max(1, int(in_sr / 8000))
        cutoff = 0.5 / max(up, down)  # normalized (Nyquist=0.5)
        h = _kaiser_lowpass(num_taps, cutoff, beta=8.6).astype(np.float32)

        # Upsample
        upsampled = np.zeros(x.size * up, dtype=np.float32)
        upsampled[::up] = x

        # Convolve (same mode)
        y = np.convolve(upsampled, h, mode="same")

        # Decimate
        y = y[::down]
        return y.astype(np.float32, copy=False)
    except Exception:
        # Naive linear resample fallback
        t_in = np.linspace(0.0, 1.0, num=x.size, endpoint=False)
        t_out = np.linspace(0.0, 1.0, num=int(round(x.size * out_sr / in_sr)), endpoint=False)
        return np.interp(t_out, t_in, x).astype(np.float32, copy=False)


def _kaiser_lowpass(num_taps: int, cutoff: float, beta: float) -> np.ndarray:
    """
    Create a simple low-pass FIR with a Kaiser window.
    cutoff is normalized to Nyquist=0.5.
    """
    if num_taps % 2 == 0:
        num_taps += 1
    n = np.arange(num_taps) - (num_taps - 1) / 2.0
    h = np.sinc(2 * cutoff * n)
    w = np.kaiser(num_taps, beta)
    h *= w
    h /= np.sum(h) if np.sum(h) != 0 else 1.0
    return h.astype(np.float32)

