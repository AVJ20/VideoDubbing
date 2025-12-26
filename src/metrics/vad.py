from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VADConfig:
    sample_rate: int = 16000
    frame_ms: float = 30.0
    hop_ms: float = 10.0

    # Smoothing + stability
    smooth_ms: float = 80.0
    hysteresis_ratio: float = 0.85
    min_speech_ms: float = 120.0
    min_silence_ms: float = 80.0

    # Thresholding strategy: threshold = max(abs_threshold, median_rms * median_ratio)
    abs_threshold: float = 0.008
    median_ratio: float = 1.6

    # Pre-filtering / denoise before VAD
    denoise: bool = True
    highpass_hz: float = 60.0
    lowpass_hz: float = 8000.0
    # ffmpeg afftdn noise floor in dB (more negative = more aggressive)
    denoise_nf_db: float = -25.0


def _ffmpeg_decode_f32le_mono(path: str, cfg: VADConfig) -> subprocess.Popen:
    # Use ffmpeg to decode any audio into mono float32 PCM at a stable sample rate.
    af_parts = []
    if cfg.highpass_hz and float(cfg.highpass_hz) > 0:
        af_parts.append(f"highpass=f={float(cfg.highpass_hz):.1f}")
    if cfg.lowpass_hz and float(cfg.lowpass_hz) > 0:
        af_parts.append(f"lowpass=f={float(cfg.lowpass_hz):.1f}")
    if cfg.denoise:
        # Built-in ffmpeg noise reduction. No external model needed.
        af_parts.append(f"afftdn=nf={float(cfg.denoise_nf_db):.1f}")

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-ac",
        "1",
        "-ar",
        str(int(cfg.sample_rate)),
    ]
    if af_parts:
        cmd.extend(["-af", ",".join(af_parts)])
    cmd.extend([
        "-f",
        "f32le",
        "-",
    ])
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if x.size == 0:
        return x
    w = int(max(1, win))
    if w == 1:
        return x
    kernel = np.ones((w,), dtype=np.float32) / float(w)
    return np.convolve(x.astype(np.float32), kernel, mode="same").astype(np.float32)


def _apply_hysteresis(x: np.ndarray, thr_on: float, thr_off: float) -> np.ndarray:
    voiced = np.zeros((x.size,), dtype=bool)
    state = False
    for i, v in enumerate(x):
        if not state:
            if v >= thr_on:
                state = True
        else:
            if v < thr_off:
                state = False
        voiced[i] = state
    return voiced


def _fill_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    if mask.size == 0:
        return mask
    out = mask.copy()
    n = out.size
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and not out[j]:
            j += 1
        gap = j - i
        if 0 < gap <= max_gap:
            left_voiced = (i - 1) >= 0 and out[i - 1]
            right_voiced = j < n and out[j]
            if left_voiced and right_voiced:
                out[i:j] = True
        i = j
    return out


def _drop_short_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    if mask.size == 0:
        return mask
    out = mask.copy()
    n = out.size
    i = 0
    while i < n:
        if not out[i]:
            i += 1
            continue
        j = i
        while j < n and out[j]:
            j += 1
        run = j - i
        if run < min_len:
            out[i:j] = False
        i = j
    return out


def compute_rms_frames(path: str, cfg: VADConfig) -> np.ndarray:
    """Compute frame RMS energy using ffmpeg decoding.

    Returns an array of RMS values per frame at hop resolution.
    """

    frame_len = max(1, int(round(cfg.sample_rate * cfg.frame_ms / 1000.0)))
    hop_len = max(1, int(round(cfg.sample_rate * cfg.hop_ms / 1000.0)))

    proc = _ffmpeg_decode_f32le_mono(path, cfg)
    assert proc.stdout is not None

    buf = np.empty(0, dtype=np.float32)
    rms: list[float] = []

    # Read chunks of float32 samples.
    chunk_bytes = 1024 * 64
    while True:
        raw = proc.stdout.read(chunk_bytes)
        if not raw:
            break
        chunk = np.frombuffer(raw, dtype=np.float32)
        if chunk.size == 0:
            continue

        if buf.size == 0:
            buf = chunk
        else:
            buf = np.concatenate([buf, chunk])

        # Process as many frames as possible.
        start = 0
        while start + frame_len <= buf.size:
            frame = buf[start:start + frame_len]
            val = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
            rms.append(val)
            start += hop_len

        # Keep unprocessed tail.
        if start > 0:
            buf = buf[start:]

    # Drain and check ffmpeg errors.
    stderr = b""
    if proc.stderr is not None:
        stderr = proc.stderr.read() or b""
    rc = proc.wait()
    if rc != 0:
        msg = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg decode failed for '{path}': {msg}")

    return np.asarray(rms, dtype=np.float32)


def voiced_frames(path: str, cfg: Optional[VADConfig] = None) -> tuple[np.ndarray, dict]:
    """Return voiced/unvoiced boolean array per hop frame and metadata."""

    cfg = cfg or VADConfig()
    rms = compute_rms_frames(path, cfg)
    if rms.size == 0:
        return np.zeros((0,), dtype=bool), {
            "threshold": None,
            "median_rms": None,
            "frames": 0,
            "hop_seconds": cfg.hop_ms / 1000.0,
        }

    # Smooth the energy curve to reduce chattering.
    smooth_frames = int(max(1, round(float(cfg.smooth_ms) / float(cfg.hop_ms))))
    rms_smooth = _moving_average(rms, smooth_frames)

    median = float(np.median(rms_smooth))
    threshold_on = max(float(cfg.abs_threshold), float(median * cfg.median_ratio))
    threshold_off = float(threshold_on) * float(cfg.hysteresis_ratio)
    voiced = _apply_hysteresis(rms_smooth, threshold_on, threshold_off)

    # Enforce minimum speech and minimum silence durations.
    min_speech_frames = int(max(1, round(float(cfg.min_speech_ms) / float(cfg.hop_ms))))
    max_gap_frames = int(max(0, round(float(cfg.min_silence_ms) / float(cfg.hop_ms))))
    if max_gap_frames > 0:
        voiced = _fill_short_gaps(voiced, max_gap_frames)
    if min_speech_frames > 1:
        voiced = _drop_short_runs(voiced, min_speech_frames)

    return voiced.astype(bool), {
        "threshold_on": float(threshold_on),
        "threshold_off": float(threshold_off),
        "median_rms": median,
        "frames": int(voiced.size),
        "hop_seconds": cfg.hop_ms / 1000.0,
        "config": {
            "sample_rate": cfg.sample_rate,
            "frame_ms": cfg.frame_ms,
            "hop_ms": cfg.hop_ms,
            "smooth_ms": cfg.smooth_ms,
            "hysteresis_ratio": cfg.hysteresis_ratio,
            "min_speech_ms": cfg.min_speech_ms,
            "min_silence_ms": cfg.min_silence_ms,
            "abs_threshold": cfg.abs_threshold,
            "median_ratio": cfg.median_ratio,
            "denoise": cfg.denoise,
            "highpass_hz": cfg.highpass_hz,
            "lowpass_hz": cfg.lowpass_hz,
            "denoise_nf_db": cfg.denoise_nf_db,
        },
    }
