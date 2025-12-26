from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class ProsodyConfig:
    sample_rate: int = 16000
    frame_ms: float = 40.0
    hop_ms: float = 10.0

    # Pitch search range
    f0_min_hz: float = 50.0
    f0_max_hz: float = 400.0

    # Voiced detection (RMS)
    abs_threshold: float = 0.008
    median_ratio: float = 1.6


def _ffmpeg_decode_segment_f32le_mono(
    path: str,
    *,
    sample_rate: int,
    start_time: float,
    duration: float,
) -> bytes:
    if duration <= 0:
        return b""

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{float(start_time):.6f}",
        "-t",
        f"{float(duration):.6f}",
        "-i",
        path,
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-f",
        "f32le",
        "-",
    ]

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        msg = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg decode failed for '{path}': {msg}")

    return proc.stdout or b""


def _frame_rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def _autocorr_f0(
    frame: np.ndarray,
    sr: int,
    f0_min: float,
    f0_max: float,
) -> Optional[float]:
    """Very lightweight F0 estimate via autocorrelation peak picking.

    Returns None for unvoiced/invalid frames.
    """

    if frame.size < 4:
        return None

    x = frame.astype(np.float64)
    x = x - np.mean(x)

    energy = float(np.dot(x, x))
    if energy <= 1e-10:
        return None

    # Autocorrelation via FFT for speed.
    n = int(2 ** math.ceil(math.log2(x.size * 2)))
    X = np.fft.rfft(x, n=n)
    ac = np.fft.irfft(np.abs(X) ** 2)
    ac = ac[: x.size]

    # Convert pitch range to lag range.
    lag_min = int(max(1, math.floor(sr / float(f0_max))))
    lag_max = int(min(x.size - 1, math.ceil(sr / float(f0_min))))
    if lag_max <= lag_min:
        return None

    # Ignore lag 0; find best peak.
    region = ac[lag_min:lag_max]
    if region.size == 0:
        return None

    peak_idx = int(np.argmax(region))
    lag = lag_min + peak_idx
    if lag <= 0:
        return None

    # Simple voicing check: peak prominence vs lag0
    peak = float(ac[lag])
    if peak <= 0:
        return None

    # Require some periodicity.
    if peak / float(ac[0] + 1e-12) < 0.1:
        return None

    return float(sr / lag)


def prosody_summary(
    *,
    audio_path: str,
    start_time: float,
    end_time: float,
    cfg: Optional[ProsodyConfig] = None,
) -> Dict:
    cfg = cfg or ProsodyConfig()

    duration = float(end_time) - float(start_time)
    if duration <= 0.0:
        return {
            "duration": 0.0,
            "frames": 0,
            "voiced_ratio": None,
            "rms_median": None,
            "rms_iqr": None,
            "f0_median_hz": None,
            "f0_iqr_hz": None,
            "f0_min_hz": None,
            "f0_max_hz": None,
        }

    raw = _ffmpeg_decode_segment_f32le_mono(
        audio_path,
        sample_rate=cfg.sample_rate,
        start_time=float(start_time),
        duration=float(duration),
    )
    if not raw:
        return {
            "duration": duration,
            "frames": 0,
            "voiced_ratio": None,
            "rms_median": None,
            "rms_iqr": None,
            "f0_median_hz": None,
            "f0_iqr_hz": None,
            "f0_min_hz": None,
            "f0_max_hz": None,
        }

    samples = np.frombuffer(raw, dtype=np.float32)
    if samples.size == 0:
        return {
            "duration": duration,
            "frames": 0,
            "voiced_ratio": None,
            "rms_median": None,
            "rms_iqr": None,
            "f0_median_hz": None,
            "f0_iqr_hz": None,
            "f0_min_hz": None,
            "f0_max_hz": None,
        }

    frame_len = max(1, int(round(cfg.sample_rate * cfg.frame_ms / 1000.0)))
    hop_len = max(1, int(round(cfg.sample_rate * cfg.hop_ms / 1000.0)))

    rms_vals = []
    f0_vals = []

    # Precompute voiced threshold from whole-segment RMS distribution.
    # First pass: RMS
    pos = 0
    while pos + frame_len <= samples.size:
        frame = samples[pos:pos + frame_len]
        rms_vals.append(_frame_rms(frame))
        pos += hop_len

    if not rms_vals:
        return {
            "duration": duration,
            "frames": 0,
            "voiced_ratio": None,
            "rms_median": None,
            "rms_iqr": None,
            "f0_median_hz": None,
            "f0_iqr_hz": None,
            "f0_min_hz": None,
            "f0_max_hz": None,
        }

    rms_arr = np.asarray(rms_vals, dtype=np.float32)
    rms_med = float(np.median(rms_arr))
    thr = max(float(cfg.abs_threshold), float(rms_med * cfg.median_ratio))
    voiced_mask = rms_arr >= thr

    # Second pass: F0 on voiced frames only
    pos = 0
    frame_index = 0
    window = np.hanning(frame_len).astype(np.float32)
    while pos + frame_len <= samples.size and frame_index < voiced_mask.size:
        if voiced_mask[frame_index]:
            frame = samples[pos:pos + frame_len] * window
            f0 = _autocorr_f0(
                frame,
                cfg.sample_rate,
                cfg.f0_min_hz,
                cfg.f0_max_hz,
            )
            if f0 is not None and cfg.f0_min_hz <= f0 <= cfg.f0_max_hz:
                f0_vals.append(float(f0))
        pos += hop_len
        frame_index += 1

    def _iqr(a: np.ndarray) -> Optional[float]:
        if a.size == 0:
            return None
        q75, q25 = np.percentile(a, [75, 25])
        return float(q75 - q25)

    rms_iqr = _iqr(rms_arr.astype(np.float64))
    voiced_ratio = float(np.mean(voiced_mask.astype(np.float32)))

    f0_arr = np.asarray(f0_vals, dtype=np.float32)

    return {
        "duration": duration,
        "frames": int(rms_arr.size),
        "voiced_ratio": voiced_ratio,
        "rms_median": float(np.median(rms_arr)),
        "rms_iqr": rms_iqr,
        "f0_median_hz": float(np.median(f0_arr)) if f0_arr.size else None,
        "f0_iqr_hz": _iqr(f0_arr.astype(np.float64)) if f0_arr.size else None,
        "f0_min_hz": float(np.min(f0_arr)) if f0_arr.size else None,
        "f0_max_hz": float(np.max(f0_arr)) if f0_arr.size else None,
        "voiced_threshold": float(thr),
    }


def prosody_similarity(a: Dict, b: Dict) -> Dict:
    """Compute a simple [0,1] similarity score from prosody summaries."""

    def rel_diff(x: Optional[float], y: Optional[float]) -> Optional[float]:
        if x is None or y is None:
            return None
        denom = abs(float(x)) + abs(float(y)) + 1e-6
        return abs(float(x) - float(y)) / denom

    diffs = {
        "voiced_ratio": rel_diff(a.get("voiced_ratio"), b.get("voiced_ratio")),
        "rms_median": rel_diff(a.get("rms_median"), b.get("rms_median")),
        "rms_iqr": rel_diff(a.get("rms_iqr"), b.get("rms_iqr")),
        "f0_median_hz": rel_diff(a.get("f0_median_hz"), b.get("f0_median_hz")),
        "f0_iqr_hz": rel_diff(a.get("f0_iqr_hz"), b.get("f0_iqr_hz")),
    }

    valid = [d for d in diffs.values() if d is not None]
    if not valid:
        return {
            "status": "unavailable",
            "score": None,
            "relative_diffs": diffs,
        }

    mean_diff = float(np.mean(np.asarray(valid, dtype=np.float32)))
    score = float(max(0.0, 1.0 - mean_diff))

    return {
        "status": "ok",
        "score": score,
        "relative_diffs": diffs,
    }
