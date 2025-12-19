import os
import subprocess
import logging
from typing import Optional

import contextlib
import wave

logger = logging.getLogger(__name__)


def get_wav_duration_seconds(wav_path: str) -> Optional[float]:
    """Return WAV duration in seconds, or None if unknown/unreadable."""
    try:
        with contextlib.closing(wave.open(wav_path, "rb")) as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            if not rate:
                return None
            return float(frames) / float(rate)
    except Exception:
        return None


def extract_audio(video_path: str, out_audio_path: str, sample_rate: int = 16000) -> str:
    """Extract audio from video using ffmpeg. Produces WAV with specified sample rate.

    Requires ffmpeg to be installed and on PATH.
    """
    os.makedirs(os.path.dirname(out_audio_path) or ".", exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        out_audio_path,
    ]
    logger.info("Extracting audio: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        logger.error("ffmpeg failed: %s", proc.stderr)
        raise RuntimeError(f"ffmpeg failed: {proc.stderr}")
    return out_audio_path


def extract_audio_clip(
    audio_path: str,
    out_audio_path: str,
    start_time: float,
    end_time: float,
    sample_rate: Optional[int] = None,
) -> str:
    """Extract a clip from an audio file using ffmpeg.

    Args:
        audio_path: Input audio path (wav recommended)
        out_audio_path: Output clip path
        start_time: Clip start time in seconds
        end_time: Clip end time in seconds
        sample_rate: Optional sample rate for the output clip

    Requires ffmpeg to be installed and on PATH.
    """

    os.makedirs(os.path.dirname(out_audio_path) or ".", exist_ok=True)

    safe_start = max(0.0, float(start_time))
    safe_end = max(safe_start, float(end_time))
    duration = max(0.05, safe_end - safe_start)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(safe_start),
        "-t",
        str(duration),
        "-i",
        audio_path,
        "-vn",
        "-ac",
        "1",
    ]
    if sample_rate:
        cmd.extend(["-ar", str(int(sample_rate))])
    cmd.append(out_audio_path)

    logger.info("Extracting audio clip: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        logger.error("ffmpeg clip extraction failed: %s", proc.stderr)
        raise RuntimeError(f"ffmpeg clip extraction failed: {proc.stderr}")

    return out_audio_path
