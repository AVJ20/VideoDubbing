import os
import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)


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
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("ffmpeg failed: %s", proc.stderr)
        raise RuntimeError(f"ffmpeg failed: {proc.stderr}")
    return out_audio_path
