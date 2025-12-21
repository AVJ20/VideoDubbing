from __future__ import annotations

import os
from typing import Optional

from ..audio import extract_video_clip


def extract_video_segment(
    input_video_path: str,
    output_video_path: str,
    *,
    start_time_seconds: float,
    end_time_seconds: Optional[float] = None,
    duration_seconds: Optional[float] = None,
) -> str:
    """Extract a small segment from a video using ffmpeg.

    This is a convenience wrapper around `src.audio.extract_video_clip()` that also
    supports specifying `duration_seconds` instead of `end_time_seconds`.

    Args:
        input_video_path: Path to the source video file.
        output_video_path: Path where the extracted segment should be written.
        start_time_seconds: Segment start time in seconds.
        end_time_seconds: Optional segment end time in seconds.
        duration_seconds: Optional segment duration in seconds. Mutually exclusive
            with `end_time_seconds`.

    Returns:
        The `output_video_path`.

    Raises:
        ValueError: If both `end_time_seconds` and `duration_seconds` are provided.
        RuntimeError: If ffmpeg fails.
        FileNotFoundError: If input video does not exist.
    """

    if end_time_seconds is not None and duration_seconds is not None:
        raise ValueError("Provide either end_time_seconds or duration_seconds, not both")

    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    end_time: Optional[float] = end_time_seconds
    if end_time is None and duration_seconds is not None:
        end_time = float(start_time_seconds) + max(0.0, float(duration_seconds))

    return extract_video_clip(
        input_video_path,
        output_video_path,
        start_time=float(start_time_seconds),
        end_time=None if end_time is None else float(end_time),
    )
