from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from .runner import MetricsConfig, compute_dubbing_metrics


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def run_metrics_from_output_dir(output_dir: str) -> Dict:
    output_dir = os.path.abspath(output_dir)

    meta_path = os.path.join(output_dir, "pipeline_metadata.json")
    segments_path = os.path.join(output_dir, "segments.json")
    translated_path = os.path.join(output_dir, "translated_segments.json")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing pipeline metadata: {meta_path}")
    if not os.path.exists(segments_path):
        raise FileNotFoundError(f"Missing segments.json: {segments_path}")
    if not os.path.exists(translated_path):
        # Allow running without translations (speech rate still works on source text).
        translated = []
    else:
        translated = _read_json(translated_path)

    meta = _read_json(meta_path)
    segments = _read_json(segments_path)

    stages = (meta or {}).get("stages", {})

    source_audio_path = _find_first_existing(
        [
            (stages.get("audio_extraction") if isinstance(stages, dict) else None),
            (meta.get("audio_path") if isinstance(meta, dict) else None),
        ]
    )

    dubbed_audio_path = _find_first_existing(
        [
            os.path.join(output_dir, "dubbed_audio.wav"),
            (stages.get("output", {}) or {}).get("dubbed_audio")
            if isinstance(stages, dict)
            else None,
        ]
    )

    if not source_audio_path:
        raise FileNotFoundError(
            "Could not locate source audio path from pipeline_metadata.json"
        )
    if not dubbed_audio_path:
        raise FileNotFoundError(
            "Could not locate dubbed audio (expected dubbed_audio.wav in output_dir)"
        )

    return compute_dubbing_metrics(
        source_audio_path=source_audio_path,
        dubbed_audio_path=dubbed_audio_path,
        segments=segments,
        translated_segments=translated,
        output_dir=output_dir,
        cfg=MetricsConfig(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute metrics from an existing pipeline output directory"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Enhanced pipeline output directory (contains pipeline_metadata.json)",
    )

    args = parser.parse_args()
    info = run_metrics_from_output_dir(args.output_dir)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
