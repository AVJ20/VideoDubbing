from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .vad import VADConfig, voiced_frames
from .prosody import ProsodyConfig, prosody_similarity, prosody_summary
from .speaker_similarity import (
    SpeakerSimConfig,
    cosine_similarity,
    speaker_embedding,
)


_WORD_RE = re.compile(r"[\w']+", flags=re.UNICODE)


def _count_words(text: str) -> int:
    if not text:
        return 0
    return len(_WORD_RE.findall(text))


def _seg_get(seg, key: str, default=None):
    if isinstance(seg, dict):
        return seg.get(key, default)
    return getattr(seg, key, default)


def _seg_id(seg) -> int:
    return int(_seg_get(seg, "id", 0) or 0)


def _seg_start(seg) -> float:
    return float(_seg_get(seg, "start_time", 0.0) or 0.0)


def _seg_end(seg) -> float:
    return float(_seg_get(seg, "end_time", 0.0) or 0.0)


def _seg_text(seg) -> str:
    return str(_seg_get(seg, "text", "") or "")


def _seg_speaker(seg):
    return _seg_get(seg, "speaker", None)


def _slice_frames(
    frames: List[bool],
    *,
    start_time: float,
    end_time: float,
    hop_seconds: float,
) -> List[bool]:
    if end_time <= start_time:
        return []
    start_idx = int(max(0, int(start_time / hop_seconds)))
    end_idx = int(max(start_idx, int(end_time / hop_seconds)))
    end_idx = min(end_idx, len(frames))
    return frames[start_idx:end_idx]


def _voiced_bounds(frames: List[bool]) -> Dict[str, Optional[int]]:
    """Return first/last voiced frame indices within the given frame list."""
    if not frames:
        return {"first": None, "last": None}
    first = None
    last = None
    for i, v in enumerate(frames):
        if v:
            first = i
            break
    for i in range(len(frames) - 1, -1, -1):
        if frames[i]:
            last = i
            break
    return {"first": first, "last": last}


def _boundary_alignment_metrics(
    *,
    start_time: float,
    end_time: float,
    hop_seconds: float,
    src_slice: List[bool],
    dub_slice: List[bool],
) -> Dict[str, Optional[float]]:
    """Compute start/end boundary alignment between source and dubbed speech.

    Uses VAD voiced frames inside the segment window to estimate:
    - speech_start_time = start_time + leading_silence
    - speech_end_time = end_time - trailing_silence
    Then reports deltas: dubbed - source (seconds).
    """
    seg_len = max(0.0, float(end_time) - float(start_time))
    src_bounds = _voiced_bounds(src_slice)
    dub_bounds = _voiced_bounds(dub_slice)

    def _speech_start(first_idx: Optional[int]) -> Optional[float]:
        if first_idx is None:
            return None
        return float(start_time) + float(first_idx) * float(hop_seconds)

    def _speech_end(last_idx: Optional[int], n_frames: int) -> Optional[float]:
        if last_idx is None or n_frames <= 0:
            return None
        trailing_frames = max(0, (n_frames - 1) - int(last_idx))
        return float(end_time) - float(trailing_frames) * float(hop_seconds)

    src_speech_start = _speech_start(src_bounds["first"])
    dub_speech_start = _speech_start(dub_bounds["first"])

    src_speech_end = _speech_end(src_bounds["last"], len(src_slice))
    dub_speech_end = _speech_end(dub_bounds["last"], len(dub_slice))

    start_delta = (
        float(dub_speech_start) - float(src_speech_start)
        if (dub_speech_start is not None and src_speech_start is not None)
        else None
    )
    end_delta = (
        float(dub_speech_end) - float(src_speech_end)
        if (dub_speech_end is not None and src_speech_end is not None)
        else None
    )

    # A rough, segment-local measure of how much the estimated voiced region
    # is shifted, independent of segment duration.
    return {
        "segment_duration": seg_len,
        "source_speech_start_time": src_speech_start,
        "dubbed_speech_start_time": dub_speech_start,
        "start_delta_seconds": start_delta,
        "abs_start_delta_seconds": (
            abs(start_delta) if start_delta is not None else None
        ),
        "source_speech_end_time": src_speech_end,
        "dubbed_speech_end_time": dub_speech_end,
        "end_delta_seconds": end_delta,
        "abs_end_delta_seconds": (
            abs(end_delta) if end_delta is not None else None
        ),
    }


def _voiced_overlap_metrics(
    src: List[bool],
    dub: List[bool],
) -> Dict[str, Optional[float]]:
    if not src and not dub:
        return {
            "voiced_iou": None,
            "source_voiced_ratio": None,
            "dubbed_voiced_ratio": None,
            "voiced_intersection_ratio": None,
        }

    n = min(len(src), len(dub))
    if n <= 0:
        return {
            "voiced_iou": None,
            "source_voiced_ratio": None,
            "dubbed_voiced_ratio": None,
            "voiced_intersection_ratio": None,
        }

    src_arr = src[:n]
    dub_arr = dub[:n]

    src_voiced = sum(1 for v in src_arr if v)
    dub_voiced = sum(1 for v in dub_arr if v)
    inter = sum(1 for a, b in zip(src_arr, dub_arr) if a and b)
    union = sum(1 for a, b in zip(src_arr, dub_arr) if a or b)

    return {
        "voiced_iou": (inter / union) if union > 0 else None,
        "source_voiced_ratio": src_voiced / n,
        "dubbed_voiced_ratio": dub_voiced / n,
        "voiced_intersection_ratio": inter / n,
    }


@dataclass
class MetricsConfig:
    vad: VADConfig = VADConfig()
    speaker: SpeakerSimConfig = SpeakerSimConfig()
    prosody: ProsodyConfig = ProsodyConfig()


def compute_dubbing_metrics(
    *,
    source_audio_path: str,
    dubbed_audio_path: str,
    segments: List,
    translated_segments: List[dict],
    output_dir: str,
    cfg: Optional[MetricsConfig] = None,
) -> Dict:
    """Compute per-segment and overall evaluation metrics.

    Args:
        source_audio_path: extracted source WAV (timeline 0-based)
        dubbed_audio_path: final dubbed WAV aligned to the same timeline
        segments: list of Segment objects or dicts (from segments.json)
        translated_segments: list of dicts created by translation stage
        output_dir: directory where a `metrics/` folder will be created
    """

    cfg = cfg or MetricsConfig()

    os.makedirs(output_dir, exist_ok=True)
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Build fast lookup for translated text.
    translated_by_id: Dict[int, dict] = {}
    for t in translated_segments or []:
        if t.get("id") is None:
            continue
        try:
            translated_by_id[int(t["id"])] = t
        except Exception:
            continue

    src_voiced, src_vad_meta = voiced_frames(source_audio_path, cfg.vad)
    dub_voiced, dub_vad_meta = voiced_frames(dubbed_audio_path, cfg.vad)
    hop_seconds = float(
        src_vad_meta.get("hop_seconds") or cfg.vad.hop_ms / 1000.0
    )

    src_list = src_voiced.tolist()
    dub_list = dub_voiced.tolist()

    segment_metrics: List[dict] = []

    segments_dir = os.path.join(output_dir, "segments")
    refs_dir = os.path.join(segments_dir, "_refs")
    speaker_scores: List[float] = []
    emotion_scores: List[float] = []
    translation_scores: List[float] = []
    start_delta_abs: List[float] = []
    end_delta_abs: List[float] = []

    for seg in segments:
        seg_id = _seg_id(seg)
        start = _seg_start(seg)
        end = _seg_end(seg)
        duration = max(0.0, end - start)

        src_slice = _slice_frames(
            src_list,
            start_time=start,
            end_time=end,
            hop_seconds=hop_seconds,
        )
        dub_slice = _slice_frames(
            dub_list,
            start_time=start,
            end_time=end,
            hop_seconds=hop_seconds,
        )

        overlap = _voiced_overlap_metrics(src_slice, dub_slice)

        boundary_alignment = _boundary_alignment_metrics(
            start_time=start,
            end_time=end,
            hop_seconds=hop_seconds,
            src_slice=src_slice,
            dub_slice=dub_slice,
        )
        if boundary_alignment.get("abs_start_delta_seconds") is not None:
            start_delta_abs.append(
                float(boundary_alignment["abs_start_delta_seconds"])
            )
        if boundary_alignment.get("abs_end_delta_seconds") is not None:
            end_delta_abs.append(
                float(boundary_alignment["abs_end_delta_seconds"])
            )

        src_words = _count_words(_seg_text(seg))
        tgt_text = (
            ((translated_by_id.get(seg_id) or {}).get("text") or "").strip()
        )
        tgt_words = _count_words(tgt_text)

        tmeta = translated_by_id.get(seg_id) or {}
        translation_quality = tmeta.get("translation_quality")
        if not isinstance(translation_quality, dict):
            translation_quality = {
                "status": "stub",
                "reason": "No reference translation/LLM judge configured yet",
                "length_ratio": (
                    (tgt_words / src_words) if src_words > 0 else None
                ),
            }
        else:
            # If an LLM quality score is present, aggregate it.
            try:
                s = translation_quality.get("score")
                if s is not None:
                    translation_scores.append(float(s))
            except Exception:
                pass

        # Speaker similarity (compare dubbed segment audio
        # to its voice prompt reference).
        speaker_id = _seg_speaker(seg)
        dubbed_segment_path = os.path.join(
            segments_dir,
            f"segment_{seg_id:04d}.wav",
        )
        reference_path = None
        if speaker_id:
            reference_path = os.path.join(
                refs_dir,
                f"seg_{seg_id:04d}_{speaker_id}_ref.wav",
            )

        speaker_similarity = {
            "status": "skipped",
            "model": cfg.speaker.model_id,
            "reference_audio": reference_path,
            "dubbed_segment_audio": dubbed_segment_path,
            "cosine_similarity": None,
            "reason": None,
        }

        if not os.path.exists(dubbed_segment_path):
            speaker_similarity["reason"] = "Dubbed segment wav not found"
        elif not reference_path or not os.path.exists(reference_path):
            speaker_similarity["reason"] = (
                "Reference (voice prompt) wav not found"
            )
        else:
            try:
                ref_emb = speaker_embedding(
                    reference_path,
                    model_id=cfg.speaker.model_id,
                    sample_rate=cfg.speaker.sample_rate,
                    device=cfg.speaker.device,
                )
                dub_emb = speaker_embedding(
                    dubbed_segment_path,
                    model_id=cfg.speaker.model_id,
                    sample_rate=cfg.speaker.sample_rate,
                    device=cfg.speaker.device,
                )
                score = cosine_similarity(ref_emb, dub_emb)
                speaker_similarity["status"] = "ok"
                speaker_similarity["cosine_similarity"] = float(score)
                speaker_scores.append(float(score))
            except Exception as e:
                speaker_similarity["status"] = "failed"
                speaker_similarity["reason"] = str(e)

        # Emotion/affect similarity (prosody-based, model-free)
        try:
            src_prosody = prosody_summary(
                audio_path=source_audio_path,
                start_time=start,
                end_time=end,
                cfg=cfg.prosody,
            )
            dub_prosody = prosody_summary(
                audio_path=dubbed_audio_path,
                start_time=start,
                end_time=end,
                cfg=cfg.prosody,
            )
            emo_sim = prosody_similarity(src_prosody, dub_prosody)
            if (
                emo_sim.get("status") == "ok"
                and emo_sim.get("score") is not None
            ):
                emotion_scores.append(float(emo_sim["score"]))
            emotion_quality = {
                "status": (
                    "ok" if emo_sim.get("status") == "ok" else "unavailable"
                ),
                "method": "prosody",
                "source": src_prosody,
                "dubbed": dub_prosody,
                "similarity": emo_sim,
            }
        except Exception as e:
            emotion_quality = {
                "status": "failed",
                "method": "prosody",
                "error": str(e),
            }

        segment_metrics.append(
            {
                "segment_id": seg_id,
                "speaker": speaker_id,
                "start_time": start,
                "end_time": end,
                "duration": duration,
                "source_text": _seg_text(seg).strip(),
                "translated_text": tgt_text,
                "isochrony": overlap,
                "boundary_alignment": boundary_alignment,
                "speech_rate": {
                    "source_words": src_words,
                    "target_words": tgt_words,
                    "source_wps": (
                        (src_words / duration) if duration > 0 else None
                    ),
                    "target_wps": (
                        (tgt_words / duration) if duration > 0 else None
                    ),
                    "source_wpm": (
                        (src_words / duration) * 60.0
                        if duration > 0
                        else None
                    ),
                    "target_wpm": (
                        (tgt_words / duration) * 60.0
                        if duration > 0
                        else None
                    ),
                },
                "translation_quality": translation_quality,
                "speaker_similarity": speaker_similarity,
                "emotion_quality": emotion_quality,
            }
        )

    # Overall metrics.
    total_end = 0.0
    if segments:
        total_end = max(_seg_end(s) for s in segments)

    src_all = _slice_frames(
        src_list,
        start_time=0.0,
        end_time=total_end,
        hop_seconds=hop_seconds,
    )
    dub_all = _slice_frames(
        dub_list,
        start_time=0.0,
        end_time=total_end,
        hop_seconds=hop_seconds,
    )

    overall_overlap = _voiced_overlap_metrics(src_all, dub_all)

    overall_boundary_alignment = _boundary_alignment_metrics(
        start_time=0.0,
        end_time=total_end,
        hop_seconds=hop_seconds,
        src_slice=src_all,
        dub_slice=dub_all,
    )

    all_src_words = sum(_count_words(_seg_text(s).strip()) for s in segments)
    all_tgt_words = sum(
        _count_words(
            (
                (translated_by_id.get(_seg_id(s)) or {}).get("text")
                or ""
            ).strip()
        )
        for s in segments
    )

    overall = {
        "duration": total_end,
        "isochrony": overall_overlap,
        "boundary_alignment": {
            "overall": overall_boundary_alignment,
            "segments_measured": int(
                max(len(start_delta_abs), len(end_delta_abs))
            ),
            "mean_abs_start_delta_seconds": (
                float(sum(start_delta_abs) / len(start_delta_abs))
                if start_delta_abs
                else None
            ),
            "max_abs_start_delta_seconds": (
                float(max(start_delta_abs)) if start_delta_abs else None
            ),
            "mean_abs_end_delta_seconds": (
                float(sum(end_delta_abs) / len(end_delta_abs))
                if end_delta_abs
                else None
            ),
            "max_abs_end_delta_seconds": (
                float(max(end_delta_abs)) if end_delta_abs else None
            ),
        },
        "speech_rate": {
            "source_words": all_src_words,
            "target_words": all_tgt_words,
            "source_wps": (
                (all_src_words / total_end) if total_end > 0 else None
            ),
            "target_wps": (
                (all_tgt_words / total_end) if total_end > 0 else None
            ),
            "source_wpm": (
                (all_src_words / total_end) * 60.0
                if total_end > 0
                else None
            ),
            "target_wpm": (
                (all_tgt_words / total_end) * 60.0
                if total_end > 0
                else None
            ),
        },
        "translation_quality": {
            "status": "ok" if translation_scores else "stub",
            "method": "llm" if translation_scores else "none",
            "segments_scored": int(len(translation_scores)),
            "mean_score": (
                float(sum(translation_scores) / len(translation_scores))
                if translation_scores
                else None
            ),
            "reason": (
                None
                if translation_scores
                else "No reference translation/LLM judge configured yet"
            ),
        },
        "speaker_similarity": {
            "status": "ok" if speaker_scores else "unavailable",
            "model": cfg.speaker.model_id,
            "segments_scored": int(len(speaker_scores)),
            "mean_cosine_similarity": float(
                sum(speaker_scores) / len(speaker_scores)
            )
            if speaker_scores
            else None,
        },
        "emotion_quality": {
            "status": "ok" if emotion_scores else "unavailable",
            "method": "prosody",
            "segments_scored": int(len(emotion_scores)),
            "mean_similarity": float(sum(emotion_scores) / len(emotion_scores))
            if emotion_scores
            else None,
        },
    }

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "source_audio_path": source_audio_path,
            "dubbed_audio_path": dubbed_audio_path,
        },
        "vad": {
            "source": src_vad_meta,
            "dubbed": dub_vad_meta,
        },
        "segment_metrics": segment_metrics,
        "overall": overall,
    }

    out_path = os.path.join(metrics_dir, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        import json

        json.dump(report, f, indent=2, ensure_ascii=False)

    return {
        "metrics_dir": metrics_dir,
        "metrics_json": out_path,
        "summary": {
            "segments": len(segment_metrics),
            "duration": total_end,
        },
    }
