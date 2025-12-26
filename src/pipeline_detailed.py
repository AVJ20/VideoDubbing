"""
Detailed Video Dubbing Pipeline

Integrates:
1. Audio Segmentation (logical + speaker-based)
2. Alignment (timing synchronization)
3. Speaker-specific TTS (voice cloning)
4. Translation

Workflow:
1. Extract audio and transcribe with speaker diarization
2. Segment audio based on ASR boundaries and speaker changes
3. Register speaker profiles with voice cloning references
4. Translate segments (preserves speaker metadata)
5. Synthesize dubbed audio with speaker-specific voices
6. Align timing between source and target
7. Generate final dubbed audio and metadata
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import json
import subprocess
import sys
from datetime import datetime, timezone

from .audio import extract_audio, get_audio_duration_seconds
from .asr import AbstractASR, WhisperWithDiarizationASR, ASRResult
from .segmentation import AudioSegmenter, SegmentationResult, SegmentationValidator, Segment
from .alignment import SegmentAligner, TimingAnalyzer, AlignmentStrategy
from .speaker_tts import (
    SpeakerTTSOrchestrator,
    AbstractSpeakerTTS,
    ChatterboxSpeakerTTS,
    ElevenLabsSpeakerTTS,
    create_speaker_profiles_from_segments
)
from .translator import (
    AbstractTranslator,
    GroqTranslator,
    OpenAITranslator,
    AzureOpenAITranslator,
)
from .cache import ensure_dir, load_json, save_json_atomic, stable_hash_dict

logger = logging.getLogger(__name__)


@dataclass
class DetailedPipelineConfig:
    """Configuration for detailed dubbing pipeline."""
    
    work_dir: str = "work"
    output_dir: str = "output"
    sample_rate: int = 16000
    
    # Segmentation
    min_segment_duration: float = 0.5
    speaker_change_threshold: float = 0.1
    
    # Alignment
    alignment_strategy: AlignmentStrategy = AlignmentStrategy.STRICT
    max_timing_drift: float = 2.0
    
    # TTS
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts_device: str = "cpu"
    preserve_speaker_identity: bool = True
    
    # Pipeline
    debug: bool = False

    # Metrics
    compute_metrics: bool = False


@dataclass
class PipelineState:
    """Tracks state throughout pipeline execution."""
    
    stage: str = "init"
    video_path: str = ""
    audio_path: str = ""
    source_language: str = "en"
    target_language: str = "en"
    
    # Results at each stage
    asr_result: Optional[ASRResult] = None
    segmentation_result: Optional[SegmentationResult] = None
    alignment_results: Optional[List] = None
    tts_results: Optional[List] = None
    
    # Metadata
    metadata: Dict = field(default_factory=dict)


class DetailedDubbingPipeline:
    """
    Comprehensive video dubbing pipeline with detailed component support.
    
    Features:
    - Speaker diarization
    - Logical + speaker-based segmentation
    - Voice cloning and speaker-specific TTS
    - Timing alignment
    - Translation with speaker preservation
    """
    
    def __init__(self,
                 asr: Optional[AbstractASR] = None,
                 translator: Optional[AbstractTranslator] = None,
                 tts: Optional[AbstractSpeakerTTS] = None,
                 config: Optional[DetailedPipelineConfig] = None):
        """
        Initialize detailed dubbing pipeline.
        
        Args:
            asr: ASR backend (defaults to WhisperWithDiarization)
            translator: Translation backend (defaults to Groq)
            tts: Speaker TTS backend (defaults to Chatterbox)
            config: Pipeline configuration
        """
        self.config = config or DetailedPipelineConfig()
        self.state = PipelineState()
        
        # Configure logging based on debug flag
        if self.config.debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logger.debug("Debug logging enabled")
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
        
        # Initialize components
        if asr is None:
            try:
                self.asr = WhisperWithDiarizationASR(whisper_model="base", 
                                                     device=self.config.tts_device)
                logger.info("Using WhisperWithDiarizationASR with speaker diarization")
            except (RuntimeError, ImportError) as e:
                logger.warning(f"Could not load WhisperWithDiarizationASR: {e}")
                raise RuntimeError("WhisperWithDiarizationASR required for detailed pipeline")
        else:
            self.asr = asr
        
        if translator is not None:
            self.translator = translator
        else:
            if (
                os.environ.get("AZURE_OPENAI_API_KEY")
                and os.environ.get("AZURE_OPENAI_ENDPOINT")
                and os.environ.get("AZURE_OPENAI_DEPLOYMENT")
            ):
                self.translator = AzureOpenAITranslator()
            elif os.environ.get("OPENAI_API_KEY"):
                model = (
                    os.environ.get("OPENAI_TRANSLATION_MODEL")
                    or "gpt-4o-mini"
                )
                self.translator = OpenAITranslator(model=model)
            else:
                self.translator = GroqTranslator()
        
        if tts is None:
            # Use Chatterbox TTS (free, open-source API)
            try:
                self.tts = ChatterboxSpeakerTTS(
                    device=self.config.tts_device
                )
                logger.info("Using ChatterboxSpeakerTTS (free cloud API)")
            except RuntimeError as e:
                logger.error(f"Chatterbox TTS initialization failed: {e}")
                # Fallback to ElevenLabs if available
                try:
                    self.tts = ElevenLabsSpeakerTTS(
                        device=self.config.tts_device
                    )
                    logger.info("Fallback: Using ElevenLabsSpeakerTTS")
                except RuntimeError:
                    raise RuntimeError(
                        "No TTS backend available. "
                        "Ensure network connection or set ELEVENLABS_API_KEY for fallback."
                    )
        else:
            self.tts = tts
        
        # Initialize processors
        self.segmenter = AudioSegmenter(
            min_segment_duration=self.config.min_segment_duration,
            speaker_change_threshold=self.config.speaker_change_threshold
        )
        
        self.aligner = SegmentAligner(
            strategy=self.config.alignment_strategy
        )
        
        self.tts_orchestrator = SpeakerTTSOrchestrator(self.tts)
        self.timing_analyzer = TimingAnalyzer()
        
        # Create output directories
        os.makedirs(self.config.work_dir, exist_ok=True)
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def run(self, 
            video_path: str,
            source_lang: str = "en",
            target_lang: str = "es",
            speaker_reference_audio: Optional[Dict[str, str]] = None) -> Dict:
        """
        Run complete dubbing pipeline.
        
        Args:
            video_path: Path to video file
            source_lang: Source language code
            target_lang: Target language code
            speaker_reference_audio: Dict mapping speaker_id to reference audio path
                                    for voice cloning (optional)
        
        Returns:
            Dict with pipeline results and metadata
        """
        self.state.source_language = source_lang
        self.state.target_language = target_lang
        self.state.video_path = video_path
        
        result = {
            "source_language": source_lang,
            "target_language": target_lang,
            "video_path": video_path,
            "stages": {}
        }
        
        try:
            # Stage 1: Extract audio
            self.state.stage = "audio_extraction"
            audio_path = self._extract_audio(video_path)
            self.state.audio_path = audio_path
            result["stages"]["audio_extraction"] = audio_path
            
            # Stage 2: ASR with diarization
            self.state.stage = "transcription"
            asr_result = self._transcribe(audio_path, source_lang)
            self.state.asr_result = asr_result
            result["stages"]["transcription"] = {
                "text": asr_result.text,
                "segments": len(asr_result.segments),
                "speakers": list(set(s.get("speaker", "Unknown") for s in asr_result.segments))
            }
            
            # Stage 3: Segmentation
            self.state.stage = "segmentation"
            segmentation_result = self._segment(asr_result)
            self.state.segmentation_result = segmentation_result
            result["stages"]["segmentation"] = {
                "segments": len(segmentation_result.segments),
                "speakers": segmentation_result.speakers,
                "total_duration": segmentation_result.total_duration
            }
            
            # Stage 4: Register speaker profiles
            self.state.stage = "speaker_registration"
            self._register_speakers(segmentation_result.segments, speaker_reference_audio)
            result["stages"]["speaker_registration"] = {
                "registered_speakers": list(self.tts_orchestrator.speaker_profiles.keys())
            }
            
            # Stage 5: Translation
            self.state.stage = "translation"
            translated_segments = self._translate_segments(
                segmentation_result.segments,
                source_lang,
                target_lang
            )
            result["stages"]["translation"] = {
                "segments_translated": len(translated_segments)
            }

            # Persist segment-level artifacts for evaluation/debugging.
            try:
                artifact_paths = self._save_segment_level_artifacts(
                    asr_result=asr_result,
                    segments=segmentation_result.segments,
                    translated_segments=translated_segments,
                )
                result["stages"]["segment_artifacts"] = artifact_paths
            except Exception as e:
                logger.warning(
                    "Failed to save segment-level artifacts: %s",
                    str(e),
                    exc_info=True,
                )
            
            # Stage 6: TTS synthesis
            self.state.stage = "tts_synthesis"
            tts_results, timing_info = self._synthesize_dubbed_audio(
                translated_segments,
                target_lang
            )
            self.state.tts_results = tts_results
            result["stages"]["tts_synthesis"] = timing_info
            
            # Stage 7: Alignment
            self.state.stage = "alignment"
            target_durations = self.tts_orchestrator.get_segment_durations()
            alignment_results = self._align_segments(
                segmentation_result.segments,
                target_durations
            )
            self.state.alignment_results = alignment_results
            timing_stats = self.timing_analyzer.analyze(alignment_results)
            result["stages"]["alignment"] = timing_stats
            
            # Stage 8: Generate final output
            self.state.stage = "output_generation"
            output_info = self._generate_output(
                segmentation_result.segments,
                tts_results,
                alignment_results
            )
            result["stages"]["output"] = output_info

            # Stage 9: Metrics (optional)
            if getattr(self.config, "compute_metrics", False):
                self.state.stage = "metrics"
                try:
                    from .metrics.runner import compute_dubbing_metrics

                    dubbed_audio_path = (output_info or {}).get("dubbed_audio")
                    if dubbed_audio_path and os.path.exists(dubbed_audio_path):
                        metrics_info = compute_dubbing_metrics(
                            source_audio_path=self.state.audio_path,
                            dubbed_audio_path=dubbed_audio_path,
                            segments=segmentation_result.segments,
                            translated_segments=translated_segments,
                            output_dir=self.config.output_dir,
                        )
                        result["stages"]["metrics"] = metrics_info
                    else:
                        result["stages"]["metrics"] = {
                            "status": "skipped",
                            "reason": (
                                "Dubbed audio missing; cannot compute metrics"
                            ),
                        }
                except Exception as e:
                    logger.warning(
                        "Failed to compute metrics: %s", str(e), exc_info=True
                    )
                    result["stages"]["metrics"] = {
                        "status": "failed",
                        "error": str(e),
                    }
            
            # Success
            self.state.stage = "complete"
            result["status"] = "success"
            logger.info("Pipeline execution complete")
            
        except Exception as e:
            self.state.stage = f"error_{self.state.stage}"
            logger.error(f"Pipeline failed at stage {self.state.stage}: {e}", exc_info=True)
            result["status"] = "failed"
            result["error"] = str(e)
        
        # Save final metadata
        self._save_metadata(result)
        
        return result

    def _save_segment_level_artifacts(
        self,
        *,
        asr_result: ASRResult,
        segments: List[Segment],
        translated_segments: List[dict],
    ) -> Dict[str, str]:
        """Write segment-level outputs used for evaluation/debugging.

        Stores:
        - Raw ASR segments (with timings + speaker if available)
        - Segmentation segments + per-segment translation
        - A combined report that links segmentation segments to overlapping
          ASR subsegments
        """

        os.makedirs(self.config.output_dir, exist_ok=True)

        clip_start_offset = float(self.state.metadata.get("clip_start", 0.0) or 0.0)
        clip_end_offset = self.state.metadata.get("clip_end", None)
        if clip_end_offset is not None:
            clip_end_offset = float(clip_end_offset)

        # Normalize ASR segments to a consistent schema.
        normalized_asr_segments: List[dict] = []
        for idx, s in enumerate(asr_result.segments or []):
            start = float(s.get("offset", s.get("start", 0.0)) or 0.0)
            duration = float(s.get("duration", 0.0) or 0.0)
            end = float(s.get("end", start + duration) or (start + duration))

            # Normalize word-level timestamps when available.
            normalized_words: List[dict] = []
            for w in (s.get("words") or []):
                if not isinstance(w, dict):
                    continue
                w_start = w.get("start_time", w.get("start"))
                w_end = w.get("end_time", w.get("end"))
                try:
                    w_start_f = float(w_start) if w_start is not None else None
                except Exception:
                    w_start_f = None
                try:
                    w_end_f = float(w_end) if w_end is not None else None
                except Exception:
                    w_end_f = None

                normalized_words.append(
                    {
                        "word": w.get("word") or w.get("text") or w.get("token") or "",
                        "start_time": w_start_f,
                        "end_time": w_end_f,
                        "start_time_global": (w_start_f + clip_start_offset) if w_start_f is not None else None,
                        "end_time_global": (w_end_f + clip_start_offset) if w_end_f is not None else None,
                        "probability": w.get("probability", w.get("prob")),
                    }
                )
            if end < start:
                end = start

            normalized_asr_segments.append(
                {
                    "index": idx,
                    "start_time": start,
                    "end_time": end,
                    "duration": max(0.0, end - start),
                    "start_time_global": start + clip_start_offset,
                    "end_time_global": end + clip_start_offset,
                    "speaker": s.get("speaker", "Unknown"),
                    "words": normalized_words,
                    "text": (s.get("text") or "").strip(),
                    "confidence": s.get("confidence", 1.0),
                }
            )

        # Index translations by segment id.
        translated_by_id: Dict[int, dict] = {
            int(t.get("id")): t for t in (translated_segments or [])
            if t.get("id") is not None
        }

        # Build per-seg report linking to overlapping ASR subsegments.
        combined_segments: List[dict] = []
        for seg in segments:
            seg_start = float(seg.start_time)
            seg_end = float(seg.end_time)
            overlap_asr = [
                a
                for a in normalized_asr_segments
                if (a["end_time"] > seg_start) and (a["start_time"] < seg_end)
            ]

            translated_entry = translated_by_id.get(int(seg.id))
            combined_segments.append(
                {
                    "id": int(seg.id),
                    "speaker": seg.speaker,
                    "start_time": seg_start,
                    "end_time": seg_end,
                    "duration": float(seg.duration),
                    "start_time_global": seg_start + clip_start_offset,
                    "end_time_global": seg_end + clip_start_offset,
                    "confidence": seg.confidence,
                    "asr_text": seg.text,
                    "translated_text": (
                        (translated_entry or {}).get("text")
                        or ""
                    ).strip(),
                    "asr_subsegments": overlap_asr,
                }
            )

        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "video_path": self.state.video_path,
            "audio_path": self.state.audio_path,
            "clip_start": clip_start_offset,
            "clip_end": clip_end_offset,
            "source_language": self.state.source_language,
            "target_language": self.state.target_language,
            "asr_segments": normalized_asr_segments,
            "segments": combined_segments,
        }

        asr_path = os.path.join(self.config.output_dir, "asr_segments.json")
        translated_path = os.path.join(
            self.config.output_dir,
            "translated_segments.json",
        )
        report_path = os.path.join(
            self.config.output_dir,
            "segment_report.json",
        )

        with open(asr_path, "w", encoding="utf-8") as f:
            json.dump(normalized_asr_segments, f, indent=2, ensure_ascii=False)
        with open(translated_path, "w", encoding="utf-8") as f:
            json.dump(translated_segments, f, indent=2, ensure_ascii=False)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return {
            "asr_segments": asr_path,
            "translated_segments": translated_path,
            "segment_report": report_path,
        }
    
    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video."""
        logger.info(f"Extracting audio from {video_path}")
        audio_path = os.path.join(
            self.config.work_dir,
            os.path.splitext(os.path.basename(video_path))[0] + ".wav"
        )
        extract_audio(video_path, audio_path, sample_rate=self.config.sample_rate)
        logger.info(f"Audio extracted to {audio_path}")
        return audio_path
    
    def _transcribe(self, audio_path: str, language: str) -> ASRResult:
        """Transcribe audio with speaker diarization."""
        logger.info(f"Transcribing audio (language: {language})")
        result = self.asr.transcribe(audio_path, language=language)
        logger.info(f"Transcription complete: {len(result.segments)} segments")
        return result
    
    def _segment(self, asr_result: ASRResult) -> SegmentationResult:
        """Segment audio based on ASR and speaker changes."""
        logger.info("Segmenting audio based on logical and speaker boundaries")
        result = self.segmenter.segment(asr_result.segments)
        
        # Validate
        is_valid, warnings = SegmentationValidator.validate(result)
        if not is_valid:
            for warning in warnings:
                logger.warning(f"Segmentation warning: {warning}")
        
        logger.info(f"Segmentation complete: {result}")
        return result
    
    def _register_speakers(self,
                          segments: List[Segment],
                          speaker_reference_audio: Optional[Dict[str, str]] = None) -> None:
        """Register speaker profiles for voice cloning."""
        logger.info("Registering speaker profiles")
        
        # Create default profiles
        profiles = create_speaker_profiles_from_segments(segments)
        
        # Update with reference audio if provided
        if speaker_reference_audio:
            for speaker_id, audio_path in speaker_reference_audio.items():
                if speaker_id in profiles:
                    profiles[speaker_id].voice_reference = audio_path
                    logger.info(f"Registered voice reference for {speaker_id}: {audio_path}")
        
        # Register all profiles
        for speaker_id, profile in profiles.items():
            success = self.tts_orchestrator.register_speaker(profile)
            if not success:
                logger.warning(f"Failed to register speaker {speaker_id}")
    
    def _translate_segments(self,
                           segments: List[Segment],
                           source_lang: str,
                           target_lang: str) -> List[dict]:
        """Translate segment text while preserving speaker metadata."""
        logger.info(f"Translating {len(segments)} segments ({source_lang} -> {target_lang})")

        # Output-dir reuse: if this output already has translations for the
        # *same* segmentation (by segment id + original_text), reuse them.
        translated_path = os.path.join(self.config.output_dir, "translated_segments.json")
        try:
            if os.path.exists(translated_path):
                existing = load_json(translated_path, default=None)
                if isinstance(existing, list) and existing:
                    by_id = {
                        int(e.get("id")): e
                        for e in existing
                        if isinstance(e, dict) and e.get("id") is not None
                    }

                    matches = True
                    for seg in segments:
                        cur_id = int(seg.id)
                        cur_src = (seg.text or "").strip()
                        prev = by_id.get(cur_id)
                        if not isinstance(prev, dict):
                            matches = False
                            break
                        prev_src = (prev.get("original_text") or "").strip()
                        if prev_src != cur_src:
                            matches = False
                            break

                    if matches:
                        logger.info(
                            "Reusing translations from output dir: %s",
                            translated_path,
                        )
                        return existing
        except Exception as e:
            logger.warning(
                "Could not reuse output-dir translations (%s): %s",
                translated_path,
                str(e),
            )

        # Persistent cache to avoid repeated paid translation calls.
        cache_root = os.path.join(self.config.work_dir, "_cache")
        ensure_dir(cache_root)
        cache_path = os.path.join(cache_root, "translation_cache.json")
        cache = load_json(cache_path, default={})
        if not isinstance(cache, dict):
            cache = {}

        translator_sig = {
            "class": self.translator.__class__.__name__,
            "model": getattr(self.translator, "model", None),
            "deployment": getattr(self.translator, "deployment_name", None),
        }

        cache_hits = 0
        cache_misses = 0
        
        translated = []
        for idx, seg in enumerate(segments):
            prev_start = max(0, idx - 5)
            next_end = min(len(segments), idx + 6)

            previous_segments = [s.text for s in segments[prev_start:idx]]
            next_segments = [s.text for s in segments[idx + 1:next_end]]

            cache_key = stable_hash_dict(
                {
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "text": seg.text or "",
                    # Note: we intentionally do NOT include context here
                    # so identical segment text reuses a prior translation.
                    "translator": translator_sig,
                }
            )

            cached_item = cache.get(cache_key)
            if isinstance(cached_item, dict) and cached_item.get("text"):
                translated_text = str(cached_item.get("text") or "").strip()
                translation_meta = cached_item
                cache_hits += 1
            else:
                cache_misses += 1
                translation_meta = None
                if hasattr(self.translator, "translate_with_context_and_quality"):
                    translation_meta = (
                        self.translator.translate_with_context_and_quality(
                            seg.text,
                            source_lang,
                            target_lang,
                            previous_segments=previous_segments,
                            next_segments=next_segments,
                        )
                        or {}
                    )
                    translated_text = (
                        str(translation_meta.get("text") or "").strip()
                    )
                elif hasattr(self.translator, "translate_with_context"):
                    translated_text = self.translator.translate_with_context(
                        seg.text,
                        source_lang,
                        target_lang,
                        previous_segments=previous_segments,
                        next_segments=next_segments,
                    )
                else:
                    translated_text = self.translator.translate(
                        seg.text,
                        source_lang,
                        target_lang,
                    )

                # Store best-effort metadata (including quality if present).
                if translation_meta is None:
                    translation_meta = {"text": translated_text}
                cache[cache_key] = translation_meta

            # Flush cache periodically to keep runs resilient.
            if (idx % 10) == 0:
                save_json_atomic(cache_path, cache)
            
            item = {
                "id": seg.id,
                "text": translated_text,
                "speaker": seg.speaker,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "original_text": seg.text,
                "duration": seg.duration,
                "confidence": seg.confidence
            }
            if translation_meta:
                if translation_meta.get("register"):
                    item["translation_register"] = translation_meta.get("register")
                if translation_meta.get("localization_notes"):
                    item["translation_localization_notes"] = translation_meta.get(
                        "localization_notes"
                    )
                if translation_meta.get("translation_quality"):
                    item["translation_quality"] = translation_meta.get(
                        "translation_quality"
                    )
            translated.append(item)

        save_json_atomic(cache_path, cache)

        logger.info(
            "Translation complete (cache hits=%s, misses=%s, cache=%s)",
            cache_hits,
            cache_misses,
            cache_path,
        )
        return translated
    
    def _synthesize_dubbed_audio(self,
                                translated_segments: List[dict],
                                target_lang: str) -> Tuple[List, Dict]:
        """Synthesize dubbed audio with speaker-specific voices."""
        logger.info(f"Synthesizing dubbed audio ({target_lang})")
        
        output_dir = os.path.join(self.config.output_dir, "segments")
        results, timing_info = self.tts_orchestrator.synthesize_segments(
            translated_segments,
            output_dir,
            language=target_lang,
            source_audio_path=self.state.audio_path,
            use_segment_audio_as_reference=True,
        )
        
        logger.info(f"TTS synthesis complete: {timing_info}")
        return results, timing_info
    
    def _align_segments(self,
                       source_segments: List[Segment],
                       target_durations: Dict[int, float]) -> List:
        """Align source and target segments for timing."""
        logger.info("Aligning segments")
        
        alignment_results = self.aligner.align_segments(
            source_segments,
            target_durations
        )
        
        logger.info(f"Alignment complete: {len(alignment_results)} results")
        return alignment_results
    
    def _generate_output(self,
                        segments: List[Segment],
                        tts_results: List,
                        alignment_results: List) -> Dict:
        """Generate final output metadata and files."""
        logger.info("Generating output files and metadata")
        
        # Save synthesis report
        synthesis_report = self.tts_orchestrator.get_synthesis_report()
        synthesis_report_path = os.path.join(self.config.output_dir, "synthesis_report.json")
        with open(synthesis_report_path, 'w') as f:
            json.dump(synthesis_report, f, indent=2)
        
        # Save segment metadata
        segments_metadata = []
        for seg in segments:
            seg_meta = {
                "id": seg.id,
                "text": seg.text,
                "speaker": seg.speaker,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "duration": seg.duration,
                "type": seg.segment_type.value,
                "confidence": seg.confidence
            }
            segments_metadata.append(seg_meta)
        
        segments_path = os.path.join(self.config.output_dir, "segments.json")
        with open(segments_path, 'w') as f:
            json.dump(segments_metadata, f, indent=2)
        
        # Save alignment information
        alignment_metadata = []
        for align_result in alignment_results:
            align_meta = {
                "segment_id": align_result.segment_id,
                "source_start": align_result.source_start,
                "source_end": align_result.source_end,
                "target_start": align_result.target_start,
                "target_end": align_result.target_end,
                "status": align_result.alignment_status,
                "metadata": align_result.metadata
            }
            alignment_metadata.append(align_meta)
        
        alignment_path = os.path.join(self.config.output_dir, "alignment.json")
        with open(alignment_path, 'w') as f:
            json.dump(alignment_metadata, f, indent=2)

        dubbed_audio_path = os.path.join(self.config.output_dir, "dubbed_audio.wav")
        dubbed_video_path = os.path.join(self.config.output_dir, "dubbed_video.mp4")

        try:
            self._compose_dubbed_audio(
                segments=segments,
                tts_results=tts_results,
                out_audio_path=dubbed_audio_path,
                max_speed_change=None,
            )

            # Preserve background ambience: mix a ducked version of the original
            # audio under the dubbed track.
            try:
                self._mix_background_ambience(
                    dubbed_audio_path=dubbed_audio_path,
                    background_audio_path=self.state.audio_path,
                )
            except Exception as e:
                logger.warning(
                    "Background ambience mix failed; continuing without it: %s",
                    str(e),
                )

            self._mux_dubbed_video(
                input_video_path=self.state.video_path,
                dubbed_audio_path=dubbed_audio_path,
                out_video_path=dubbed_video_path,
            )
        except Exception as e:
            logger.error("Failed to generate final dubbed audio/video: %s", str(e), exc_info=True)
        
        return {
            "synthesis_report": synthesis_report_path,
            "segments_metadata": segments_path,
            "alignment_metadata": alignment_path,
            "dubbed_audio": dubbed_audio_path,
            "dubbed_video": dubbed_video_path,
            "output_directory": self.config.output_dir
        }

    @staticmethod
    def _run_ffmpeg(cmd: List[str]) -> None:
        logger.info("Running ffmpeg: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise RuntimeError(f"ffmpeg failed: {stderr}")

    def _compose_dubbed_audio(
        self,
        segments: List[Segment],
        tts_results: List,
        out_audio_path: str,
                max_speed_change: Optional[float] = None,
    ) -> str:
        """Compose a full dubbed audio track aligned to the original timeline.

                Each segment is forced to fit exactly within its original time slot
                ($start_time \rightarrow end_time$) by adjusting playback speed ("speaking
                rate") and then padding/trimming to the slot duration.

                - When `max_speed_change` is None, the required tempo change is applied
                    (via chained ffmpeg `atempo` filters) to best match the slot.
                - When `max_speed_change` is set (e.g. 0.15 for ±15%), the tempo change is
                    clamped, and any remaining mismatch is resolved by trim/pad so segment
                    boundaries still align.
        """
        os.makedirs(os.path.dirname(out_audio_path) or ".", exist_ok=True)

        tts_by_id: Dict[int, object] = {
            int(r.segment_id): r
            for r in tts_results
            if getattr(r, "success", False) and getattr(r, "output_path", None)
        }

        def _atempo_chain(factor: float) -> str:
            """Return a comma-separated atempo chain for a desired overall factor.

            ffmpeg's `atempo` accepts values in [0.5, 2.0]. Chaining multiplies.
            """
            try:
                f = float(factor)
            except Exception:
                f = 1.0
            if f <= 0.0:
                f = 1.0

            parts: List[str] = []
            remaining = f
            while remaining < 0.5:
                parts.append("atempo=0.5")
                remaining /= 0.5
            while remaining > 2.0:
                parts.append("atempo=2.0")
                remaining /= 2.0
            parts.append(f"atempo={remaining:.5f}")
            return ",".join(parts)

        inputs: List[Tuple[str, float, int, float]] = []
        for seg in segments:
            tts_res = tts_by_id.get(int(seg.id))
            if not tts_res:
                continue

            src_dur = max(0.05, float(seg.end_time) - float(seg.start_time))
            tts_dur = getattr(tts_res, "duration", None)
            if tts_dur is None or float(tts_dur) <= 0.0:
                tts_dur = get_audio_duration_seconds(tts_res.output_path) or src_dur

            # atempo factor: output_duration = input_duration / atempo
            # For exact slot fit, we want output_duration == src_dur.
            required_atempo = float(tts_dur) / float(src_dur) if src_dur > 0 else 1.0

            applied_atempo = required_atempo
            if max_speed_change is not None:
                min_atempo = 1.0 - float(max_speed_change)
                max_atempo = 1.0 + float(max_speed_change)
                if applied_atempo < min_atempo:
                    applied_atempo = min_atempo
                elif applied_atempo > max_atempo:
                    applied_atempo = max_atempo

            delay_ms = int(round(float(seg.start_time) * 1000.0))
            inputs.append((tts_res.output_path, applied_atempo, delay_ms, src_dur))

        if not inputs:
            raise RuntimeError("No successful TTS segments to compose dubbed audio")

        total_duration = get_audio_duration_seconds(self.state.audio_path)
        if total_duration is None:
            total_duration = max(float(s.end_time) for s in segments) if segments else None

        cmd: List[str] = ["ffmpeg", "-y"]
        for path, _, _, _ in inputs:
            cmd.extend(["-i", path])

        filter_parts: List[str] = []
        mix_labels: List[str] = []
        for i, (_, atempo, delay_ms, slot_dur) in enumerate(inputs):
            label = f"a{i}"
            mix_labels.append(f"[{label}]")
            # Force exact slot duration: time-scale (speaking rate), then pad/trim.
            # `apad` extends with silence; `atrim` cuts/pads to exactly slot_dur.
            atempo_filters = _atempo_chain(atempo)
            filter_parts.append(
                f"[{i}:a]{atempo_filters},apad,atrim=duration={float(slot_dur):.5f},adelay={delay_ms}|{delay_ms}[{label}]"
            )

        mix_in = "".join(mix_labels)
        filter_parts.append(f"{mix_in}amix=inputs={len(inputs)}:normalize=0[aout]")
        filter_complex = ";".join(filter_parts)

        cmd.extend(["-filter_complex", filter_complex, "-map", "[aout]", "-ac", "1"])
        if total_duration is not None:
            cmd.extend(["-t", str(float(total_duration))])
        cmd.append(out_audio_path)

        self._run_ffmpeg(cmd)
        return out_audio_path

    def _mux_dubbed_video(
        self,
        input_video_path: str,
        dubbed_audio_path: str,
        out_video_path: str,
    ) -> str:
        """Mux dubbed audio with the original video stream."""
        os.makedirs(os.path.dirname(out_video_path) or ".", exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_video_path,
            "-i",
            dubbed_audio_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            out_video_path,
        ]
        self._run_ffmpeg(cmd)
        return out_video_path

    def _mix_background_ambience(
        self,
        dubbed_audio_path: str,
        background_audio_path: str,
    ) -> str:
        """Mix background ambience from the original audio under the dubbed track.

        This uses sidechain compression so when the dubbed voice is present, the
        original audio is strongly attenuated (reducing source speech bleed) while
        still preserving room tone / ambience.
        """

        if not (dubbed_audio_path and os.path.exists(dubbed_audio_path)):
            raise RuntimeError("Dubbed audio not found")
        if not (background_audio_path and os.path.exists(background_audio_path)):
            raise RuntimeError("Background audio not found")

        use_demucs = str(
            os.environ.get("DEMUCS_BACKGROUND", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}

        background_bed_path = background_audio_path
        if use_demucs:
            try:
                background_bed_path = self._extract_background_bed_demucs(
                    background_audio_path
                )
                logger.info(
                    "Using Demucs background bed: %s",
                    background_bed_path,
                )
            except Exception as e:
                logger.warning(
                    "Demucs background extraction failed; falling back to "
                    "ducked-original mix: %s",
                    str(e),
                )
                background_bed_path = background_audio_path

        tmp_out = dubbed_audio_path + ".tmp.wav"

        # Filtergraph notes:
        # - Use the original audio as a background bed.
        # - Duck it using the dubbed audio as the sidechain.
        # - Mix the ducked bed under the dubbed track.
        filter_complex = (
            "[1:a]highpass=f=60,lowpass=f=8000[bg];"
            "[bg][0:a]sidechaincompress=threshold=0.02:ratio=20:attack=20:release=250[bgduck];"
            "[bgduck]volume=0.25[bgv];"
            "[0:a][bgv]amix=inputs=2:normalize=0[aout]"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            dubbed_audio_path,
            "-i",
            background_bed_path,
            "-filter_complex",
            filter_complex,
            "-map",
            "[aout]",
            "-ac",
            "1",
            tmp_out,
        ]

        self._run_ffmpeg(cmd)
        os.replace(tmp_out, dubbed_audio_path)
        return dubbed_audio_path

    def _extract_background_bed_demucs(self, audio_path: str) -> str:
        """Extract a background-only bed using Demucs.

        Produces a `no_vocals.wav` track using `--two-stems vocals`.
        This is much better than sidechain ducking for removing source speech.
        """

        if not (audio_path and os.path.exists(audio_path)):
            raise RuntimeError("audio not found for demucs")

        # Use the current interpreter so Demucs is resolved in this env.
        py = sys.executable

        out_root = os.path.join(self.config.output_dir, "segments", "_demucs")
        os.makedirs(out_root, exist_ok=True)

        model = os.environ.get("DEMUCS_MODEL") or "htdemucs"
        device = os.environ.get("DEMUCS_DEVICE")  # e.g. "cpu" or "cuda"

        # Demucs output layout:
        #   <out_root>/separated/<model>/<basename>/no_vocals.wav
        # Depending on version, it may create <out_root>/<model>/<basename>/.
        # We'll search for the expected file.
        cmd: List[str] = [
            py,
            "-m",
            "demucs.separate",
            "--two-stems",
            "vocals",
            "-n",
            model,
            "-o",
            out_root,
            audio_path,
        ]
        if device:
            cmd.extend(["-d", device])

        logger.info("Running Demucs to extract background bed")
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"demucs failed: {stderr}")

        base = os.path.splitext(os.path.basename(audio_path))[0]
        candidates = [
            os.path.join(out_root, "separated", model, base, "no_vocals.wav"),
            os.path.join(out_root, model, base, "no_vocals.wav"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c

        # Last resort: search in out_root for no_vocals.wav
        for root, _, files in os.walk(out_root):
            for fn in files:
                if fn.lower() == "no_vocals.wav":
                    return os.path.join(root, fn)

        raise RuntimeError("demucs output no_vocals.wav not found")
    
    def _save_metadata(self, result: Dict) -> None:
        """Save complete pipeline execution metadata."""
        metadata_path = os.path.join(self.config.output_dir, "pipeline_metadata.json")
        
        # Convert non-serializable objects
        serializable_result = self._make_serializable(result)
        
        with open(metadata_path, 'w') as f:
            json.dump(serializable_result, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    @staticmethod
    def _make_serializable(obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: DetailedDubbingPipeline._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DetailedDubbingPipeline._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
