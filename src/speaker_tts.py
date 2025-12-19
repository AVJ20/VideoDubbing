"""
Speaker-Specific TTS Module

Handles text-to-speech synthesis with speaker-specific characteristics:
1. Preserves speaker identity through voice cloning or voice selection
2. Manages speaker profiles (voice characteristics, language, emotion)
3. Generates audio with segment-level speaker metadata
4. Supports multiple TTS backends for different speaker requirements
"""

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json

logger = logging.getLogger(__name__)


class VoiceCloneMethod(Enum):
    """Methods for preserving speaker identity."""
    
    ZERO_SHOT = "zero_shot"  # Clone from reference audio (best quality)
    VOICE_NAME = "voice_name"  # Use predefined voice name
    STYLE_TRANSFER = "style_transfer"  # Transfer style from source
    PROSODY_MATCHING = "prosody_matching"  # Match prosody of source


@dataclass
class SpeakerProfile:
    """Profile of a speaker with TTS characteristics."""
    
    speaker_id: str
    language: str
    name: Optional[str] = None
    voice_reference: Optional[str] = None  # Path to reference audio for cloning
    voice_name: Optional[str] = None  # Predefined voice name
    clone_method: VoiceCloneMethod = VoiceCloneMethod.ZERO_SHOT
    
    # Prosody parameters (can be overridden per segment)
    pace: float = 1.0  # Speaking rate multiplier
    pitch: float = 1.0  # Pitch multiplier
    energy: float = 1.0  # Energy/loudness multiplier
    
    # Style/emotion
    emotion: Optional[str] = None  # "neutral", "happy", "sad", "angry", etc.
    style: Optional[str] = None  # Style indication
    
    metadata: Dict = field(default_factory=dict)  # Additional metadata


@dataclass
class TTSSegment:
    """Segment to synthesize with speaker information."""
    
    segment_id: int
    text: str
    speaker_id: str
    speaker_profile: SpeakerProfile
    start_time: float
    end_time: float
    language: str
    
    # Output
    output_path: Optional[str] = None
    duration: Optional[float] = None  # Actual duration of synthesized audio
    confidence: float = 1.0


@dataclass
class TTSResult:
    """Result of TTS synthesis for a segment."""
    
    segment_id: int
    speaker_id: str
    success: bool
    output_path: Optional[str] = None
    duration: Optional[float] = None
    text: str = ""
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class AbstractSpeakerTTS(ABC):
    """Abstract base class for speaker-specific TTS backends."""
    
    @abstractmethod
    def synthesize_segment(self, tts_segment: TTSSegment) -> TTSResult:
        """Synthesize a single segment with speaker information."""
        pass
    
    @abstractmethod
    def register_speaker(self, profile: SpeakerProfile) -> bool:
        """Register a speaker profile for voice cloning."""
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        pass


class CoquiSpeakerTTS(AbstractSpeakerTTS):
    """
    Speaker-specific TTS using Coqui TTS with voice cloning.
    
    Features:
    - Zero-shot voice cloning from reference audio
    - Multilingual support
    - Prosody control (pitch, rate, energy)
    - Emotion and style transfer
    """
    
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", 
                 device: str = "cpu"):
        """
        Args:
            model_name: Coqui TTS model (XTTS v2 recommended for voice cloning)
            device: 'cpu' or 'cuda'
        """
        try:
            from TTS.api import TTS as CoquiTTSAPI
        except ImportError:
            raise RuntimeError(
                "Coqui TTS is not installed. Install with: pip install TTS torch torchaudio"
            )
        
        logger.info(f"Loading Coqui TTS model: {model_name} (device: {device})")
        self.tts = CoquiTTSAPI(model_name=model_name, gpu=(device == "cuda"), progress_bar=False)
        self.device = device
        self.speakers: Dict[str, SpeakerProfile] = {}
        self.supported_languages = self._get_supported_languages()
    
    def register_speaker(self, profile: SpeakerProfile) -> bool:
        """
        Register a speaker for voice cloning.
        
        Validates that reference audio exists if using zero-shot cloning.
        """
        if profile.clone_method == VoiceCloneMethod.ZERO_SHOT:
            if not profile.voice_reference or not os.path.exists(profile.voice_reference):
                logger.error(f"Voice reference not found for speaker {profile.speaker_id}: "
                           f"{profile.voice_reference}")
                return False
            
            logger.info(f"Registered speaker {profile.speaker_id} with voice reference "
                       f"{profile.voice_reference}")
        
        self.speakers[profile.speaker_id] = profile
        return True
    
    def synthesize_segment(self, tts_segment: TTSSegment) -> TTSResult:
        """
        Synthesize a segment with speaker-specific voice.
        
        Args:
            tts_segment: TTSSegment with text and speaker info
        
        Returns:
            TTSResult with output path and duration
        """
        try:
            speaker = tts_segment.speaker_profile
            text = tts_segment.text
            output_path = tts_segment.output_path
            
            if not output_path:
                output_path = f"tts_segment_{tts_segment.segment_id:04d}.wav"
            
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            
            logger.info(f"Synthesizing segment {tts_segment.segment_id} for speaker "
                       f"{tts_segment.speaker_id}")
            
            # Determine synthesis method
            if speaker.clone_method == VoiceCloneMethod.ZERO_SHOT and speaker.voice_reference:
                # Zero-shot voice cloning
                logger.debug(f"Using zero-shot cloning with reference: {speaker.voice_reference}")
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker_wav=speaker.voice_reference,
                    language=tts_segment.language
                )
            elif speaker.voice_name:
                # Use predefined voice name
                logger.debug(f"Using voice name: {speaker.voice_name}")
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker=speaker.voice_name,
                    language=tts_segment.language
                )
            else:
                # Default synthesis
                logger.debug("Using default voice")
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    language=tts_segment.language
                )
            
            # Get actual duration
            duration = self._get_audio_duration(output_path)
            
            logger.info(f"Segment {tts_segment.segment_id} synthesized: "
                       f"{output_path} ({duration:.2f}s)")
            
            return TTSResult(
                segment_id=tts_segment.segment_id,
                speaker_id=tts_segment.speaker_id,
                success=True,
                output_path=output_path,
                duration=duration,
                text=text,
                metadata={
                    "clone_method": speaker.clone_method.value,
                    "voice_reference": speaker.voice_reference,
                    "language": tts_segment.language
                }
            )
        
        except Exception as e:
            logger.error(f"TTS synthesis failed for segment {tts_segment.segment_id}: {str(e)}")
            return TTSResult(
                segment_id=tts_segment.segment_id,
                speaker_id=tts_segment.speaker_id,
                success=False,
                error=str(e),
                text=text
            )
    
    def _get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        # XTTS v2 supports: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh, hu, ko, ja, hi
        return [
            "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", 
            "ar", "zh", "hu", "ko", "ja", "hi"
        ]
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.supported_languages
    
    @staticmethod
    def _get_audio_duration(audio_path: str) -> float:
        """Get duration of audio file in seconds."""
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            return float(duration)
        except Exception:
            # Fallback: use wave module
            try:
                import wave
                with wave.open(audio_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration = frames / float(rate)
                    return duration
            except Exception as e:
                logger.warning(f"Could not determine audio duration: {e}")
                return 0.0


class ChatterboxSpeakerTTS(AbstractSpeakerTTS):
    """
    Speaker-specific TTS using Chatterbox (local, open-source).
    
    Features:
    - Local model (no API calls)
    - Voice cloning support
    - High-quality synthesis
    - GPU acceleration support
    - Free and open-source
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Args:
            device: 'cpu' or 'cuda' for GPU acceleration
        """
        try:
            from src.tts import ChatterboxTTS
        except ImportError:
            raise RuntimeError(
                "ChatterboxTTS not available. Install with: pip install chatterbox-tts torch torchaudio"
            )
        
        logger.info(f"Initializing ChatterboxSpeakerTTS (device: {device})")
        try:
            self.tts = ChatterboxTTS(device=device)
            self.speakers: Dict[str, SpeakerProfile] = {}
            self.supported_languages = self._get_supported_languages()
        except RuntimeError as e:
            logger.error(f"Failed to initialize Chatterbox TTS: {e}")
            raise
    
    def register_speaker(self, profile: SpeakerProfile) -> bool:
        """Register a speaker profile."""
        if profile.clone_method == VoiceCloneMethod.ZERO_SHOT:
            if profile.voice_reference and os.path.exists(profile.voice_reference):
                logger.info(f"Registered speaker {profile.speaker_id} with voice reference "
                           f"{profile.voice_reference}")
            else:
                logger.warning(f"Speaker {profile.speaker_id} configured for voice cloning but "
                              f"reference audio not found: {profile.voice_reference}")
        else:
            logger.info(f"Registered speaker {profile.speaker_id}")
        
        self.speakers[profile.speaker_id] = profile
        return True
    
    def synthesize_segment(self, tts_segment: TTSSegment) -> TTSResult:
        """Synthesize a segment using Chatterbox."""
        try:
            speaker = tts_segment.speaker_profile
            text = tts_segment.text
            output_path = tts_segment.output_path
            
            if not output_path:
                output_path = f"tts_segment_{tts_segment.segment_id:04d}.wav"
            
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            
            logger.info(f"Synthesizing segment {tts_segment.segment_id} for speaker "
                       f"{tts_segment.speaker_id}")
            
            # Synthesize with voice cloning if reference available
            voice_reference = None
            if speaker.voice_reference and os.path.exists(speaker.voice_reference):
                voice_reference = speaker.voice_reference
                logger.debug(f"Using voice cloning with reference: {voice_reference}")
            
            self.tts.synthesize(
                text=text,
                out_path=output_path,
                voice=voice_reference,
                language=tts_segment.language
            )
            
            # Get actual duration
            duration = self._get_audio_duration(output_path)
            
            logger.info(f"Segment {tts_segment.segment_id} synthesized: "
                       f"{output_path} ({duration:.2f}s)")
            
            return TTSResult(
                segment_id=tts_segment.segment_id,
                speaker_id=tts_segment.speaker_id,
                success=True,
                output_path=output_path,
                duration=duration,
                text=text,
                metadata={
                    "tts_backend": "chatterbox-turbo",
                    "voice_cloning": voice_reference is not None,
                    "language": tts_segment.language
                }
            )
        
        except Exception as e:
            logger.error(f"Chatterbox TTS synthesis failed for segment {tts_segment.segment_id}: {str(e)}")
            return TTSResult(
                segment_id=tts_segment.segment_id,
                speaker_id=tts_segment.speaker_id,
                success=False,
                error=str(e),
                text=text
            )
    
    def _get_supported_languages(self) -> List[str]:
        """Get list of supported languages by Chatterbox."""
        return ["en", "es", "fr", "de", "it", "pt"]
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.supported_languages
    
    @staticmethod
    def _get_audio_duration(audio_path: str) -> float:
        """Get duration of audio file in seconds."""
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            return float(duration)
        except Exception:
            # Fallback: use wave module
            try:
                import wave
                with wave.open(audio_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration = frames / float(rate)
                    return duration
            except Exception as e:
                logger.warning(f"Could not determine audio duration: {e}")
                return 0.0


class MultiEnvWorkerSpeakerTTS(AbstractSpeakerTTS):
    """Speaker-specific TTS that runs synthesis via the multi-env worker.

    This avoids importing or initializing heavy TTS dependencies (and Hugging Face
    downloads) in the main ASR environment. Instead, it delegates each synthesis
    call to `workers/tts_worker.py` running under the `tts` conda environment.
    """

    def __init__(self, device: str = "cpu", tts_backend: str = "chatterbox"):
        from workers.env_manager import EnvManager

        self._env = EnvManager
        self.device = device
        self.tts_backend = tts_backend
        self.speakers: Dict[str, SpeakerProfile] = {}
        self.supported_languages = self._get_supported_languages()

    def register_speaker(self, profile: SpeakerProfile) -> bool:
        # We just store profiles here; the worker uses voice reference paths per call.
        if profile.clone_method == VoiceCloneMethod.ZERO_SHOT and profile.voice_reference:
            if not os.path.exists(profile.voice_reference):
                logger.warning(
                    "Speaker %s voice_reference not found: %s",
                    profile.speaker_id,
                    profile.voice_reference,
                )
        self.speakers[profile.speaker_id] = profile
        return True

    def synthesize_segment(self, tts_segment: TTSSegment) -> TTSResult:
        text = tts_segment.text
        try:
            speaker = tts_segment.speaker_profile
            output_path = tts_segment.output_path or f"tts_segment_{tts_segment.segment_id:04d}.wav"
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            voice_reference = None
            if speaker.clone_method == VoiceCloneMethod.ZERO_SHOT and speaker.voice_reference:
                if os.path.exists(speaker.voice_reference):
                    voice_reference = speaker.voice_reference

            self._env.run_tts(
                text=text,
                language=tts_segment.language,
                output_audio=output_path,
                voice=voice_reference,
                device=self.device,
                tts_backend=self.tts_backend,
            )

            duration = self._get_audio_duration(output_path)

            return TTSResult(
                segment_id=tts_segment.segment_id,
                speaker_id=tts_segment.speaker_id,
                success=True,
                output_path=output_path,
                duration=duration,
                text=text,
                metadata={
                    "tts_backend": f"worker:{self.tts_backend}",
                    "voice_cloning": voice_reference is not None,
                    "language": tts_segment.language,
                },
            )
        except Exception as e:
            logger.error(
                "Worker TTS synthesis failed for segment %s: %s",
                tts_segment.segment_id,
                str(e),
            )
            return TTSResult(
                segment_id=tts_segment.segment_id,
                speaker_id=tts_segment.speaker_id,
                success=False,
                error=str(e),
                text=text,
            )

    def _get_supported_languages(self) -> List[str]:
        # Chatterbox turbo is multilingual; keep a conservative list (can be expanded).
        return [
            "ar",
            "da",
            "de",
            "el",
            "en",
            "es",
            "fi",
            "fr",
            "he",
            "hi",
            "it",
            "ja",
            "ko",
            "ms",
            "nl",
            "no",
            "pl",
            "pt",
            "ru",
            "sv",
            "sw",
            "tr",
        ]

    def get_supported_languages(self) -> List[str]:
        return self.supported_languages

    @staticmethod
    def _get_audio_duration(audio_path: str) -> float:
        # Reuse the same duration logic.
        try:
            import librosa

            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            return float(duration)
        except Exception:
            try:
                import wave

                with wave.open(audio_path, "rb") as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    return frames / float(rate)
            except Exception as e:
                logger.warning("Could not determine audio duration: %s", e)
                return 0.0


class ElevenLabsSpeakerTTS(AbstractSpeakerTTS):
    """
    Speaker-specific TTS using ElevenLabs API.
    
    Features:
    - High quality voices
    - Voice cloning support
    - Multilingual
    - Stable, reliable API
    
    Requires: ELEVENLABS_API_KEY environment variable
    """
    
    def __init__(self, api_key: Optional[str] = None, device: str = "cpu"):
        """
        Args:
            api_key: ElevenLabs API key (defaults to ELEVENLABS_API_KEY env var)
            device: Ignored, for API compatibility
        """
        try:
            from src.tts import ElevenLabsTTS
        except ImportError:
            raise RuntimeError(
                "ElevenLabsTTS not available. Ensure elevenlabs library is installed."
            )
        
        api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ElevenLabs API key required. Set ELEVENLABS_API_KEY environment variable."
            )
        
        logger.info("Initializing ElevenLabsSpeakerTTS")
        self.tts = ElevenLabsTTS(api_key=api_key)
        self.speakers: Dict[str, SpeakerProfile] = {}
        self.supported_languages = self._get_supported_languages()
    
    def register_speaker(self, profile: SpeakerProfile) -> bool:
        """Register a speaker profile."""
        logger.info(f"Registered speaker {profile.speaker_id} for ElevenLabs TTS")
        self.speakers[profile.speaker_id] = profile
        return True
    
    def synthesize_segment(self, tts_segment: TTSSegment) -> TTSResult:
        """Synthesize a segment using ElevenLabs API."""
        try:
            speaker = tts_segment.speaker_profile
            text = tts_segment.text
            output_path = tts_segment.output_path
            
            if not output_path:
                output_path = f"tts_segment_{tts_segment.segment_id:04d}.wav"
            
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            
            logger.info(f"Synthesizing segment {tts_segment.segment_id} for speaker "
                       f"{tts_segment.speaker_id} (ElevenLabs)")
            
            # Use voice name if available, otherwise use speaker ID
            voice = speaker.voice_name or speaker.speaker_id
            self.tts.synthesize(
                text=text,
                out_path=output_path,
                voice=voice,
                language=tts_segment.language
            )
            
            # Get actual duration
            duration = self._get_audio_duration(output_path)
            
            logger.info(f"Segment {tts_segment.segment_id} synthesized: "
                       f"{output_path} ({duration:.2f}s)")
            
            return TTSResult(
                segment_id=tts_segment.segment_id,
                speaker_id=tts_segment.speaker_id,
                success=True,
                output_path=output_path,
                duration=duration,
                text=text,
                metadata={
                    "tts_backend": "elevenlabs",
                    "voice": voice,
                    "language": tts_segment.language
                }
            )
        
        except Exception as e:
            logger.error(f"ElevenLabs TTS synthesis failed for segment {tts_segment.segment_id}: {str(e)}")
            return TTSResult(
                segment_id=tts_segment.segment_id,
                speaker_id=tts_segment.speaker_id,
                success=False,
                error=str(e),
                text=text
            )
    
    def _get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return [
            "en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh", 
            "ko", "ar", "hi", "th", "vi", "tr", "pl", "nl", "sv"
        ]
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.supported_languages
    
    @staticmethod
    def _get_audio_duration(audio_path: str) -> float:
        """Get duration of audio file in seconds."""
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            return float(duration)
        except Exception:
            # Fallback: use wave module
            try:
                import wave
                with wave.open(audio_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration = frames / float(rate)
                    return duration
            except Exception as e:
                logger.warning(f"Could not determine audio duration: {e}")
                return 0.0


class SpeakerTTSOrchestrator:
    """
    Orchestrates TTS synthesis for multiple segments with different speakers.
    
    Features:
    - Manages speaker registration
    - Batch synthesis
    - Handles speaker-specific configurations
    - Generates alignment information
    """
    
    def __init__(self, tts_backend: AbstractSpeakerTTS):
        """
        Args:
            tts_backend: TTS backend implementation
        """
        self.tts = tts_backend
        self.synthesis_results: Dict[int, TTSResult] = {}
        self.speaker_profiles: Dict[str, SpeakerProfile] = {}
    
    def register_speaker(self, profile: SpeakerProfile) -> bool:
        """Register a speaker profile."""
        success = self.tts.register_speaker(profile)
        if success:
            self.speaker_profiles[profile.speaker_id] = profile
            logger.info(f"Speaker registered: {profile.speaker_id}")
        return success
    
    def synthesize_segments(
        self,
        segments: List[dict],
        output_dir: str,
        language: str = "en",
        source_audio_path: Optional[str] = None,
        use_segment_audio_as_reference: bool = False,
        reference_max_seconds: float = 10.0,
        reference_min_seconds: float = 6.0,
    ) -> Tuple[List[TTSResult], Dict]:
        """
        Synthesize multiple segments.
        
        Args:
            segments: List of segment dicts with text, speaker, start_time, end_time
            output_dir: Directory for output audio files
            language: Target language
        
        Returns:
            (list of TTSResult, dict with timing info)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        timing_info = {
            "total_segments": len(segments),
            "successful": 0,
            "failed": 0,
            "total_duration": 0.0,
            "segment_durations": {}
        }
        
        logger.info(f"Starting synthesis of {len(segments)} segments")
        
        reference_dir = os.path.join(output_dir, "_refs")
        if use_segment_audio_as_reference and source_audio_path:
            os.makedirs(reference_dir, exist_ok=True)

        source_audio_duration: Optional[float] = None
        if use_segment_audio_as_reference and source_audio_path and os.path.exists(source_audio_path):
            try:
                from .audio import get_wav_duration_seconds

                source_audio_duration = get_wav_duration_seconds(source_audio_path)
            except Exception:
                source_audio_duration = None

        for seg in segments:
            segment_id = seg.get("id", 0)
            speaker_id = seg.get("speaker", "Unknown")
            text = seg.get("text", "")
            start_time = seg.get("start_time", 0.0)
            end_time = seg.get("end_time", 0.0)
            
            # Get speaker profile
            profile = self.speaker_profiles.get(speaker_id)
            if not profile:
                logger.warning(f"No profile for speaker {speaker_id}, using default")
                profile = SpeakerProfile(
                    speaker_id=speaker_id,
                    language=language
                )
            
            # Create (optional) per-segment reference audio for zero-shot cloning.
            profile_for_segment = profile
            if profile.voice_reference and os.path.exists(profile.voice_reference):
                # Prefer explicit per-speaker reference audio when provided.
                profile_for_segment = replace(
                    profile,
                    clone_method=VoiceCloneMethod.ZERO_SHOT,
                )
            elif use_segment_audio_as_reference and source_audio_path:
                try:
                    from .audio import extract_audio_clip, get_wav_duration_seconds

                    seg_start = max(0.0, float(start_time))
                    seg_end = max(seg_start, float(end_time))
                    seg_len = seg_end - seg_start

                    min_len = float(reference_min_seconds or 0.0)
                    max_len = float(reference_max_seconds or 0.0)
                    if max_len and min_len and max_len < min_len:
                        max_len = min_len

                    desired_len = max(min_len, seg_len if seg_len > 0 else min_len)
                    if max_len:
                        desired_len = min(desired_len, max_len)

                    # Center the reference window around the segment.
                    center = (seg_start + seg_end) / 2.0
                    clip_start = center - (desired_len / 2.0)
                    clip_end = clip_start + desired_len

                    # Clamp to [0, duration] when possible.
                    if clip_start < 0.0:
                        clip_start = 0.0
                        clip_end = clip_start + desired_len

                    if source_audio_duration is not None:
                        if source_audio_duration < min_len:
                            raise RuntimeError(
                                f"Source audio too short for voice prompt ({source_audio_duration:.2f}s < {min_len:.2f}s)"
                            )
                        if clip_end > source_audio_duration:
                            clip_end = source_audio_duration
                            clip_start = max(0.0, clip_end - desired_len)

                    ref_path = os.path.join(
                        reference_dir,
                        f"seg_{segment_id:04d}_{speaker_id}_ref.wav",
                    )
                    extract_audio_clip(
                        source_audio_path,
                        ref_path,
                        clip_start,
                        clip_end,
                    )

                    ref_dur = get_wav_duration_seconds(ref_path)
                    if ref_dur is not None and ref_dur <= 5.1:
                        raise RuntimeError(
                            f"Extracted prompt too short for Chatterbox ({ref_dur:.2f}s)."
                        )

                    profile_for_segment = replace(
                        profile,
                        clone_method=VoiceCloneMethod.ZERO_SHOT,
                        voice_reference=ref_path,
                    )
                except Exception as e:
                    logger.warning(
                        "Could not extract reference audio for segment %s (%s): %s",
                        segment_id,
                        speaker_id,
                        str(e),
                    )

            # Create TTS segment
            output_path = os.path.join(output_dir, f"segment_{segment_id:04d}.wav")
            tts_seg = TTSSegment(
                segment_id=segment_id,
                text=text,
                speaker_id=speaker_id,
                speaker_profile=profile_for_segment,
                start_time=start_time,
                end_time=end_time,
                language=language,
                output_path=output_path
            )
            
            # Synthesize
            result = self.tts.synthesize_segment(tts_seg)
            results.append(result)
            self.synthesis_results[segment_id] = result
            
            if result.success:
                timing_info["successful"] += 1
                if result.duration:
                    timing_info["total_duration"] += result.duration
                    timing_info["segment_durations"][segment_id] = result.duration
                logger.debug(f"Segment {segment_id}: {result.duration:.2f}s")
            else:
                timing_info["failed"] += 1
                logger.error(f"Segment {segment_id} failed: {result.error}")
        
        logger.info(
            f"Synthesis complete: {timing_info['successful']}/{timing_info['total_segments']} "
            f"successful, total duration: {timing_info['total_duration']:.1f}s"
        )
        
        return results, timing_info
    
    def get_segment_durations(self) -> Dict[int, float]:
        """Get synthesized duration for each segment."""
        durations = {}
        for seg_id, result in self.synthesis_results.items():
            if result.success and result.duration:
                durations[seg_id] = result.duration
        return durations
    
    def get_synthesis_report(self) -> Dict:
        """Generate synthesis report with all metadata."""
        report = {
            "total_segments": len(self.synthesis_results),
            "successful": sum(1 for r in self.synthesis_results.values() if r.success),
            "failed": sum(1 for r in self.synthesis_results.values() if not r.success),
            "segments": {}
        }
        
        for seg_id, result in self.synthesis_results.items():
            report["segments"][seg_id] = {
                "success": result.success,
                "speaker": result.speaker_id,
                "duration": result.duration,
                "output_path": result.output_path,
                "error": result.error,
                "metadata": result.metadata
            }
        
        return report


def create_speaker_profiles_from_segments(segments: List) -> Dict[str, SpeakerProfile]:
    """
    Create default speaker profiles from segmentation results.
    
    Extracts unique speakers and creates basic profiles.
    Can be enhanced with voice reference audio.
    """
    profiles = {}
    speakers = set(seg.speaker for seg in segments)
    
    for speaker_id in speakers:
        profile = SpeakerProfile(
            speaker_id=speaker_id,
            language="en",
            name=speaker_id,
            clone_method=VoiceCloneMethod.VOICE_NAME
        )
        profiles[speaker_id] = profile
    
    logger.info(f"Created default profiles for {len(profiles)} speakers: {list(speakers)}")
    return profiles
