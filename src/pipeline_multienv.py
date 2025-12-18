"""Enhanced Pipeline using separate ASR and TTS environments.

This pipeline uses:
- 'asr' conda environment for Whisper + Pyannote
- 'tts' conda environment for Coqui TTS

This avoids dependency conflicts between ASR and TTS.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

from .downloader import VideoDownloader
from .audio import extract_audio
from .translator import AbstractTranslator, GroqTranslator
from workers.env_manager import EnvManager

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    work_dir: str = "work"
    sample_rate: int = 16000
    use_worker_envs: bool = True  # Use separate ASR/TTS envs
    tts_device: str = "cpu"  # 'cpu' or 'cuda' for TTS
    tts_backend: str = "pyttsx3"  # TTS backend: 'pyttsx3' (default, local) or 'coqui'


class EnvAwarePipeline:
    """Dubbing pipeline using separate conda environments for ASR and TTS."""

    def __init__(
        self,
        translator: Optional[AbstractTranslator] = None,
        config: PipelineConfig = PipelineConfig(),
    ):
        self.downloader = VideoDownloader()
        self.translator = translator or GroqTranslator()
        self.config = config
        os.makedirs(self.config.work_dir, exist_ok=True)
        
        # Verify environments exist
        if self.config.use_worker_envs:
            env_status = EnvManager.check_envs()
            if not env_status.get(EnvManager.ASR_ENV):
                logger.warning(f"ASR environment '{EnvManager.ASR_ENV}' not found. Create with: conda create -n asr python=3.10 && conda activate asr && pip install -r requirements-asr.txt")
            if not env_status.get(EnvManager.TTS_ENV):
                logger.warning(f"TTS environment '{EnvManager.TTS_ENV}' not found. Create with: conda create -n tts python=3.10 && conda activate tts && pip install TTS")

    def run(self, source_lang: str, target_lang: str, url: Optional[str] = None, video_path: Optional[str] = None) -> dict:
        """Run the dubbing pipeline.
        
        Args:
            source_lang: Source language (ISO code or 'auto')
            target_lang: Target language (ISO code)
            url: Optional URL to download video from (if not using local file)
            video_path: Optional local file path to use (if not downloading)
            
        Returns:
            Dictionary with pipeline outputs
            
        Raises:
            ValueError: If neither url nor video_path is provided
        """
        if not url and not video_path:
            raise ValueError("Either 'url' or 'video_path' must be provided")
        
        out = {"source_lang": source_lang, "target_lang": target_lang, "steps": {}}

        # 1) Download or use local file
        if video_path:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Local video file not found: {video_path}")
            out["video_path"] = video_path
            logger.info("Using local video file: %s", video_path)
        else:
            out["url"] = url
            video_path = self.downloader.download(url, self.config.work_dir)
            out["steps"]["downloaded_video"] = video_path

        # 2) Extract audio
        audio_path = os.path.splitext(video_path)[0] + ".wav"
        audio_path = os.path.join(self.config.work_dir, os.path.basename(audio_path))
        extract_audio(video_path, audio_path, sample_rate=self.config.sample_rate)
        out["steps"]["audio"] = audio_path

        # 3) ASR (using asr environment)
        logger.info("Step 3: Speech Recognition (ASR)")
        try:
            asr_result = EnvManager.run_asr(audio_path, source_lang)
            transcript = asr_result.get("text", "")
            out["steps"]["transcript"] = transcript
            if asr_result.get("segments"):
                out["steps"]["segments"] = asr_result["segments"]
            logger.info("Transcription complete: %d characters", len(transcript))
        except Exception as e:
            logger.error("ASR failed: %s", e)
            raise

        # 4) Translation
        logger.info("Step 4: Translation")
        try:
            translated = self.translator.translate(transcript, source_lang, target_lang)
            out["steps"]["translation"] = translated
            logger.info("Translation complete: %d characters", len(translated))
        except Exception as e:
            logger.error("Translation failed: %s", e)
            raise

        # 5) TTS (using tts environment)
        logger.info("Step 5: Text-to-Speech (TTS)")
        try:
            tts_out = os.path.join(self.config.work_dir, f"dubbed_{target_lang}.wav")
            tts_result = EnvManager.run_tts(
                translated, 
                target_lang, 
                tts_out,
                device=self.config.tts_device,
                tts_backend=self.config.tts_backend
            )
            out["steps"]["tts_audio"] = tts_result.get("audio", tts_out)
            logger.info("TTS complete: %s", tts_out)
        except Exception as e:
            logger.error("TTS failed: %s", e)
            raise

        logger.info("Pipeline finished; outputs: %s", out["steps"])
        return out
