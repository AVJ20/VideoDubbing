import os
import logging
from dataclasses import dataclass
from typing import Optional

from .downloader import VideoDownloader
from .audio import extract_audio
from .asr import AbstractASR, StubASR, WhisperWithDiarizationASR, ASRResult
from .translator import (
    AbstractTranslator,
    GroqTranslator,
)
from .tts import AbstractTTS, StubTTS

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    work_dir: str = "work"
    sample_rate: int = 16000


class DubbingPipeline:
    def __init__(
        self,
        downloader: Optional[VideoDownloader] = None,
        asr: Optional[AbstractASR] = None,
        translator: Optional[AbstractTranslator] = None,
        tts: Optional[AbstractTTS] = None,
        config: PipelineConfig = PipelineConfig(),
    ):
        self.downloader = downloader or VideoDownloader()
        # Try to use WhisperWithDiarizationASR (best free ASR with speaker diarization)
        # Falls back to StubASR if dependencies not installed
        if asr is None:
            try:
                self.asr = WhisperWithDiarizationASR(whisper_model="base")
                logger.info("Using WhisperWithDiarizationASR (speaker diarization enabled)")
            except (RuntimeError, ImportError) as e:
                logger.warning(
                    "Could not load WhisperWithDiarizationASR: %s. "
                    "Install with: pip install openai-whisper pyannote.audio torch torchaudio",
                    str(e)
                )
                self.asr = StubASR()
        else:
            self.asr = asr
        self.translator = translator or GroqTranslator()
        self.tts = tts or StubTTS()
        self.config = config
        os.makedirs(self.config.work_dir, exist_ok=True)

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

        # 1) download or use local file
        if video_path:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Local video file not found: {video_path}")
            out["video_path"] = video_path
            logger.info("Using local video file: %s", video_path)
        else:
            out["url"] = url
            video_path = self.downloader.download(url, self.config.work_dir)
            out["steps"]["downloaded_video"] = video_path

        # 2) extract audio
        audio_path = os.path.splitext(video_path)[0] + ".wav"
        audio_path = os.path.join(self.config.work_dir, os.path.basename(audio_path))
        extract_audio(video_path, audio_path, sample_rate=self.config.sample_rate)
        out["steps"]["audio"] = audio_path

        # 3) ASR
        asr_result: ASRResult = self.asr.transcribe(audio_path, language=source_lang)
        out["steps"]["transcript"] = asr_result.text

        # 4) Translation
        translated = self.translator.translate(asr_result.text, source_lang, target_lang)
        out["steps"]["translation"] = translated

        # 5) TTS
        tts_out = os.path.join(self.config.work_dir, f"tts_{target_lang}.wav")
        self.tts.synthesize(translated, tts_out, language=target_lang)
        out["steps"]["tts_audio"] = tts_out

        logger.info("Pipeline finished; outputs: %s", out["steps"]) 
        return out
