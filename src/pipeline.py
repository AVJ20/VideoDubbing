import os
import logging
from dataclasses import dataclass
from typing import Optional

from .downloader import VideoDownloader
from .audio import extract_audio
from .asr import AbstractASR, StubASR, ASRResult
from .translator import AbstractTranslator, IdentityTranslator
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
        self.asr = asr or StubASR()
        self.translator = translator or IdentityTranslator()
        self.tts = tts or StubTTS()
        self.config = config
        os.makedirs(self.config.work_dir, exist_ok=True)

    def run(self, url: str, source_lang: str, target_lang: str) -> dict:
        out = {"url": url, "steps": {}}

        # 1) download
        video_path = self.downloader.download(url, self.config.work_dir)
        out["steps"]["downloaded_video"] = video_path

        # 2) extract audio
        audio_path = os.path.splitext(video_path)[0] + ".wav"
        audio_path = os.path.join(self.config.work_dir, os.path.basename(audio_path))
        extract_audio(video_path, audio_path, sample_rate=self.config.sample_rate)
        out["steps"]["audio"] = audio_path

        # 3) ASR
        asr_result: ASRResult = self.asr.transcribe(audio_path)
        out["steps"]["transcript"] = asr_result.text

        # 4) Translation
        translated = self.translator.translate(asr_result.text, source_lang, target_lang)
        out["steps"]["translation"] = translated

        # 5) TTS
        tts_out = os.path.join(self.config.work_dir, f"tts_{target_lang}.wav")
        self.tts.synthesize(translated, tts_out)
        out["steps"]["tts_audio"] = tts_out

        logger.info("Pipeline finished; outputs: %s", out["steps"]) 
        return out
