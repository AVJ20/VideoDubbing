import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import pyttsx3
except Exception:  # pragma: no cover - optional dependency
    pyttsx3 = None


class AbstractTTS(ABC):
    @abstractmethod
    def synthesize(self, text: str, out_path: str, voice: Optional[str] = None) -> str:
        pass


class Pyttsx3TTS(AbstractTTS):
    def __init__(self):
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 is not installed; install requirements to use Pyttsx3TTS")
        self.engine = pyttsx3.init()

    def synthesize(self, text: str, out_path: str, voice: Optional[str] = None) -> str:
        logger.info("Synthesizing to %s with pyttsx3", out_path)
        if voice:
            try:
                self.engine.setProperty("voice", voice)
            except Exception:
                logger.warning("Failed to set voice %s", voice)
        # pyttsx3 supports saving to file
        self.engine.save_to_file(text, out_path)
        self.engine.runAndWait()
        return out_path


class StubTTS(AbstractTTS):
    def synthesize(self, text: str, out_path: str, voice: Optional[str] = None) -> str:
        logger.warning("Using StubTTS: writing text into a .txt placeholder at %s", out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        return out_path


class AzureSpeechTTS(AbstractTTS):
    """Azure Cognitive Services Text-to-Speech adapter.

    Expects environment variables AZURE_SPEECH_KEY and AZURE_SPEECH_REGION
    or accepts them via constructor arguments.
    """

    def __init__(self, subscription_key: str | None = None, region: str | None = None, voice: Optional[str] = None):
        try:
            import azure.cognitiveservices.speech as speechsdk
        except Exception:  # pragma: no cover - optional dependency
            raise RuntimeError("azure.cognitiveservices.speech is not installed")

        self.speechsdk = speechsdk
        self.key = subscription_key or __import__("os").environ.get("AZURE_SPEECH_KEY")
        self.region = region or __import__("os").environ.get("AZURE_SPEECH_REGION")
        if not self.key or not self.region:
            raise RuntimeError(
                "Azure Speech subscription key and region are required. "
                "Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION or pass them in."
            )

        self.speech_config = speechsdk.SpeechConfig(subscription=self.key, region=self.region)
        if voice:
            # voice example: "en-US-JennyNeural"
            self.speech_config.speech_synthesis_voice_name = voice

    def synthesize(self, text: str, out_path: str, voice: Optional[str] = None) -> str:
        speechsdk = self.speechsdk
        if voice:
            # override voice for this call
            self.speech_config.speech_synthesis_voice_name = voice

        audio_config = speechsdk.audio.AudioOutputConfig(filename=out_path)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=audio_config)

        logger.info("AzureSpeechTTS: synthesizing to %s", out_path)
        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return out_path
        else:
            err = getattr(result, "error_details", None)
            raise RuntimeError(f"Azure Speech synthesis failed: {err}")

