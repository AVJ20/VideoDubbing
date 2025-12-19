import os
import logging
from abc import ABC, abstractmethod
from typing import Optional
import json
import requests

logger = logging.getLogger(__name__)


def _allow_anonymous_hf_downloads_if_needed(hf_token: str | None = None) -> None:
    """Coerce `token=True` Hugging Face downloads to anonymous.

    Some downstream libraries (or specific versions) call into `huggingface_hub`
    with `token=True` unconditionally. In modern `huggingface_hub`, that raises
    `LocalTokenNotFoundError` when the user isn't logged in, even for public
    repos. Since download helpers are decorated at import-time, we wrap the
    download entry points directly.

    If the target repo is gated/private, anonymous download will still fail with
    401/403 and the user must authenticate.
    """

    try:
        try:
            import huggingface_hub.utils._headers as headers

            orig_get = getattr(headers, "get_token_to_send", None)
            if orig_get is not None and not getattr(orig_get, "__vd_token_true_patch__", False):

                def patched_get_token_to_send(token):
                    if token is True:
                        token = hf_token if hf_token else False
                    return orig_get(token)

                patched_get_token_to_send.__vd_token_true_patch__ = True
                headers.get_token_to_send = patched_get_token_to_send
        except Exception:
            pass

        import inspect
        import huggingface_hub

        def _wrap(module, attr: str) -> None:
            fn = getattr(module, attr, None)
            if fn is None:
                return
            if getattr(fn, "__vd_token_true_patch__", False):
                return

            sig = None
            try:
                sig = inspect.signature(fn)
            except Exception:
                sig = None

            def wrapped(*args, **kwargs):
                if sig is not None:
                    try:
                        bound = sig.bind_partial(*args, **kwargs)
                        if bound.arguments.get("token") is True:
                            bound.arguments["token"] = hf_token if hf_token else False
                        if bound.arguments.get("use_auth_token") is True:
                            bound.arguments["use_auth_token"] = hf_token if hf_token else False
                        return fn(*bound.args, **bound.kwargs)
                    except Exception:
                        pass

                if kwargs.get("token") is True:
                    kwargs["token"] = hf_token if hf_token else False
                if kwargs.get("use_auth_token") is True:
                    kwargs["use_auth_token"] = hf_token if hf_token else False
                return fn(*args, **kwargs)

            wrapped.__vd_token_true_patch__ = True
            setattr(module, attr, wrapped)

        _wrap(huggingface_hub, "hf_hub_download")
        _wrap(huggingface_hub, "snapshot_download")

        try:
            import huggingface_hub.file_download as file_download

            _wrap(file_download, "hf_hub_download")
        except Exception:
            pass

        try:
            import huggingface_hub._snapshot_download as snapshot_mod

            _wrap(snapshot_mod, "snapshot_download")
        except Exception:
            pass

    except Exception:
        return

try:
    import pyttsx3
except Exception:  # pragma: no cover - optional dependency
    pyttsx3 = None


class AbstractTTS(ABC):
    @abstractmethod
    def synthesize(self, text: str, out_path: str, voice: Optional[str] = None, language: Optional[str] = None) -> str:
        pass


class Pyttsx3TTS(AbstractTTS):
    """Local TTS using pyttsx3 (cross-platform, no internet needed)."""
    
    def __init__(self):
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 is not installed; install requirements to use Pyttsx3TTS")
        self.engine = pyttsx3.init()

    def synthesize(self, text: str, out_path: str, voice: Optional[str] = None, language: Optional[str] = None) -> str:
        logger.info("Synthesizing to %s with pyttsx3", out_path)
        if voice:
            try:
                self.engine.setProperty("voice", voice)
            except Exception:
                logger.warning("Failed to set voice %s", voice)
        self.engine.save_to_file(text, out_path)
        self.engine.runAndWait()
        return out_path


class CoquiTTS(AbstractTTS):
    """Coqui TTS - best open-source TTS with natural voices and emotion support.
    
    Features:
    - Multilingual support
    - Natural prosody and emotions
    - Zero-shot voice cloning (with reference audio)
    - Fast synthesis
    
    Installation: pip install TTS torch
    """

    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", device: str = "cpu"):
        """
        Args:
            model_name: TTS model to use (default: XTTS v2 - multilingual, high quality)
            device: 'cpu' or 'cuda' for GPU acceleration
        """
        try:
            from TTS.api import TTS as CoquiTTSAPI
        except ImportError:
            raise RuntimeError(
                "Coqui TTS is not installed. Install with: pip install TTS torch torchaudio"
            )

        logger.info("Loading Coqui TTS model: %s (device: %s)", model_name, device)
        self.tts = CoquiTTSAPI(model_name=model_name, gpu=(device == "cuda"))
        self.device = device

    def synthesize(self, text: str, out_path: str, voice: Optional[str] = None, language: Optional[str] = None) -> str:
        """Synthesize text to speech.
        
        Args:
            text: Text to synthesize
            out_path: Output audio file path
            voice: Speaker name or path to reference audio for zero-shot cloning
            language: Target language code (e.g., 'en', 'es', 'fr')
        """
        logger.info("Coqui TTS: synthesizing to %s (language: %s)", out_path, language or "auto")
        
        # Use speaker/reference audio if provided
        if voice and os.path.exists(voice):
            # Zero-shot voice cloning mode
            self.tts.tts_to_file(
                text=text,
                file_path=out_path,
                speaker_wav=voice,  # Reference audio for cloning
                language=language or "en"
            )
        else:
            # Standard synthesis with speaker name
            self.tts.tts_to_file(
                text=text,
                file_path=out_path,
                speaker=voice or "default",
                language=language or "en"
            )
        
        logger.info("Coqui TTS synthesis complete: %s", out_path)
        return out_path


class StubTTS(AbstractTTS):
    def synthesize(self, text: str, out_path: str, voice: Optional[str] = None, language: Optional[str] = None) -> str:
        logger.warning("Using StubTTS: writing text into a .txt placeholder at %s", out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        return out_path


class AzureSpeechTTS(AbstractTTS):
    """Azure Cognitive Services Text-to-Speech adapter.

    Features:
    - 400+ voices in 140+ languages
    - Neural voices with emotions
    - SSML support for advanced prosody control
    
    Expects environment variables AZURE_SPEECH_KEY and AZURE_SPEECH_REGION
    or accepts them via constructor arguments.
    """

    def __init__(self, subscription_key: str | None = None, region: str | None = None, voice: Optional[str] = None):
        try:
            import azure.cognitiveservices.speech as speechsdk
        except Exception:  # pragma: no cover - optional dependency
            raise RuntimeError("azure.cognitiveservices.speech is not installed")

        self.speechsdk = speechsdk
        self.key = subscription_key or os.environ.get("AZURE_SPEECH_KEY")
        self.region = region or os.environ.get("AZURE_SPEECH_REGION")
        if not self.key or not self.region:
            raise RuntimeError(
                "Azure Speech subscription key and region are required. "
                "Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION or pass them in."
            )

        self.speech_config = speechsdk.SpeechConfig(subscription=self.key, region=self.region)
        if voice:
            # voice example: "en-US-JennyNeural"
            self.speech_config.speech_synthesis_voice_name = voice

    def synthesize(self, text: str, out_path: str, voice: Optional[str] = None, language: Optional[str] = None) -> str:
        speechsdk = self.speechsdk
        if voice:
            self.speech_config.speech_synthesis_voice_name = voice

        audio_config = speechsdk.audio.AudioOutputConfig(filename=out_path)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=audio_config)

        logger.info("AzureSpeechTTS: synthesizing to %s (voice: %s)", out_path, voice or "default")
        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return out_path
        else:
            err = getattr(result, "error_details", None)
            raise RuntimeError(f"Azure Speech synthesis failed: {err}")


class ElevenLabsTTS(AbstractTTS):
    """ElevenLabs Text-to-Speech adapter.
    
    Features:
    - Highest quality AI voices
    - Natural emotion and prosody
    - Voice cloning capabilities
    - Premium voice library
    
    Requires: ElevenLabs API key (set ELEVENLABS_API_KEY environment variable)
    """

    def __init__(self, api_key: str | None = None, voice_id: str = "21m00Tcm4TlvDq8ikWAM"):
        """
        Args:
            api_key: ElevenLabs API key (defaults to ELEVENLABS_API_KEY env var)
            voice_id: Default voice ID (default is Rachel, a popular female voice)
        """
        try:
            from elevenlabs import ElevenLabs, Voice, VoiceSettings
        except ImportError:
            raise RuntimeError(
                "ElevenLabs SDK is not installed. Install with: pip install elevenlabs"
            )

        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "ElevenLabs API key required. Set ELEVENLABS_API_KEY or pass api_key argument."
            )

        self.client = ElevenLabs(api_key=self.api_key)
        self.voice_id = voice_id
        self.Voice = Voice
        self.VoiceSettings = VoiceSettings

    def synthesize(self, text: str, out_path: str, voice: Optional[str] = None, language: Optional[str] = None) -> str:
        """Synthesize text to speech using ElevenLabs.
        
        Args:
            text: Text to synthesize
            out_path: Output audio file path
            voice: Voice ID or name (defaults to class default)
            language: Language code (ElevenLabs auto-detects by default)
        """
        voice_id = voice or self.voice_id
        
        logger.info("ElevenLabsTTS: synthesizing to %s (voice: %s)", out_path, voice_id)

        audio = self.client.generate(
            text=text,
            voice=self.Voice(voice_id=voice_id)
        )

        # Save audio to file
        with open(out_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        logger.info("ElevenLabsTTS synthesis complete: %s", out_path)
        return out_path


class GoogleCloudTTS(AbstractTTS):
    """Google Cloud Text-to-Speech adapter.
    
    Features:
    - 400+ voices in 100+ languages
    - Neural voices (WaveNet, Neural2)
    - SSML support
    - Multilingual support
    
    Requires: Google Cloud credentials file (set GOOGLE_APPLICATION_CREDENTIALS env var)
    """

    def __init__(self, project_id: str | None = None, voice_name: str = "en-US-Neural2-A"):
        """
        Args:
            project_id: Google Cloud project ID
            voice_name: Voice to use (e.g., 'en-US-Neural2-A', 'es-ES-Neural2-A')
        """
        try:
            from google.cloud import texttospeech
        except ImportError:
            raise RuntimeError(
                "Google Cloud TTS is not installed. Install with: pip install google-cloud-texttospeech"
            )

        self.texttospeech = texttospeech
        self.client = texttospeech.TextToSpeechClient()
        self.voice_name = voice_name

    def synthesize(self, text: str, out_path: str, voice: Optional[str] = None, language: Optional[str] = None) -> str:
        """Synthesize text to speech using Google Cloud.
        
        Args:
            text: Text to synthesize
            out_path: Output audio file path
            voice: Voice name (e.g., 'en-US-Neural2-A')
            language: Language code (extracted from voice_name if not provided)
        """
        voice_name = voice or self.voice_name
        
        # Extract language from voice name (e.g., 'en-US' from 'en-US-Neural2-A')
        lang_code = voice_name.split('-')[0] + '-' + voice_name.split('-')[1] if '-' in voice_name else "en-US"
        
        logger.info("GoogleCloudTTS: synthesizing to %s (voice: %s)", out_path, voice_name)

        input_text = self.texttospeech.SynthesisInput(text=text)
        voice = self.texttospeech.VoiceSelectionParams(
            language_code=lang_code,
            name=voice_name
        )
        audio_config = self.texttospeech.AudioConfig(
            audio_encoding=self.texttospeech.AudioEncoding.LINEAR16
        )

        response = self.client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )

        # Save audio to file
        with open(out_path, "wb") as out:
            out.write(response.audio_content)

        logger.info("GoogleCloudTTS synthesis complete: %s", out_path)
        return out_path


class ChatterboxTTS(AbstractTTS):
    """Chatterbox TTS - Local, open-source text-to-speech with voice cloning.
    
    Features:
    - Local model (no API calls needed)
    - Voice cloning support (requires reference audio)
    - High-quality synthesis
    - Open-source and free
    - Supports GPU acceleration (CUDA)
    - Multilingual support
    
    Installation: pip install chatterbox-tts torch torchaudio
    """

    def __init__(self, device: str = "cpu"):
        """
        Args:
            device: 'cpu' or 'cuda' for GPU acceleration
        """
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            import chatterbox.tts_turbo as tts_turbo_mod
            import torchaudio as ta
        except ImportError:
            raise RuntimeError(
                "Chatterbox TTS is not installed. Install with: pip install chatterbox-tts torch torchaudio"
            )
        
        self.ChatterboxTurboTTS = ChatterboxTurboTTS
        self.ta = ta
        self.device = device
        
        logger.info(f"Loading ChatterboxTurboTTS model (device: {device})")
        try:
            hf_token = (
                os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
                or os.environ.get("HUGGINGFACE_TOKEN")
            )

            # Apply the workaround early so it also covers chatterbox versions
            # that ignore token=False and force token=True internally.
            _allow_anonymous_hf_downloads_if_needed(hf_token)

            # Some chatterbox versions import `snapshot_download` into their module
            # namespace and call it directly. Patch that symbol too.
            try:
                import inspect

                orig_snapshot = getattr(tts_turbo_mod, "snapshot_download", None)
                if orig_snapshot is not None and not getattr(orig_snapshot, "__vd_token_true_patch__", False):
                    sig = None
                    try:
                        sig = inspect.signature(orig_snapshot)
                    except Exception:
                        sig = None

                    def snapshot_no_token(*args, **kwargs):
                        if sig is not None:
                            try:
                                bound = sig.bind_partial(*args, **kwargs)
                                if bound.arguments.get("token") is True:
                                    bound.arguments["token"] = hf_token if hf_token else False
                                return orig_snapshot(*bound.args, **bound.kwargs)
                            except Exception:
                                pass

                        if kwargs.get("token") is True:
                            kwargs["token"] = hf_token if hf_token else False
                        return orig_snapshot(*args, **kwargs)

                    snapshot_no_token.__vd_token_true_patch__ = True
                    tts_turbo_mod.snapshot_download = snapshot_no_token
            except Exception:
                pass

            # Some chatterbox-tts builds call into huggingface_hub with token=True by default,
            # which raises LocalTokenNotFoundError when you're not logged in.
            # We prefer:
            # - use an explicit token if provided
            # - otherwise avoid requiring a token (token=False)
            try:
                # Prefer hf_kwargs if supported (newer chatterbox-tts).
                self.model = ChatterboxTurboTTS.from_pretrained(
                    device=device,
                    hf_kwargs={"token": hf_token if hf_token else False},
                )
            except TypeError:
                # Back-compat: older chatterbox-tts may not accept token=...
                self.model = ChatterboxTurboTTS.from_pretrained(device=device)
            self.sr = self.model.sr  # Sample rate
            logger.info(f"ChatterboxTurboTTS loaded successfully (sample rate: {self.sr})")
        except Exception as e:
            raise RuntimeError(
                "Failed to load ChatterboxTurboTTS. "
                "If this is a Hugging Face token error, either run `hf auth login` "
                "or set `HF_TOKEN` / `HUGGINGFACEHUB_API_TOKEN` in the environment. "
                f"Original error: {e}"
            )

    def synthesize(self, text: str, out_path: str, voice: Optional[str] = None, language: Optional[str] = None) -> str:
        """Synthesize text to speech using Chatterbox Turbo.
        
        Args:
            text: Text to synthesize
            out_path: Output audio file path
            voice: Path to reference audio clip for voice cloning (10s recommended)
            language: Language code (not used by Chatterbox, but kept for compatibility)
        """
        try:
            logger.info("ChatterboxTTS: synthesizing to %s", out_path)
            
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            
            # Generate audio with voice cloning if reference provided
            if voice and os.path.exists(voice):
                logger.info(f"Synthesizing with voice cloning from: {voice}")
                wav = self.model.generate(text, audio_prompt_path=voice)
            else:
                # Standard synthesis without voice cloning
                logger.info("Synthesizing with default voice")
                wav = self.model.generate(text)
            
            # Save audio file
            self.ta.save(out_path, wav, self.sr)
            logger.info("ChatterboxTTS synthesis complete: %s", out_path)
            return out_path
        
        except Exception as e:
            logger.error(f"ChatterboxTTS synthesis failed: {e}")
            raise RuntimeError(f"ChatterboxTTS failed: {e}")


class PyttartoTTS(AbstractTTS):
    """Fallback: Pyttsx3 Local TTS (no internet, no setup needed).
    
    Uses pyttsx3 for local text-to-speech synthesis.
    Best for quick testing when APIs are unavailable.
    """

    def __init__(self):
        """Initialize with pyttsx3."""
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 is not installed")
        self.engine = pyttsx3.init()
        logger.info("Pyttsx3 TTS initialized (fallback mode)")

    def synthesize(self, text: str, out_path: str, voice: Optional[str] = None, language: Optional[str] = None) -> str:
        """Synthesize using pyttsx3."""
        logger.info("Pyttsx3: synthesizing to %s (fallback, local mode)", out_path)
        
        if voice:
            try:
                self.engine.setProperty("voice", voice)
            except Exception:
                logger.warning("Failed to set voice %s", voice)
        
        self.engine.save_to_file(text, out_path)
        self.engine.runAndWait()
        return out_path


