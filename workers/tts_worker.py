"""TTS Worker - runs in 'tts' conda environment.

This script synthesizes speech using various TTS backends.
It's designed to be called from the main pipeline via subprocess.

Usage:
    python tts_worker.py <text> <language> <output_audio> [--voice VOICE_ID] [--tts BACKEND]
    
Example:
    python tts_worker.py "Hola mundo" "es" "output.wav" --tts chatterbox
    python tts_worker.py "Hello" "en" "output.wav" --tts coqui --voice reference_audio.wav
"""

import sys
import json
import logging
import argparse
import os
from pathlib import Path

# Add parent directory to path so we can import src
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _allow_anonymous_hf_downloads_if_needed(hf_token: str | None = None) -> None:
    """Work around upstream callers passing `token=True` unconditionally.

    Newer `huggingface_hub` versions validate `token=True` and raise
    `LocalTokenNotFoundError` when the user isn't logged in, even for public repos.

    Important: patching validators is not sufficient because `hf_hub_download` and
    friends are decorated at import-time. We instead wrap the public download
    entry points and coerce `token=True` to anonymous (`token=None`).

    If the model is gated/private, the download will still fail (401/403) and the
    user must authenticate via `hf auth login` or an env token.
    """

    try:
        # Patch the exact function that raises the error seen in the traceback:
        # huggingface_hub.utils._headers.get_token_to_send.
        # This is more robust than patching callers because downstream libs may
        # import snapshot_download into their own module namespace.
        try:
            import huggingface_hub.utils._headers as headers

            orig_get = getattr(headers, "get_token_to_send", None)
            if orig_get is not None and not getattr(orig_get, "__vd_token_true_patch__", False):

                def patched_get_token_to_send(token):
                    if token is True:
                        # Prefer an explicitly provided token; otherwise allow anonymous.
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
                # Handle both keyword and positional token arguments.
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

        # Common entry points used by downstream libs.
        _wrap(huggingface_hub, "hf_hub_download")
        _wrap(huggingface_hub, "snapshot_download")

        # Some code paths import directly from submodules.
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


def main():
    parser = argparse.ArgumentParser(description="TTS Worker for multiple TTS backends")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("language", help="Language code (e.g., 'en', 'es', 'fr')")
    parser.add_argument("output_audio", help="Output audio file path")
    parser.add_argument("--voice", default=None, help="Voice/reference audio (optional)")
    parser.add_argument("--device", default="cpu", help="Device: 'cpu' or 'cuda' (for Chatterbox/Coqui)")
    parser.add_argument(
        "--tts",
        default="pyttsx3",
        help="TTS backend: pyttsx3 (default), chatterbox, or coqui (if available)",
    )
    
    args = parser.parse_args()
    
    logger.info(f"TTS Worker: Synthesizing '{args.text}' (language: {args.language}, backend: {args.tts})")
    
    try:
        if args.tts.lower() == "chatterbox":
            logger.info(f"Loading ChatterboxTurboTTS model (device: {args.device})...")

            try:
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                import chatterbox.tts_turbo as tts_turbo_mod
                import torchaudio as ta

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

                # Prefer hf_kwargs if supported (newer chatterbox-tts).
                try:
                    model = ChatterboxTurboTTS.from_pretrained(
                        device=args.device,
                        hf_kwargs={"token": hf_token if hf_token else False},
                    )
                except TypeError:
                    # Back-compat: older chatterbox-tts may not accept hf_kwargs/token.
                    model = ChatterboxTurboTTS.from_pretrained(device=args.device)

                logger.info("✓ ChatterboxTurboTTS model loaded successfully")

                # Generate audio (with optional voice cloning)
                if args.voice and os.path.exists(args.voice):
                    logger.info(f"Synthesizing with voice cloning: {args.voice}")
                    wav = model.generate(args.text, audio_prompt_path=args.voice)
                else:
                    wav = model.generate(args.text)

                sr = getattr(model, "sr", 24000)
                ta.save(args.output_audio, wav, sr)
                backend_used = "chatterbox-turbo"

            except Exception as e:
                logger.exception("Chatterbox TTS backend exception")
                msg = str(e)
                if "Audio prompt" in msg and "longer than" in msg and "5" in msg:
                    raise RuntimeError(
                        "Chatterbox voice-clone prompt too short. "
                        "Provide a reference clip longer than 5 seconds (6-10s recommended), "
                        f"or disable per-segment voice cloning. Original error: {e}"
                    )

                raise RuntimeError(
                    "Chatterbox TTS failed to initialize/run. "
                    "If the model download is gated/private, run `hf auth login` in the 'tts' env "
                    "or set `HF_TOKEN` / `HUGGINGFACEHUB_API_TOKEN`. "
                    f"Original error: {e}"
                )

        elif args.tts.lower() == "coqui":
            logger.info(f"Loading Coqui TTS xtts_v2 model (device: {args.device})...")
            
            try:
                from TTS.api import TTS
                
                # Use xtts_v2 - best quality multilingual model with zero-shot voice cloning
                tts = TTS(
                    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                    gpu=(args.device == "cuda"),
                    progress_bar=False
                )
                logger.info("✓ xtts_v2 model loaded successfully")
                
                # Synthesize with xtts_v2
                if args.voice and os.path.exists(args.voice):
                    # Zero-shot voice cloning with reference audio
                    logger.info(f"Synthesizing with voice cloning: {args.voice}")
                    tts.tts_to_file(
                        text=args.text,
                        file_path=args.output_audio,
                        speaker_wav=args.voice,
                        language=args.language
                    )
                else:
                    # Standard synthesis with default voice
                    logger.info(f"Synthesizing for language: {args.language}")
                    tts.tts_to_file(
                        text=args.text,
                        file_path=args.output_audio,
                        language=args.language
                    )
                backend_used = "coqui"
                
            except ImportError as e:
                logger.warning(f"Coqui TTS not available: {e}. Falling back to pyttsx3...")
                from src.tts import Pyttsx3TTS
                tts = Pyttsx3TTS()
                tts.synthesize(args.text, args.output_audio, voice=args.voice, language=args.language)
                backend_used = "pyttsx3 (fallback)"
        else:
            # Default to pyttsx3 (local, no dependencies)
            logger.info("Using pyttsx3 TTS (local, no setup needed)...")
            from src.tts import Pyttsx3TTS
            
            tts = Pyttsx3TTS()
            logger.info("✓ Pyttsx3 TTS loaded successfully")
            
            # Synthesize
            tts.synthesize(args.text, args.output_audio, voice=args.voice, language=args.language)
            backend_used = "pyttsx3"
        
        # Return success
        output_data = {
            "status": "success",
            "audio": args.output_audio,
            "text": args.text,
            "language": args.language,
            "tts_backend": backend_used
        }
        
        logger.info(f"TTS Worker: Synthesis complete. Saved to {args.output_audio}")
        print(json.dumps(output_data))
        
    except Exception as e:
        error_data = {
            "status": "error",
            "error": str(e)
        }
        
        logger.error(f"TTS Worker failed: {e}")
        print(json.dumps(error_data))

        sys.exit(1)


if __name__ == "__main__":
    main()

