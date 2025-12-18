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


def main():
    parser = argparse.ArgumentParser(description="TTS Worker for multiple TTS backends")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("language", help="Language code (e.g., 'en', 'es', 'fr')")
    parser.add_argument("output_audio", help="Output audio file path")
    parser.add_argument("--voice", default=None, help="Voice/reference audio (optional)")
    parser.add_argument("--device", default="cpu", help="Device: 'cpu' or 'cuda' (for Coqui)")
    parser.add_argument("--tts", default="pyttsx3", help="TTS backend: pyttsx3 (default, local), or coqui (if available)")
    
    args = parser.parse_args()
    
    logger.info(f"TTS Worker: Synthesizing '{args.text}' (language: {args.language}, backend: {args.tts})")
    
    try:
        if args.tts.lower() == "coqui":
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

