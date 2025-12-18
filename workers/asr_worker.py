"""ASR Worker - runs in 'asr' conda environment.

This script transcribes audio using Whisper + Pyannote (with speaker diarization).
It's designed to be called from the main pipeline via subprocess.

Usage:
    python asr_worker.py <audio_path> <language> <output_json>
    
Example:
    python asr_worker.py "audio.wav" "en" "transcript.json"
"""

import sys
import json
import logging
from pathlib import Path

# Add parent directory to path so we can import src
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 4:
        print("Usage: python asr_worker.py <audio_path> <language> <output_json>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    language = sys.argv[2]
    output_json = sys.argv[3]
    
    logger.info(f"ASR Worker: Transcribing {audio_path} (language: {language})")
    
    try:
        from src.asr import WhisperWithDiarizationASR
        
        # Initialize ASR
        asr = WhisperWithDiarizationASR(whisper_model="base", device="cpu")
        
        # Transcribe
        result = asr.transcribe(audio_path, language=language)
        
        # Output as JSON
        output_data = {
            "text": result.text,
            "segments": result.segments,
            "status": "success"
        }
        
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"ASR Worker: Transcription complete. Saved to {output_json}")
        print(json.dumps(output_data))  # Also print to stdout
        
    except Exception as e:
        error_data = {
            "status": "error",
            "error": str(e)
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(error_data, f, indent=2)
        
        logger.error(f"ASR Worker failed: {e}")
        print(json.dumps(error_data))
        sys.exit(1)


if __name__ == "__main__":
    main()
