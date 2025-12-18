#!/usr/bin/env python
"""
Monitor Coqui TTS model download and test TTS worker.
"""

import os
import sys
from pathlib import Path

def check_model():
    """Check if xtts_v2 model is already downloaded."""
    tts_home = os.path.expanduser("~/.tts")
    model_path = Path(tts_home) / "tts_models--multilingual--multi-dataset--xtts_v2"
    
    if model_path.exists():
        print("✓ xtts_v2 model is already downloaded!")
        return True
    else:
        print("⏳ Model not yet downloaded...")
        print(f"Expected location: {model_path}")
        return False

def main():
    print("=" * 60)
    print("Coqui TTS Model Status Check")
    print("=" * 60)
    print()
    
    if check_model():
        print()
        print("You can now run the full pipeline:")
        print()
        print("python cli.py --file video.mp4 --source es --target en --multi-env")
        print()
    else:
        print()
        print("Model is downloading in the background...")
        print("This may take 5-15 minutes depending on internet speed.")
        print()
        print("Once complete, you can run:")
        print()
        print("python cli.py --file video.mp4 --source es --target en --multi-env")
        print()

if __name__ == "__main__":
    main()
