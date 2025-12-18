#!/usr/bin/env python
"""Create a simple test video with audio for testing the pipeline."""

from pathlib import Path
import wave
import struct
import math

# Create work directory
Path("work").mkdir(exist_ok=True)

# Create a test WAV file with simple tone
wav_path = Path("work") / "test_en.wav"
print(f"Creating test audio: {wav_path}")

# Parameters
sample_rate = 16000
duration = 2  # seconds
frequency = 440  # Hz (A note)

# Create WAV file
with wave.open(str(wav_path), 'w') as wav_file:
    wav_file.setnchannels(1)  # Mono
    wav_file.setsampwidth(2)  # 16-bit
    wav_file.setframerate(sample_rate)
    
    # Generate sine wave
    for i in range(int(sample_rate * duration)):
        # Simple sine wave
        value = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * i / sample_rate))
        packed_value = struct.pack('<h', value)
        wav_file.writeframes(packed_value)

print(f"✓ Test audio created: {wav_path}")
print(f"  Duration: {duration} seconds")
print(f"  Sample rate: {sample_rate} Hz")
print(f"  Frequency: {frequency} Hz")
print()
print("Test with:")
print(f"  python cli.py --file work/test_en.wav --source en --target es --multi-env")
