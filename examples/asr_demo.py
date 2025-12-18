#!/usr/bin/env python3
"""
ASR Demo: Whisper + Pyannote Speaker Diarization Example

This script demonstrates how to use the WhisperWithDiarizationASR class
for transcription with speaker identification.

Usage:
    python examples/asr_demo.py <audio_file>
    
Example:
    python examples/asr_demo.py work/extracted_audio.wav
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_basic_transcription(audio_path: str):
    """Demo 1: Basic transcription with speaker diarization."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Transcription with Speaker Diarization")
    print("=" * 70)
    
    from src.asr import WhisperWithDiarizationASR
    
    # Initialize ASR (models loaded on first run)
    print("\n[1/3] Loading Whisper and Pyannote models...")
    asr = WhisperWithDiarizationASR(whisper_model="base", device="cpu")
    
    # Transcribe
    print(f"[2/3] Transcribing audio: {audio_path}")
    result = asr.transcribe(audio_path)
    
    # Display results
    print(f"[3/3] Transcription complete!\n")
    print(f"Full text ({len(result.text)} characters):")
    print("-" * 70)
    print(result.text)
    print("-" * 70)
    
    print(f"\nSegments ({len(result.segments)} total):")
    print("-" * 70)
    for i, segment in enumerate(result.segments, 1):
        duration = segment.get('duration', 0)
        offset = segment.get('offset', 0)
        speaker = segment.get('speaker', 'Unknown')
        text = segment.get('text', '')[:60]  # First 60 chars
        
        print(f"\n[Segment {i}]")
        print(f"  Time: {offset:.2f}s - {offset + duration:.2f}s ({duration:.2f}s)")
        print(f"  Speaker: {speaker}")
        print(f"  Text: {text}{'...' if len(segment.get('text', '')) > 60 else ''}")
    
    print("\n" + "=" * 70)


def demo_model_comparison(audio_path: str):
    """Demo 2: Compare different Whisper model sizes."""
    print("\n" + "=" * 70)
    print("DEMO 2: Model Size Comparison")
    print("=" * 70)
    
    from src.asr import WhisperWithDiarizationASR
    import time
    
    models = ["tiny", "base"]  # Just test two for speed
    
    for model_name in models:
        print(f"\n📊 Testing model: {model_name}")
        print("-" * 70)
        
        asr = WhisperWithDiarizationASR(
            whisper_model=model_name,
            device="cpu"
        )
        
        start = time.time()
        result = asr.transcribe(audio_path)
        elapsed = time.time() - start
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Segments: {len(result.segments)}")
        print(f"  Text length: {len(result.text)} characters")
        print(f"  Sample: {result.text[:80]}...")
    
    print("\n" + "=" * 70)


def demo_segment_analysis(audio_path: str):
    """Demo 3: Deep dive into segment analysis."""
    print("\n" + "=" * 70)
    print("DEMO 3: Segment Analysis")
    print("=" * 70)
    
    from src.asr import WhisperWithDiarizationASR
    
    asr = WhisperWithDiarizationASR(whisper_model="base", device="cpu")
    result = asr.transcribe(audio_path)
    
    # Analyze speakers
    speakers = set()
    speaker_segments = {}
    
    for segment in result.segments:
        speaker = segment.get('speaker', 'Unknown')
        speakers.add(speaker)
        
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append(segment)
    
    print(f"\n👥 Speaker Analysis:")
    print("-" * 70)
    print(f"Total speakers detected: {len(speakers)}")
    
    for speaker in sorted(speakers):
        segments = speaker_segments[speaker]
        total_time = sum(s.get('duration', 0) for s in segments)
        total_words = sum(len(s.get('text', '').split()) for s in segments)
        
        print(f"\n  {speaker}:")
        print(f"    - Segments: {len(segments)}")
        print(f"    - Total time: {total_time:.2f}s")
        print(f"    - Total words: {total_words}")
        print(f"    - Avg words/segment: {total_words / len(segments):.1f}")
    
    # Timeline visualization
    print(f"\n📈 Timeline (approximate):")
    print("-" * 70)
    
    # Find max time
    max_time = max((s.get('offset', 0) + s.get('duration', 0)) 
                   for s in result.segments) if result.segments else 0
    
    # Create simple timeline (scaled to 70 chars)
    timeline = ['.' for _ in range(70)]
    speaker_chars = {}
    char_idx = 0
    
    for speaker in sorted(speakers):
        speaker_chars[speaker] = chr(65 + char_idx)  # A, B, C, etc.
        char_idx += 1
    
    for segment in result.segments:
        speaker = segment.get('speaker', 'Unknown')
        offset = segment.get('offset', 0)
        duration = segment.get('duration', 0)
        
        if max_time > 0:
            start_pos = int((offset / max_time) * 69)
            end_pos = int(((offset + duration) / max_time) * 69)
            char = speaker_chars.get(speaker, '?')
            
            for pos in range(max(0, start_pos), min(70, end_pos + 1)):
                timeline[pos] = char
    
    timeline_str = ''.join(timeline)
    print(f"  {timeline_str}")
    print(f"  {0:.0f}s{''.join([' ' * 8 for _ in range(8)])}{max_time:.0f}s")
    
    # Legend
    print(f"\n  Legend:")
    for speaker in sorted(speakers):
        char = speaker_chars[speaker]
        print(f"    {char} = {speaker}")
    
    print("\n" + "=" * 70)


def demo_use_in_pipeline(audio_path: str):
    """Demo 4: Integration with DubbingPipeline."""
    print("\n" + "=" * 70)
    print("DEMO 4: Using ASR in DubbingPipeline")
    print("=" * 70)
    
    from src.pipeline import DubbingPipeline, PipelineConfig
    from src.asr import WhisperWithDiarizationASR
    
    print("\n[1/2] Creating pipeline with WhisperWithDiarizationASR...")
    config = PipelineConfig(work_dir="work")
    
    # The pipeline automatically detects and uses WhisperWithDiarizationASR
    # But we can also explicitly pass it:
    asr = WhisperWithDiarizationASR(whisper_model="base", device="cpu")
    pipeline = DubbingPipeline(asr=asr, config=config)
    
    print(f"[2/2] ASR configured: {pipeline.asr.__class__.__name__}\n")
    
    # Show what segments contain
    print("Sample segment structure:")
    print("""
    {
        "text": "Hello, how are you?",
        "speaker": "Speaker_1",
        "offset": 1.23,           # Start time in seconds
        "duration": 2.5,          # Duration in seconds
        "confidence": 0.95,       # Whisper confidence
        "words": [...]            # Word-level details (optional)
    }
    """)
    
    print("Use segments in your code:")
    print("""
    result = asr.transcribe("audio.wav")
    for segment in result.segments:
        print(f"Speaker {segment['speaker']}: {segment['text']}")
        
        # Can be used for:
        # - Emotion-aware translation per speaker
        # - Lip-sync alignment
        # - Speaker-specific TTS
        # - Audio mixing by speaker
    """)
    
    print("=" * 70)


def main():
    """Run ASR demonstrations."""
    
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n❌ Error: Audio file path required")
        print("Usage: python examples/asr_demo.py <audio_file>")
        print("\nExample with your work directory:")
        print("  python examples/asr_demo.py work/extracted_audio.wav")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    if not Path(audio_path).exists():
        print(f"\n❌ Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    print(f"🎙️  ASR Demo - Using audio: {audio_path}\n")
    
    # Run demonstrations
    try:
        demo_basic_transcription(audio_path)
        
        # Only run other demos if explicitly requested
        if "--full" in sys.argv:
            demo_model_comparison(audio_path)
            demo_segment_analysis(audio_path)
            demo_use_in_pipeline(audio_path)
        else:
            print("\n💡 Tip: Run with --full flag to see more demos:")
            print("   python examples/asr_demo.py <audio_file> --full")
    
    except Exception as e:
        logger.error(f"Error during demo: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
