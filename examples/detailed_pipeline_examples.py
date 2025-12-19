"""
Detailed Dubbing Pipeline Example

Demonstrates how to use the key components:
1. Audio segmentation (logical + speaker-based)
2. Alignment
3. Speaker-specific TTS
4. Complete dubbing workflow
"""

import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_segmentation():
    """Example: Segment audio based on ASR and speaker changes."""
    from src.segmentation import AudioSegmenter, SegmentationValidator
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Audio Segmentation")
    print("="*80)
    
    # Simulated ASR segments (from Whisper + diarization)
    asr_segments = [
        {
            "text": "Hello, this is speaker one.",
            "speaker": "Speaker_1",
            "offset": 0.0,
            "duration": 2.0,
            "confidence": 0.95,
            "words": []
        },
        {
            "text": "I'm here to discuss the project.",
            "speaker": "Speaker_1",
            "offset": 2.1,
            "duration": 2.5,
            "confidence": 0.92,
            "words": []
        },
        {
            "text": "That's interesting. Tell me more.",
            "speaker": "Speaker_2",
            "offset": 4.7,
            "duration": 2.0,
            "confidence": 0.90,
            "words": []
        },
    ]
    
    # Simulated speaker diarization segments
    speaker_segments = [
        {"start": 0.0, "end": 4.6, "speaker": "Speaker_1"},
        {"start": 4.6, "end": 6.7, "speaker": "Speaker_2"},
    ]
    
    # Create segmenter
    segmenter = AudioSegmenter(min_segment_duration=0.5)
    
    # Perform segmentation
    result = segmenter.segment(asr_segments, speaker_segments)
    
    print(f"\nSegmentation Result:")
    print(f"  Total segments: {len(result.segments)}")
    print(f"  Speakers: {result.speakers}")
    print(f"  Total duration: {result.total_duration:.2f}s")
    
    print(f"\nSegments:")
    for seg in result.segments:
        print(f"  {seg}")
    
    # Validate
    is_valid, warnings = SegmentationValidator.validate(result)
    print(f"\nValidation: {'PASSED' if is_valid else 'WARNINGS'}")
    for warning in warnings:
        print(f"  ⚠️  {warning}")


def example_alignment():
    """Example: Align segments between source and target."""
    from src.alignment import SegmentAligner, TimingAnalyzer, AlignmentStrategy
    from src.segmentation import Segment, SegmentType
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Segment Alignment")
    print("="*80)
    
    # Create sample segments
    segments = [
        Segment(
            id=0,
            text="Hello, this is speaker one.",
            speaker="Speaker_1",
            start_time=0.0,
            end_time=2.0,
            segment_type=SegmentType.LOGICAL,
            confidence=0.95
        ),
        Segment(
            id=1,
            text="I'm here to discuss the project.",
            speaker="Speaker_1",
            start_time=2.1,
            end_time=4.6,
            segment_type=SegmentType.LOGICAL,
            confidence=0.92
        ),
        Segment(
            id=2,
            text="That's interesting. Tell me more.",
            speaker="Speaker_2",
            start_time=4.7,
            end_time=6.7,
            segment_type=SegmentType.LOGICAL,
            confidence=0.90
        ),
    ]
    
    # Simulated TTS output durations (may differ from source)
    target_durations = {
        0: 2.5,  # Stretched: longer TTS output
        1: 3.8,  # Stretched
        2: 2.1,  # Slightly longer
    }
    
    # Create aligner
    aligner = SegmentAligner(strategy=AlignmentStrategy.ADAPTIVE)
    
    # Perform alignment
    alignment_results = aligner.align_segments(segments, target_durations)
    
    print(f"\nAlignment Results:")
    for result in alignment_results:
        print(f"\n  Segment {result.segment_id}:")
        print(f"    Status: {result.alignment_status}")
        print(f"    Source:  {result.source_start:.2f}s - {result.source_end:.2f}s "
              f"({result.metadata['source_duration']:.2f}s)")
        print(f"    Target:  {result.target_start:.2f}s - {result.target_end:.2f}s "
              f"({result.metadata['target_duration']:.2f}s)")
        print(f"    Scaling: {result.metadata['scaling_factor']:.2f}x")
    
    # Analyze timing
    analyzer = TimingAnalyzer()
    stats = analyzer.analyze(alignment_results)
    
    print(f"\nTiming Analysis:")
    print(f"  Source duration: {stats['total_source_duration']:.2f}s")
    print(f"  Target duration: {stats['total_target_duration']:.2f}s")
    print(f"  Average scaling: {stats['average_scaling_factor']:.2f}x")
    print(f"  Status distribution: {stats['status_distribution']}")


def example_speaker_tts():
    """Example: Configure and use speaker-specific TTS."""
    from src.speaker_tts import (
        SpeakerProfile, 
        VoiceCloneMethod,
        create_speaker_profiles_from_segments
    )
    from src.segmentation import Segment, SegmentType
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Speaker-Specific TTS Configuration")
    print("="*80)
    
    # Create sample segments
    segments = [
        Segment(id=0, text="Text", speaker="Speaker_1", start_time=0.0, 
                end_time=2.0, segment_type=SegmentType.LOGICAL),
        Segment(id=1, text="Text", speaker="Speaker_2", start_time=2.1, 
                end_time=4.6, segment_type=SegmentType.LOGICAL),
    ]
    
    # Create default profiles
    profiles = create_speaker_profiles_from_segments(segments)
    
    print(f"\nDefault Speaker Profiles:")
    for speaker_id, profile in profiles.items():
        print(f"\n  {speaker_id}:")
        print(f"    Language: {profile.language}")
        print(f"    Clone method: {profile.clone_method.value}")
        print(f"    Voice reference: {profile.voice_reference}")
    
    # Customize profiles with voice cloning
    print(f"\nCustomizing profiles with voice cloning...")
    
    # Example: Add voice reference for Speaker_1
    if "Speaker_1" in profiles:
        profiles["Speaker_1"].voice_reference = "reference_audio/speaker1.wav"
        profiles["Speaker_1"].clone_method = VoiceCloneMethod.ZERO_SHOT
        print(f"  Speaker_1: Added reference audio for voice cloning")
    
    # Create a custom profile with specific voice
    custom_profile = SpeakerProfile(
        speaker_id="Narrator",
        language="es",
        name="Professional Narrator",
        voice_name="professional_male",
        clone_method=VoiceCloneMethod.VOICE_NAME,
        pace=0.95,
        pitch=1.1,
        emotion="professional"
    )
    
    print(f"\n  Custom Narrator profile:")
    print(f"    Voice: {custom_profile.voice_name}")
    print(f"    Pace: {custom_profile.pace}x")
    print(f"    Pitch: {custom_profile.pitch}x")
    print(f"    Emotion: {custom_profile.emotion}")


def example_detailed_pipeline():
    """Example: Run the complete detailed dubbing pipeline."""
    from src.pipeline_detailed import (
        DetailedDubbingPipeline, 
        DetailedPipelineConfig
    )
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Complete Detailed Dubbing Pipeline")
    print("="*80)
    
    # Configuration
    config = DetailedPipelineConfig(
        work_dir="work",
        output_dir="output",
        sample_rate=16000,
        min_segment_duration=0.5,
        alignment_strategy="adaptive",
        tts_device="cpu",  # Use GPU if available: "cuda"
        preserve_speaker_identity=True,
        debug=True
    )
    
    print(f"\nPipeline Configuration:")
    print(f"  Work directory: {config.work_dir}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Sample rate: {config.sample_rate} Hz")
    print(f"  TTS device: {config.tts_device}")
    print(f"  Preserve speaker identity: {config.preserve_speaker_identity}")
    
    # Initialize pipeline
    print(f"\nInitializing pipeline...")
    try:
        pipeline = DetailedDubbingPipeline(config=config)
        print(f"  ✓ Pipeline initialized successfully")
    except Exception as e:
        print(f"  ✗ Pipeline initialization failed: {e}")
        print(f"\n  Note: Some components may require additional dependencies:")
        print(f"    - Whisper: pip install openai-whisper")
        print(f"    - Pyannote: pip install pyannote.audio")
        print(f"    - Coqui TTS: pip install TTS torch torchaudio")
        return
    
    print(f"\nPipeline Stages:")
    print(f"  1. Audio extraction from video")
    print(f"  2. Transcription with speaker diarization")
    print(f"  3. Segmentation (logical + speaker boundaries)")
    print(f"  4. Speaker registration (voice cloning setup)")
    print(f"  5. Translation (preserving speaker metadata)")
    print(f"  6. TTS synthesis (speaker-specific voices)")
    print(f"  7. Alignment (timing synchronization)")
    print(f"  8. Output generation (metadata + audio)")
    
    print(f"\nUsage:")
    print(f"""
    # Run the pipeline on a video file
    result = pipeline.run(
        video_path="input_video.mp4",
        source_lang="en",
        target_lang="es",
        speaker_reference_audio={{
            "Speaker_1": "reference_audio/speaker1.wav",
            "Speaker_2": "reference_audio/speaker2.wav"
        }}
    )
    
    # Access results
    if result["status"] == "success":
        print(f"Pipeline completed successfully!")
        print(f"Output saved to: {{result['stages']['output']['output_directory']}}")
    else:
        print(f"Pipeline failed: {{result.get('error')}}")
    """)


def example_workflow_walkthrough():
    """Walkthrough of a complete dubbing workflow."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Complete Dubbing Workflow Walkthrough")
    print("="*80)
    
    print("""
WORKFLOW OVERVIEW
=================

1. INPUT
   ├─ Video file (e.g., meeting_video.mp4)
   └─ Configuration (source language, target language)

2. AUDIO EXTRACTION
   ├─ Extract audio track from video
   └─ Resample to standard rate (16000 Hz)

3. ASR + SPEAKER DIARIZATION
   ├─ Transcribe audio using Whisper
   ├─ Identify speakers using Pyannote diarization
   └─ Output: Segments with text and speaker labels
       Example:
         Segment 0: "Hello, I'm Alice" (Speaker_1, 0.0s-2.0s)
         Segment 1: "Hi Alice, how are you?" (Speaker_2, 2.1s-4.0s)

4. SEGMENTATION
   ├─ Combine ASR segments with speaker boundaries
   ├─ Split segments when speaker changes
   └─ Output: Refined segments respecting both logical and speaker boundaries
       Example:
         Segment 0: "Hello, I'm Alice" (Speaker_1, 0.0s-2.0s, LOGICAL)
         Segment 1: "Hi Alice, how are you?" (Speaker_2, 2.1s-4.0s, LOGICAL)

5. SPEAKER PROFILE REGISTRATION
   ├─ Create speaker profiles for TTS
   ├─ Register voice reference audio for voice cloning (optional)
   ├─ Configure voice characteristics (pace, pitch, emotion)
   └─ Output: Speaker profiles ready for TTS
       Example Profile:
         Speaker_1: voice_reference="alice.wav", language="en"
         Speaker_2: voice_reference="bob.wav", language="en"

6. TRANSLATION
   ├─ Translate each segment's text
   ├─ Preserve speaker metadata
   └─ Output: Translated segments with speaker info
       Example (English → Spanish):
         Segment 0: "Hola, soy Alice" (Speaker_1, 0.0s-2.0s)
         Segment 1: "Hola Alice, ¿cómo estás?" (Speaker_2, 2.1s-4.0s)

7. TEXT-TO-SPEECH SYNTHESIS
   ├─ Synthesize each segment using speaker-specific voice
   ├─ Use voice cloning if reference available
   ├─ Output: Audio file for each segment
   └─ Timing Info: Actual duration of synthesized audio
       Example Output:
         segment_0.wav: 2.3s (slightly longer than source 2.0s)
         segment_1.wav: 2.1s (shorter than source 2.5s)

8. ALIGNMENT
   ├─ Compare source and target segment durations
   ├─ Adjust timing to avoid overlaps/gaps
   ├─ Generate timing maps for video synchronization
   └─ Output: Alignment metadata with scaling factors
       Example Alignment:
         Segment 0: stretch from 2.0s to 2.3s (scale 1.15x)
         Segment 1: compress from 2.5s to 2.1s (scale 0.84x)

9. OUTPUT GENERATION
   ├─ Generate synthesis report (success rates, metadata)
   ├─ Save segment information (text, speaker, timing)
   ├─ Save alignment information (timing maps)
   └─ Output files:
       - synthesis_report.json: TTS synthesis details
       - segments.json: Segment metadata
       - alignment.json: Timing alignment info
       - segments/segment_*.wav: Dubbed audio segments

10. FINAL ASSEMBLY (Future Stage)
    ├─ Combine all segments into single dubbed audio track
    ├─ Synchronize with video using alignment information
    └─ Output: Final dubbed video file


KEY FEATURES
============

✓ Logical Segmentation
  - Respects ASR sentence boundaries
  - Preserves sentence context

✓ Speaker-Based Segmentation
  - Detects speaker changes automatically
  - Splits segments when speaker changes
  - Ensures each segment is monospeak (single speaker)

✓ Speaker-Specific TTS
  - Voice cloning from reference audio
  - Preserves original speaker characteristics
  - Supports multiple speakers in single video

✓ Timing Alignment
  - Handles TTS duration differences
  - Maintains video-audio synchronization
  - Provides timing maps for video editors

✓ Translation with Context
  - Preserves speaker metadata through translation
  - Maintains segment boundaries during translation
  - Supports multiple languages


EXAMPLE: Meeting Translation (English → Spanish)
==================================================

SOURCE VIDEO: business_meeting.mp4
├─ Duration: 10 minutes
├─ Speakers: 3 people
└─ Language: English

AFTER PROCESSING:

1. Segmentation identifies 47 segments
   ├─ Speaker A: 20 segments
   ├─ Speaker B: 18 segments
   └─ Speaker C: 9 segments

2. Translation produces Spanish equivalent
   └─ Total translated text: ~5000 characters

3. Voice cloning for each speaker
   ├─ Speaker A: Cloned from reference_A.wav
   ├─ Speaker B: Cloned from reference_B.wav
   └─ Speaker C: Cloned from reference_C.wav

4. TTS synthesis generates 47 audio segments
   ├─ Total dubbed audio: 9:45 (vs 10:00 original)
   └─ Alignment adjusts timing to match video

5. Final dubbed video
   └─ Spanish audio track synchronized to video
   └─ Speaker identities preserved

OUTPUT FILES
============
output/
├─ segments/
│  ├─ segment_0000.wav
│  ├─ segment_0001.wav
│  └─ ...
├─ synthesis_report.json      # TTS synthesis info
├─ segments.json              # Segment metadata
├─ alignment.json             # Timing information
└─ pipeline_metadata.json     # Complete execution log
    """)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DETAILED DUBBING PIPELINE - EXAMPLES AND WALKTHROUGHS")
    print("="*80)
    
    # Run examples
    example_segmentation()
    example_alignment()
    example_speaker_tts()
    example_detailed_pipeline()
    example_workflow_walkthrough()
    
    print("\n" + "="*80)
    print("END OF EXAMPLES")
    print("="*80)
    print("\nNext steps:")
    print("  1. Install dependencies: pip install -r requirements-detailed.txt")
    print("  2. Prepare reference audio for voice cloning (optional)")
    print("  3. Run the pipeline on your video file")
    print("  4. Monitor output in 'output/' directory")
