# Detailed Dubbing Pipeline Components

Complete documentation for speaker segmentation, alignment, and speaker-specific TTS components.

## Overview

The detailed dubbing pipeline introduces three key components that work together to enable professional-quality video dubbing with speaker identity preservation:

```
Video Input
    ↓
Audio Extraction
    ↓
ASR + Speaker Diarization
    ↓
[SEGMENTATION] ← Combines logical + speaker boundaries
    ↓
Segment Metadata
    ↓
[SPEAKER PROFILE REGISTRATION] ← Voice cloning setup
    ↓
Translation (preserves speaker info)
    ↓
[SPEAKER-SPECIFIC TTS] ← Synthesizes with speaker voices
    ↓
[ALIGNMENT] ← Synchronizes timing
    ↓
Final Dubbed Audio + Metadata
```

---

## 1. Audio Segmentation Module (`src/segmentation.py`)

### Purpose
Intelligently segments audio based on both:
- **Logical boundaries**: ASR-detected sentence/phrase boundaries
- **Speaker boundaries**: Speaker change detection from diarization

### Key Classes

#### `Segment`
Represents a contiguous audio segment with consistent properties.

```python
@dataclass
class Segment:
    id: int                    # Segment ID
    text: str                  # Transcribed text
    speaker: str               # Speaker identifier
    start_time: float          # Start in seconds
    end_time: float            # End in seconds
    segment_type: SegmentType  # LOGICAL, SPEAKER_CHANGE, or COMBINED
    confidence: float          # ASR confidence (0-1)
    words: List[dict]          # Word-level timing info
```

#### `SegmentationType`
```python
enum SegmentType:
    LOGICAL = "logical"              # ASR boundary
    SPEAKER_CHANGE = "speaker_change" # Diarization boundary
    COMBINED = "combined"             # Both logical and speaker
```

#### `AudioSegmenter`
Main segmentation engine with intelligent boundary detection.

**Constructor:**
```python
segmenter = AudioSegmenter(
    min_segment_duration=0.5,      # Merge shorter segments
    speaker_change_threshold=0.1   # Confidence threshold
)
```

**Main Method:**
```python
result = segmenter.segment(
    asr_segments=[...],            # From Whisper
    speaker_segments=[...]          # From Pyannote (optional)
)
# Returns: SegmentationResult
```

### Example Usage

```python
from src.segmentation import AudioSegmenter, SegmentationValidator
from src.asr import WhisperWithDiarizationASR

# Get ASR results with speaker diarization
asr = WhisperWithDiarizationASR()
asr_result = asr.transcribe("audio.wav", language="en")

# Segment audio
segmenter = AudioSegmenter(min_segment_duration=0.5)
result = segmenter.segment(asr_result.segments)

# Validate segmentation
is_valid, warnings = SegmentationValidator.validate(result)
print(f"Segments: {len(result.segments)}")
print(f"Speakers: {result.speakers}")
print(f"Total duration: {result.total_duration:.1f}s")

# Iterate through segments
for seg in result.segments:
    print(f"{seg.id}: [{seg.speaker}] {seg.text}")
    print(f"  Time: {seg.start_time:.2f}s - {seg.end_time:.2f}s")
    print(f"  Type: {seg.segment_type.value}")
```

### Segmentation Algorithm

1. **Start with ASR segments** - Each ASR segment becomes a potential segment
2. **Detect speaker changes** - For each ASR segment:
   - Find all speaker diarization segments that overlap
   - If multiple speakers found, create sub-segments for each
3. **Merge short segments** - Segments < min_segment_duration are merged with adjacent segments
4. **Renumber segments** - Ensure sequential IDs

### When Segmentation Occurs

A new segment is created when:
- ✓ ASR detects a sentence/phrase boundary (logical segment)
- ✓ Speaker changes (speaker diarization boundary)
- ✓ Both conditions occur simultaneously

### Validation

`SegmentationValidator` checks for:
- Empty segments
- Invalid timing (duration ≤ 0)
- Overlapping segments
- Gaps between segments
- No segments found

---

## 2. Alignment Module (`src/alignment.py`)

### Purpose
Synchronizes timing between source and target audio when TTS produces different durations than the source.

### Problem it Solves

```
Source Audio Segment:     0.0s ----2.0s---- 4.0s
                                 "Hello"

TTS Output Segment:       0.0s ------2.5s------ 4.5s
                                 "Hola"   (longer!)

Alignment:                0.0s ----2.0s---- 4.0s (preserve source timing)
                          OR
                          0.0s ------2.5s------ 4.5s (allow stretching)
```

### Key Classes

#### `AlignmentStrategy`
```python
enum AlignmentStrategy:
    STRICT = "strict"         # Target must match source duration exactly
    FLEXIBLE = "flexible"     # Allow full duration changes
    ADAPTIVE = "adaptive"     # Preserve start time, adjust end time
```

#### `TimingMap`
Maps source timing to target timing for a segment.

```python
@dataclass
class TimingMap:
    source_start: float       # Original audio start
    source_end: float         # Original audio end
    target_start: float       # Dubbed audio start
    target_end: float         # Dubbed audio end
    scaling_factor: float     # target_duration / source_duration
```

#### `SegmentAligner`
Main alignment engine.

**Constructor:**
```python
aligner = SegmentAligner(
    strategy=AlignmentStrategy.ADAPTIVE,
    slack_time=0.1  # Allow 100ms difference before adjustment
)
```

**Main Method:**
```python
alignment_results = aligner.align_segments(
    source_segments=[...],              # Segment objects
    target_durations={seg_id: duration} # From TTS synthesis
)
```

### Example Usage

```python
from src.alignment import (
    SegmentAligner, 
    TimingAnalyzer, 
    AlignmentStrategy,
    SyncValidator
)

# Assume we have:
# - source_segments: List of Segment objects from segmentation
# - target_durations: Dict mapping segment_id to TTS output duration

# Align segments
aligner = SegmentAligner(strategy=AlignmentStrategy.ADAPTIVE)
alignment_results = aligner.align_segments(
    source_segments,
    target_durations
)

# Analyze timing statistics
analyzer = TimingAnalyzer()
stats = analyzer.analyze(alignment_results)

print(f"Source total: {stats['total_source_duration']:.2f}s")
print(f"Target total: {stats['total_target_duration']:.2f}s")
print(f"Average scaling: {stats['average_scaling_factor']:.2f}x")
print(f"Problematic segments: {len(stats['problematic_segments'])}")

# Validate synchronization
is_valid, issues = SyncValidator.validate_sync(
    alignment_results,
    max_drift=2.0  # Maximum allowed timing drift
)

for issue in issues:
    print(f"⚠️  {issue}")
```

### Alignment Strategies Explained

#### STRICT Strategy
- Target audio duration is forced to match source
- **Pros:** Maintains original timing
- **Cons:** May require speed-up/slow-down, audio quality impact
- **Use when:** Timing precision is critical

```
Source: 0.0s --2.0s-- 4.0s    TTS: 2.5s
Result: 0.0s --2.0s-- 4.0s    (stretch TTS from 2.5s to 2.0s)
```

#### FLEXIBLE Strategy
- Target audio duration is fully accepted
- **Pros:** Preserves natural TTS quality
- **Cons:** Timing may shift significantly
- **Use when:** Audio quality is most important

```
Source: 0.0s --2.0s-- 4.0s    TTS: 2.5s
Result: 0.0s ----2.5s---- 4.5s (shift next segment)
```

#### ADAPTIVE Strategy (Recommended)
- Preserves segment start time
- Adjusts end time based on TTS duration
- Propagates adjustments to subsequent segments
- **Pros:** Balances quality and timing
- **Cons:** Timing may drift over many segments
- **Use for:** Professional dubbing

```
Source: 0.0s --2.0s-- 4.0s    TTS: 2.5s
Result: 0.0s ----2.5s---- 4.5s (start time preserved, end adjusted)
```

---

## 3. Speaker-Specific TTS Module (`src/speaker_tts.py`)

### Purpose
Generate dubbed audio while preserving speaker characteristics through voice cloning.

### Key Classes

#### `VoiceCloneMethod`
```python
enum VoiceCloneMethod:
    ZERO_SHOT = "zero_shot"              # Clone from reference audio
    VOICE_NAME = "voice_name"            # Use predefined voice name
    STYLE_TRANSFER = "style_transfer"    # Transfer style from source
    PROSODY_MATCHING = "prosody_matching" # Match prosody of source
```

#### `SpeakerProfile`
Configuration for a speaker's voice.

```python
@dataclass
class SpeakerProfile:
    speaker_id: str                      # Unique identifier
    language: str                        # Language code (e.g., "en", "es")
    name: Optional[str] = None           # Human-readable name
    voice_reference: Optional[str] = None # Path to reference audio for cloning
    voice_name: Optional[str] = None     # Predefined voice name
    clone_method: VoiceCloneMethod       # How to preserve speaker identity
    
    # Prosody parameters
    pace: float = 1.0                    # Speaking rate (1.0 = normal)
    pitch: float = 1.0                   # Pitch modifier (1.0 = normal)
    energy: float = 1.0                  # Loudness (1.0 = normal)
    
    # Style
    emotion: Optional[str] = None        # "neutral", "happy", "sad", etc.
    style: Optional[str] = None          # Style indication
    
    metadata: Dict = field(default_factory=dict) # Additional data
```

#### `TTSSegment`
A segment to synthesize with speaker information.

```python
@dataclass
class TTSSegment:
    segment_id: int
    text: str
    speaker_id: str
    speaker_profile: SpeakerProfile
    start_time: float
    end_time: float
    language: str
    output_path: Optional[str] = None
    duration: Optional[float] = None
```

#### `AbstractSpeakerTTS`
Abstract base class for TTS backends.

```python
class AbstractSpeakerTTS(ABC):
    @abstractmethod
    def synthesize_segment(self, tts_segment: TTSSegment) -> TTSResult:
        """Synthesize with speaker-specific voice"""
        pass
    
    @abstractmethod
    def register_speaker(self, profile: SpeakerProfile) -> bool:
        """Register speaker for voice cloning"""
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """List supported languages"""
        pass
```

#### `CoquiSpeakerTTS`
Implementation using Coqui TTS (XTTS v2 model).

**Features:**
- Zero-shot voice cloning from reference audio
- Multilingual (16+ languages)
- Prosody control
- Emotion and style support

**Constructor:**
```python
tts = CoquiSpeakerTTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    device="cpu"  # or "cuda" for GPU
)
```

#### `SpeakerTTSOrchestrator`
Orchestrates TTS synthesis for multiple segments.

**Constructor:**
```python
orchestrator = SpeakerTTSOrchestrator(tts_backend)
```

**Main Methods:**
```python
# Register speakers
orchestrator.register_speaker(profile)

# Synthesize segments
results, timing_info = orchestrator.synthesize_segments(
    segments=[...],
    output_dir="output/dubbed",
    language="es"
)

# Get results
durations = orchestrator.get_segment_durations()
report = orchestrator.get_synthesis_report()
```

### Example Usage

#### Basic TTS Synthesis

```python
from src.speaker_tts import (
    CoquiSpeakerTTS, 
    SpeakerTTSOrchestrator,
    SpeakerProfile,
    VoiceCloneMethod
)

# Initialize TTS backend
tts = CoquiSpeakerTTS(device="cuda")  # GPU if available

# Create orchestrator
orchestrator = SpeakerTTSOrchestrator(tts)

# Create speaker profiles
profile_alice = SpeakerProfile(
    speaker_id="Alice",
    language="es",
    name="Alice (Spanish)",
    voice_reference="reference_audio/alice_english.wav",
    clone_method=VoiceCloneMethod.ZERO_SHOT
)

# Register speaker
orchestrator.register_speaker(profile_alice)

# Synthesize segments
segments = [
    {
        "id": 0,
        "text": "Hola, soy Alice",
        "speaker": "Alice",
        "start_time": 0.0,
        "end_time": 2.0
    }
]

results, timing_info = orchestrator.synthesize_segments(
    segments,
    output_dir="output/dubbed",
    language="es"
)

print(f"Success: {timing_info['successful']}/{timing_info['total_segments']}")
print(f"Total duration: {timing_info['total_duration']:.1f}s")
```

#### Voice Cloning from Reference

```python
# Setup speaker with voice cloning
profile = SpeakerProfile(
    speaker_id="Speaker_1",
    language="es",
    voice_reference="reference_audio/speaker1_english.wav",
    clone_method=VoiceCloneMethod.ZERO_SHOT
)

orchestrator.register_speaker(profile)

# TTS will now use the reference audio to match the speaker's voice
# The synthesized Spanish audio will sound like the original speaker!
```

#### Multiple Speakers with Different Voices

```python
# Create profiles for multiple speakers
profiles = {
    "Alice": SpeakerProfile(
        speaker_id="Alice",
        language="es",
        voice_reference="ref_alice.wav",
        clone_method=VoiceCloneMethod.ZERO_SHOT,
        emotion="professional"
    ),
    "Bob": SpeakerProfile(
        speaker_id="Bob",
        language="es",
        voice_reference="ref_bob.wav",
        clone_method=VoiceCloneMethod.ZERO_SHOT,
        emotion="casual"
    )
}

# Register all
for profile in profiles.values():
    orchestrator.register_speaker(profile)

# Synthesize (each speaker uses their own voice)
results, info = orchestrator.synthesize_segments(
    translated_segments,
    output_dir="output/dubbed",
    language="es"
)
```

### Voice Cloning Methods Explained

#### ZERO_SHOT (Recommended for Quality)
- Uses a reference audio sample from the original speaker
- TTS synthesizes in the target language but with the original speaker's voice
- **Pros:** Best quality, preserves speaker identity
- **Cons:** Requires reference audio
- **Use when:** You have sample audio from each speaker

```python
profile = SpeakerProfile(
    speaker_id="Alice",
    voice_reference="alice_sample.wav",
    clone_method=VoiceCloneMethod.ZERO_SHOT
)
```

#### VOICE_NAME (Predefined Voices)
- Uses a built-in voice name from the TTS system
- Fast and doesn't require reference audio
- **Pros:** No reference audio needed, consistent
- **Cons:** Less personalized, may not match original speaker
- **Use when:** You want generic but consistent voices

```python
profile = SpeakerProfile(
    speaker_id="Narrator",
    voice_name="professional_male",
    clone_method=VoiceCloneMethod.VOICE_NAME
)
```

---

## 4. Detailed Dubbing Pipeline (`src/pipeline_detailed.py`)

### Purpose
Orchestrates all components in a complete dubbing workflow.

### Workflow Stages

1. **Audio Extraction** - Extract audio from video
2. **Transcription** - Transcribe with speaker diarization
3. **Segmentation** - Combine logical and speaker boundaries
4. **Speaker Registration** - Setup voice cloning profiles
5. **Translation** - Translate segments preserving speaker info
6. **TTS Synthesis** - Synthesize dubbed audio
7. **Alignment** - Synchronize timing
8. **Output Generation** - Save all metadata and results

### Main Class: `DetailedDubbingPipeline`

**Constructor:**
```python
pipeline = DetailedDubbingPipeline(
    asr=None,          # Auto-initialize WhisperWithDiarzation
    translator=None,   # Auto-initialize GroqTranslator
    tts=None,          # Auto-initialize CoquiSpeakerTTS
    config=None        # Use default config
)
```

**Main Method:**
```python
result = pipeline.run(
    video_path="input.mp4",
    source_lang="en",
    target_lang="es",
    speaker_reference_audio={
        "Speaker_1": "ref_speaker1.wav",
        "Speaker_2": "ref_speaker2.wav"
    }
)
```

### Configuration: `DetailedPipelineConfig`

```python
config = DetailedPipelineConfig(
    # Directories
    work_dir="work",
    output_dir="output",
    
    # Audio
    sample_rate=16000,
    
    # Segmentation
    min_segment_duration=0.5,
    speaker_change_threshold=0.1,
    
    # Alignment
    alignment_strategy=AlignmentStrategy.ADAPTIVE,
    max_timing_drift=2.0,
    
    # TTS
    tts_model="tts_models/multilingual/multi-dataset/xtts_v2",
    tts_device="cpu",  # or "cuda"
    preserve_speaker_identity=True,
    
    # Debug
    debug=True
)
```

### Example: Complete Workflow

```python
from src.pipeline_detailed import (
    DetailedDubbingPipeline,
    DetailedPipelineConfig
)

# Configure
config = DetailedPipelineConfig(
    work_dir="work",
    output_dir="output",
    tts_device="cuda",  # Use GPU
    preserve_speaker_identity=True
)

# Initialize
pipeline = DetailedDubbingPipeline(config=config)

# Run
result = pipeline.run(
    video_path="meeting.mp4",
    source_lang="en",
    target_lang="es",
    speaker_reference_audio={
        "Speaker_1": "ref_audio/speaker1.wav",
        "Speaker_2": "ref_audio/speaker2.wav"
    }
)

# Check result
if result["status"] == "success":
    print("✓ Dubbing complete!")
    print(f"Segments: {result['stages']['segmentation']['segments']}")
    print(f"Speakers: {result['stages']['segmentation']['speakers']}")
    print(f"Output: {result['stages']['output']['output_directory']}")
else:
    print(f"✗ Failed: {result['error']}")
```

### Output Files

```
output/
├─ segments/
│  ├─ segment_0000.wav          # Dubbed audio for segment 0
│  ├─ segment_0001.wav
│  └─ ...
├─ synthesis_report.json        # TTS synthesis details
│  └─ Success rates, durations, speaker assignments
├─ segments.json                # Segment metadata
│  └─ Text, speaker, timing for each segment
├─ alignment.json               # Timing alignment info
│  └─ Source/target timing, scaling factors
└─ pipeline_metadata.json       # Complete execution log
```

---

## Integration Example: End-to-End Dubbing

```python
"""
Complete example: Dub an English video to Spanish
with speaker identity preservation
"""

from src.pipeline_detailed import (
    DetailedDubbingPipeline,
    DetailedPipelineConfig
)
import os

# Step 1: Prepare reference audio for voice cloning
# Place reference audio files:
# - reference_audio/alice.wav    (sample from speaker Alice in English)
# - reference_audio/bob.wav      (sample from speaker Bob in English)

# Step 2: Configure pipeline
config = DetailedPipelineConfig(
    work_dir="work",
    output_dir="output",
    sample_rate=16000,
    min_segment_duration=0.5,
    tts_device="cuda",  # Use GPU if available
    preserve_speaker_identity=True,
    debug=True
)

# Step 3: Initialize pipeline
pipeline = DetailedDubbingPipeline(config=config)

# Step 4: Run dubbing
result = pipeline.run(
    video_path="input_video.mp4",
    source_lang="en",
    target_lang="es",
    speaker_reference_audio={
        "Speaker_1": "reference_audio/alice.wav",
        "Speaker_2": "reference_audio/bob.wav"
    }
)

# Step 5: Check results
if result["status"] == "success":
    print("✓ Dubbing successful!")
    
    # Access results
    output_dir = result["stages"]["output"]["output_directory"]
    
    print(f"\nSegmentation:")
    print(f"  Total segments: {result['stages']['segmentation']['segments']}")
    print(f"  Speakers: {result['stages']['segmentation']['speakers']}")
    
    print(f"\nTTS Synthesis:")
    print(f"  Successful: {result['stages']['tts_synthesis']['successful']}")
    print(f"  Failed: {result['stages']['tts_synthesis']['failed']}")
    
    print(f"\nAlignment:")
    timing = result["stages"]["alignment"]
    print(f"  Source duration: {timing['total_source_duration']:.1f}s")
    print(f"  Target duration: {timing['total_target_duration']:.1f}s")
    print(f"  Scaling factor: {timing['average_scaling_factor']:.2f}x")
    
    print(f"\nOutput files:")
    print(f"  Directory: {output_dir}")
    print(f"  Segments: {len([f for f in os.listdir(f'{output_dir}/segments') if f.endswith('.wav')])} files")
    
else:
    print(f"✗ Dubbing failed: {result.get('error')}")
```

---

## Dependencies

Install required packages:

```bash
# Core components
pip install openai-whisper
pip install pyannote.audio
pip install TTS torch torchaudio
pip install librosa

# Translation
pip install groq

# Optional: GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Performance Tips

### 1. GPU Acceleration
```python
# Use CUDA for faster processing
config = DetailedPipelineConfig(tts_device="cuda")
```

### 2. Reference Audio Quality
- Use 1-3 second samples from each speaker
- Ensure clear, background-noise-free audio
- Sample should match the content (e.g., formal for business, casual for conversation)

### 3. Segmentation Tuning
```python
# Adjust for different content types
segmenter = AudioSegmenter(
    min_segment_duration=0.3,  # Shorter for fast speech
    speaker_change_threshold=0.05  # Lower for sensitive detection
)
```

### 4. Alignment Strategy Selection
- **STRICT**: Use for music videos where timing is critical
- **FLEXIBLE**: Use when audio quality is paramount
- **ADAPTIVE**: Recommended for professional dubbing

---

## Troubleshooting

### Issue: Speaker voices not distinct
**Solution:** 
- Provide higher-quality reference audio (1-3 seconds, noise-free)
- Ensure speaker reference is from the original language

### Issue: Long durations causing timing drift
**Solution:**
- Use STRICT alignment strategy
- Break long videos into shorter segments
- Check segmentation quality

### Issue: Out of memory errors
**Solution:**
- Process shorter videos
- Use CPU instead of GPU (slower but uses less memory)
- Reduce batch sizes

### Issue: Poor translation quality
**Solution:**
- Use a better translation backend
- Provide context to translator
- Check source transcription quality

---

## Future Enhancements

1. **Audio Mixing** - Combine segments with proper crossfading
2. **Background Music** - Preserve/adjust background audio
3. **Emotion Transfer** - Match prosody from source
4. **Video Synchronization** - Automatic lip-sync adjustment
5. **Quality Assessment** - Automatic quality scoring
6. **Multi-Speaker Optimization** - Better speaker distinction
