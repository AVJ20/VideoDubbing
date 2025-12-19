# Detailed Components Quick Start

Get up and running with speaker segmentation, alignment, and speaker-specific TTS in minutes.

## 5-Minute Setup

### 1. Install Dependencies
```bash
pip install openai-whisper pyannote.audio TTS torch torchaudio librosa groq
```

### 2. Run Your First Segmentation
```python
from src.pipeline_detailed import DetailedDubbingPipeline, DetailedPipelineConfig

# Simple configuration
config = DetailedPipelineConfig(
    work_dir="work",
    output_dir="output",
    tts_device="cpu"  # Change to "cuda" if you have GPU
)

# Create pipeline
pipeline = DetailedDubbingPipeline(config=config)

# Run on your video
result = pipeline.run(
    video_path="your_video.mp4",
    source_lang="en",
    target_lang="es"
)

# Check results
if result["status"] == "success":
    print("✓ Success!")
    print(f"Segments: {result['stages']['segmentation']['segments']}")
    print(f"Speakers: {result['stages']['segmentation']['speakers']}")
    print(f"Output: {result['stages']['output']['output_directory']}")
```

## Key Components Overview

### 1. Audio Segmentation
**What it does:** Splits audio into segments based on:
- Logical sentence/phrase boundaries (from ASR)
- Speaker changes (from diarization)

**When to use:** Always, for intelligent segment-based processing

**Example:**
```python
from src.segmentation import AudioSegmenter
from src.asr import WhisperWithDiarizationASR

# Get transcription with speaker info
asr = WhisperWithDiarizationASR(whisper_model="base")
asr_result = asr.transcribe("audio.wav")

# Segment
segmenter = AudioSegmenter()
segments = segmenter.segment(asr_result.segments)

# Results
for seg in segments.segments:
    print(f"{seg.id}: [{seg.speaker}] {seg.text[:30]}...")
    print(f"  {seg.start_time:.2f}s - {seg.end_time:.2f}s")
```

### 2. Alignment
**What it does:** Synchronizes segment timing between source and target audio when durations differ

**When to use:** After TTS synthesis to adjust for timing differences

**Example:**
```python
from src.alignment import SegmentAligner, TimingAnalyzer, AlignmentStrategy

# Create aligner
aligner = SegmentAligner(strategy=AlignmentStrategy.ADAPTIVE)

# Get TTS durations (from synthesis)
target_durations = {0: 2.5, 1: 3.1, 2: 2.2}  # segment_id: duration

# Align
alignment = aligner.align_segments(segments.segments, target_durations)

# Analyze
analyzer = TimingAnalyzer()
stats = analyzer.analyze(alignment)
print(f"Average scaling: {stats['average_scaling_factor']:.2f}x")
```

### 3. Speaker-Specific TTS
**What it does:** Synthesizes speech in target language while preserving speaker identity

**When to use:** For professional dubbing with voice cloning

**Example:**
```python
from src.speaker_tts import CoquiSpeakerTTS, SpeakerTTSOrchestrator, SpeakerProfile

# Setup TTS
tts = CoquiSpeakerTTS(device="cuda")  # GPU recommended
orchestrator = SpeakerTTSOrchestrator(tts)

# Register speakers with reference audio
profile = SpeakerProfile(
    speaker_id="Speaker_1",
    language="es",
    voice_reference="reference_audio/speaker1.wav"
)
orchestrator.register_speaker(profile)

# Synthesize
results, info = orchestrator.synthesize_segments(
    segments=[{"id": 0, "text": "Hola", "speaker": "Speaker_1", ...}],
    output_dir="output",
    language="es"
)
print(f"Success: {info['successful']}/{info['total_segments']}")
```

## Common Workflows

### Workflow 1: Subtitle-Based Dubbing
When you already have translated text for each segment:

```python
from src.speaker_tts import SpeakerTTSOrchestrator, CoquiSpeakerTTS

tts = CoquiSpeakerTTS()
orch = SpeakerTTSOrchestrator(tts)

# For each segment with speaker info
segments = [
    {"id": 0, "text": "Translated text 1", "speaker": "Alice"},
    {"id": 1, "text": "Translated text 2", "speaker": "Bob"},
]

results, info = orch.synthesize_segments(segments, "output")
```

### Workflow 2: Full Dubbing Pipeline
With automatic transcription, translation, and synthesis:

```python
from src.pipeline_detailed import DetailedDubbingPipeline

pipeline = DetailedDubbingPipeline()
result = pipeline.run(
    video_path="input.mp4",
    source_lang="en",
    target_lang="es",
    speaker_reference_audio={
        "Speaker_1": "ref1.wav",
        "Speaker_2": "ref2.wav"
    }
)
```

### Workflow 3: Segmentation + Timing Analysis Only
Without synthesis, just analyze structure and timing:

```python
from src.segmentation import AudioSegmenter
from src.asr import WhisperWithDiarizationASR

asr = WhisperWithDiarizationASR()
asr_result = asr.transcribe("audio.wav")

segmenter = AudioSegmenter()
seg_result = segmenter.segment(asr_result.segments)

# Analyze
print(f"Segments: {len(seg_result.segments)}")
print(f"Duration: {seg_result.total_duration:.1f}s")
print(f"Speakers: {seg_result.speakers}")

# No synthesis needed
```

## Configuration Reference

### For Quick Results (Default)
```python
config = DetailedPipelineConfig(
    tts_device="cpu",  # Slower but works everywhere
    preserve_speaker_identity=True
)
```

### For Professional Dubbing (Recommended)
```python
config = DetailedPipelineConfig(
    tts_device="cuda",  # GPU acceleration
    min_segment_duration=0.5,  # Merge very short segments
    alignment_strategy="adaptive",  # Smooth timing
    preserve_speaker_identity=True  # Voice cloning enabled
)
```

### For Speed (Minimal Quality)
```python
config = DetailedPipelineConfig(
    tts_device="cuda",
    min_segment_duration=1.0,  # Merge more segments
    preserve_speaker_identity=False  # Generic voices
)
```

### For Maximum Quality (Slow)
```python
config = DetailedPipelineConfig(
    tts_device="cuda",
    tts_model="tts_models/multilingual/multi-dataset/xtts_v2",  # Best model
    min_segment_duration=0.3,  # Keep all segments
    preserve_speaker_identity=True,  # Full voice cloning
)
```

## Output Structure

After running the pipeline, check:

```
output/
├─ segments/                    # Dubbed audio segments
│  ├─ segment_0000.wav         # Speaker 1, "Hello..."
│  ├─ segment_0001.wav         # Speaker 2, "Hi..."
│  └─ ...
├─ synthesis_report.json        # TTS success/failure info
├─ segments.json               # Segment text & timing
├─ alignment.json              # Timing adjustment info
└─ pipeline_metadata.json      # Complete execution log
```

## Troubleshooting

### "Module not found" error
```bash
pip install --upgrade openai-whisper pyannote.audio TTS torch
```

### Memory error on CPU
```python
# Use smaller Whisper model
asr = WhisperWithDiarizationASR(whisper_model="tiny")
```

### GPU not used despite installation
```python
# Verify CUDA installation
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name())  # Should show your GPU

# Then use GPU
config = DetailedPipelineConfig(tts_device="cuda")
```

### Voice cloning not working
```python
# Ensure reference audio:
# 1. Is in WAV format
# 2. Is 1-3 seconds long
# 3. Has clear speech
# 4. Matches original speaker language

profile = SpeakerProfile(
    speaker_id="Alice",
    voice_reference="reference_audio/alice_clear_sample.wav"
)
```

### Translation issues
```python
# Check source transcription first
from src.asr import WhisperWithDiarizationASR
asr = WhisperWithDiarizationASR()
result = asr.transcribe("audio.wav")
print(result.text)  # Verify correctness before translation
```

## Next Steps

1. **Explore examples:** Run `python examples/detailed_pipeline_examples.py`
2. **Read full documentation:** See [DETAILED_COMPONENTS.md](DETAILED_COMPONENTS.md)
3. **Implement agentic framework:** Build agents that optimize segmentation, translation, TTS selection
4. **Add audio mixing:** Combine segments with proper crossfading
5. **Implement lip-sync:** Adjust video to match dubbed audio timing

## Performance Benchmarks

Typical processing times on a modern machine:

| Task | CPU | GPU |
|------|-----|-----|
| Extract audio | <1s | <1s |
| Transcribe (1 min) | 5-10s | 2-3s |
| Segmentation | <1s | <1s |
| TTS (10 segments) | 30-60s | 10-20s |
| Alignment | <1s | <1s |
| **Total (1 min video)** | **2-3 min** | **30-45s** |

---

## API Reference

### AudioSegmenter
```python
segmenter.segment(asr_segments, speaker_segments) → SegmentationResult
```

### SegmentAligner
```python
aligner.align_segments(source_segments, target_durations) → List[AlignmentResult]
```

### SpeakerTTSOrchestrator
```python
orchestrator.register_speaker(profile) → bool
orchestrator.synthesize_segments(segments, output_dir, language) → (results, timing_info)
orchestrator.get_segment_durations() → Dict[int, float]
orchestrator.get_synthesis_report() → Dict
```

### DetailedDubbingPipeline
```python
pipeline.run(video_path, source_lang, target_lang, speaker_reference_audio) → Dict
```

---

For detailed documentation, see [DETAILED_COMPONENTS.md](DETAILED_COMPONENTS.md)
