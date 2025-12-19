# ASR (Speech-to-Text) Setup Guide

## Overview

VideoDubbing now uses **Whisper + Pyannote** as the default ASR (Automatic Speech Recognition) system. This combination provides:

- ✅ **Free & Open Source** (no API keys needed)
- ✅ **Speaker Diarization** (identifies who is speaking when)
- ✅ **Batch Transcription** (processes entire audio in one go)
- ✅ **High Accuracy** (trained on 680K hours of multilingual data)
- ✅ **Multilingual** (100+ languages supported)
- ✅ **Segment-Level Timestamps** (word-level precision available)
- ✅ **Runs Locally** (no cloud dependency, full privacy)

---

## Installation

### Step 1: Install Core Dependencies

```bash
pip install openai-whisper pyannote.audio torch torchaudio
```

**Installation time:** ~3-5 minutes (depends on internet speed)

**Models downloaded on first run:**
- Whisper model (~400MB, cached)
- Pyannote diarization model (~300MB, cached)
- PyTorch (~500MB)

### Step 2: Accept Pyannote License (One-time)

Pyannote requires you to accept their model license on Hugging Face:

1. Visit: https://huggingface.co/pyannote/speaker-diarization-3.1
2. Click **"Agree and access repository"**
3. Authenticate locally:
   ```bash
   huggingface-cli login
   ```
   - This saves your token to `~/.cache/huggingface/`
   - Only needs to be done once per machine

> If you skip this step, diarization will be skipped with a warning, but transcription still works.

### Step 3: Verify Installation

```bash
python -c "import whisper; import pyannote; print('✅ All ASR dependencies installed!')"
```

---

## Usage

### Automatic (Default in Pipeline)

The pipeline automatically uses `WhisperWithDiarizationASR`:

```python
from src.pipeline import DubbingPipeline, PipelineConfig

config = PipelineConfig(work_dir="./work")
pipeline = DubbingPipeline(config=config)

# Automatically uses Whisper + Pyannote ASR
result = pipeline.run(
    source_lang="en",
    target_lang="es",
    video_path="my_video.mp4"
)

print("Transcript:", result["steps"]["transcript"])
```

### Manual ASR Usage

```python
from src.asr import WhisperWithDiarizationASR

# Initialize ASR (loads models on first run)
asr = WhisperWithDiarizationASR(
    whisper_model="base",  # tiny, base, small, medium, large
    device="cpu"           # or "cuda" for GPU
)

# Transcribe audio file
result = asr.transcribe("audio.wav")

# Access results
print("Full text:", result.text)
print("\nSegments with speaker info:")
for segment in result.segments:
    print(f"  [{segment['offset']:.2f}s - {segment['offset'] + segment['duration']:.2f}s] "
          f"Speaker {segment['speaker']}: {segment['text']}")
```

### Example Output

```
[0.50s - 3.20s] Speaker Speaker_1: Hello, how are you today?
[3.45s - 5.80s] Speaker Speaker_2: I'm doing great, thanks for asking!
[5.95s - 8.10s] Speaker Speaker_1: That's wonderful to hear.
```

---

## Whisper Model Sizes

Choose based on your needs:

| Model | Size | Speed | Accuracy | RAM | VRAM |
|-------|------|-------|----------|-----|------|
| **tiny** | 39M | ⚡⚡⚡ Fast | Good | 1GB | 2GB |
| **base** | 140M | ⚡⚡ Medium | Good | 1GB | 3GB |
| **small** | 244M | ⚡ Slower | Very Good | 2GB | 4GB |
| **medium** | 769M | Slow | Excellent | 5GB | 6GB |
| **large** | 1.5B | Very Slow | Excellent | 10GB | 10GB |

**Default:** `base` (best balance of speed and accuracy)

### Change Model Size

```python
asr = WhisperWithDiarizationASR(whisper_model="small")  # More accurate
asr = WhisperWithDiarizationASR(whisper_model="tiny")   # Faster
```

---

## GPU Acceleration

If you have NVIDIA GPU with CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then use GPU in code:

```python
asr = WhisperWithDiarizationASR(
    whisper_model="base",
    device="cuda"  # Uses GPU instead of CPU
)
```

**Speed improvement:** ~3-5x faster transcription

---

## Advanced Options

### Customize Speaker Diarization Model

```python
from pyannote.audio import Pipeline
import torch

# Load different diarization model
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=True
)
pipeline.to(torch.device("cuda"))

asr = WhisperWithDiarizationASR(
    whisper_model="base",
    device="cuda"
)
asr.diarization_pipeline = pipeline  # Override
```

### Disable Diarization (If Having Issues)

```python
from src.asr import WhisperASR

# Use plain Whisper without diarization
asr = WhisperASR(model="base")
# Segments won't have speaker info, but transcription still works
```

---

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'whisper'`

**Solution:**
```bash
pip install openai-whisper
```

### Error: `Could not load Pyannote model`

This usually means:
1. You haven't accepted the license at https://huggingface.co/pyannote/speaker-diarization-3.1
2. You haven't logged in with `huggingface-cli login`

**Solution:**
```bash
huggingface-cli login
```

### Error: `CUDA out of memory`

You're using a model too large for your GPU.

**Solutions:**
- Use smaller model: `whisper_model="tiny"` or `"base"`
- Use CPU instead: `device="cpu"`
- Reduce batch size in Pyannote (advanced configuration)

### Slow Transcription

**Why:** Using CPU or large model size

**Solutions:**
1. Use GPU: `device="cuda"`
2. Use smaller model: `whisper_model="tiny"` or `"base"`
3. Use multi-threading (see examples/asr_demo.py)

---

## Comparison with Other ASR Options

| Option | Free | Diarization | Batch | Local | Accuracy |
|--------|------|-------------|-------|-------|----------|
| **Whisper + Pyannote** ✅ | ✅ | ✅ | ✅ | ✅ | Excellent |
| Google Cloud Speech | ❌ | ✅ | ✅ | ❌ | Excellent |
| Azure Batch ASR | ❌ | ✅ | ✅ | ❌ | Excellent |
| Deepgram | ❌ | ✅ | ✅ | ❌ | Excellent |
| AssemblyAI | ❌ | ✅ | ✅ | ❌ | Excellent |
| AWS Transcribe | ❌ | ✅ | ✅ | ❌ | Very Good |

---

## Next Steps

### Phase 1: Segment Processing
The ASR now provides segment-level data with timestamps and speaker info, perfect for:
- Emotion detection per segment
- Speaker-aware translation
- Lip-sync alignment

### Phase 2: Emotion-Aware Translation
Use segment information to preserve emotional tone:
```python
for segment in asr_result.segments:
    translated = translator.translate(
        segment['text'],
        source_lang="en",
        target_lang="es",
        speaker=segment['speaker'],  # Can use for context
        emotion=detect_emotion(segment)  # Future feature
    )
```

---

## Resources

- **Whisper:** https://github.com/openai/whisper
- **Pyannote:** https://github.com/pyannote/pyannote-audio
- **Hugging Face Models:** https://huggingface.co/pyannote

---

## Support

For issues:
1. Check troubleshooting section above
2. Run demo: `python examples/asr_demo.py`
3. Check logs: Enable `logging.DEBUG` for verbose output
