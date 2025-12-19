# ASR Testing Quick Start

## Install Dependencies (2 minutes)

```bash
# Option 1: Install from main requirements (includes all features)
pip install -r requirements.txt

# Option 2: Install only ASR dependencies
pip install -r requirements-asr.txt
```

## Accept Pyannote License (1-time, 1 minute)

```bash
# Visit this URL in browser:
# https://huggingface.co/pyannote/speaker-diarization-3.1

# Click: "Agree and access repository"

# Then login in terminal:
huggingface-cli login
# Paste your HuggingFace token and press Enter
```

## Verify Installation (30 seconds)

```bash
python -c "
import whisper
import pyannote
print('✅ Whisper:', whisper.__version__)
print('✅ Pyannote loaded')
print('✅ ASR ready to use!')
"
```

## Test ASR with Your Video (First run: 5-10 minutes, subsequent: 1-2 minutes)

### Option 1: Use Existing Video File

```bash
# If you have a video file:
python -c "
from src.asr import WhisperWithDiarizationASR

asr = WhisperWithDiarizationASR(whisper_model='base')
result = asr.transcribe('your_video.wav')  # or .mp3, .mp4

print('=== TRANSCRIPT ===')
print(result.text)

print('\n=== SPEAKERS ===')
for segment in result.segments:
    print(f'[{segment[\"offset\"]:.1f}s] {segment[\"speaker\"]}: {segment[\"text\"][:50]}...')
"
```

### Option 2: Run Full Pipeline with Sample Video

```bash
# Create a test video first (or use your own)
# Then run pipeline:
python cli.py --file your_video.mp4 --source en --target es

# This will:
# 1. Extract audio
# 2. Transcribe with Whisper + speaker diarization
# 3. Translate with Groq
# 4. Synthesize with TTS
```

### Option 3: Run Interactive Demo

```bash
# First, you need audio file in work/ directory
# Extract from your video:
ffmpeg -i your_video.mp4 -q:a 9 -n work/audio.wav

# Then run demo:
python examples/asr_demo.py work/audio.wav
python examples/asr_demo.py work/audio.wav --full  # Full analysis
```

## Expected Output Example

```
=== TRANSCRIPT ===
Welcome to the video. I'm excited to show you something interesting today.
Let me explain the concept in more detail...

=== SPEAKERS ===
[0.5s] Speaker_1: Welcome to the video.
[2.3s] Speaker_1: I'm excited to show you something interesting today.
[5.1s] Speaker_2: That's great!
[6.2s] Speaker_1: Let me explain the concept in more detail...
```

## Debug Launch Configurations in VS Code

We have 5 pre-configured debug profiles:

1. **VideoDubbing: Local File (Groq)**
   - Uses sample_video.mp4 (en → es)
   - Click F5 after creating sample file

2. **VideoDubbing: Custom File**
   - Interactive prompts for file path & languages
   - Best for testing your own videos

3. **VideoDubbing: Test Translation Only**
   - Runs just the translator without video
   - Fast way to test Groq

4. **VideoDubbing: Full Debug Mode**
   - Full debugging with breakpoints
   - Slowest mode but most detailed

5. **VideoDubbing: YouTube URL (Groq)**
   - Downloads and processes YouTube videos
   - Requires internet

## Performance: What to Expect

**First Run:**
- Download Whisper model (400MB) - ~2 minutes
- Download Pyannote model (300MB) - ~2 minutes
- Download PyTorch - ~1 minute
- Process audio - ~5-10 minutes (base model on CPU)
- Total: ~10-15 minutes

**Subsequent Runs:**
- Process audio only - ~1-2 minutes (base model on CPU)

**With GPU (if available):**
- Process audio - ~30 seconds (3-5x faster!)

## Troubleshooting Quick Fixes

### Error: "No module named 'whisper'"
```bash
pip install openai-whisper
```

### Error: "Could not load Pyannote model"
```bash
# 1. Accept license: https://huggingface.co/pyannote/speaker-diarization-3.1
# 2. Login:
huggingface-cli login
```

### Error: "CUDA out of memory"
```python
# Use smaller model and CPU:
asr = WhisperWithDiarizationASR(
    whisper_model='tiny',  # Use tiny instead
    device='cpu'           # Use CPU instead
)
```

### Slow processing?
```python
# Use GPU if available:
asr = WhisperWithDiarizationASR(
    whisper_model='base',
    device='cuda'  # GPU acceleration
)
```

## What Data You Get

Each segment contains:
- **text**: The transcribed words
- **speaker**: Who spoke (Speaker_1, Speaker_2, etc.)
- **offset**: When it started (in seconds)
- **duration**: How long it lasted (in seconds)
- **confidence**: How confident Whisper was (0-1)
- **words**: Optional word-level timing details

## Next: Use in Your Code

```python
from src.asr import WhisperWithDiarizationASR

asr = WhisperWithDiarizationASR(whisper_model='base')
result = asr.transcribe('audio.wav')

# Use segments for:
# - Emotion-aware translation (Phase 2)
# - Speaker-specific TTS voices (Phase 2)
# - Lip-sync alignment (Phase 3)
# - Audio mixing (Phase 4)

for segment in result.segments:
    print(f"{segment['speaker']} ({segment['offset']:.1f}s): {segment['text']}")
    # Access timing for precise audio processing
    start_time = segment['offset']
    end_time = segment['offset'] + segment['duration']
```

## Model Selection Guide

**For Testing:**
- Use `tiny` (fastest, 39MB)
- Command: `asr = WhisperWithDiarizationASR(whisper_model='tiny')`

**For Production:**
- Use `base` (good balance, 140MB) - **RECOMMENDED**
- Command: `asr = WhisperWithDiarizationASR(whisper_model='base')`

**For High Accuracy:**
- Use `small` (more accurate, 244MB)
- Command: `asr = WhisperWithDiarizationASR(whisper_model='small')`

**For Very High Accuracy (slow):**
- Use `medium` (769MB) or `large` (1.5GB)
- Command: `asr = WhisperWithDiarizationASR(whisper_model='medium')`

---

## 30-Second Setup

```bash
# 1. Install (one-time)
pip install -r requirements.txt

# 2. Accept Pyannote license (one-time)
huggingface-cli login

# 3. Test
python examples/asr_demo.py your_audio.wav

# 4. Use in code
python -c "
from src.asr import WhisperWithDiarizationASR
asr = WhisperWithDiarizationASR()
result = asr.transcribe('your_audio.wav')
print(result.text)
"
```

That's it! You now have production-grade ASR with speaker diarization. 🎉

---

For more details, see:
- `ASR_SETUP.md` - Complete installation guide
- `ASR_INTEGRATION_SUMMARY.md` - Technical overview
- `examples/asr_demo.py` - Interactive examples
