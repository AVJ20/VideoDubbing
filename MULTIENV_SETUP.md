# Multi-Environment Setup Guide

This project now uses **separate conda environments** for ASR and TTS to avoid dependency conflicts.

## Architecture

```
┌─────────────────────────────────────────────────┐
│        Main Pipeline (videodub or any env)      │
│  - Video download                               │
│  - Audio extraction                             │
│  - Translation (Groq)                           │
│  - Orchestrates ASR & TTS                       │
└──────────┬──────────────────────────┬───────────┘
           │                          │
           ▼                          ▼
    ┌─────────────┐           ┌─────────────┐
    │ ASR Worker  │           │ TTS Worker  │
    │ (subprocess)│           │ (subprocess)│
    └──────┬──────┘           └──────┬──────┘
           │                          │
           ▼                          ▼
    ┌─────────────┐           ┌─────────────┐
    │  asr env    │           │  tts env    │
    │             │           │             │
    │ - Whisper   │           │ - Coqui TTS │
    │ - Pyannote  │           │ - torch     │
    │ - torch 1.13│           │ - torchaudio│
    └─────────────┘           └─────────────┘
```

---

## Setup (5 minutes)

### 1. Create ASR Environment

```powershell
# Create new conda env for ASR
conda create -n asr python=3.10 -y
conda activate asr

# Install ASR dependencies
cd C:\Codebase\VD\VideoDubbing
pip install -r requirements-asr.txt

# Verify
python -c "import whisper, pyannote.audio; print('✓ ASR ready')"
```

**Expected output:**
```
PyTorch: 1.13.1
torchaudio: 0.13.1
✓ ASR ready
```

### 2. Create TTS Environment

```powershell
# Create new conda env for TTS
conda create -n tts python=3.10 -y
conda activate tts

# Install TTS
pip install TTS==0.22.0

# Verify
python -c "from TTS.api import TTS; print('✓ TTS ready')"
```

### 3. (Optional) Setup Translation

```powershell
# In main environment (videodub or base)
pip install groq

# Set Groq API key
$env:GROQ_API_KEY="your-api-key"
```

---

## Usage

### Method 1: Using EnvAwarePipeline (Recommended)

```python
from src.pipeline_multienv import EnvAwarePipeline, PipelineConfig

# Configure
config = PipelineConfig(
    work_dir="work",
    tts_device="cuda"  # Use 'cuda' if you have GPU
)

# Create pipeline
pipeline = EnvAwarePipeline(config=config)

# Run
result = pipeline.run(
    source_lang="en",
    target_lang="es",
    video_path="video.mp4"
)

# Outputs
print(f"Transcript: {result['steps']['transcript']}")
print(f"Dubbed audio: {result['steps']['tts_audio']}")
```

### Method 2: Using EnvManager Directly

```python
from workers.env_manager import EnvManager

# Transcribe
asr_result = EnvManager.run_asr("audio.wav", language="en")
print(f"Transcript: {asr_result['text']}")

# Translate (your choice of service)
# ... translation logic ...

# Synthesize
tts_result = EnvManager.run_tts(
    "Hola mundo",
    language="es",
    output_audio="output.wav",
    device="cpu"
)
print(f"Audio saved: {tts_result['audio']}")
```

### Method 3: CLI

```bash
# Using Python
python -c "from src.pipeline_multienv import EnvAwarePipeline; p = EnvAwarePipeline(); p.run(source_lang='en', target_lang='es', video_path='video.mp4')"
```

---

## File Structure

```
VideoDubbing/
├── workers/
│   ├── asr_worker.py          # ASR subprocess handler (runs in 'asr' env)
│   ├── tts_worker.py          # TTS subprocess handler (runs in 'tts' env)
│   └── env_manager.py         # Environment coordinator
│
├── src/
│   ├── pipeline_multienv.py   # Main pipeline using separate envs
│   ├── pipeline.py            # Original pipeline (still works)
│   ├── asr.py                 # ASR classes
│   ├── tts.py                 # TTS classes
│   └── ...
│
├── requirements.txt           # Main dependencies
├── requirements-asr.txt       # ASR-only dependencies
└── requirements-tts.txt       # TTS-only dependencies (optional)
```

---

## Environment Paths

The workers look for conda environments here:

```
C:\Users\vijoshi\AppData\Local\anaconda3\envs\asr\
C:\Users\vijoshi\AppData\Local\anaconda3\envs\tts\
```

**To use different paths**, edit `workers/env_manager.py`:

```python
CONDA_BASE = Path("C:\\path\\to\\your\\anaconda3")
```

---

## Performance Tips

### ASR (Whisper + Pyannote)
- **GPU:** To enable GPU in `asr` env:
  ```bash
  conda activate asr
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
  Then in code: `EnvManager.run_asr(...)`  # Will auto-detect CUDA

- **CPU:** Default, no GPU needed

### TTS (Coqui)
- **GPU:** Much faster (5-10x speedup)
  ```bash
  conda activate tts
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
  Then pass `device="cuda"` to `run_tts()`

- **CPU:** Use `device="cpu"` (default, slower)

---

## Troubleshooting

### "asr environment not found"
- Create it: `conda create -n asr python=3.10 && conda activate asr && pip install -r requirements-asr.txt`
- Update path in `env_manager.py` if using non-standard location

### "tts environment not found"
- Create it: `conda create -n tts python=3.10 && conda activate tts && pip install TTS`

### "ModuleNotFoundError: whisper"
- Verify ASR env is created: `conda info --envs`
- Check Python path in `env_manager.py`

### TTS "CUDA out of memory"
- Use `device="cpu"`
- Reduce batch size or text length

### Slow TTS synthesis
- Check if `device="cuda"` is being used
- Larger text = slower (normal)

---

## Next Steps

1. ✓ Create `asr` environment
2. ✓ Create `tts` environment
3. Set up Groq API key for translation
4. Test with sample video
5. (Optional) Set up Azure/ElevenLabs/Google for TTS

---

## Example: Complete Dubbing Workflow

```python
import os
from src.pipeline_multienv import EnvAwarePipeline, PipelineConfig

# Setup
os.environ["GROQ_API_KEY"] = "your-key"  # For translation

config = PipelineConfig(
    work_dir="work",
    tts_device="cuda"  # GPU if available
)

pipeline = EnvAwarePipeline(config=config)

# Dub English video to Spanish
result = pipeline.run(
    source_lang="en",
    target_lang="es",
    video_path="english_video.mp4"
)

# Access outputs
print(f"✓ Transcript: {result['steps']['transcript'][:100]}...")
print(f"✓ Translation: {result['steps']['translation'][:100]}...")
print(f"✓ Dubbed audio: {result['steps']['tts_audio']}")
```

---

## API Reference

### EnvManager.run_asr()

```python
result = EnvManager.run_asr(
    audio_path: str,      # Path to audio file
    language: str = "en"  # Language code
) -> Dict[str, Any]

# Returns
{
    "status": "success",
    "text": "Transcribed text",
    "segments": [  # Speaker diarization
        {
            "text": "...",
            "speaker": "Speaker 0",
            "offset": 0.0,
            "duration": 1.5
        }
    ]
}
```

### EnvManager.run_tts()

```python
result = EnvManager.run_tts(
    text: str,                     # Text to synthesize
    language: str,                 # Language code
    output_audio: str,             # Output file path
    voice: Optional[str] = None,   # Reference audio or voice ID
    device: str = "cpu"            # 'cpu' or 'cuda'
) -> Dict[str, Any]

# Returns
{
    "status": "success",
    "audio": "output.wav",
    "text": "...",
    "language": "es"
}
```

---

## Support

For issues, check:
- Are both `asr` and `tts` environments created?
- Are they at the correct paths?
- Do all imports work individually?

Test individually:
```bash
conda activate asr && python -c "import whisper, pyannote.audio; print('ASR OK')"
conda activate tts && python -c "from TTS.api import TTS; print('TTS OK')"
```
