# 🎬 Video Dubbing with Multi-Environment - Visual Guide

## The Big Picture

```
🎥 Your Video
    ↓
[cli.py --multi-env]
    ↓
┌──────────────────────────────────────────┐
│         MAIN PROCESS (cli.py)            │
│                                          │
│  args = parse_arguments()                │
│  if args.multi_env:                      │
│      pipeline = EnvAwarePipeline()       │
│  else:                                   │
│      pipeline = DubbingPipeline()        │
│                                          │
│  result = pipeline.run(video, lang)      │
└────────────────────┬─────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
    [ASR Worker]            [TTS Worker]
    (asr env)               (tts env)
        │                         │
        ▼                         ▼
   Whisper +              Coqui TTS
   Pyannote              + Voice Clone
        │                         │
        └────────────┬────────────┘
                     │
                     ▼
            📊 Dubbed Audio
            (dubbed_audio.wav)
```

---

## Command Line Usage

### 🟢 Multi-Env (Recommended)
```bash
python cli.py --file video.mp4 --source en --target es --multi-env
                                                        ↑
                                          NEW FLAG - Use this!
```

### 🟡 Single-Env (Legacy)
```bash
python cli.py --file video.mp4 --source en --target es
                                         
                    (no --multi-env flag, uses original setup)
```

### 🔵 With GPU Acceleration
```bash
python cli.py --file video.mp4 --source en --target es --multi-env --tts-device cuda
                                                                    ↑
                                                    GPU for faster TTS synthesis
```

---

## Process Execution Flow

### With --multi-env Flag

```
Step 1: Main Process Starts
    ├─ Python: asr environment
    ├─ Parse args: --multi-env = True
    └─ Load EnvAwarePipeline

Step 2: ASR (Speech → Text)
    ├─ Call EnvManager.run_asr()
    ├─ Spawn subprocess
    ├─ Run: asr_worker.py in 'asr' environment
    │   ├─ Load Whisper model
    │   ├─ Load Pyannote model
    │   └─ Transcribe audio
    └─ Receive: transcript.json

Step 3: Translation (Text → Text)
    ├─ Use Groq API (main process)
    ├─ Translate transcript
    └─ Return: translated text

Step 4: TTS (Text → Audio)
    ├─ Call EnvManager.run_tts()
    ├─ Spawn subprocess
    ├─ Run: tts_worker.py in 'tts' environment
    │   ├─ Load Coqui TTS model
    │   ├─ Synthesize speech
    │   └─ Apply voice clone (optional)
    └─ Receive: dubbed_audio.wav

Step 5: Complete
    ├─ Aggregate results
    ├─ Save to work/ directory
    └─ Report success
```

### Without --multi-env Flag (Legacy)

```
Step 1: Main Process Starts
    ├─ Python: current/videodub environment
    ├─ Parse args: --multi-env = False
    └─ Load DubbingPipeline

Step 2-5: Same as above, but all in one process
    ├─ No subprocess calls
    ├─ Direct Python imports
    └─ Risk: dependency conflicts
```

---

## File Organization

```
VideoDubbing/
│
├── 🎯 ENTRY POINT
│   └── cli.py (updated)
│       ├─ --file <path>
│       ├─ --url <url>
│       ├─ --source <lang>
│       ├─ --target <lang>
│       ├─ --multi-env ✨ NEW
│       └─ --tts-device <cpu|cuda> ✨ NEW
│
├── 📦 SOURCE CODE
│   └── src/
│       ├── pipeline.py (original)
│       ├── pipeline_multienv.py ✨ NEW
│       ├── asr.py
│       ├── tts.py
│       └── ... others
│
├── 🔧 WORKERS (subprocess handlers)
│   └── workers/
│       ├── env_manager.py ✨ Coordinates
│       ├── asr_worker.py ✨ Runs in 'asr' env
│       └── tts_worker.py ✨ Runs in 'tts' env
│
├── ✅ TESTING
│   └── test_multienv.py ✨ NEW
│
├── 📚 DOCUMENTATION (7 files)
│   ├── START_HERE.md ✨ NEW
│   ├── QUICK_REFERENCE.md ✨ NEW
│   ├── CLI_MULTIENV_GUIDE.md ✨ NEW
│   ├── MULTIENV_CLI_INTEGRATION.md ✨ NEW
│   ├── INTEGRATION_COMPLETE.md ✨ NEW
│   ├── MULTIENV_CHECKLIST.md ✨ NEW
│   ├── CHANGES.md ✨ NEW
│   └── MULTIENV_SETUP.md (existing)
│
├── ⚙️ VS CODE CONFIG
│   └── .vscode/
│       ├── settings.json (updated)
│       └── launch.json (updated)
│
└── 📁 OUTPUTS
    └── work/
        ├── audio.wav
        ├── transcript.json
        ├── translation.json
        └── dubbed_audio.wav ← Your result!
```

---

## Environment Separation

### What's Different?

```
┌─────────────────────────────────────────┐
│           DEPENDENCY CONFLICT           │
├─────────────────────────────────────────┤
│                                         │
│  ASR needs:        TTS needs:           │
│  ├─ torch 1.13.1   ├─ torch 2.9.1      │
│  ├─ torchaudio     ├─ torchaudio 2.9.1 │
│  │  0.13.1         └─ ...              │
│  └─ ...                                 │
│                                         │
│  ❌ Can't have both in same env!       │
│  ✅ Solution: Separate environments    │
│                                         │
└─────────────────────────────────────────┘
```

### Solution: Isolated Environments

```
Windows Conda Directory
│
├─ asr/                           ├─ tts/
│  ├─ python.exe                  │  ├─ python.exe
│  ├─ lib/                        │  ├─ lib/
│  ├─ Scripts/                    │  ├─ Scripts/
│  └─ Packages:                   │  └─ Packages:
│     ├─ whisper                  │     ├─ TTS
│     ├─ pyannote.audio           │     ├─ torch 2.9.1
│     ├─ torch 1.13.1  ✓          │     ├─ torchaudio 2.9.1
│     ├─ torchaudio 0.13.1 ✓      │     └─ ...
│     └─ ...                      │
│                                 │
│  Subprocess 1                   │  Subprocess 2
│  EnvManager.run_asr()           │  EnvManager.run_tts()
│
├─ ✓ No conflicts
├─ ✓ Both can run
└─ ✓ Problem solved!
```

---

## Data Flow During Execution

```
Input Video
    │
    ├─ [Extract Audio]
    │   └─ work/audio.wav
    │
    ├─ [Subprocess 1: ASR]  🟢 asr environment
    │   ├─ Whisper transcription
    │   ├─ Pyannote diarization
    │   └─ work/transcript.json
    │       {
    │         "text": "Hello world",
    │         "segments": [...]
    │       }
    │
    ├─ [Translation]  🔵 main process
    │   ├─ Groq API call
    │   └─ work/translation.json
    │       {
    │         "original": "Hello world",
    │         "translated": "Hola mundo"
    │       }
    │
    ├─ [Subprocess 2: TTS]  🔴 tts environment
    │   ├─ Coqui TTS synthesis
    │   ├─ Voice cloning (optional)
    │   └─ work/dubbed_audio.wav ✨ YOUR OUTPUT!
    │
    └─ [Complete]
        └─ Log summary to console
```

---

## Command Examples

### Quick Start
```bash
# Test setup
python test_multienv.py

# Simple video (English to Spanish)
python cli.py --file my_video.mp4 --source en --target es --multi-env
```

### Real-World Examples
```bash
# YouTube video (auto-detect source)
python cli.py --url "https://youtube.com/watch?v=..." --target en --multi-env

# Spanish video to French (GPU acceleration)
python cli.py --file spanish_video.mp4 --source es --target fr --multi-env --tts-device cuda

# Custom output directory
python cli.py --file video.mp4 --source auto --target de --multi-env --work-dir ./results

# Legacy single-env (if needed)
python cli.py --file video.mp4 --source en --target es
```

### All Options
```bash
python cli.py \
  --file video.mp4 \           # or --url
  --source en \                # or 'auto'
  --target es \                # required
  --multi-env \                # NEW: recommended
  --tts-device cpu \           # NEW: cpu or cuda
  --work-dir ./work            # optional
```

---

## Performance Timeline

```
First Run:
└─ Models Downloaded (5-10 min total)
   ├─ ASR starts: "Downloading Whisper model..." (3-5 min, ~3GB)
   └─ TTS starts: "Downloading Coqui TTS..." (2-3 min, ~1-2GB)

Subsequent Runs:
└─ Cached Models (much faster, depends on video length)
   ├─ ASR: ~30-60 seconds
   └─ TTS: ~1-2 minutes
   └─ With GPU: ~20-40 seconds for TTS

Example: 5-minute video
├─ First run: ~15-20 minutes (includes downloads)
└─ Later runs: ~2-3 minutes (models cached)
```

---

## Troubleshooting Decision Tree

```
Issue?
├─ "ModuleNotFoundError: No module 'src'"
│  └─ Solution: Run from project root
│
├─ "conda: command not found"
│  └─ Solution: Restart terminal
│
├─ "asr environment not found"
│  └─ Solution: Recreate with requirements-asr.txt
│
├─ CUDA/GPU errors
│  └─ Solution: Use --tts-device cpu
│
├─ Script hangs
│  └─ Solution: Check if models are downloading
│
└─ Something else?
   └─ Solution: Check MULTIENV_CLI_INTEGRATION.md
```

---

## Key Takeaways

✅ **Easy to Use**: Single command with optional flag  
✅ **Reliable**: No dependency conflicts  
✅ **Fast**: Cached models for fast subsequent runs  
✅ **Flexible**: GPU acceleration optional  
✅ **Backwards Compatible**: Old way still works  
✅ **Well Documented**: 7 comprehensive guides  

---

## Quick Start Checklist

- [ ] Read `START_HERE.md` (5 min)
- [ ] Run `python test_multienv.py` (1 min)
- [ ] Try `python cli.py --file test.mp4 --source en --target es --multi-env` (5 min)
- [ ] Check `work/dubbed_audio.wav` ✓
- [ ] Use for your videos! 🚀

---

**You're ready to go!** 🎉

Try: `python cli.py --file video.mp4 --source en --target es --multi-env`
