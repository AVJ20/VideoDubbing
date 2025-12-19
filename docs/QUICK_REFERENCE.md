# Quick Reference: Multi-Environment Setup

## One-Liner Commands

### Run with Multi-Environment (Recommended)
```bash
python cli.py --file video.mp4 --source en --target es --multi-env
```

### Run with Single Environment (Original)
```bash
python cli.py --file video.mp4 --source en --target es
```

### Test Everything Works
```bash
python test_multienv.py
```

---

## Common Language Codes

```
English:   en      German:    de      Italian:   it
Spanish:   es      French:    fr      Portuguese: pt
Japanese:  ja      Chinese:   zh      Korean:    ko
Russian:   ru      Arabic:    ar      Hindi:     hi
```

---

## Full CLI Reference

```bash
# Basic usage
python cli.py --file VIDEO_FILE --source LANG --target LANG [OPTIONS]
python cli.py --url VIDEO_URL --source LANG --target LANG [OPTIONS]

# Options
--multi-env       Use separate environments (recommended)
--tts-device cpu  Use CPU for TTS (default)
--tts-device cuda Use GPU for TTS (faster, requires CUDA)
--work-dir PATH   Output directory (default: work/)
```

---

## VS Code Debug Shortcuts

1. Press `F5` or click Run
2. Select one of:
   - "VideoDubbing: Local File (Groq) - Multi-Env" ← **Use this**
   - "VideoDubbing: YouTube URL (Groq) - Multi-Env" ← **Use this**
   - "VideoDubbing: Custom File - Multi-Env" ← **Use this**
   - (Original single-env configs still available for backwards compatibility)

---

## Environments

| Environment | Location | Purpose |
|---|---|---|
| `asr` | `.../envs/asr` | Speech-to-text (Whisper) |
| `tts` | `.../envs/tts` | Text-to-speech (Coqui) |

Activate manually (if needed):
```bash
conda activate asr    # For debugging ASR
conda activate tts    # For debugging TTS
```

---

## Common Issues & Fixes

| Problem | Fix |
|---------|-----|
| `conda: command not found` | Restart terminal after conda install |
| `No module named 'src'` | Run from project root directory |
| Environment not found | Run `conda env list` to verify |
| CUDA errors | Use `--tts-device cpu` instead |
| Out of memory | Reduce video resolution or use CPU |

---

## Output Files

When you run the pipeline, check `work/` directory for:

```
work/
├── video.mp4                  # Downloaded video (if from URL)
├── audio.wav                  # Extracted audio
├── transcript.json            # ASR output
├── translation.json           # Translation output
└── dubbed_audio.wav           # Final dubbed audio
```

---

## Performance Tips

1. **First run takes longer** (model downloads)
   - ASR: ~5-10 min (3GB download)
   - TTS: ~2-3 min (1-2GB download)

2. **Use GPU for faster TTS** (if available)
   ```bash
   python cli.py --file video.mp4 --source en --target es --multi-env --tts-device cuda
   ```

3. **Test with short videos first**
   - Easier to debug
   - Faster to see results

---

## Files to Know About

| File | Purpose |
|------|---------|
| `cli.py` | Main entry point |
| `src/pipeline_multienv.py` | Multi-env orchestration |
| `workers/env_manager.py` | Environment coordination |
| `workers/asr_worker.py` | ASR subprocess handler |
| `workers/tts_worker.py` | TTS subprocess handler |
| `test_multienv.py` | Verification tests |
| `.vscode/settings.json` | VS Code config |
| `.vscode/launch.json` | Debug configurations |

---

## Support

For detailed documentation, see:
- `CLI_MULTIENV_GUIDE.md` - Comprehensive user guide
- `MULTIENV_SETUP.md` - Architecture & setup details
- `MULTIENV_CLI_INTEGRATION.md` - Integration summary

---

**Ready to use!** 🚀

Example: `python cli.py --file my_video.mp4 --source en --target es --multi-env`
