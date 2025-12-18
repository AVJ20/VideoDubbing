# Multi-Environment CLI Integration

Your CLI is now updated to support both single-environment and multi-environment execution!

## Quick Start

### Option 1: Multi-Environment (Recommended) ✓
Runs ASR in the `asr` environment and TTS in the `tts` environment, avoiding dependency conflicts:

```bash
python cli.py --file video.mp4 --source en --target es --multi-env
python cli.py --url "https://example.com/video.mp4" --source en --target es --multi-env
```

### Option 2: Single Environment (Backwards Compatible)
Uses original setup (all in one environment):

```bash
python cli.py --file video.mp4 --source en --target es
python cli.py --url "https://example.com/video.mp4" --source en --target es
```

## CLI Options

```
--file PATH              Local video file path
--url URL                Video URL to download
--source LANG            Source language (ISO code, default: 'auto')
--target LANG            Target language (ISO code, REQUIRED)
--work-dir DIR           Working directory (default: 'work')
--multi-env              Use separate conda environments (default: False)
--tts-device DEVICE      TTS device: 'cpu' or 'cuda' (default: 'cpu')
```

## Examples

### English to Spanish using multi-env:
```bash
python cli.py --file english_video.mp4 --source en --target es --multi-env
```

### Auto-detect source, translate to French using GPU:
```bash
python cli.py --url "https://example.com/video.mp4" --source auto --target fr --multi-env --tts-device cuda
```

### Original single-env (for backwards compatibility):
```bash
python cli.py --file video.mp4 --source en --target es
```

## Environment Setup

Both environments have been created:
- **ASR Environment**: `C:\Users\vijoshi\AppData\Local\anaconda3\envs\asr`
  - Contains: Whisper, Pyannote.audio, torch 1.13.1, torchaudio 0.13.1
  - For speech-to-text with speaker diarization

- **TTS Environment**: `C:\Users\vijoshi\AppData\Local\anaconda3\envs\tts`
  - Contains: Coqui TTS, torch 2.9.1, torchaudio 2.9.1
  - For text-to-speech synthesis

## VS Code Configuration

The default Python interpreter has been updated to use the `asr` environment:
```
python.defaultInterpreterPath: C:\Users\vijoshi\AppData\Local\anaconda3\envs\asr\python.exe
```

This ensures VS Code debugger uses the correct environment.

## Testing

Run the test script to verify everything is working:
```bash
python test_multienv.py
```

This will check:
- Both conda environments exist
- Worker scripts can be called
- JSON communication works
- EnvAwarePipeline initializes correctly

## Architecture

When using `--multi-env`:

```
CLI (main.py)
    ↓
EnvAwarePipeline (src/pipeline_multienv.py)
    ├─ EnvManager (workers/env_manager.py)
    │   ├─ Calls asr_worker.py in 'asr' environment
    │   └─ Calls tts_worker.py in 'tts' environment
    ├─ Translation (groq API)
    └─ Results aggregation
```

## Troubleshooting

### "conda: command not found"
- Ensure conda is in your PATH
- Restart VS Code terminal after conda installation

### "asr" or "tts" environment not found
- Create missing environments: see MULTIENV_SETUP.md
- Or recreate: `conda env create -f requirements-asr.txt -n asr`

### Worker script import errors
- Worker scripts now auto-add parent directory to sys.path
- Ensure you're running from project root directory

### CUDA device errors
- If TTS fails with CUDA, use `--tts-device cpu`
- Or ensure nvidia-cuda-toolkit is installed for your torch version

## Next Steps

1. Test the setup: `python test_multienv.py`
2. Try a small video: `python cli.py --file test.mp4 --source en --target es --multi-env`
3. Monitor the output for any errors
4. Check `work/` directory for intermediate files (transcripts, translations, audio)

## File Changes Summary

Updated:
- `cli.py` - Added `--multi-env` and `--tts-device` flags
- `workers/asr_worker.py` - Fixed sys.path for imports
- `workers/tts_worker.py` - Fixed sys.path for imports
- `.vscode/settings.json` - Updated default interpreter to `asr` environment
- `test_multienv.py` - Created comprehensive test script

Ready to use! 🚀
