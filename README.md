# VideoDubbing

Modular video dubbing pipeline.

Features
- Download video (yt-dlp)
- Extract audio (ffmpeg)
- ASR (optional whisper or stub)
- Translation (OpenAI or stub)
- TTS (pyttsx3 or stub)

Quickstart

1. Install dependencies (recommended in a venv):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Ensure `ffmpeg` is installed and on PATH.

3. Run the CLI (example):

```powershell
python cli.py --url "https://www.youtube.com/watch?v=..." --source en --target es --work-dir work
```

Notes
- The code is modular: swap implementations in `src/` modules (ASR, translator, tts).
- For production, provide proper error handling, segmentation, alignment and lip-sync logic.
