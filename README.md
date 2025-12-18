# VideoDubbing

Modular video dubbing pipeline.

Features
- Download video (yt-dlp)
- Extract audio (ffmpeg)
- ASR (optional whisper or stub)
- Translation (Groq API by default, or OpenAI/Azure/Ollama)
- TTS (pyttsx3 or stub)

Quickstart

1. Install dependencies (recommended in a venv):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Ensure `ffmpeg` is installed and on PATH.

3. Choose a translation backend (see [TRANSLATION_GUIDE.md](TRANSLATION_GUIDE.md)):
   - **Groq** (default, free): `export GROQ_API_KEY="your-key"`
   - **Ollama** (free, local): Install Ollama and run `ollama serve`
   - **Azure OpenAI** (paid, $200 free trial): Set Azure environment variables
   - **OpenAI** (paid): `export OPENAI_API_KEY="your-key"`

4. Run the CLI with a local video file (example):

```powershell
# Set up Groq API key (default translator)
$env:GROQ_API_KEY = "your-api-key-from-console.groq.com"

# Run pipeline with local video
python cli.py --file "path/to/video.mp4" --source en --target es --work-dir work
```

   Or download from URL:

```powershell
python cli.py --url "https://www.youtube.com/watch?v=..." --source en --target es --work-dir work
```

Notes
- The code is modular: swap implementations in `src/` modules (ASR, translator, tts).
- Use `--file` for local videos or `--url` to download from YouTube/web.
- For production, provide proper error handling, segmentation, alignment and lip-sync logic.
