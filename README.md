# VideoDubbing

Modular video dubbing pipeline.

Features
- Download video (yt-dlp)
- Extract audio (ffmpeg)
- ASR (Whisper + optional speaker diarization)
- Translation (Groq by default, or OpenAI/Azure/Ollama)
- TTS (Chatterbox voice cloning via worker, or pyttsx3 fallback)

Quickstart (single env)

1. Install dependencies (recommended in a venv):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Ensure `ffmpeg` is installed and on PATH.

3. Choose a translation backend (see [docs/TRANSLATION_GUIDE.md](docs/TRANSLATION_GUIDE.md)):
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

Multi-env (recommended for enhanced pipeline)

This repo supports running ASR and TTS in separate conda envs:
- `asr` env: Whisper/pyannote + orchestration
- `tts` env: Chatterbox TTS worker (isolates heavy deps)

Create envs:

```powershell
conda create -n asr python=3.10 -y
conda create -n tts python=3.10 -y

conda activate asr
pip install -r requirements-asr.txt

conda activate tts
pip install -r requirements-tts.txt
```

Run enhanced pipeline (from `asr` env):

```powershell
conda activate asr
python cli.py --file "path\to\video.mp4" --source en --target es --enhanced --multi-env --work-dir work --output-dir work\output_enhanced
```

More details: [docs/MULTIENV_SETUP.md](docs/MULTIENV_SETUP.md)

Notes
- The code is modular: swap implementations in `src/` modules (ASR, translator, tts).
- Use `--file` for local videos or `--url` to download from YouTube/web.
- For production, provide proper error handling, segmentation, alignment and lip-sync logic.
