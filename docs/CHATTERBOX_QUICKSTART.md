# Quick Start: Using Chatterbox TTS

## TL;DR - Get Started in 30 Seconds

### 1. No setup needed! Just run:
```bash
python cli.py --file your_video.mp4 --source en --target es --multi-env
```

That's it! 🎉

## More Examples

### Using URL instead of local file:
```bash
python cli.py --url "https://example.com/video.mp4" --source en --target es --multi-env
```

### Auto-detect source language:
```bash
python cli.py --file video.mp4 --source auto --target es --multi-env
```

### Different languages:
```bash
# English to French
python cli.py --file video.mp4 --source en --target fr --multi-env

# Auto-detect to German
python cli.py --file video.mp4 --source auto --target de --multi-env

# Spanish to Russian
python cli.py --file video.mp4 --source es --target ru --multi-env
```

## Using from Python

```python
from src.tts import ChatterboxTTS

# Simple usage
tts = ChatterboxTTS()
tts.synthesize("Hello world", "hello.wav", language="en")

# With specific voice
tts = ChatterboxTTS(voice="male")
tts.synthesize("Hola mundo", "spanish.wav", language="es")

# With API key (for higher rate limits)
tts = ChatterboxTTS(api_key="your-api-key-here")
```

## If You Want Better Quality

Chatterbox is great for quick, free dubbing. But if you need premium quality, use Coqui:

```bash
python cli.py --file video.mp4 --source en --target es --multi-env --tts-backend coqui
```

**Trade-off:** Coqui takes longer to setup and run, but produces excellent quality.

## Supported Languages

| Code | Language | Code | Language |
|------|----------|------|----------|
| en | English | de | German |
| es | Spanish | it | Italian |
| fr | French | pt | Portuguese |
| ru | Russian | ja | Japanese |
| zh | Chinese | ko | Korean |
| ar | Arabic | hi | Hindi |
| ... | Many more! | ... | ... |

## Troubleshooting

### "Connection error" or "API unavailable"?
→ Check your internet connection

### "Rate limited" or "Too many requests"?
→ Wait a bit or use `--tts-backend coqui` instead

### Want faster processing?
→ Use `--tts-backend chatterbox` (already default)

### Want better quality?
→ Use `--tts-backend coqui` (slower but excellent)

## What's Happening?

When you run the pipeline:

1. **Download** - Downloads video from URL (if provided)
2. **Extract** - Extracts audio from video
3. **ASR** - Transcribes speech to text (using Whisper)
4. **Translate** - Translates text to target language (using Groq/OpenAI)
5. **TTS** - Synthesizes translated text to speech (using Chatterbox 🎉)
6. **Output** - Creates dubbed video

## Performance

- **Chatterbox:** ~5-10 seconds per minute of video (fast!)
- **Coqui:** ~2-5 minutes per minute of video (slower, but amazing quality)

## API Limits

Chatterbox free tier: ~100 requests per hour per IP

That's enough for:
- 100 short clips per hour
- Or ~5-10 full videos per hour depending on length

Need more? Get a free API key or use `--tts-backend coqui`

## Next Steps

1. ✅ Try it: `python cli.py --file video.mp4 --target es --multi-env`
2. 📚 Read full guide: See `CHATTERBOX_TTS_GUIDE.md`
3. 🔧 Customize: Check `PipelineConfig` in `src/pipeline_multienv.py`
4. 🚀 Deploy: Set up on your server with proper error handling

## Need Help?

Check these files:
- `CHATTERBOX_TTS_GUIDE.md` - Comprehensive guide
- `CHATTERBOX_INTEGRATION_SUMMARY.md` - What changed
- `README.md` - General project info
