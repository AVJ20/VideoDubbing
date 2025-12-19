# Chatterbox TTS Integration Guide

## Overview
Chatterbox API is now integrated as the **default TTS backend** for the video dubbing pipeline. It's completely **free** and offers modern text-to-speech synthesis with multiple language support.

## Features
✅ **Free** - No authentication required (optional API key for higher rate limits)  
✅ **Latest Models** - State-of-the-art TTS technology  
✅ **Multi-language** - Supports multiple languages  
✅ **Fast** - Cloud-based synthesis  
✅ **No Local Dependencies** - No need to download large models like Coqui  

## Installation

1. **Update requirements** (already done):
   ```bash
   pip install -r requirements.txt
   ```

2. The `requests` library is already included in requirements.txt

## Usage

### Command Line

**Default (using Chatterbox):**
```bash
python cli.py --file video.mp4 --source en --target es --multi-env
```

**Using Chatterbox explicitly:**
```bash
python cli.py --file video.mp4 --source en --target es --multi-env --tts-backend chatterbox
```

**Using Coqui instead:**
```bash
python cli.py --file video.mp4 --source en --target es --multi-env --tts-backend coqui
```

### Python Code

```python
from src.tts import ChatterboxTTS

# Create instance
tts = ChatterboxTTS(voice="default")

# Synthesize speech
output_path = tts.synthesize(
    text="Hola mundo",
    out_path="output.wav",
    voice="default",
    language="es"
)
```

### Through Worker (Multi-Environment)

```python
from workers.env_manager import EnvManager

result = EnvManager.run_tts(
    text="Hello world",
    language="en",
    output_audio="output.wav",
    tts_backend="chatterbox"  # Default
)
```

## Configuration

### Environment Variables (Optional)

If you want higher rate limits, set an API key:

```bash
# .env file
CHATTERBOX_API_KEY=your_api_key_here
```

```python
# Or in code
from src.tts import ChatterboxTTS

tts = ChatterboxTTS(api_key="your_api_key_here")
```

## Supported Languages

Chatterbox supports many languages. Common examples:
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `ja` - Japanese
- `zh` - Chinese
- And many more...

## Voice Options

Default voices supported:
- `default` - Default voice
- `male` - Male voice
- `female` - Female voice
- And other voice variants depending on API capabilities

## Troubleshooting

### Network Issues
If you get connection errors, check your internet connection and firewall settings.

### Rate Limiting
- Free tier: ~100 requests per hour per IP
- With API key: Higher limits available
- If rate limited, consider upgrading or using Coqui as fallback

### API Unavailability
If Chatterbox API is unavailable, switch to Coqui:
```bash
python cli.py --file video.mp4 --source en --target es --tts-backend coqui
```

## Comparison with Other Backends

| Feature | Chatterbox | Coqui | ElevenLabs | Azure |
|---------|-----------|-------|-----------|-------|
| **Cost** | Free | Free | Paid | Paid |
| **Quality** | Good | Excellent | Premium | Good |
| **Setup** | Easy | Complex | API Key | API Key |
| **Local** | No | Yes | No | No |
| **Languages** | Many | Many | 30+ | 100+ |
| **Voice Cloning** | No | Yes | Yes | No |

## Switching Backends

The pipeline supports multiple TTS backends:

1. **Chatterbox** (default, free, cloud-based)
2. **Coqui** (free, local, slower but excellent quality)

Switch anytime with `--tts-backend` flag or in `PipelineConfig`.

## API Details

**Chatterbox Endpoint:** https://chatterbox.tech/api/tts

**Request Format:**
```json
{
    "text": "Text to synthesize",
    "voice": "default",
    "language": "en",
    "speed": 1.0,
    "api_key": "optional"
}
```

**Response:** Direct audio data (WAV format)

## Performance Tips

1. **Batch Requests** - Synthesize multiple texts to maximize API quota
2. **Reasonable Text Length** - Avoid extremely long texts in one request
3. **Language Correct** - Use correct language codes for better results
4. **Rate Management** - Space out requests if hitting rate limits

## Next Steps

- ✅ Chatterbox is now the default backend
- Try it: `python cli.py --file video.mp4 --source auto --target es --multi-env`
- If you need higher quality, switch to Coqui: `--tts-backend coqui`
- For advanced features (voice cloning), use Coqui

## Support

For issues:
1. Check Chatterbox API status at https://chatterbox.tech
2. Verify internet connection and firewall settings
3. Try with `--tts-backend coqui` as fallback
4. Check logs for detailed error messages
