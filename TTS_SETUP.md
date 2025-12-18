# Text-to-Speech (TTS) Setup Guide

This project supports multiple TTS backends with different capabilities. Choose based on your needs.

## TTS Options Comparison

| TTS Backend | Cost | Quality | Emotions | Zero-Shot | Languages | Local |
|---|---|---|---|---|---|---|
| **Coqui TTS** (Recommended Open-Source) | Free | ⭐⭐⭐⭐ | ✓ | ✓ | 10+ | ✓ |
| **Pyttsx3** | Free | ⭐⭐ | ✗ | ✗ | Limited | ✓ |
| **Azure Speech** | Paid | ⭐⭐⭐⭐ | ✓ | ✗ | 140+ | ✗ |
| **ElevenLabs** | Paid | ⭐⭐⭐⭐⭐ | ✓ | ✓ | 30+ | ✗ |
| **Google Cloud** | Paid | ⭐⭐⭐⭐ | ✓ | ✗ | 100+ | ✗ |

---

## 1. Coqui TTS (Best Open-Source Choice)

**Features:**
- Best free option with high-quality voices
- Natural emotions and prosody control
- Zero-shot voice cloning (with reference audio)
- Multilingual support
- Works offline (no API key needed)

**Installation:**
```bash
pip install TTS torch torchaudio
```

**Usage in Pipeline:**
```python
from src.tts import CoquiTTS

tts = CoquiTTS(device="cpu")  # or "cuda" for GPU
tts.synthesize("Hello world", "output.wav", language="en")
```

**Voice Cloning Example:**
```python
# Use reference audio for zero-shot cloning
tts.synthesize(
    "Hello world",
    "output.wav",
    voice="path/to/reference_audio.wav",  # Reference audio file
    language="en"
)
```

**Supported Languages:**
- English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu), Korean (ko)

---

## 2. Pyttsx3 (Local, Lightweight)

**Features:**
- No internet required
- Lightweight and fast
- Works on all platforms

**Installation:**
```bash
pip install pyttsx3
```

**Usage:**
```python
from src.tts import Pyttsx3TTS

tts = Pyttsx3TTS()
tts.synthesize("Hello world", "output.wav")
```

---

## 3. Azure Speech TTS (Enterprise)

**Features:**
- 400+ neural voices in 140+ languages
- Emotion and prosody control
- SSML support for advanced control
- Enterprise support

**Setup:**
1. Create Azure Cognitive Services account
2. Get subscription key and region from Azure Portal
3. Set environment variables:
   ```powershell
   $env:AZURE_SPEECH_KEY="your-key"
   $env:AZURE_SPEECH_REGION="eastus"  # or your region
   ```

**Installation:**
```bash
pip install azure-cognitiveservices-speech
```

**Usage:**
```python
from src.tts import AzureSpeechTTS

tts = AzureSpeechTTS()
tts.synthesize("Hello world", "output.wav", voice="en-US-JennyNeural")
```

**Popular Voices:**
- English: `en-US-JennyNeural`, `en-US-GuyNeural`, `en-GB-ThomasNeural`
- Spanish: `es-ES-AlvaroNeural`, `es-ES-ElviraNeural`
- French: `fr-FR-DeniseNeural`, `fr-FR-HenriNeural`

[See all voices](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=text-to-speech)

---

## 4. ElevenLabs TTS (Highest Quality)

**Features:**
- Highest quality AI voices (premium voices)
- Zero-shot voice cloning
- Natural emotions and prosody
- Real-time streaming support

**Setup:**
1. Create ElevenLabs account at https://elevenlabs.io
2. Get API key from account settings
3. Set environment variable:
   ```powershell
   $env:ELEVENLABS_API_KEY="your-api-key"
   ```

**Installation:**
```bash
pip install elevenlabs
```

**Usage:**
```python
from src.tts import ElevenLabsTTS

tts = ElevenLabsTTS()
tts.synthesize("Hello world", "output.wav", voice="21m00Tcm4TlvDq8ikWAM")
```

**Popular Voice IDs:**
- Rachel (Female): `21m00Tcm4TlvDq8ikWAM`
- Clyde (Male): `2EiwWnXFnvU5JabPnXlBw`
- Domi (Female): `AZnzlk1XvdvUBZXUNXHP`
- Bella (Female): `EXAVITQu4vLQe8LHuRG`

[Get all voice IDs](https://api.elevenlabs.io/v1/voices)

---

## 5. Google Cloud TTS

**Features:**
- 400+ voices in 100+ languages
- WaveNet (premium) and Neural2 voices
- SSML support
- Natural and expressive voices

**Setup:**
1. Create Google Cloud account and enable Text-to-Speech API
2. Create service account and download JSON credentials
3. Set environment variable:
   ```powershell
   $env:GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
   ```

**Installation:**
```bash
pip install google-cloud-texttospeech
```

**Usage:**
```python
from src.tts import GoogleCloudTTS

tts = GoogleCloudTTS()
tts.synthesize("Hello world", "output.wav", voice="en-US-Neural2-A")
```

**Popular Voices:**
- English: `en-US-Neural2-A`, `en-US-Neural2-C`
- Spanish: `es-ES-Neural2-A`, `es-ES-Neural2-B`
- French: `fr-FR-Neural2-A`, `fr-FR-Neural2-B`

[See all voices](https://cloud.google.com/text-to-speech/docs/voices)

---

## Using TTS in the Pipeline

By default, the pipeline uses `StubTTS` (placeholder). To use a real TTS:

```python
from src.pipeline import DubbingPipeline
from src.tts import CoquiTTS

# Use Coqui TTS
tts = CoquiTTS(device="cuda")  # GPU acceleration
pipeline = DubbingPipeline(tts=tts)

# Run pipeline
result = pipeline.run(
    source_lang="en",
    target_lang="es",
    video_path="input.mp4"
)
```

Or pass via CLI:
```bash
python cli.py --file video.mp4 --source en --target es --tts coqui
```

---

## Performance Tips

### Coqui TTS
- **GPU Acceleration:** Use `device="cuda"` for 5-10x speedup
- **Model Selection:** XTTS v2 (default) is best for multilingual
- **Batch Processing:** Cache model for multiple files

### Azure Speech
- Use neural voices for better quality (`*Neural`)
- Region affects latency; choose closest region

### ElevenLabs
- Monitor API credits (pay per character)
- Use voice cloning for consistent dubbing voices

### Google Cloud
- Neural2 voices are better quality than Standard
- Monitor costs (billed per 1M characters)

---

## Troubleshooting

### Coqui TTS Memory Issues
```bash
# Use CPU with offloading
device = "cpu"

# For GPU, use smaller model (if available)
model_name = "tts_models/en/ljspeech/glow-tts"
```

### Azure/Google Auth Errors
- Verify credentials and region
- Check API key validity
- Ensure API is enabled in cloud console

### ElevenLabs Rate Limiting
- Implement retry logic
- Split long text into chunks
- Monitor API usage

---

## Cost Comparison (Monthly for 1M characters)

- **Coqui TTS:** Free (local)
- **Pyttsx3:** Free (local)
- **Azure:** ~$15 (neural voices)
- **Google Cloud:** ~$16 (Neural2 voices)
- **ElevenLabs:** ~$200+ (premium voices)

*Costs vary by region and voice type. Check official pricing.*

---

## Recommendations

**For Cost-Conscious Projects:** Coqui TTS (free, offline, good quality)

**For Enterprise:** Azure Speech or Google Cloud (scalable, reliable)

**For Highest Quality:** ElevenLabs (premium voices, emotion control)

**For Quick Testing:** Pyttsx3 (lightweight, no setup)

---

## Next Steps

1. Install your chosen TTS backend
2. Set up API keys if needed
3. Test with a small audio sample
4. Integrate into pipeline
5. Monitor quality and costs

See [SETUP_API_KEYS.md](SETUP_API_KEYS.md) for detailed API key setup instructions.
