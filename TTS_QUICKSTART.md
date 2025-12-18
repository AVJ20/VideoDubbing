# Quick TTS Setup Reference

## Best Open-Source: Coqui TTS

**Why Coqui?** 
- ✓ Best quality open-source TTS
- ✓ Emotion & prosody control
- ✓ Zero-shot voice cloning
- ✓ Free & offline
- ✓ Multilingual

### Install & Use (5 minutes)

```bash
pip install TTS torch torchaudio
```

```python
from src.tts import CoquiTTS

# Initialize
tts = CoquiTTS(device="cuda")  # Use GPU for speed

# Synthesize
tts.synthesize("Hola mundo", "audio.wav", language="es")

# Use in pipeline
from src.pipeline import DubbingPipeline
pipeline = DubbingPipeline(tts=tts)
result = pipeline.run(
    source_lang="en",
    target_lang="es",
    video_path="video.mp4"
)
```

---

## Commercial APIs Quick Setup

### Azure Speech
```bash
pip install azure-cognitiveservices-speech

# Set env vars
$env:AZURE_SPEECH_KEY="your-key"
$env:AZURE_SPEECH_REGION="eastus"
```

```python
from src.tts import AzureSpeechTTS
tts = AzureSpeechTTS()
tts.synthesize("Hello", "audio.wav", voice="en-US-JennyNeural", language="en")
```

### ElevenLabs
```bash
pip install elevenlabs

# Set env var
$env:ELEVENLABS_API_KEY="your-key"
```

```python
from src.tts import ElevenLabsTTS
tts = ElevenLabsTTS()
tts.synthesize("Hello", "audio.wav", voice="21m00Tcm4TlvDq8ikWAM", language="en")
```

### Google Cloud
```bash
pip install google-cloud-texttospeech

# Set credentials
$env:GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
```

```python
from src.tts import GoogleCloudTTS
tts = GoogleCloudTTS()
tts.synthesize("Hello", "audio.wav", voice="en-US-Neural2-A", language="en")
```

---

## Language Codes

```
English:   "en"
Spanish:   "es"
French:    "fr"
German:    "de"
Italian:   "it"
Portuguese:"pt"
Chinese:   "zh-cn"
Japanese:  "ja"
Russian:   "ru"
Arabic:    "ar"
```

---

## API Key Links

- **Azure:** https://portal.azure.com → Cognitive Services → Speech
- **ElevenLabs:** https://elevenlabs.io → API Keys
- **Google Cloud:** https://console.cloud.google.com → Text-to-Speech

---

## Features Summary

| Feature | Coqui | Azure | ElevenLabs | Google |
|---------|-------|-------|-----------|--------|
| Free | ✓ | ✗ | ✗ | ✗ |
| Offline | ✓ | ✗ | ✗ | ✗ |
| Emotions | ✓ | ✓ | ✓ | ✓ |
| Zero-Shot Cloning | ✓ | ✗ | ✓ | ✗ |
| Setup Time | 5 min | 10 min | 5 min | 15 min |
| Voice Quality | 4/5 | 4.5/5 | 5/5 | 4.5/5 |

---

## Troubleshooting

**"CUDA out of memory"** → Use `device="cpu"` or reduce batch size

**"API key not found"** → Set environment variables correctly (restart terminal)

**"Module not found"** → Install package: `pip install TTS` (or specific package)

**"Voice not found"** → Check voice ID/name is correct for that TTS service
