# Chatterbox TTS Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI (cli.py)                           │
│  python cli.py --file video.mp4 --target es --multi-env    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            Pipeline Config (PipelineConfig)                 │
│  - work_dir: str                                            │
│  - tts_backend: "chatterbox" (default) or "coqui"          │
│  - tts_device: "cpu" (default) or "cuda"                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         EnvAwarePipeline (pipeline_multienv.py)             │
│                                                             │
│  1. VideoDownloader - Download video from URL              │
│  2. extract_audio() - Extract audio from video             │
│  3. EnvManager.run_asr() - Transcribe audio                │
│  4. Translator - Translate transcript                      │
│  5. EnvManager.run_tts() - Synthesize speech               │
└──────────────────────┬──────────────────────────────────────┘
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
    (ASR Environment)    (TTS Environment)
    python 3.10          python 3.10
    
    - Whisper            - requests
    - Pyannote           - (Coqui optional)
    
            │                     │
            ▼                     ▼
    asr_worker.py         tts_worker.py
            │                     │
            └──────────┬──────────┘
                       │
                       ▼
        EnvManager.run_tts(..., tts_backend="chatterbox")
                       │
                       ▼
        ┌──────────────────────────┐
        │  tts_worker.py           │
        │  --tts chatterbox        │
        └────────────┬─────────────┘
                     │
                     ▼
        ┌──────────────────────────────────────────────┐
        │         src/tts.py                           │
        │                                              │
        │  if backend == "chatterbox":                 │
        │      from src.tts import ChatterboxTTS       │
        │      tts = ChatterboxTTS(...)                │
        │      tts.synthesize(...)                     │
        │  else:                                       │
        │      from TTS.api import TTS                 │
        │      tts = TTS(model="xtts_v2")              │
        │      tts.tts_to_file(...)                    │
        └────────────┬─────────────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────────────────┐
        │        ChatterboxTTS Class                   │
        │  https://chatterbox.tech/api/tts             │
        │                                              │
        │  - voice: default/male/female                │
        │  - language: en/es/fr/de/etc                 │
        │  - speed: 1.0 (default)                      │
        │  - api_key: optional (higher limits)         │
        └────────────┬─────────────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────────────────┐
        │  HTTP POST to Chatterbox API                 │
        │  Payload: {                                  │
        │    "text": "...",                            │
        │    "voice": "default",                       │
        │    "language": "es",                         │
        │    "speed": 1.0                              │
        │  }                                           │
        └────────────┬─────────────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────────────────┐
        │  Chatterbox Cloud Service                    │
        │  (Free, No Auth Required)                    │
        │                                              │
        │  - Latest TTS Models                         │
        │  - Multi-language support                    │
        │  - Fast processing (~5-10s/min video)        │
        └────────────┬─────────────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────────────────┐
        │  Response: Audio Data (WAV format)           │
        │  or JSON with audio_url/audio_data           │
        └────────────┬─────────────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────────────────┐
        │  Save to: output_audio (work/dubbed_es.wav)  │
        └──────────────────────────────────────────────┘
```

## Component Interaction Flow

### Request Flow

```
CLI Input
  ↓
PipelineConfig (tts_backend="chatterbox")
  ↓
EnvAwarePipeline.run()
  ↓
1. Video Download/Extract
  ↓
2. ASR (Speech → Text)
  ↓
3. Translation (Text → Translated Text)
  ↓
4. EnvManager.run_tts(
     text="translated text",
     language="target_lang",
     tts_backend="chatterbox"
   )
  ↓
5. Subprocess: tts_worker.py
  ↓
6. ChatterboxTTS.synthesize()
  ↓
7. HTTP Request to Chatterbox API
  ↓
8. Save Audio to File
  ↓
Output: dubbed_video_path
```

### Response Flow

```
Chatterbox API
  ↓
Audio Data (WAV)
  ↓
ChatterboxTTS.synthesize() receives and saves
  ↓
Return output_path
  ↓
tts_worker.py returns JSON:
{
  "status": "success",
  "audio": "work/dubbed_es.wav",
  "text": "original text",
  "language": "es",
  "tts_backend": "chatterbox"
}
  ↓
EnvManager.run_tts() parses JSON
  ↓
Pipeline stores result
  ↓
User gets dubbed video
```

## Class Hierarchy

```
AbstractTTS (ABC)
├── Pyttsx3TTS
│   └── Local synthesis (no internet)
├── CoquiTTS
│   └── Local models (excellent quality)
├── AzureSpeechTTS
│   └── Azure Cognitive Services
├── ElevenLabsTTS
│   └── Premium quality (paid)
├── GoogleCloudTTS
│   └── Google's TTS (paid)
└── ChatterboxTTS ✨ NEW (default, free)
    └── Cloud-based (free, latest models)
```

## Configuration Chain

```
CLI Arguments
  └─ --tts-backend "chatterbox" (default)
  └─ --tts-device "cpu"
  
       ↓
       
PipelineConfig
  └─ tts_backend = "chatterbox"
  └─ tts_device = "cpu"
  
       ↓
       
EnvManager.run_tts(..., tts_backend="chatterbox")
  
       ↓
       
tts_worker.py [text] [language] [output.wav] --tts chatterbox
  
       ↓
       
ChatterboxTTS(voice="default", api_key=None)
  
       ↓
       
HTTP Request to https://chatterbox.tech/api/tts
```

## Environment Separation

### ASR Environment (asr conda env)
```
Python 3.10
├── openai-whisper (speech-to-text)
└── pyannote.audio (speaker diarization)
```

### TTS Environment (tts conda env)
```
Python 3.10
├── requests (for Chatterbox API)
├── TTS (optional, for Coqui fallback)
└── torch (optional, for Coqui)
```

### Main Environment (base conda env)
```
Python 3.10
├── yt-dlp (video download)
├── pydub (audio processing)
├── ffmpeg (video/audio processing)
├── groq (translation)
├── requests (API calls)
└── All TTS packages (for direct usage)
```

## Data Flow Example

```
Input: URL to Spanish video "Hola, ¿cómo estás?"
       Source: Spanish, Target: English

       ↓ Step 1: Download
       
Video: downloaded_video.mp4

       ↓ Step 2: Extract Audio
       
Audio: work/audio.wav

       ↓ Step 3: ASR (asr env)
       
Transcript: "Hola, ¿cómo estás?"

       ↓ Step 4: Translate
       
English: "Hello, how are you?"

       ↓ Step 5: TTS with Chatterbox (tts env)
       
Chatterbox Request:
{
  "text": "Hello, how are you?",
  "voice": "default",
  "language": "en",
  "speed": 1.0
}

       ↓
       
Chatterbox Response:
Audio Data (WAV)

       ↓ Step 6: Save
       
Audio: work/dubbed_en.wav

       ↓ Output
       
dubbed_video_en.mp4 (with English audio)
```

## Alternative Flow: Using Coqui

```
Same as above until Step 5...

       ↓ Step 5: TTS with Coqui (tts env)
       
TTS Coqui Request:
model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.tts_to_file(
  text="Hello, how are you?",
  file_path="work/dubbed_en.wav",
  language="en"
)

       ↓
       
Coqui Local Synthesis (takes ~2-5 min/min video)
Downloads model from huggingface (~4GB)

       ↓
       
Audio: work/dubbed_en.wav (excellent quality)
```

## Error Handling Flow

```
Error Occurs
  ↓
Exception caught in try/except
  ↓
Log error with context
  ↓
tts_worker.py returns JSON:
{
  "status": "error",
  "error": "error message"
}
  ↓
EnvManager.run_tts() checks status
  ↓
Raises RuntimeError with details
  ↓
Pipeline catches and logs
  ↓
User sees clear error message
  ↓
Can retry with --tts-backend coqui
```

## Performance Characteristics

### Chatterbox TTS
- **Setup Time:** ~1 second (no downloads)
- **Synthesis Time:** 5-10 seconds per minute of video
- **Quality:** Good (modern AI models)
- **Languages:** 100+
- **Cost:** Free (with optional paid tier)
- **Reliability:** Depends on internet and API availability

### Coqui TTS
- **Setup Time:** ~5 minutes first run (model download ~4GB)
- **Synthesis Time:** 2-5 minutes per minute of video
- **Quality:** Excellent (state-of-the-art)
- **Languages:** 100+
- **Cost:** Free (no API limits)
- **Reliability:** High (local processing, no API dependency)

## Security Considerations

### Chatterbox
- API calls over HTTPS
- Text sent to Chatterbox servers (privacy consideration)
- Optional API key for rate limit increase
- No authentication required (free)

### Coqui
- All processing local (no data sent externally)
- Requires local GPU/CPU resources
- No API keys needed

## Scalability

### Single Video Processing
- Sequential pipeline (ASR → Translate → TTS)
- ~10-30 minutes total with Chatterbox
- ~1-2 hours total with Coqui

### Batch Processing (Future Enhancement)
- Could process multiple videos in parallel
- Would need separate TTS environment instances
- Current design supports this with subprocess calls

## Technology Stack

```
Chatterbox TTS Integration
├── Frontend
│   └── cli.py (command-line interface)
├── Core
│   ├── pipeline_multienv.py (orchestration)
│   ├── tts.py (ChatterboxTTS implementation)
│   └── env_manager.py (subprocess management)
├── External Services
│   ├── Chatterbox API (default TTS)
│   ├── Groq (translation)
│   └── OpenAI Whisper (ASR)
└── Infrastructure
    ├── requests library (HTTP)
    ├── subprocess (environment isolation)
    └── JSON (data serialization)
```

This architecture ensures:
✅ Modular design
✅ Easy to add new TTS backends
✅ Clean separation of concerns
✅ Proper error handling
✅ Flexible configuration
✅ Easy to test and debug
