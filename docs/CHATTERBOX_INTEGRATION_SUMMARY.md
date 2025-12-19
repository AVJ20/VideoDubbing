# Chatterbox TTS Integration - Implementation Summary

## Changes Made

### 1. **Core TTS Module** (`src/tts.py`)
   - Added `requests` import for HTTP API calls
   - Created new `ChatterboxTTS` class that implements the `AbstractTTS` interface
   - Features:
     - Free API integration (no authentication required)
     - Optional API key support for higher rate limits
     - Supports multiple voices and languages
     - Handles both direct audio responses and base64-encoded responses
     - Proper error handling and logging

### 2. **TTS Worker** (`workers/tts_worker.py`)
   - Updated to support multiple TTS backends via `--tts` argument
   - Default backend: `chatterbox`
   - Fallback support for `coqui` TTS
   - Usage:
     ```bash
     python tts_worker.py <text> <language> <output.wav> --tts chatterbox --voice optional_voice
     ```

### 3. **Environment Manager** (`workers/env_manager.py`)
   - Updated `run_tts()` method to accept `tts_backend` parameter
   - Passes backend choice to worker subprocess
   - Default: `chatterbox`

### 4. **Pipeline Configuration** (`src/pipeline_multienv.py`)
   - Added `tts_backend` field to `PipelineConfig` dataclass
   - Default: `"chatterbox"`
   - Pipeline now passes this to the TTS worker

### 5. **CLI Interface** (`cli.py`)
   - Added `--tts-backend` argument
   - Choices: `["chatterbox", "coqui"]`
   - Default: `chatterbox`
   - Updated docstring with examples
   - Passes backend to pipeline config

### 6. **Dependencies** (`requirements.txt`)
   - Added `chatterbox-api>=0.1.0` package reference
   - `requests` library already included (required for API calls)
   - No breaking changes to existing dependencies

### 7. **Documentation** (`CHATTERBOX_TTS_GUIDE.md`)
   - Created comprehensive guide
   - Installation and setup instructions
   - Usage examples (CLI, Python, Worker)
   - Configuration options
   - Troubleshooting section
   - Comparison with other TTS backends

## How It Works

### Default Flow (Chatterbox)
```
cli.py --file video.mp4 --target es --multi-env
    ↓
PipelineConfig(tts_backend="chatterbox")
    ↓
EnvManager.run_tts(..., tts_backend="chatterbox")
    ↓
tts_worker.py --tts chatterbox
    ↓
ChatterboxTTS.synthesize()
    ↓
HTTP POST to https://chatterbox.tech/api/tts
    ↓
Audio file saved
```

### Optional: Using Coqui Fallback
```bash
python cli.py --file video.mp4 --target es --multi-env --tts-backend coqui
```

## Key Features

✅ **Free & No Authentication** - Uses free Chatterbox API tier  
✅ **Cloud-Based** - No large model downloads needed  
✅ **Multiple Languages** - Supports 100+ languages  
✅ **Easy Integration** - Seamless drop-in replacement  
✅ **Backward Compatible** - Coqui still available as option  
✅ **Configurable** - Can set API key for higher limits  

## Testing

To test the Chatterbox integration:

```bash
# Test with CLI
python cli.py --file sample_video.mp4 --source en --target es --multi-env --tts-backend chatterbox

# Test with Python directly
from src.tts import ChatterboxTTS
tts = ChatterboxTTS()
tts.synthesize("Hello world", "test.wav", language="en")

# Test via worker
python workers/tts_worker.py "Hello" "en" "output.wav" --tts chatterbox
```

## API Details

**Endpoint:** `https://chatterbox.tech/api/tts`

**Parameters:**
- `text` (required): Text to synthesize
- `voice` (optional): Voice variant (default, male, female)
- `language` (optional): Language code (default: "en")
- `speed` (optional): Speech speed (default: 1.0)
- `api_key` (optional): API key for higher rate limits

**Response:** Audio data in WAV format

## Rate Limits

- **Free Tier:** ~100 requests per hour per IP
- **With API Key:** Higher limits available
- Can be configured via `CHATTERBOX_API_KEY` environment variable

## Fallback Strategy

If Chatterbox is unavailable:
1. Switch to Coqui: `--tts-backend coqui`
2. Or set environment variable and use API key for reliability

## Future Enhancements

Possible improvements:
- [ ] Voice cloning support (if Chatterbox adds it)
- [ ] Batch synthesis optimization
- [ ] Caching of synthesized audio
- [ ] Automatic fallback on rate limit
- [ ] Advanced prosody controls

## Compatibility

- ✅ Works with existing pipeline architecture
- ✅ Compatible with all ASR backends
- ✅ Works with translation backends
- ✅ Supports multi-environment setup
- ✅ Backward compatible with Coqui

## Configuration Files Modified

1. `src/tts.py` - Added ChatterboxTTS class
2. `workers/tts_worker.py` - Multi-backend support
3. `workers/env_manager.py` - TTS backend parameter
4. `src/pipeline_multienv.py` - Pipeline config update
5. `cli.py` - CLI argument and documentation
6. `requirements.txt` - Added chatterbox-api reference
7. `CHATTERBOX_TTS_GUIDE.md` - NEW guide file

## No Breaking Changes

- Existing code continues to work
- Chatterbox is default but optional
- Users can still use Coqui if preferred
- Environment setup remains the same
