# Phase 2 Preview: Emotion-Aware Translation & Speaker-Specific TTS

## Overview

The ASR integration (Phase 1) provides segment-level speaker information. Phase 2 will build on this to add emotion awareness and speaker-specific audio synthesis.

**Current State (Phase 1):**
```python
segment = {
    "text": "I love this!",
    "speaker": "Speaker_1",
    "offset": 5.2,
    "duration": 1.5,
    "confidence": 0.98
}
```

**Phase 2 Vision:**
```python
segment = {
    "text": "I love this!",
    "speaker": "Speaker_1",
    "offset": 5.2,
    "duration": 1.5,
    "confidence": 0.98,
    # NEW in Phase 2:
    "emotion": "joy",           # Detected emotion
    "energy": 0.8,              # Audio energy level (0-1)
    "pitch_mean": 220.5,        # Average pitch (Hz)
    "speaker_voice": {...}      # Speaker voice characteristics
}
```

---

## Phase 2 Component: Emotion Detection

### Proposed Implementation

```python
from src.emotion import AudioEmotionDetector

class AudioEmotionDetector:
    """Detect emotion from audio segments."""
    
    def __init__(self, model: str = "wav2vec2-emotion"):
        # Use pre-trained emotion detection model
        # Models available: wav2vec2-emotion, hubert-emotion, etc.
        pass
    
    def detect_emotion(self, 
                      audio_path: str, 
                      segment_start: float, 
                      segment_end: float) -> dict:
        """
        Detect emotion in audio segment.
        
        Returns:
        {
            "emotion": "joy",  # Primary emotion
            "confidence": 0.92,
            "emotions_all": {
                "joy": 0.92,
                "sadness": 0.05,
                "anger": 0.02,
                "fear": 0.01
            },
            "energy": 0.8,
            "pitch_mean": 220.5,
            "pitch_range": (150, 290)
        }
        """
        pass
    
    def detect_emotion_batch(self, audio_path: str, 
                            segments: List[Dict]) -> List[Dict]:
        """Detect emotion for multiple segments efficiently."""
        # Process all segments at once for better performance
        pass
```

### Integration with ASR

```python
from src.asr import WhisperWithDiarizationASR
from src.emotion import AudioEmotionDetector

# Transcribe with speaker diarization
asr = WhisperWithDiarizationASR()
asr_result = asr.transcribe("audio.wav")

# Detect emotion for each segment
emotion_detector = AudioEmotionDetector()
for segment in asr_result.segments:
    emotion = emotion_detector.detect_emotion(
        "audio.wav",
        segment["offset"],
        segment["offset"] + segment["duration"]
    )
    segment["emotion"] = emotion["emotion"]
    segment["energy"] = emotion["energy"]
    segment["pitch_mean"] = emotion["pitch_mean"]

# Now segments have emotion information
for segment in asr_result.segments:
    print(f"{segment['speaker']} ({segment['emotion']}): {segment['text']}")
```

---

## Phase 2 Component: Emotion-Aware Translation

### Proposed Implementation

```python
from src.translator import EmotionAwareTranslator

class EmotionAwareTranslator(AbstractTranslator):
    """Translator that preserves emotional tone and energy."""
    
    def translate(self, 
                 text: str, 
                 source_lang: str, 
                 target_lang: str,
                 emotion: str = None,
                 energy: float = None,
                 speaker: str = None) -> str:
        """
        Translate text while preserving emotional intent.
        
        Example:
            text = "I absolutely LOVE this!"  # High energy, joy
            
            # Without emotion awareness:
            result = translator.translate(text, "en", "es")
            # "Me encanta esto!" (literal translation)
            
            # With emotion awareness:
            result = translator.translate(
                text, "en", "es",
                emotion="joy",
                energy=0.9
            )
            # "¡Me encanta MUCHO esto!" (preserves enthusiasm)
        """
        pass
```

### Enhanced Groq Translator

```python
# In src/translator.py, GroqTranslator enhancement:

def translate(self, text: str, source_lang: str, target_lang: str,
             emotion: str = None, energy: float = None,
             speaker: str = None) -> str:
    
    # Build emotion-aware system prompt
    system_prompt = f"""
    You are a professional translator. Translate the following text from 
    {source_lang} to {target_lang}.
    
    CRITICAL: Preserve the emotional tone and energy of the original text.
    """
    
    if emotion:
        system_prompt += f"""
    
    The original text has the emotion: {emotion}
    - If emotion is 'joy', 'excitement', 'enthusiasm': use exclamations, 
      emphasis, and positive language
    - If emotion is 'sadness', 'melancholy': use softer language, 
      reflect the somber tone
    - If emotion is 'anger', 'frustration': use sharp, direct language
    - If emotion is 'fear', 'anxiety': use cautious, reserved language
    """
    
    if energy is not None:
        system_prompt += f"""
    
    The original has energy level {energy:.1f}/1.0:
    - High energy (>0.7): Use emphatic expressions, exclamations
    - Medium energy (0.3-0.7): Use normal conversational tone
    - Low energy (<0.3): Use subdued, quiet language
    """
    
    # Make translation API call with emotion-aware prompt
    response = self.client.messages.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    )
    
    return response.content[0].text
```

---

## Phase 2 Component: Speaker-Specific TTS

### Proposed Implementation

```python
from src.tts import SpeakerAwareTTS

class SpeakerAwareTTS(AbstractTTS):
    """TTS that adapts voice and style per speaker."""
    
    def __init__(self):
        # Initialize with speaker voice profiles
        # Build voice banks for different speakers
        pass
    
    def register_speaker(self, speaker_id: str, 
                        voice_profile: Dict = None) -> None:
        """Register a speaker with specific voice characteristics."""
        pass
    
    def synthesize(self, 
                  text: str, 
                  output_path: str,
                  speaker: str = None,
                  emotion: str = None,
                  energy: float = None,
                  pitch_shift: float = 0.0) -> None:
        """
        Synthesize speech with speaker-specific characteristics.
        
        Args:
            text: Text to synthesize
            output_path: Where to save the audio
            speaker: Speaker ID (determines voice)
            emotion: Emotion to express
            energy: Energy level (affects speech rate and loudness)
            pitch_shift: Pitch adjustment (semitones)
        """
        pass
    
    def synthesize_batch(self, segments: List[Dict], 
                        audio_path: str) -> str:
        """Synthesize multiple segments with speaker awareness."""
        # Process all segments
        # Use appropriate voice for each speaker
        # Mix them together
        # Return combined audio path
        pass
```

### Azure Speech Integration for Phase 2

```python
from src.tts import AzureSpeakerAwareTTS

class AzureSpeakerAwareTTS(SpeakerAwareTTS):
    """Use Azure Speech for high-quality speaker-aware TTS."""
    
    def __init__(self):
        self.client = TextToSpeechClient()
        self.voice_profiles = {}
    
    def synthesize(self, text: str, output_path: str,
                  speaker: str = None, 
                  emotion: str = None,
                  energy: float = None) -> None:
        
        # Select voice based on speaker
        voice = self.select_voice(speaker, emotion)
        
        # Adjust prosody based on emotion
        ssml = self.build_ssml(
            text=text,
            voice=voice,
            emotion=emotion,
            energy=energy,
            pitch=self.calculate_pitch(emotion),
            rate=self.calculate_rate(energy)
        )
        
        # Synthesize using Azure
        result = self.client.synthesize_speech_from_text(ssml)
        
        with open(output_path, 'wb') as f:
            f.write(result.audio_data)
```

---

## Phase 2 Integration: Full Pipeline

```python
from src.asr import WhisperWithDiarizationASR
from src.emotion import AudioEmotionDetector
from src.translator import EmotionAwareTranslator
from src.tts import AzureSpeakerAwareTTS

class EmotionAwareDubbingPipeline:
    """Full pipeline with emotion awareness and speaker-specific synthesis."""
    
    def run(self, source_lang: str, target_lang: str, 
            video_path: str) -> dict:
        
        # Phase 1: Extract audio and transcribe
        audio_path = extract_audio(video_path)
        asr = WhisperWithDiarizationASR()
        asr_result = asr.transcribe(audio_path)
        
        # Phase 2: Detect emotions
        emotion_detector = AudioEmotionDetector()
        for segment in asr_result.segments:
            emotion = emotion_detector.detect_emotion(
                audio_path,
                segment["offset"],
                segment["offset"] + segment["duration"]
            )
            segment["emotion"] = emotion["emotion"]
            segment["energy"] = emotion["energy"]
        
        # Phase 2: Emotion-aware translation
        translator = EmotionAwareTranslator()
        translations = []
        for segment in asr_result.segments:
            translated = translator.translate(
                segment["text"],
                source_lang,
                target_lang,
                emotion=segment["emotion"],
                energy=segment["energy"],
                speaker=segment["speaker"]
            )
            translations.append({
                "original": segment["text"],
                "translated": translated,
                "emotion": segment["emotion"],
                "offset": segment["offset"],
                "duration": segment["duration"],
                "speaker": segment["speaker"]
            })
        
        # Phase 2: Speaker-aware TTS
        tts = AzureSpeakerAwareTTS()
        for trans in translations:
            audio = tts.synthesize(
                trans["translated"],
                speaker=trans["speaker"],
                emotion=trans["emotion"],
                energy=segment.get("energy", 0.5)
            )
            trans["audio"] = audio
        
        # Mix audio segments respecting original timing
        final_audio = mix_audio_segments(
            translations,
            original_duration=get_video_duration(video_path)
        )
        
        return {
            "transcript": asr_result.text,
            "segments": translations,
            "audio": final_audio,
            "emotions": [s["emotion"] for s in translations],
            "speakers": list(set(s["speaker"] for s in translations))
        }
```

---

## Phase 2 Example Output

**Input:** Multi-speaker video with varied emotions

```
Video: interview.mp4
Speakers: Interviewer, Guest
Duration: 5 minutes
```

**ASR (Phase 1):**
```
[0.5s] Interviewer: So tell me about your project?
[3.2s] Guest: I'm so excited about it! It's going to change everything!
[6.1s] Interviewer: That's wonderful. What challenges did you face?
[8.9s] Guest: Well, honestly it was difficult... lots of late nights.
```

**With Emotion Detection (Phase 2):**
```
[0.5s] Interviewer (neutral): So tell me about your project?
[3.2s] Guest (joy, energy=0.95): I'm so excited about it! It's going to change everything!
[6.1s] Interviewer (interested): That's wonderful. What challenges did you face?
[8.9s] Guest (sadness, energy=0.4): Well, honestly it was difficult... lots of late nights.
```

**Translated Spanish with Emotion Preservation (Phase 2):**
```
[0.5s] Interviewer: ¿Cuéntame sobre tu proyecto?
[3.2s] Guest: ¡Estoy TAN emocionado! ¡Esto va a cambiar TODO!
         ↑ Preserves excitement with emphasis marks and exclamations
[6.1s] Interviewer: Qué maravilloso. ¿Qué desafíos enfrentaste?
[8.9s] Guest: Bueno, honestamente fue difícil... muchas noches sin dormir.
         ↑ Preserves melancholy with subdued language
```

**Dubbed Audio (Phase 2):**
- Interviewer's segments use steady, professional voice
- Guest's excited segments use higher pitch, faster speech rate
- Guest's sad segments use lower pitch, slower speech rate
- All timing synced to original video

---

## Technical Dependencies for Phase 2

```python
# Emotion detection models
# Option 1: wav2vec2-emotion (HuggingFace)
transformers>=4.30.0
torch>=2.0.0

# Option 2: librosa-based emotion detection
librosa>=0.10.0
numpy>=1.21.0

# Advanced TTS with prosody
# Azure Speech (high quality, paid)
azure-cognitiveservices-speech>=1.23.0

# Or open-source option
# espeakng (free, lower quality)
espeak-ng>=1.50

# Audio processing
soundfile>=0.12.0
pydub>=0.25.1
librosa>=0.10.0
```

---

## Phase 2 Roadmap

### Week 1: Emotion Detection
- [ ] Implement `AudioEmotionDetector` class
- [ ] Integrate with ASR pipeline
- [ ] Test on sample videos
- [ ] Create examples

### Week 2: Emotion-Aware Translation
- [ ] Enhance all translators with emotion awareness
- [ ] Test Groq with emotion prompts
- [ ] Test OpenAI with emotion preservation
- [ ] Create translation examples

### Week 3: Speaker-Aware TTS
- [ ] Implement `SpeakerAwareTTS` base class
- [ ] Create Azure TTS with prosody control
- [ ] Speaker voice profile management
- [ ] Voice bank creation

### Week 4: Integration & Testing
- [ ] Combine all components into `EmotionAwareDubbingPipeline`
- [ ] Test end-to-end on multi-speaker videos
- [ ] Optimize timing and synchronization
- [ ] Create comprehensive documentation

---

## Success Metrics for Phase 2

✅ **Emotion Detection:**
- 85%+ accuracy on emotion classification
- Segment-level emotion consistency

✅ **Translation:**
- Emotion preserved in translated text
- No loss of emphasis or sentiment markers

✅ **TTS:**
- Speaker-differentiated audio
- Emotion reflected in prosody (pitch, rate)
- Proper timing synchronization

✅ **User Experience:**
- Dubbed video sounds natural
- Speaker characteristics maintained
- Emotional tone preserved

---

## Next Steps

1. **Review Phase 1 (Current)**
   - Verify ASR with speaker diarization works
   - Test on your sample videos
   - Ensure segment data quality

2. **Plan Phase 2**
   - Decide on emotion detection model
   - Choose TTS backend (Azure vs. open-source)
   - Plan speaker profile management

3. **Start Phase 2**
   - Implement emotion detection
   - Enhance translators
   - Create TTS with prosody

---

## Resources & References

- **Emotion Detection Models**: https://huggingface.co/models?task=audio-classification
- **Azure Speech SSML**: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup
- **Speaker Diarization**: https://github.com/pyannote/pyannote-audio
- **Prosody Control**: https://github.com/mozilla/TTS

---

This Phase 2 preview shows how the ASR foundation enables emotion-aware, speaker-specific dubbing! 🎉
