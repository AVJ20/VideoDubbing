# Architecture and Component Relationships

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VIDEO DUBBING PIPELINE                           │
│                                                                         │
│  ┌──────────┐  ┌─────────┐  ┌────────────┐  ┌─────────┐  ┌──────────┐ │
│  │  Video   │→ │ Audio   │→ │    ASR +   │→ │Segment  │→ │ Speaker  │ │
│  │ Input    │  │Extract  │  │Diarization │  │-ation   │  │ Profile  │ │
│  └──────────┘  └─────────┘  └────────────┘  └─────────┘  └──────────┘ │
│                                                              │            │
│                                                              ↓            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ ┌──────────┐   │
│  │ Dubbed   │← │ Combine  │← │ Alignment│← │   TTS    │ │Translate │   │
│  │ Output   │  │ Segments │  │          │  │Synthesis │ │ Segments │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ └──────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Detailed Component Interaction

```
                    INPUT PIPELINE
                    ═══════════════

Video File (MP4, MKV, etc.)
        │
        ↓
    [Audio Extraction]
        │
        ↓
    Raw Audio (WAV, 16kHz)
        │
        ├─────────────────────────────────────────────────┐
        │                                                 │
        ↓                                                 │
    [ASR: Whisper]                                       │
        │                                                 │
        ├──→ Transcribed Text                             │
        │                                                 │
        ↓                                                 │
    [Diarization: Pyannote]                              │
        │                                                 │
        ├──→ Speaker Labels                               │
        │    (Speaker_1, Speaker_2, etc.)                 │
        │                                                 │
        ↓                                                 │
    [ASR Result]                                          │
    ├─ text: full transcript                              │
    └─ segments:                                          │
       ├─ {text, speaker, offset, duration, ...}        │
       ├─ {text, speaker, offset, duration, ...}        │
       └─ {...}                                           │
        │                                                 │
        ↓                                                 │
    ═══════════════════════════════════════════════════════

                 PROCESSING PIPELINE
                 ═══════════════════

    [SEGMENTATION MODULE]
    ┌─────────────────────────────────────────┐
    │ Combines logical + speaker boundaries    │
    │                                         │
    │ Input: ASR segments + speaker segments  │
    │ Output: Refined segments                │
    │                                         │
    │ Algorithm:                              │
    │ 1. Start with ASR segments              │
    │ 2. Detect speaker changes within       │
    │ 3. Split on speaker boundaries         │
    │ 4. Merge short segments                │
    │ 5. Validate and renumber               │
    └─────────────────────────────────────────┘
        │
        ↓
    [Segment Data]
    ├─ Segment 0: "Hello" [Speaker_1, 0.0s-2.0s]
    ├─ Segment 1: "Hi there" [Speaker_2, 2.1s-3.6s]
    ├─ Segment 2: "Good to see you" [Speaker_1, 3.7s-5.2s]
    └─ ...
        │
        ├─────────────────────────────────────┬────────────┐
        │                                     │            │
        ↓                                     ↓            ↓
    [Speaker Profiles]               [Translation]    (parallel)
    ├─ Speaker_1:                       │
    │  ├─ language: es                  ↓
    │  ├─ voice_reference: ref.wav   [TRANSLATED SEGMENTS]
    │  └─ emotion: professional      ├─ "Hola" [Speaker_1]
    │                                ├─ "Hola ahí" [Speaker_2]
    └─ Speaker_2:                    └─ ...
       ├─ language: es
       ├─ voice_reference: ref2.wav
       └─ emotion: casual
        │
        └─────────────────────────────────────┐
                                              │
        ┌─────────────────────────────────────┘
        │
        ↓
    [TTS SYNTHESIS MODULE]
    ┌────────────────────────────────────────────┐
    │ Synthesizes dubbed audio with voice cloning │
    │                                            │
    │ For each segment:                          │
    │ 1. Get speaker profile                     │
    │ 2. Load voice reference (if available)     │
    │ 3. Synthesize text with speaker voice      │
    │ 4. Get actual duration of output audio     │
    │ 5. Save segment audio file                 │
    └────────────────────────────────────────────┘
        │
        ↓
    [TTS Results]
    ├─ segment_0000.wav: 2.3s (Speaker_1)
    ├─ segment_0001.wav: 1.9s (Speaker_2)
    ├─ segment_0002.wav: 3.0s (Speaker_1)
    └─ ...
    
    Also: segment_durations = {0: 2.3, 1: 1.9, 2: 3.0, ...}
        │
        ├────────────────────────────────────────┐
        │                                        │
        ↓                                        ↓
    [Dubbed Audio Segments]          [ALIGNMENT MODULE]
                                     ┌──────────────────────────────┐
                                     │ Synchronizes timing          │
                                     │                              │
                                     │ Input:                       │
                                     │ ├─ Source segments           │
                                     │ └─ Target durations          │
                                     │                              │
                                     │ Process:                     │
                                     │ 1. Compare durations         │
                                     │ 2. Choose alignment strategy │
                                     │ 3. Adjust timing             │
                                     │ 4. Propagate changes         │
                                     │ 5. Validate sync             │
                                     └──────────────────────────────┘
        │                                        │
        └────────────────────────────────────────┤
                                                │
                                                ↓
                                    [Alignment Results]
                                    ├─ Segment 0: 0.0s-2.0s→2.0s-2.3s (scale: 1.15x)
                                    ├─ Segment 1: 2.1s-3.6s→2.4s-4.3s (scale: 0.88x)
                                    └─ ...
                                                │
                                                ↓
                                    [FINAL OUTPUT]
                                    ├─ output/segments/
                                    │  ├─ segment_0000.wav
                                    │  ├─ segment_0001.wav
                                    │  └─ ...
                                    ├─ output/synthesis_report.json
                                    ├─ output/segments.json
                                    ├─ output/alignment.json
                                    └─ output/pipeline_metadata.json

    ═══════════════════════════════════════════════════════════════════
```

## Component Dependencies Graph

```
┌──────────────────┐
│   Video Input    │
└────────┬─────────┘
         │
         ↓
┌────────────────────┐         ┌──────────────────┐
│ Audio Extraction   │────────→│ Raw Audio (WAV)  │
└────────────────────┘         └────────┬─────────┘
                                        │
                   ┌────────────────────┼────────────────────┐
                   │                    │                    │
                   ↓                    ↓                    ↓
         ┌──────────────────┐  ┌──────────────────┐ ┌────────────────┐
         │  ASR (Whisper)   │  │Diarization       │ │   (Internal)   │
         │                  │  │(Pyannote)        │ │ Synchronization│
         └────────┬─────────┘  └────────┬─────────┘ └────────────────┘
                  │                    │
                  └────────┬───────────┘
                           │
                           ↓
            ┌──────────────────────────────┐
            │  ASR Result                  │
            │  ├─ Full text                │
            │  └─ segments (with speaker)  │
            └──────────────────┬───────────┘
                               │
                               ↓
            ┌──────────────────────────────┐
            │  SEGMENTATION               │ ◄─── NEW COMPONENT
            │  ├─ Combines boundaries      │
            │  ├─ Splits on speaker change │
            │  └─ Merges short segments    │
            └──────────────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                │                            │
                ↓                            ↓
    ┌──────────────────────┐      ┌──────────────────┐
    │  Segmentation Result │      │ Translation      │
    │  ├─ Segments         │      │ (Groq/OpenAI)    │
    │  ├─ Speakers         │      │                  │
    │  └─ Timing info      │      └────────┬─────────┘
    └──────────┬───────────┘               │
               │                           │
               ↓                           ↓
    ┌──────────────────────┐      ┌──────────────────┐
    │ Speaker Profile      │      │ Translated       │
    │ Registration         │      │ Segments         │
    │ ├─ Language          │      │ ├─ text          │
    │ ├─ Voice reference   │      │ ├─ speaker       │
    │ ├─ Emotion/style     │      │ └─ timing        │
    │ └─ Prosody params    │      └────────┬─────────┘
    └──────────┬───────────┘               │
               │                           │
               └───────────────┬───────────┘
                               │
                               ↓
            ┌──────────────────────────────┐
            │  TTS SYNTHESIS              │ ◄─── NEW COMPONENT
            │  ├─ Voice cloning            │
            │  ├─ Multilingual synthesis   │
            │  └─ Batch processing         │
            └──────────────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                │                            │
                ↓                            ↓
    ┌──────────────────────┐      ┌──────────────────┐
    │  TTS Results         │      │ Segment Durations│
    │  ├─ Audio files      │      │ {seg_id: time}   │
    │  ├─ Success rates    │      └────────┬─────────┘
    │  └─ Metadata         │               │
    └──────────┬───────────┘               │
               │                           │
               └───────────────┬───────────┘
                               │
                               ↓
            ┌──────────────────────────────┐
            │  ALIGNMENT                  │ ◄─── NEW COMPONENT
            │  ├─ Compare durations        │
            │  ├─ Adjust timing            │
            │  ├─ Validate sync            │
            │  └─ Generate timing maps     │
            └──────────────────┬───────────┘
                               │
                               ↓
            ┌──────────────────────────────┐
            │  FINAL OUTPUT                │
            │  ├─ Segment audio files      │
            │  ├─ Metadata (JSON)          │
            │  ├─ Timing information       │
            │  └─ Execution log            │
            └──────────────────────────────┘
```

## Class Hierarchy

```
Segmentation Classes:
  ├─ SegmentType (Enum)
  │  ├─ LOGICAL
  │  ├─ SPEAKER_CHANGE
  │  └─ COMBINED
  │
  ├─ Segment (DataClass)
  │  ├─ id: int
  │  ├─ text: str
  │  ├─ speaker: str
  │  ├─ start_time: float
  │  ├─ end_time: float
  │  ├─ segment_type: SegmentType
  │  ├─ confidence: float
  │  └─ words: List[dict]
  │
  ├─ SegmentationResult (DataClass)
  │  ├─ segments: List[Segment]
  │  ├─ total_duration: float
  │  ├─ speaker_count: int
  │  └─ speakers: List[str]
  │
  ├─ AudioSegmenter (Main Engine)
  │  ├─ segment()
  │  ├─ _detect_speaker_changes()
  │  ├─ _find_overlapping_speakers()
  │  ├─ _split_on_speaker_boundaries()
  │  ├─ _merge_short_segments()
  │  └─ _renumber_segments()
  │
  └─ SegmentationValidator
     └─ validate()

Alignment Classes:
  ├─ AlignmentStrategy (Enum)
  │  ├─ STRICT
  │  ├─ FLEXIBLE
  │  └─ ADAPTIVE
  │
  ├─ TimingMap (DataClass)
  │  ├─ source_start/end: float
  │  ├─ target_start/end: float
  │  └─ scaling_factor: float
  │
  ├─ AlignmentResult (DataClass)
  │  ├─ segment_id: int
  │  ├─ source_start/end: float
  │  ├─ target_start/end: float
  │  ├─ alignment_status: str
  │  ├─ confidence: float
  │  └─ timing_maps: List[TimingMap]
  │
  ├─ SegmentAligner (Main Engine)
  │  ├─ align_segments()
  │  ├─ _align_single_segment()
  │  └─ _validate_and_propagate()
  │
  ├─ TimingAnalyzer
  │  └─ analyze()
  │
  └─ SyncValidator
     └─ validate_sync()

Speaker TTS Classes:
  ├─ VoiceCloneMethod (Enum)
  │  ├─ ZERO_SHOT
  │  ├─ VOICE_NAME
  │  ├─ STYLE_TRANSFER
  │  └─ PROSODY_MATCHING
  │
  ├─ SpeakerProfile (DataClass)
  │  ├─ speaker_id: str
  │  ├─ language: str
  │  ├─ voice_reference: str
  │  ├─ voice_name: str
  │  ├─ clone_method: VoiceCloneMethod
  │  ├─ pace/pitch/energy: float
  │  ├─ emotion/style: str
  │  └─ metadata: Dict
  │
  ├─ TTSSegment (DataClass)
  │  ├─ segment_id: int
  │  ├─ text: str
  │  ├─ speaker_id: str
  │  ├─ speaker_profile: SpeakerProfile
  │  ├─ start_time/end_time: float
  │  ├─ language: str
  │  ├─ output_path: str
  │  └─ duration: float
  │
  ├─ TTSResult (DataClass)
  │  ├─ segment_id: int
  │  ├─ speaker_id: str
  │  ├─ success: bool
  │  ├─ output_path: str
  │  ├─ duration: float
  │  ├─ text: str
  │  ├─ error: str
  │  └─ metadata: Dict
  │
  ├─ AbstractSpeakerTTS (ABC)
  │  ├─ synthesize_segment()
  │  ├─ register_speaker()
  │  └─ get_supported_languages()
  │
  ├─ CoquiSpeakerTTS (implements AbstractSpeakerTTS)
  │  ├─ synthesize_segment()
  │  ├─ register_speaker()
  │  └─ get_supported_languages()
  │
  └─ SpeakerTTSOrchestrator (Coordinator)
     ├─ register_speaker()
     ├─ synthesize_segments()
     ├─ get_segment_durations()
     └─ get_synthesis_report()

Pipeline Classes:
  ├─ DetailedPipelineConfig (DataClass)
  │  ├─ work_dir: str
  │  ├─ output_dir: str
  │  ├─ sample_rate: int
  │  ├─ Segmentation config
  │  ├─ Alignment config
  │  ├─ TTS config
  │  └─ debug: bool
  │
  ├─ PipelineState (DataClass)
  │  ├─ stage: str
  │  ├─ video_path: str
  │  ├─ audio_path: str
  │  ├─ source/target_language: str
  │  ├─ asr_result: ASRResult
  │  ├─ segmentation_result: SegmentationResult
  │  ├─ alignment_results: List
  │  ├─ tts_results: List
  │  └─ metadata: Dict
  │
  └─ DetailedDubbingPipeline (Main Orchestrator)
     ├─ __init__()
     ├─ run()
     ├─ _extract_audio()
     ├─ _transcribe()
     ├─ _segment()
     ├─ _register_speakers()
     ├─ _translate_segments()
     ├─ _synthesize_dubbed_audio()
     ├─ _align_segments()
     └─ _generate_output()
```

## Data Flow Example: 30-Second Conversation

```
Input: video_conversation.mp4

Stage 1: Transcription
─────────────────────
ASR Output:
├─ [0.0s-2.0s] "Hello, my name is Alice" [Speaker: Speaker_1]
├─ [2.1s-3.5s] "Hi Alice, I'm Bob" [Speaker: Speaker_2]
├─ [3.6s-5.5s] "Nice to meet you" [Speaker: Speaker_1]
└─ [5.6s-6.8s] "You too" [Speaker: Speaker_2]

Diarization: 2 speakers detected ✓

Stage 2: Segmentation
──────────────────────
Segmentation Output:
├─ Segment 0: Alice, 0.0s-2.0s, "Hello, my name is Alice" [LOGICAL]
├─ Segment 1: Bob, 2.1s-3.5s, "Hi Alice, I'm Bob" [LOGICAL]
├─ Segment 2: Alice, 3.6s-5.5s, "Nice to meet you" [LOGICAL]
└─ Segment 3: Bob, 5.6s-6.8s, "You too" [LOGICAL]

Speakers: [Alice, Bob]
Total Duration: 6.8s

Stage 3: Speaker Registration
──────────────────────────────
Profiles Created:
├─ Alice:
│  ├─ language: es
│  ├─ voice_reference: alice_english.wav (provided)
│  └─ clone_method: ZERO_SHOT
│
└─ Bob:
   ├─ language: es
   ├─ voice_reference: bob_english.wav (provided)
   └─ clone_method: ZERO_SHOT

Stage 4: Translation (en→es)
────────────────────────────
Translated Segments:
├─ Segment 0: "Hola, me llamo Alice"
├─ Segment 1: "Hola Alice, soy Bob"
├─ Segment 2: "Encantada de conocerte"
└─ Segment 3: "Igualmente"

Stage 5: TTS Synthesis
──────────────────────
TTS Output:
├─ segment_0000.wav: 2.3s ← Source 2.0s (Alice's voice)
├─ segment_0001.wav: 1.8s ← Source 1.4s (Bob's voice)
├─ segment_0002.wav: 2.2s ← Source 1.9s (Alice's voice)
└─ segment_0003.wav: 0.9s ← Source 1.2s (Bob's voice)

TTS Success: 4/4 ✓

Stage 6: Alignment
──────────────────
Alignment (Strategy: ADAPTIVE):
├─ Segment 0: 0.0s-2.0s → 0.0s-2.3s (stretch 1.15x)
├─ Segment 1: 2.1s-3.5s → 2.4s-4.2s (compress 0.86x)
├─ Segment 2: 3.6s-5.5s → 4.3s-6.5s (stretch 0.95x)
└─ Segment 3: 5.6s-6.8s → 6.6s-7.5s (compress 0.75x)

Timing Analysis:
├─ Source total: 6.8s
├─ Target total: 7.5s
├─ Average scale: 0.97x
└─ Status: ✓ VALID SYNC

Stage 7: Output
────────────────
output/
├─ segments/
│  ├─ segment_0000.wav (2.3s, Alice in Spanish)
│  ├─ segment_0001.wav (1.8s, Bob in Spanish)
│  ├─ segment_0002.wav (2.2s, Alice in Spanish)
│  └─ segment_0003.wav (0.9s, Bob in Spanish)
│
├─ synthesis_report.json
│  └─ Success: 4/4, speakers: [Alice, Bob]
│
├─ segments.json
│  └─ Original + translated text with speaker/timing
│
├─ alignment.json
│  └─ Timing maps and scaling factors
│
└─ pipeline_metadata.json
   └─ Complete execution trace

Final Result: 7.5s of dubbed Spanish audio ✓
All speaker identities preserved through voice cloning ✓
```

## Processing Pipeline Flow

```
┌────────────────────────────────────────────────────────────────────┐
│ DETAILED DUBBING PIPELINE - COMPLETE FLOW                         │
└────────────────────────────────────────────────────────────────────┘

Input: Video File
    │
    ├─[PARALLEL] Audio Extraction & Validation
    │
    ├─ Extract WAV @ 16kHz
    ├─ Validate audio quality
    │
    ├─[SEQUENTIAL] ASR + Diarization
    │
    ├─ Transcribe with Whisper
    ├─ Diarize with Pyannote
    ├─ Merge results
    │
    ├─[PARALLEL] Segmentation & Translation Setup
    │
    ├─ Segment (logical + speaker)
    ├─ Prepare speaker profiles
    │
    ├─[SEQUENTIAL] Translation + Speaker Registration
    │
    ├─ Translate each segment
    ├─ Register speaker voices
    │
    ├─[PARALLEL] TTS Synthesis
    │
    ├─ Synthesize each segment
    ├─ Collect durations
    │
    ├─[SEQUENTIAL] Alignment
    │
    ├─ Align segments
    ├─ Validate sync
    │
    ├─ Generate Output
    │
    └─ Final Dubbed Audio + Metadata

Duration: ~2-3 min (CPU) or 30-45s (GPU) for 1 minute video
```

This architecture enables the next phase: **Agentic Framework** where multiple agents can:
- Explore different segmentation parameters
- Evaluate quality at each step
- Optimize component selection
- Iteratively improve results
