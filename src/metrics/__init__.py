"""Evaluation metrics for dubbing quality.

This package computes per-segment and overall metrics to help an evaluation agent
provide feedback to the generation pipeline.

Current metrics:
- Isochrony: voiced overlap between source and dubbed audio
- Speech rate: words/sec and words/min from transcripts

Additional implemented metrics:
- Speaker similarity: SpeechBrain ECAPA embeddings + cosine similarity
- Emotion/affect similarity: prosody similarity (pitch/energy/voicing features)

Planned (stubs included):
- Translation quality
- Higher-fidelity speaker similarity aggregation
- Optional SER model (emotion classification)
"""
