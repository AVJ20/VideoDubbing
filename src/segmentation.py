"""
Audio Segmentation Module

Handles segmentation of audio based on:
1. Logical segments from ASR (sentence/phrase boundaries)
2. Speaker changes (diarization boundaries)

Segments are created when either:
- A logical end is detected (ASR segment boundary)
- A speaker change occurs (different speaker detected)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class SegmentType(Enum):
    """Type of segment boundary detection."""
    LOGICAL = "logical"  # ASR-based logical segment
    SPEAKER_CHANGE = "speaker_change"  # Speaker diarization boundary
    COMBINED = "combined"  # Both logical and speaker change


@dataclass
class Segment:
    """Represents a contiguous segment of audio with consistent properties."""
    
    id: int
    text: str
    speaker: str
    start_time: float  # seconds
    end_time: float  # seconds
    segment_type: SegmentType
    confidence: float = 1.0
    words: List[dict] = field(default_factory=list)  # Word-level timing from ASR
    
    @property
    def duration(self) -> float:
        """Duration of segment in seconds."""
        return self.end_time - self.start_time
    
    def __repr__(self) -> str:
        return (
            f"Segment(id={self.id}, speaker='{self.speaker}', "
            f"time=[{self.start_time:.2f}s-{self.end_time:.2f}s], "
            f"type={self.segment_type.value}, text='{self.text[:30]}...')"
        )


@dataclass
class SegmentationResult:
    """Result of audio segmentation."""
    
    segments: List[Segment]
    total_duration: float
    speaker_count: int
    speakers: List[str] = field(default_factory=list)
    
    def __repr__(self) -> str:
        return (
            f"SegmentationResult(segments={len(self.segments)}, "
            f"duration={self.total_duration:.1f}s, "
            f"speakers={self.speaker_count}, speakers={self.speakers})"
        )


class AudioSegmenter:
    """
    Segments audio based on both logical (ASR) and speaker boundaries.
    
    Strategy:
    1. Start with ASR segments (logical boundaries)
    2. Detect speaker changes within and across ASR segments
    3. Split segments when speaker change is detected
    4. Merge very short segments if needed
    """
    
    def __init__(self, 
                 min_segment_duration: float = 0.5,
                 speaker_change_threshold: float = 0.1):
        """
        Args:
            min_segment_duration: Minimum segment duration in seconds
            speaker_change_threshold: Minimum confidence for speaker change detection
        """
        self.min_segment_duration = min_segment_duration
        self.speaker_change_threshold = speaker_change_threshold
    
    def segment(self, 
                asr_segments: List[dict],
                speaker_segments: Optional[List[dict]] = None) -> SegmentationResult:
        """
        Segment audio combining ASR and speaker diarization.
        
        Args:
            asr_segments: Segments from ASR (Whisper), each with:
                - text: transcribed text
                - speaker: speaker label (may be 'Unknown')
                - offset: start time in seconds
                - duration: segment duration
                - confidence: transcription confidence
                - words: word-level timing information
            
            speaker_segments: Optional raw speaker diarization, each with:
                - start: start time in seconds
                - end: end time in seconds
                - speaker: speaker label
        
        Returns:
            SegmentationResult with combined segments
        """
        if not asr_segments:
            return SegmentationResult(
                segments=[],
                total_duration=0.0,
                speaker_count=0,
                speakers=[]
            )
        
        logger.info(f"Segmenting {len(asr_segments)} ASR segments")
        
        # Build initial segments from ASR
        segments = []
        segment_id = 0
        
        for asr_seg in asr_segments:
            start_time = asr_seg.get("offset", 0.0)
            duration = asr_seg.get("duration", 0.0)
            end_time = start_time + duration
            
            segment = Segment(
                id=segment_id,
                text=asr_seg.get("text", ""),
                speaker=asr_seg.get("speaker", "Unknown"),
                start_time=start_time,
                end_time=end_time,
                segment_type=SegmentType.LOGICAL,
                confidence=asr_seg.get("confidence", 1.0),
                words=asr_seg.get("words", [])
            )
            segments.append(segment)
            segment_id += 1
        
        # If we have speaker diarization, check for speaker changes within ASR segments
        if speaker_segments:
            logger.info(f"Detecting speaker changes within {len(segments)} segments")
            segments = self._detect_speaker_changes(segments, speaker_segments, segment_id)
        
        # Post-processing
        segments = self._merge_short_segments(segments)
        segments = self._renumber_segments(segments)
        
        # Gather stats
        total_duration = max([s.end_time for s in segments], default=0.0)
        speakers = sorted(set(s.speaker for s in segments))
        
        result = SegmentationResult(
            segments=segments,
            total_duration=total_duration,
            speaker_count=len(speakers),
            speakers=speakers
        )
        
        logger.info(f"Segmentation complete: {result}")
        return result
    
    def _detect_speaker_changes(self, 
                               asr_segments: List[Segment],
                               speaker_segments: List[dict],
                               start_id: int) -> List[Segment]:
        """
        Detect speaker changes within ASR segments and split accordingly.
        
        Algorithm:
        1. For each ASR segment, find all speaker diarization segments that overlap
        2. If multiple speakers are found, create sub-segments for each speaker
        3. Update segment type to SPEAKER_CHANGE or COMBINED
        """
        new_segments = []
        segment_id = start_id
        
        for asr_seg in asr_segments:
            # Find all speaker segments overlapping with this ASR segment
            overlapping_speakers = self._find_overlapping_speakers(
                asr_seg.start_time, 
                asr_seg.end_time,
                speaker_segments
            )
            
            if len(overlapping_speakers) <= 1:
                # No speaker change in this segment
                new_segments.append(asr_seg)
            else:
                # Multiple speakers detected - split the segment
                logger.debug(
                    f"Speaker change detected in segment {asr_seg.id}: "
                    f"{overlapping_speakers} at {asr_seg.start_time:.2f}s"
                )
                
                sub_segments = self._split_on_speaker_boundaries(
                    asr_seg,
                    overlapping_speakers,
                    segment_id
                )
                new_segments.extend(sub_segments)
                segment_id += len(sub_segments)
        
        return new_segments
    
    def _find_overlapping_speakers(self,
                                  start_time: float,
                                  end_time: float,
                                  speaker_segments: List[dict]) -> List[Tuple[str, float, float]]:
        """
        Find all speakers and their time ranges that overlap with [start_time, end_time].
        
        Returns list of (speaker_label, overlap_start, overlap_end) tuples.
        """
        overlapping = []
        seen_speakers = set()
        
        for ds in speaker_segments:
            ds_start = ds.get("start", 0.0)
            ds_end = ds.get("end", 0.0)
            ds_speaker = ds.get("speaker", "Unknown")
            
            # Check for overlap
            overlap_start = max(start_time, ds_start)
            overlap_end = min(end_time, ds_end)
            
            if overlap_end > overlap_start:
                # If we haven't seen this speaker yet, or we have a larger overlap,
                # update the tracking
                if ds_speaker not in seen_speakers:
                    overlapping.append((ds_speaker, overlap_start, overlap_end))
                    seen_speakers.add(ds_speaker)
        
        return overlapping
    
    def _split_on_speaker_boundaries(self,
                                    asr_seg: Segment,
                                    overlapping_speakers: List[Tuple[str, float, float]],
                                    start_id: int) -> List[Segment]:
        """
        Split an ASR segment into sub-segments based on speaker boundaries.
        
        Create a sub-segment for each speaker within the original segment.
        """
        # Sort by start time
        sorted_speakers = sorted(overlapping_speakers, key=lambda x: x[1])
        
        sub_segments = []
        segment_id = start_id
        
        for speaker, speaker_start, speaker_end in sorted_speakers:
            # Sub-segment is bounded by both ASR segment and speaker segment
            sub_start = max(asr_seg.start_time, speaker_start)
            sub_end = min(asr_seg.end_time, speaker_end)
            
            if sub_end > sub_start:  # Valid segment
                # For now, keep the full text (could be refined to split text as well)
                sub_seg = Segment(
                    id=segment_id,
                    text=asr_seg.text,
                    speaker=speaker,
                    start_time=sub_start,
                    end_time=sub_end,
                    segment_type=SegmentType.COMBINED,
                    confidence=asr_seg.confidence,
                    words=asr_seg.words
                )
                sub_segments.append(sub_seg)
                segment_id += 1
        
        # If no valid sub-segments were created, keep the original
        if not sub_segments:
            sub_segments = [asr_seg]
        
        return sub_segments
    
    def _merge_short_segments(self, segments: List[Segment]) -> List[Segment]:
        """
        Merge segments shorter than min_segment_duration with adjacent segments.
        Prioritizes merging with same speaker if possible.
        """
        if not segments:
            return segments
        
        merged = []
        i = 0
        
        while i < len(segments):
            current = segments[i]
            
            # Check if current segment is too short
            if current.duration < self.min_segment_duration and i < len(segments) - 1:
                # Try to merge with next segment
                next_seg = segments[i + 1]
                
                merged_seg = Segment(
                    id=current.id,
                    text=current.text + " " + next_seg.text,
                    speaker=current.speaker if current.speaker != "Unknown" else next_seg.speaker,
                    start_time=current.start_time,
                    end_time=next_seg.end_time,
                    segment_type=SegmentType.COMBINED,
                    confidence=min(current.confidence, next_seg.confidence),
                    words=current.words + next_seg.words
                )
                
                merged.append(merged_seg)
                i += 2  # Skip both segments
                
                logger.debug(f"Merged short segment {current.id} with {next_seg.id}")
            else:
                merged.append(current)
                i += 1
        
        return merged
    
    def _renumber_segments(self, segments: List[Segment]) -> List[Segment]:
        """Renumber segments sequentially after merging/splitting."""
        for i, seg in enumerate(segments):
            seg.id = i
        return segments


class SegmentationValidator:
    """Validates segmentation results for consistency and quality."""
    
    @staticmethod
    def validate(result: SegmentationResult) -> Tuple[bool, List[str]]:
        """
        Validate segmentation result.
        
        Returns:
            (is_valid, list_of_warnings)
        """
        warnings = []
        
        if not result.segments:
            warnings.append("No segments found in segmentation result")
        
        # Check for gaps or overlaps
        for i in range(len(result.segments) - 1):
            curr = result.segments[i]
            next_seg = result.segments[i + 1]
            
            if curr.end_time > next_seg.start_time:
                warnings.append(
                    f"Segment {i} and {i+1} overlap: "
                    f"{curr.end_time:.2f}s vs {next_seg.start_time:.2f}s"
                )
            elif curr.end_time < next_seg.start_time:
                gap = next_seg.start_time - curr.end_time
                if gap > 0.1:  # Allow small gaps
                    warnings.append(
                        f"Gap between segment {i} and {i+1}: {gap:.2f}s"
                    )
        
        # Check for empty segments
        for seg in result.segments:
            if not seg.text.strip():
                warnings.append(f"Segment {seg.id} has empty text")
            if seg.duration <= 0:
                warnings.append(f"Segment {seg.id} has invalid duration: {seg.duration}s")
        
        is_valid = len(warnings) == 0
        return is_valid, warnings
