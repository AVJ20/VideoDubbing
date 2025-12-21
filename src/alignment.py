"""
Audio Alignment Module

Handles alignment between:
1. Source audio and transcribed segments
2. Source segments and target (translated/dubbed) segments
3. Segment timing and duration preservation

Provides methods for:
- Verifying segment timing
- Adjusting segment boundaries based on content
- Calculating time scaling factors when durations change
- Generating alignment maps for video editing
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class AlignmentStrategy(Enum):
    """Strategy for aligning source and target segments."""
    
    STRICT = "strict"  # Target duration must match source
    FLEXIBLE = "flexible"  # Allow duration changes with scaling
    ADAPTIVE = "adaptive"  # Preserve timing where possible, compress/expand as needed


@dataclass
class TimingMap:
    """Maps source timing to target timing."""
    
    source_start: float
    source_end: float
    target_start: float
    target_end: float
    scaling_factor: float = 1.0  # target_duration / source_duration
    
    @property
    def source_duration(self) -> float:
        return self.source_end - self.source_start
    
    @property
    def target_duration(self) -> float:
        return self.target_end - self.target_start


@dataclass
class AlignmentResult:
    """Result of alignment process."""
    
    segment_id: int
    source_start: float
    source_end: float
    target_start: float
    target_end: float
    confidence: float
    alignment_status: str  # "aligned", "stretched", "compressed", "modified"
    timing_maps: List[TimingMap] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class SegmentAligner:
    """
    Aligns segments ensuring consistency and validity.
    
    Features:
    - Validates segment timing
    - Handles timing adjustments when TTS produces different durations
    - Supports different alignment strategies
    - Generates timing maps for video sync
    """
    
    def __init__(self, 
                 strategy: AlignmentStrategy = AlignmentStrategy.ADAPTIVE,
                 slack_time: float = 0.1):
        """
        Args:
            strategy: Alignment strategy to use
            slack_time: Allowed timing difference before adjustment (seconds)
        """
        self.strategy = strategy
        self.slack_time = slack_time
    
    def align_segments(self,
                      source_segments: List,  # From segmentation
                      target_durations: Dict[int, float]) -> List[AlignmentResult]:
        """
        Align source and target segments.
        
        Args:
            source_segments: List of Segment objects from segmentation
            target_durations: Dict mapping segment_id to TTS output duration
        
        Returns:
            List of AlignmentResult objects
        """
        alignment_results = []
        
        for seg in source_segments:
            target_duration = target_durations.get(seg.id, seg.duration)
            
            result = self._align_single_segment(
                seg.id,
                seg.start_time,
                seg.end_time,
                target_duration,
                seg.confidence
            )
            
            alignment_results.append(result)
        
        # Validate and propagate timing adjustments
        self._validate_and_propagate(alignment_results)
        
        logger.info(f"Aligned {len(alignment_results)} segments")
        return alignment_results
    
    def _align_single_segment(self,
                             segment_id: int,
                             source_start: float,
                             source_end: float,
                             target_duration: float,
                             confidence: float) -> AlignmentResult:
        """
        Align a single segment and determine timing strategy.
        """
        source_duration = source_end - source_start
        raw_scaling_factor = (
            target_duration / source_duration if source_duration > 0 else 1.0
        )
        # ffmpeg `atempo` factor that would make output_duration == source_duration
        # given an input duration of `target_duration`.
        required_atempo = raw_scaling_factor
        
        # Determine alignment status
        duration_diff = abs(target_duration - source_duration)
        
        if duration_diff < self.slack_time:
            status = "aligned"
            target_start = source_start
            target_end = source_end
            actual_scaling = 1.0
        elif self.strategy == AlignmentStrategy.STRICT:
            # Force per-segment timing to match the source slot.
            # Any duration mismatch should be resolved via speaking-rate/time-scaling.
            status = "rate_adjusted"
            target_start = source_start
            target_end = source_end
            actual_scaling = raw_scaling_factor
        elif self.strategy == AlignmentStrategy.FLEXIBLE:
            # Allow full duration change
            status = "stretched" if target_duration > source_duration else "compressed"
            target_start = source_start
            target_end = source_start + target_duration
            actual_scaling = raw_scaling_factor
        else:  # ADAPTIVE
            # Preserve start time, adjust end time
            status = "modified"
            target_start = source_start
            target_end = source_start + target_duration
            actual_scaling = raw_scaling_factor
        
        # Create timing map
        timing_map = TimingMap(
            source_start=source_start,
            source_end=source_end,
            target_start=target_start,
            target_end=target_end,
            scaling_factor=actual_scaling
        )
        
        result = AlignmentResult(
            segment_id=segment_id,
            source_start=source_start,
            source_end=source_end,
            target_start=target_start,
            target_end=target_end,
            confidence=confidence,
            alignment_status=status,
            timing_maps=[timing_map],
            metadata={
                "source_duration": source_duration,
                "target_duration": target_duration,
                "scaling_factor": actual_scaling,
                "required_atempo": required_atempo,
                "duration_diff": duration_diff,
            }
        )
        
        return result
    
    def _validate_and_propagate(self, alignment_results: List[AlignmentResult]) -> None:
        """
        Validate alignment and propagate timing adjustments if needed.
        
        For FLEXIBLE and ADAPTIVE strategies, this may adjust timing to avoid
        overlaps or gaps.
        """
        if not alignment_results:
            return
        
        # Check for overlaps and adjust
        for i in range(len(alignment_results) - 1):
            curr = alignment_results[i]
            next_result = alignment_results[i + 1]
            
            if curr.target_end > next_result.target_start:
                overlap = curr.target_end - next_result.target_start
                logger.warning(
                    f"Overlap detected between segment {curr.segment_id} and "
                    f"{next_result.segment_id}: {overlap:.2f}s. "
                    f"Adjusting timing..."
                )
                
                # Adjust next segment to start where current ends
                adjustment = overlap + 0.01  # Small gap
                next_result.target_start += adjustment
                next_result.target_end += adjustment
                
                # Update timing map
                if next_result.timing_maps:
                    next_result.timing_maps[0].target_start = next_result.target_start
                    next_result.timing_maps[0].target_end = next_result.target_end


class TimingAnalyzer:
    """Analyzes timing and generates statistics about alignment."""
    
    @staticmethod
    def analyze(alignment_results: List[AlignmentResult]) -> Dict:
        """
        Analyze alignment results and generate statistics.
        
        Returns:
            Dict with timing statistics
        """
        if not alignment_results:
            return {
                "total_segments": 0,
                "total_source_duration": 0.0,
                "total_target_duration": 0.0,
                "average_scaling_factor": 1.0,
                "status_distribution": {},
                "problematic_segments": []
            }
        
        stats = {
            "total_segments": len(alignment_results),
            "total_source_duration": 0.0,
            "total_target_duration": 0.0,
            "scaling_factors": [],
            "status_distribution": {},
            "problematic_segments": [],
            "duration_changes": {
                "stretched": 0,
                "compressed": 0,
                "aligned": 0
            }
        }
        
        for result in alignment_results:
            stats["total_source_duration"] += result.metadata["source_duration"]
            stats["total_target_duration"] += result.metadata["target_duration"]
            stats["scaling_factors"].append(result.metadata["scaling_factor"])
            
            status = result.alignment_status
            stats["status_distribution"][status] = stats["status_distribution"].get(status, 0) + 1
            
            if status in ["stretched", "compressed", "modified"]:
                stats["duration_changes"][status] += 1
            elif status == "aligned":
                stats["duration_changes"]["aligned"] += 1
            
            # Flag problematic segments (extreme scaling)
            scale = result.metadata["scaling_factor"]
            if scale > 1.5 or scale < 0.7:
                stats["problematic_segments"].append({
                    "segment_id": result.segment_id,
                    "scaling_factor": scale,
                    "source_duration": result.metadata["source_duration"],
                    "target_duration": result.metadata["target_duration"]
                })
        
        # Calculate average
        if stats["scaling_factors"]:
            stats["average_scaling_factor"] = sum(stats["scaling_factors"]) / len(stats["scaling_factors"])
            stats["scaling_range"] = (min(stats["scaling_factors"]), max(stats["scaling_factors"]))
        
        logger.info(
            f"Timing analysis: {stats['total_segments']} segments, "
            f"source={stats['total_source_duration']:.1f}s, "
            f"target={stats['total_target_duration']:.1f}s, "
            f"avg_scale={stats['average_scaling_factor']:.2f}x"
        )
        
        if stats["problematic_segments"]:
            logger.warning(
                f"Found {len(stats['problematic_segments'])} problematic segments "
                f"with extreme scaling"
            )
        
        return stats


class SyncValidator:
    """Validates audio synchronization quality."""
    
    @staticmethod
    def validate_sync(alignment_results: List[AlignmentResult],
                     max_drift: float = 2.0) -> Tuple[bool, List[str]]:
        """
        Validate that segments are properly synchronized.
        
        Args:
            alignment_results: List of AlignmentResult objects
            max_drift: Maximum allowed timing drift in seconds
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        if not alignment_results:
            issues.append("No alignment results to validate")
            return False, issues
        
        cumulative_drift = 0.0
        
        for i, result in enumerate(alignment_results):
            # Check individual segment timing
            source_dur = result.source_end - result.source_start
            target_dur = result.target_end - result.target_start
            
            # Check cumulative drift
            cumulative_drift += (target_dur - source_dur)
            
            if abs(cumulative_drift) > max_drift:
                issues.append(
                    f"Segment {result.segment_id}: cumulative drift {cumulative_drift:.2f}s "
                    f"exceeds threshold {max_drift}s"
                )
            
            # Check for gaps or overlaps with next segment
            if i < len(alignment_results) - 1:
                next_result = alignment_results[i + 1]
                if result.target_end > next_result.target_start:
                    overlap = result.target_end - next_result.target_start
                    issues.append(
                        f"Overlap between segment {result.segment_id} and "
                        f"{next_result.segment_id}: {overlap:.2f}s"
                    )
        
        is_valid = len(issues) == 0
        return is_valid, issues
