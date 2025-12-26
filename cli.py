"""Video Dubbing Pipeline CLI

Usage:
  python cli.py --file <VIDEO_FILE> --source en --target es [--multi-env]
  python cli.py --url <VIDEO_URL> --source en --target es [--multi-env]

Options:
  --multi-env      Use separate conda environments for ASR and TTS (recommended)
                   Requires: conda envs 'asr' and 'tts' to be created
  --tts-device     Device for TTS: 'cpu' (default) or 'cuda' (GPU)
"""
import argparse
import logging
import os
import sys
import re

from src.env_loader import load_env

# Load environment variables from .env file
load_env()

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


_TIME_RE = re.compile(
    r"^(?:(?P<h>\d+):)?(?P<m>[0-5]?\d):(?P<s>[0-5]?\d(?:\.\d+)?)$"
)


def _parse_time_seconds(value: str) -> float:
    """Parse seconds or HH:MM:SS[.ms] into float seconds."""
    if value is None:
        raise ValueError("time value is None")
    v = str(value).strip()
    if not v:
        raise ValueError("empty time value")
    try:
        return float(v)
    except ValueError:
        pass

    m = _TIME_RE.match(v)
    if not m:
        raise ValueError(
            f"Invalid time '{value}'. Use seconds (e.g. 12.5) "
            "or HH:MM:SS(.ms) (e.g. 00:01:12.500)."
        )
    hours = float(m.group("h") or 0)
    minutes = float(m.group("m") or 0)
    seconds = float(m.group("s") or 0)
    return hours * 3600.0 + minutes * 60.0 + seconds


def main():
    parser = argparse.ArgumentParser(description="Video Dubbing Pipeline")
    parser.add_argument("--url", default=None, help="Video URL to download and dub")
    parser.add_argument("--file", default=None, help="Local video file path (alternative to --url)")
    parser.add_argument("--source", default="auto", help="Source language (ISO code or 'auto')")
    parser.add_argument("--target", required=True, help="Target language (ISO code)")
    parser.add_argument("--work-dir", default="work", help="Working directory")
    parser.add_argument("--enhanced", action="store_true",
                       help="Use enhanced pipeline with speaker segmentation, alignment, and speaker-specific TTS")
    parser.add_argument("--multi-env", action="store_true", 
                       help="Use separate conda environments for ASR and TTS (recommended)")
    parser.add_argument("--tts-device", default="cpu", choices=["cpu", "cuda"],
                       help="Device for TTS synthesis: 'cpu' or 'cuda' (GPU)")
    parser.add_argument("--output-dir", default=None, help="Output directory for results (default: work/output)")
    parser.add_argument(
        "--start",
        default=None,
        help=(
            "Start time for dubbing (seconds or HH:MM:SS[.ms]). "
            "If set, only this part of the video is dubbed."
        ),
    )
    parser.add_argument(
        "--end",
        default=None,
        help=(
            "End time for dubbing (seconds or HH:MM:SS[.ms]). "
            "Requires --start."
        ),
    )

    parser.add_argument(
        "--metrics",
        action="store_true",
        help=(
            "Compute evaluation metrics (isochrony, speech rate) and write to "
            "<output_dir>/metrics/ (enhanced pipeline only)."
        ),
    )

    args = parser.parse_args()
    
    # Validate that either --url or --file is provided
    if not args.url and not args.file:
        parser.error("Either --url or --file must be provided")
    if args.url and args.file:
        parser.error("Provide only one of --url or --file, not both")
    
    os.makedirs(args.work_dir, exist_ok=True)

    clip_start = None
    clip_end = None
    if args.start is not None:
        clip_start = _parse_time_seconds(args.start)
        if clip_start < 0:
            parser.error("--start must be >= 0")
        if args.end is not None:
            clip_end = _parse_time_seconds(args.end)
            if clip_end <= clip_start:
                parser.error("--end must be greater than --start")
    elif args.end is not None:
        parser.error("--end requires --start")

    try:
        if args.enhanced:
            # Use enhanced pipeline with speaker segmentation, alignment, and speaker-specific TTS
            logger.info("Using enhanced pipeline with speaker segmentation and alignment")
            from src.pipeline_detailed import DetailedDubbingPipeline, DetailedPipelineConfig
            if args.multi_env:
                from src.speaker_tts import MultiEnvWorkerSpeakerTTS
            
            output_dir = args.output_dir or os.path.join(args.work_dir, "output_enhanced")
            os.makedirs(output_dir, exist_ok=True)
            
            cfg = DetailedPipelineConfig(
                work_dir=args.work_dir,
                output_dir=output_dir,
                tts_device=args.tts_device,
                debug=False,
                compute_metrics=bool(args.metrics),
            )
            if args.multi_env:
                # Run TTS per segment in the separate `tts` environment.
                pipeline = DetailedDubbingPipeline(
                    config=cfg,
                    tts=MultiEnvWorkerSpeakerTTS(device=args.tts_device, tts_backend="chatterbox"),
                )
            else:
                pipeline = DetailedDubbingPipeline(config=cfg)
        elif args.multi_env:
            # Use multi-environment pipeline (separate ASR and TTS envs)
            logger.info("Using multi-environment setup (separate ASR and TTS)")
            from src.pipeline_multienv import EnvAwarePipeline, PipelineConfig
            
            cfg = PipelineConfig(
                work_dir=args.work_dir,
                tts_device=args.tts_device
            )
            pipeline = EnvAwarePipeline(config=cfg)
        else:
            # Use original single-environment pipeline
            logger.info("Using single-environment setup (all in one env)")
            from src.pipeline import DubbingPipeline, PipelineConfig
            
            cfg = PipelineConfig(work_dir=args.work_dir)
            pipeline = DubbingPipeline(config=cfg)
        
        # Run pipeline
        logger.info("Starting dubbing pipeline: %s → %s", args.source, args.target)
        
        # Prepare pipeline input
        if args.enhanced:
            # Enhanced pipeline works with video files only (no URL download)
            if args.url:
                # Need to download URL first
                logger.info("Downloading video from URL...")
                from src.downloader import download_video
                video_file = os.path.join(args.work_dir, "downloaded_video.mp4")
                os.makedirs(args.work_dir, exist_ok=True)
                download_video(args.url, video_file)
                args.file = video_file
            
            if not args.file:
                logger.error("No video file or URL provided")
                sys.exit(1)

            # If a clip is requested, generate a temporary clipped video and
            # run the pipeline on that clip.
            if clip_start is not None:
                from src.audio import extract_video_clip

                clips_dir = os.path.join(args.work_dir, "clips")
                os.makedirs(clips_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(args.file))[0]
                end_label = f"{clip_end:.3f}" if clip_end is not None else "end"
                clipped_path = os.path.join(
                    clips_dir,
                    f"{base}_clip_{clip_start:.3f}_{end_label}.mp4",
                )
                logger.info(
                    "Clipping video: start=%s end=%s -> %s",
                    clip_start,
                    clip_end,
                    clipped_path,
                )
                extract_video_clip(
                    args.file,
                    clipped_path,
                    start_time=clip_start,
                    end_time=clip_end,
                )
                args.file = clipped_path
            
            result = pipeline.run(
                video_path=args.file,
                source_lang=args.source,
                target_lang=args.target,
            )
        else:
            # Standard pipelines support both URL and file
            if args.file and clip_start is not None:
                from src.audio import extract_video_clip

                clips_dir = os.path.join(args.work_dir, "clips")
                os.makedirs(clips_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(args.file))[0]
                end_label = f"{clip_end:.3f}" if clip_end is not None else "end"
                clipped_path = os.path.join(
                    clips_dir,
                    f"{base}_clip_{clip_start:.3f}_{end_label}.mp4",
                )
                logger.info(
                    "Clipping video: start=%s end=%s -> %s",
                    clip_start,
                    clip_end,
                    clipped_path,
                )
                extract_video_clip(
                    args.file,
                    clipped_path,
                    start_time=clip_start,
                    end_time=clip_end,
                )
                args.file = clipped_path

            result = pipeline.run(
                source_lang=args.source,
                target_lang=args.target,
                url=args.url,
                video_path=args.file,
            )
        
        # Print results
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)
        
        if args.enhanced:
            # Enhanced pipeline outputs detailed results
            logger.info("Results (Enhanced Pipeline):")
            logger.info(f"  Source Language: {result.get('source_language', 'N/A')}")
            logger.info(f"  Target Language: {result.get('target_language', 'N/A')}")
            logger.info(f"  Video: {result.get('video_path', 'N/A')}")

            stages = result.get('stages', {})
            seg_stage = stages.get('segmentation', {})
            out_stage = stages.get('output', {})

            logger.info(f"  Segments: {seg_stage.get('segments', 'N/A')}")
            logger.info(f"  Output Directory: {out_stage.get('output_directory', output_dir)}")
            logger.info(f"  Segment Audio Files: {output_dir}/segments/")
            logger.info(f"  Dubbed Audio: {out_stage.get('dubbed_audio', 'N/A')}")
            logger.info(f"  Dubbed Video: {out_stage.get('dubbed_video', 'N/A')}")
            logger.info(f"  Synthesis Report: {out_stage.get('synthesis_report', 'N/A')}")
            logger.info(f"  Segments Metadata: {out_stage.get('segments_metadata', 'N/A')}")
            logger.info(f"  Alignment Metadata: {out_stage.get('alignment_metadata', 'N/A')}")
        else:
            # Standard pipeline outputs
            logger.info("Results (Standard Pipeline):")
            logger.info(f"  Source Language: {result['source_lang']}")
            logger.info(f"  Target Language: {result['target_lang']}")
            logger.info(f"  Transcript: {result['steps'].get('transcript', 'N/A')[:100]}...")
            logger.info(f"  Translation: {result['steps'].get('translation', 'N/A')[:100]}...")
            logger.info(f"  Dubbed Audio: {result['steps'].get('tts_audio', 'N/A')}")
        logger.info("=" * 60)
        
        return result
        
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

