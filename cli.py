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

from src.env_loader import load_env

# Load environment variables from .env file
load_env()

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Video Dubbing Pipeline")
    parser.add_argument("--url", default=None, help="Video URL to download and dub")
    parser.add_argument("--file", default=None, help="Local video file path (alternative to --url)")
    parser.add_argument("--source", default="auto", help="Source language (ISO code or 'auto')")
    parser.add_argument("--target", required=True, help="Target language (ISO code)")
    parser.add_argument("--work-dir", default="work", help="Working directory")
    parser.add_argument("--multi-env", action="store_true", 
                       help="Use separate conda environments for ASR and TTS (recommended)")
    parser.add_argument("--tts-device", default="cpu", choices=["cpu", "cuda"],
                       help="Device for TTS synthesis: 'cpu' or 'cuda' (GPU)")

    args = parser.parse_args()
    
    # Validate that either --url or --file is provided
    if not args.url and not args.file:
        parser.error("Either --url or --file must be provided")
    if args.url and args.file:
        parser.error("Provide only one of --url or --file, not both")
    
    os.makedirs(args.work_dir, exist_ok=True)

    try:
        if args.multi_env:
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
        logger.info("Results:")
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

