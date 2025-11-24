"""Simple CLI to run the video dubbing pipeline.

Usage: python cli.py --url <VIDEO_URL> --source en --target es
"""
import argparse
import logging
import os

from src.pipeline import DubbingPipeline, PipelineConfig

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Video URL to download and dub")
    parser.add_argument("--source", default="auto", help="Source language (ISO code or 'auto')")
    parser.add_argument("--target", required=True, help="Target language (ISO code)")
    parser.add_argument("--work-dir", default="work", help="Working directory")

    args = parser.parse_args()
    cfg = PipelineConfig(work_dir=args.work_dir)
    os.makedirs(cfg.work_dir, exist_ok=True)

    pipeline = DubbingPipeline(config=cfg)
    result = pipeline.run(args.url, args.source, args.target)
    print("Pipeline result:")
    print(result)


if __name__ == "__main__":
    main()
