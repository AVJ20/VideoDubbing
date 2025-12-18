#!/usr/bin/env python
"""Quick test to verify multi-environment setup works.

This tests:
1. Both conda environments exist
2. Worker scripts can be called
3. JSON communication between processes works
4. EnvAwarePipeline can initialize
"""

import sys
import json
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_environments():
    """Check if both conda environments exist."""
    logger.info("Testing conda environments...")
    from workers.env_manager import EnvManager
    
    try:
        EnvManager.check_envs()
        logger.info("✓ Both 'asr' and 'tts' environments found")
        return True
    except Exception as e:
        logger.error(f"✗ Environment check failed: {e}")
        return False


def test_asr_worker():
    """Test ASR worker with a simple audio file."""
    logger.info("Testing ASR worker...")
    from workers.env_manager import EnvManager
    
    # Create a dummy audio file for testing (very short silence)
    test_audio = Path("work") / "test_audio.wav"
    test_audio.parent.mkdir(exist_ok=True)
    
    # Create minimal WAV file (1 second of silence)
    import wave
    with wave.open(str(test_audio), 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b'\x00\x00' * 16000)
    
    try:
        result = EnvManager.run_asr(str(test_audio), "en")
        logger.info(f"✓ ASR worker ran successfully")
        return True
    except Exception as e:
        logger.error(f"✗ ASR worker failed: {e}")
        return False


def test_tts_worker():
    """Test TTS worker with a simple phrase."""
    logger.info("Testing TTS worker...")
    from workers.env_manager import EnvManager
    
    try:
        result = EnvManager.run_tts(
            text="Hello, this is a test",
            language="en",
            output_audio="work/test_output.wav",
            device="cpu"
        )
        logger.info(f"✓ TTS worker ran successfully")
        return True
    except Exception as e:
        logger.error(f"✗ TTS worker failed: {e}")
        return False


def test_pipeline():
    """Test EnvAwarePipeline initialization."""
    logger.info("Testing EnvAwarePipeline...")
    
    try:
        from src.pipeline_multienv import EnvAwarePipeline, PipelineConfig
        
        cfg = PipelineConfig(work_dir="work", tts_device="cpu")
        pipeline = EnvAwarePipeline(config=cfg)
        logger.info("✓ EnvAwarePipeline initialized successfully")
        return True
    except Exception as e:
        logger.error(f"✗ EnvAwarePipeline initialization failed: {e}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("Multi-Environment Setup Test")
    logger.info("=" * 60)
    
    results = []
    
    # Test 1: Environments
    results.append(("Conda Environments", test_environments()))
    
    # Test 2: Pipeline
    results.append(("EnvAwarePipeline", test_pipeline()))
    
    # Test 3: ASR Worker (only if envs exist)
    if results[0][1]:
        results.append(("ASR Worker", test_asr_worker()))
    
    # Test 4: TTS Worker (only if envs exist)
    if results[0][1]:
        results.append(("TTS Worker", test_tts_worker()))
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Test Results:")
    logger.info("=" * 60)
    
    passed = 0
    failed = 0
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {name}")
        if success:
            passed += 1
        else:
            failed += 1
    
    logger.info("=" * 60)
    logger.info(f"Total: {passed} passed, {failed} failed")
    logger.info("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
