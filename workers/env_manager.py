"""Multi-Environment Worker Manager

Coordinates ASR (asr env) and TTS (tts env) by calling them via subprocess.
"""

import subprocess
import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class EnvManager:
    """Manage multiple conda environments for ASR and TTS."""
    
    # Update these paths to your conda installation
    CONDA_BASE = Path("C:\\Users\\vijoshi\\AppData\\Local\\anaconda3")
    ASR_ENV = "asr"
    TTS_ENV = "tts"
    
    @classmethod
    def get_python_exe(cls, env_name: str) -> str:
        """Get full path to Python executable in conda environment."""
        # On Windows, python.exe is in the env root, not in Scripts
        python_exe = cls.CONDA_BASE / "envs" / env_name / "python.exe"
        if python_exe.exists():
            return str(python_exe)
        # Fallback to Scripts if direct doesn't work
        python_exe_scripts = cls.CONDA_BASE / "envs" / env_name / "Scripts" / "python.exe"
        if python_exe_scripts.exists():
            return str(python_exe_scripts)
        # If neither exists, return the expected path (for error messages)
        return str(cls.CONDA_BASE / "envs" / env_name / "python.exe")
    
    @classmethod
    def run_asr(cls, audio_path: str, language: str = "en") -> Dict[str, Any]:
        """Transcribe audio using ASR environment.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            Dictionary with transcript and segments
        """
        # Convert to absolute path
        audio_path = str(Path(audio_path).absolute())
        logger.info(f"Running ASR: {audio_path} (language: {language})")
        
        # Verify audio file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Create temp JSON file for output
        output_json = f"{Path(audio_path).stem}_transcript.json"
        
        # Get absolute paths
        python_exe = cls.get_python_exe(cls.ASR_ENV)
        worker_script = Path(__file__).parent / "asr_worker.py"
        
        # Make output_json absolute path in same directory as audio
        output_json = str(Path(audio_path).parent / f"{Path(audio_path).stem}_transcript.json")
        
        cmd = [python_exe, str(worker_script), audio_path, language, output_json]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse JSON output
            output_data = json.loads(result.stdout)
            
            if output_data.get("status") == "success":
                logger.info("ASR completed successfully")
                return output_data
            else:
                raise RuntimeError(f"ASR failed: {output_data.get('error')}")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"ASR subprocess failed: {e.stderr}")
            raise RuntimeError(f"ASR worker error: {e.stderr}")
        finally:
            # Clean up temp JSON
            if os.path.exists(output_json):
                os.remove(output_json)
    
    @classmethod
    def run_tts(cls, text: str, language: str, output_audio: str, 
                voice: Optional[str] = None, device: str = "cpu", tts_backend: str = "chatterbox") -> Dict[str, Any]:
        """Synthesize speech using TTS environment.
        
        Args:
            text: Text to synthesize
            language: Language code (e.g., 'en', 'es', 'fr')
            output_audio: Output audio file path
            voice: Optional voice/reference audio path
            device: 'cpu' or 'cuda' (used for Coqui)
            tts_backend: TTS backend to use ('chatterbox' or 'coqui')
            
        Returns:
            Dictionary with synthesis status and audio path
        """
        # Convert to absolute paths
        output_audio = str(Path(output_audio).absolute())
        logger.info(f"Running TTS: {text} (language: {language}, backend: {tts_backend}, device: {device})")
        
        # Get absolute paths
        python_exe = cls.get_python_exe(cls.TTS_ENV)
        worker_script = Path(__file__).parent / "tts_worker.py"
        
        cmd = [python_exe, str(worker_script), text, language, output_audio, 
               "--device", device, "--tts", tts_backend]
        
        if voice:
            voice = str(Path(voice).absolute())
            cmd.extend(["--voice", voice])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse JSON output
            output_data = json.loads(result.stdout)
            
            if output_data.get("status") == "success":
                logger.info("TTS completed successfully")
                return output_data
            else:
                raise RuntimeError(f"TTS failed: {output_data.get('error')}")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"TTS subprocess failed: {e.stderr}")
            raise RuntimeError(f"TTS worker error: {e.stderr}")
    
    @classmethod
    def check_envs(cls) -> Dict[str, bool]:
        """Verify both environments are installed."""
        status = {}
        
        for env_name in [cls.ASR_ENV, cls.TTS_ENV]:
            python_exe = cls.get_python_exe(env_name)
            status[env_name] = os.path.exists(python_exe)
        
        return status


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Check environments
    env_status = EnvManager.check_envs()
    print(f"Environment status: {env_status}")
    
    if not all(env_status.values()):
        print("Error: Not all environments are set up. Please create 'asr' and 'tts' environments.")
        exit(1)
    
    # Test ASR
    print("\n--- Testing ASR ---")
    try:
        asr_result = EnvManager.run_asr("test_audio.wav", "en")
        print(f"Transcript: {asr_result['text']}")
    except Exception as e:
        print(f"ASR test failed: {e}")
    
    # Test TTS
    print("\n--- Testing TTS ---")
    try:
        tts_result = EnvManager.run_tts("Hello world", "en", "test_output.wav")
        print(f"Audio saved: {tts_result['audio']}")
    except Exception as e:
        print(f"TTS test failed: {e}")
