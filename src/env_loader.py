"""Load environment variables from .env file"""
import os
from pathlib import Path


def load_env():
    """Load .env file into os.environ if it exists"""
    env_file = Path(__file__).parent.parent / ".env"
    
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                # Parse KEY=VALUE
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
    else:
        print(f"Warning: .env file not found at {env_file}")
        print("Copy .env.example to .env and fill in your API keys")
