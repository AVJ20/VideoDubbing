"""Load environment variables from .env file"""
import os
from pathlib import Path


def _strip_quotes(value: str) -> str:
    v = (value or "").strip()
    if len(v) >= 2 and ((v[0] == v[-1]) and v[0] in {'"', "'"}):
        return v[1:-1].strip()
    return v


def _strip_inline_comment(value: str) -> str:
    """Remove trailing inline comments for simple KEY=VALUE lines.

    Only strips when the comment marker is preceded by whitespace to avoid
    breaking values that legitimately contain '#'.
    """

    v = value or ""
    for marker in (" #", "\t#"):
        idx = v.find(marker)
        if idx != -1:
            return v[:idx].rstrip()
    return v.strip()


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
                    k = key.strip()
                    v = _strip_inline_comment(value)
                    v = _strip_quotes(v)
                    if k:
                        os.environ[k] = v
    else:
        print(f"Warning: .env file not found at {env_file}")
        print("Copy .env.example to .env and fill in your API keys")
