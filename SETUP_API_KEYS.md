# Setup Guide: Managing API Keys Securely

## Overview

This guide explains how to securely set up API keys without accidentally committing them to git.

## ⚠️ Important: Never Commit API Keys

API keys are secrets that give access to your accounts. **Never push them to git!**

```bash
# ❌ BAD - Don't do this
git add .env
git commit -m "Add API keys"

# ✅ GOOD - Use .env.example and .gitignore
```

## Setup Steps

### 1. Copy the Example File

```bash
# Copy the template
cp .env.example .env
```

### 2. Edit .env with Your API Keys

Open `.env` and fill in your actual keys:

```bash
# Using PowerShell on Windows
notepad .env

# Using VS Code
code .env

# Using nano on Linux/Mac
nano .env
```

**Example .env file:**
```
GROQ_API_KEY=gsk_abc123xyz...
AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=key-123-abc...
```

### 3. Verify .env is Protected

The `.gitignore` file already excludes:
- `.env` - Your actual keys (never commit this)
- `.env.local` - Environment-specific overrides
- `work/` - Generated output files

Check that .env is ignored:
```bash
# On Windows PowerShell
Get-Content .gitignore | Select-String ".env"

# On Mac/Linux
grep ".env" .gitignore
```

### 4. Share .env.example with Team

Only commit `.env.example`:
```bash
git add .env.example
git commit -m "Add environment variable template"
git push
```

Team members will:
1. Clone the repo
2. Copy `.env.example` to `.env`
3. Fill in their own API keys
4. Never commit `.env`

## How the Code Loads Keys

The pipeline automatically loads your `.env` file:

```python
# cli.py automatically calls this
from src.env_loader import load_env
load_env()  # Reads .env and sets os.environ variables
```

Then the translators read from environment:
```python
# In translator.py
self.api_key = api_key or os.environ.get("GROQ_API_KEY")
```

## Usage

Once you've set up `.env`, just run normally:

```powershell
# API keys are loaded automatically from .env
python cli.py --file "video.mp4" --source en --target es
```

## Alternative: Direct Environment Variables

If you prefer not to use `.env`, set environment variables directly:

**Windows PowerShell:**
```powershell
$env:GROQ_API_KEY = "your-key-here"
python cli.py --file "video.mp4" --source en --target es
```

**Linux/Mac Bash:**
```bash
export GROQ_API_KEY="your-key-here"
python cli.py --file "video.mp4" --source en --target es
```

## Safety Checklist

- [ ] Created `.env` from `.env.example`
- [ ] Filled in your actual API keys in `.env`
- [ ] `.env` is listed in `.gitignore`
- [ ] Never committed `.env` to git
- [ ] Only `.env.example` is in git repository
- [ ] Tested that pipeline loads keys correctly

## What If I Accidentally Committed an API Key?

**Immediately:**

1. **Revoke the key** in the service (Groq, Azure, OpenAI console)
2. **Generate a new key**
3. **Update .env** with the new key
4. **Remove from git history:**
   ```bash
   # This removes it from all commits
   git rm --cached .env
   git commit --amend -m "Remove .env from git"
   git push --force-with-lease
   ```

## Recommended: Use a Password Manager

Store API keys in a password manager (1Password, Bitwarden, etc.) and copy them to `.env` when needed.

## Summary

| What | Where | Commit? |
|------|-------|---------|
| API Keys | `.env` | ❌ NO |
| Template | `.env.example` | ✅ YES |
| Ignore Rules | `.gitignore` | ✅ YES |
| Code | `src/`, `cli.py` | ✅ YES |

**Golden Rule:** If it contains secrets, don't commit it.
