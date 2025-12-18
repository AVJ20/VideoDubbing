# Translation Guide: Free & Paid Options

The VideoDubbing pipeline supports multiple translation backends. Choose based on your needs and budget.

## **1. OpenAI (Paid - Most Accurate)**

**Cost:** ~$0.15/hour of video (with gpt-4o-mini)

**Setup:**
```bash
pip install openai
export OPENAI_API_KEY="your-key-here"
```

**Get API Key:**
- Go to https://platform.openai.com/account/api-keys
- Create new secret key
- **Note:** OpenAI requires a paid account (no free tier for new users as of 2025)

**Usage:**
```python
from src.translator import OpenAITranslator

translator = OpenAITranslator(model="gpt-4o-mini")
result = translator.translate("Hello", "en", "es")
```

---

## **2. Groq (FREE - Fast)**

**Cost:** FREE tier available (fast inference)

**Setup:**
```bash
pip install groq
export GROQ_API_KEY="your-key-here"
```

**Get Free API Key:**
1. Visit https://console.groq.com
2. Sign up (free)
3. Go to API Keys section
4. Copy your API key

**Models available:** 
- `llama-3.1-8b-instant` (fast, good quality - recommended)
- `llama-3.3-70b-versatile` (larger, better quality but slower)
- `mixtral-8x7b-32768` (older, may not be available)

**Usage:**
```python
from src.translator import GroqTranslator

translator = GroqTranslator(model="llama-3.1-8b-instant")
result = translator.translate("Hello", "en", "es")
```

**Pros:** Free, fast, excellent quality
**Cons:** Requires internet, rate limits on free tier

---

## **3. Azure OpenAI (Paid - $200 Free Trial)**

**Cost:** ~$0.01-0.05 per video (same as OpenAI), but with $200 free credit

**Setup:**
```bash
pip install openai
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-key-here"
export AZURE_OPENAI_DEPLOYMENT="gpt-4-turbo"  # or your deployment name
```

**Get Free Credits:**
1. Go to https://azure.microsoft.com/en-us/free/
2. Sign up with credit card (verification only, won't charge)
3. Get $200 credit valid for 30 days
4. Create Azure OpenAI resource in Azure portal
5. Deploy a model (e.g., gpt-4-turbo)
6. Copy endpoint and API key from resource

**Deployment Names:** Your custom names in Azure (e.g., `gpt-4-turbo`, `gpt-35-turbo`)

**Usage:**
```python
from src.translator import AzureOpenAITranslator

translator = AzureOpenAITranslator()  # Reads from environment variables
result = translator.translate("Hello", "en", "es")
```

**Pros:** Enterprise support, $200 free trial, integrates with Azure ecosystem
**Cons:** Requires Azure account setup, free credits expire after 30 days

---

## **4. Ollama (Free - Local/Offline)**

**Cost:** FREE

**Setup:**
1. Download Ollama from https://ollama.ai
2. Install and run: `ollama serve`
3. Pull a model: `ollama pull mistral`
4. Install Python package:
```bash
pip install ollama
```

**Models to use:**
- `mistral` (7B, balanced)
- `neural-chat` (7B, conversational)
- `llama2` (7B or 13B)

**Usage:**
```python
from src.translator import OllamaTranslator

translator = OllamaTranslator(model="mistral")
result = translator.translate("Hello", "en", "es")
```

**Pros:** Completely free, runs locally (no internet), offline privacy
**Cons:** Requires local GPU/CPU, slower than cloud APIs

---

## **4. Identity (Stub - No Translation)**

**Cost:** FREE

For testing without translation:
```python
from src.translator import IdentityTranslator

translator = IdentityTranslator()  # Returns original text unchanged
```

---

## **Quick Comparison**

| Option | Cost | Speed | Quality | Internet | Setup |
|--------|------|-------|---------|----------|-------|
| OpenAI | Paid | Fast | Excellent | Yes | API key |
| Azure OpenAI | Paid ($200 free trial) | Fast | Excellent | Yes | Azure account |
| Groq | FREE | Very Fast | Excellent | Yes | API key (free) |
| Ollama | FREE | Slow | Good | No | Local setup |
| Identity | FREE | Instant | None | No | Builtin |

---

## **Recommendation for Budget Users**

**Start with Groq** - it's free, fast, and easy to set up.

```bash
# 1. Get free API key from https://console.groq.com
# 2. Set environment variable
export GROQ_API_KEY="your-key"

# 3. Run pipeline
python cli.py --file video.mp4 --source en --target es
```

The pipeline will use Groq by default if you configure it in your code.

**If you want to try Azure first (limited time):**
- Sign up at https://azure.microsoft.com/en-us/free/
- Get $200 credit (valid 30 days)
- Then switch to Groq for long-term use (completely free)

---

## **Using Different Translators in Pipeline**

Edit your code to swap translators:

```python
from src.pipeline import DubbingPipeline
from src.translator import GroqTranslator  # or OpenAITranslator, OllamaTranslator

translator = GroqTranslator()
pipeline = DubbingPipeline(translator=translator)
result = pipeline.run(
    source_lang="en",
    target_lang="es",
    video_path="video.mp4"
)
```

Or modify `cli.py` to accept a `--translator` flag for easy switching.
