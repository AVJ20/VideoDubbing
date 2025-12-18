#!/usr/bin/env python
"""
Example: Using different translation backends with VideoDubbing

Run any of these examples to test translation without full video processing.
"""

import logging

logging.basicConfig(level=logging.INFO)


def example_groq():
    """Example: Using Groq (FREE, recommended)"""
    print("\n=== GROQ TRANSLATOR (FREE) ===")
    from src.translator import GroqTranslator

    translator = GroqTranslator(model="mixtral-8x7b-32768")
    text = "Hello, how are you today?"
    result = translator.translate(text, "en", "es")
    print(f"Input:  {text}")
    print(f"Output: {result}")


def example_openai():
    """Example: Using OpenAI (PAID)"""
    print("\n=== OPENAI TRANSLATOR (PAID) ===")
    from src.translator import OpenAITranslator

    translator = OpenAITranslator(model="gpt-4o-mini")
    text = "Hello, how are you today?"
    result = translator.translate(text, "en", "es")
    print(f"Input:  {text}")
    print(f"Output: {result}")


def example_azure_openai():
    """Example: Using Azure OpenAI (PAID, but with $200 free credit)"""
    print("\n=== AZURE OPENAI TRANSLATOR (PAID - $200 FREE TRIAL) ===")
    from src.translator import AzureOpenAITranslator

    # Requires environment variables:
    # - AZURE_OPENAI_ENDPOINT
    # - AZURE_OPENAI_API_KEY
    # - AZURE_OPENAI_DEPLOYMENT
    translator = AzureOpenAITranslator()
    text = "Hello, how are you today?"
    result = translator.translate(text, "en", "es")
    print(f"Input:  {text}")
    print(f"Output: {result}")


def example_ollama():
    """Example: Using Ollama (FREE, LOCAL)"""
    print("\n=== OLLAMA TRANSLATOR (FREE, LOCAL) ===")
    from src.translator import OllamaTranslator

    translator = OllamaTranslator(model="mistral")
    text = "Hello, how are you today?"
    result = translator.translate(text, "en", "es")
    print(f"Input:  {text}")
    print(f"Output: {result}")


def example_identity():
    """Example: Using Identity (STUB, returns original text)"""
    print("\n=== IDENTITY TRANSLATOR (STUB) ===")
    from src.translator import IdentityTranslator

    translator = IdentityTranslator()
    text = "Hello, how are you today?"
    result = translator.translate(text, "en", "es")
    print(f"Input:  {text}")
    print(f"Output: {result}")


def example_with_pipeline():
    """Example: Using a translator in the full pipeline"""
    print("\n=== FULL PIPELINE WITH GROQ ===")
    from src.pipeline import DubbingPipeline, PipelineConfig
    from src.translator import GroqTranslator

    # Create translator
    translator = GroqTranslator()

    # Create pipeline with custom translator
    cfg = PipelineConfig(work_dir="work")
    pipeline = DubbingPipeline(translator=translator, config=cfg)

    # Run with local video
    result = pipeline.run(
        source_lang="en",
        target_lang="es",
        video_path="path/to/video.mp4",  # Change to your video path
    )

    print(f"Pipeline result: {result}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == "groq":
            example_groq()
        elif example == "openai":
            example_openai()
        elif example == "azure":
            example_azure_openai()
        elif example == "ollama":
            example_ollama()
        elif example == "identity":
            example_identity()
        elif example == "pipeline":
            example_with_pipeline()
        else:
            print(f"Unknown example: {example}")
    else:
        print("Translation Examples")
        print("====================")
        print("Run with one of:")
        print("  python examples/translate.py groq       # Free (recommended)")
        print("  python examples/translate.py azure      # Free trial $200")
        print("  python examples/translate.py openai     # Paid")
        print("  python examples/translate.py ollama     # Free local")
        print("  python examples/translate.py identity   # No translation")
        print("  python examples/translate.py pipeline   # Full pipeline")
        print("\nSee TRANSLATION_GUIDE.md for setup instructions.")
