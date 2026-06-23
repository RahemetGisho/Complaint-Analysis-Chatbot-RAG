"""
LLM Generation Layer (FIXED + PRODUCTION SAFE)

Supports:
- HuggingFace Inference API (Qwen, Mistral, Llama, etc.)
- Local Transformers Pipeline

Fixes:
- correct HF pipeline usage
- correct task selection
- better compatibility across models
- safer error handling
"""

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
try:
    from config import (
        GENERATION_BACKEND,
        HF_GENERATION_MODEL,
        LOCAL_GENERATION_MODEL,
        MAX_NEW_TOKENS,
        TEMPERATURE,
    )
except ImportError:
    GENERATION_BACKEND = "hf_inference_api"
    HF_GENERATION_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    LOCAL_GENERATION_MODEL = "google/flan-t5-base"
    MAX_NEW_TOKENS = 256
    TEMPERATURE = 0.2


# -----------------------------
# CACHE
# -----------------------------
_LOCAL_PIPELINE_CACHE = {}


# -----------------------------
# PUBLIC API
# -----------------------------
def generate_answer(
    prompt: str,
    backend: str = GENERATION_BACKEND,
    model_name: Optional[str] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:

    if not prompt or not isinstance(prompt, str):
        raise ValueError("Prompt cannot be empty.")

    if backend == "hf_inference_api":
        return _generate_hf(prompt, model_name, max_new_tokens)

    if backend == "local_pipeline":
        return _generate_local(prompt, model_name, max_new_tokens)

    raise ValueError(f"Unsupported backend: {backend}")


# -----------------------------
# HF INFERENCE API
# -----------------------------
def _generate_hf(prompt: str, model_name: Optional[str], max_new_tokens: int) -> str:
    try:
        from huggingface_hub import InferenceClient

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN missing in .env")

        model = model_name or HF_GENERATION_MODEL

        client = InferenceClient(model=model, token=hf_token)

        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=TEMPERATURE,
        )

        if not response or not response.choices:
            raise RuntimeError("Empty HF response")

        return response.choices[0].message.content.strip()

    except Exception as e:
        raise RuntimeError(f"HuggingFace generation failed: {e}") from e


# -----------------------------
# LOCAL PIPELINE (FIXED)
# -----------------------------
def _generate_local(prompt: str, model_name: Optional[str], max_new_tokens: int) -> str:
    try:
        from transformers import pipeline

        model = model_name or LOCAL_GENERATION_MODEL

        generator = _LOCAL_PIPELINE_CACHE.get(model)

        if generator is None:
            print(f"[INFO] Loading local model: {model}")

            # ✅ FIX 1: correct task (this fixes your error)
            generator = pipeline(
                task="text-generation",
                model=model,
            )

            _LOCAL_PIPELINE_CACHE[model] = generator

        output = generator(
            prompt,
            max_new_tokens=max_new_tokens,
        )

        if not output:
            raise RuntimeError("Empty model output")

        # text-generation format
        if "generated_text" in output[0]:
            return output[0]["generated_text"].strip()

        # fallback safety
        return str(output[0]).strip()

    except Exception as e:
        raise RuntimeError(f"Local generation failed: {e}") from e