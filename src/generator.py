import os
import threading
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# CONFIG
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


# CACHE 
_LOCAL_PIPELINE_CACHE = {}
_LOCAL_MODEL_CACHE = {}  


# PUBLIC API - non-streaming (unchanged behavior)
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


# PUBLIC API - streaming (NEW)
def generate_answer_stream(
    prompt: str,
    backend: str = GENERATION_BACKEND,
    model_name: Optional[str] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
):
   
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Prompt cannot be empty.")

    if backend == "hf_inference_api":
        yield from _generate_hf_stream(prompt, model_name, max_new_tokens)
    elif backend == "local_pipeline":
        yield from _generate_local_stream(prompt, model_name, max_new_tokens)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


# HF INFERENCE API
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


def _generate_hf_stream(prompt: str, model_name: Optional[str], max_new_tokens: int):
    from huggingface_hub import InferenceClient

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN missing in .env")

    model = model_name or HF_GENERATION_MODEL
    client = InferenceClient(model=model, token=hf_token)

    try:
        stream = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            stream=True,
        )
        got_any = False
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                got_any = True
                yield delta
        if not got_any:
            raise RuntimeError("HF stream completed with no content")
    except Exception as e:
        print(f"[WARNING] HF streaming failed ({e}); falling back to a single non-streamed call.")
        yield _generate_hf(prompt, model_name, max_new_tokens)


# LOCAL PIPELINE 
def _resolve_local_model_class(model_id: str):
    
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id)
    is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
    task = "text2text-generation" if is_encoder_decoder else "text-generation"
    return is_encoder_decoder, task


def _generate_local(prompt: str, model_name: Optional[str], max_new_tokens: int) -> str:
    try:
        from transformers import pipeline

        model = model_name or LOCAL_GENERATION_MODEL
        generator = _LOCAL_PIPELINE_CACHE.get(model)

        if generator is None:
            _, task = _resolve_local_model_class(model)
            print(f"[INFO] Loading local model: {model} (task={task})")
            generator = pipeline(task=task, model=model)
            _LOCAL_PIPELINE_CACHE[model] = generator

        output = generator(prompt, max_new_tokens=max_new_tokens)

        if not output:
            raise RuntimeError("Empty model output")

        if "generated_text" in output[0]:
            return output[0]["generated_text"].strip()

        return str(output[0]).strip()

    except Exception as e:
        raise RuntimeError(f"Local generation failed: {e}") from e


def _load_local_streaming_model(model_id: str):
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    cached = _LOCAL_MODEL_CACHE.get(model_id)
    if cached is not None:
        return cached

    is_encoder_decoder, _ = _resolve_local_model_class(model_id)
    print(f"[INFO] Loading local model for streaming: {model_id} "
          f"({'seq2seq' if is_encoder_decoder else 'causal'})")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_cls = AutoModelForSeq2SeqLM if is_encoder_decoder else AutoModelForCausalLM
    model = model_cls.from_pretrained(model_id)

    _LOCAL_MODEL_CACHE[model_id] = (tokenizer, model)
    return tokenizer, model


def _generate_local_stream(prompt: str, model_name: Optional[str], max_new_tokens: int):
    from transformers import TextIteratorStreamer

    model_id = model_name or LOCAL_GENERATION_MODEL

    try:
        tokenizer, model = _load_local_streaming_model(model_id)
        inputs = tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(**inputs, max_new_tokens=max_new_tokens, streamer=streamer)
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        got_any = False
        for piece in streamer:
            if piece:
                got_any = True
                yield piece
        thread.join()

        if not got_any:
            raise RuntimeError("Local stream completed with no content")

    except Exception as e:
        print(f"[WARNING] Local streaming failed ({e}); falling back to a single non-streamed call.")
        yield _generate_local(prompt, model_name, max_new_tokens)