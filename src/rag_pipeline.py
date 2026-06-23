"""
RAG Orchestration Layer (Production-grade FIXED)
"""

from typing import Optional, Dict, Any

# -----------------------------
# CONFIG IMPORT (STRICT - NO SILENT FALLBACK)
# -----------------------------
try:
    from src.config import (
        TOP_K,
        GENERATION_BACKEND,
        HF_GENERATION_MODEL,
        LOCAL_GENERATION_MODEL,
    )
except Exception as e:
    raise RuntimeError(f"[CONFIG ERROR] Failed to import config: {e}")


# -----------------------------
# SAFE IMPORTS
# -----------------------------
from src.embedding import load_model as load_embedding_model
from src.generator import generate_answer
from src.prompt import build_prompt
from src.retriever import retrieve
from src.vector_store_loader import get_or_build_full_vector_store


VALID_BACKENDS = {"hf_inference_api", "local_pipeline"}


class RAGPipeline:
    def __init__(
        self,
        collection=None,
        embed_model=None,
        k: int = TOP_K,
        generation_backend: str = GENERATION_BACKEND,
        hf_model: str = HF_GENERATION_MODEL,
        local_model: str = LOCAL_GENERATION_MODEL,
        debug: bool = True,
    ):

        if generation_backend not in VALID_BACKENDS:
            raise ValueError(f"Invalid backend: {generation_backend}")

        self.debug = debug

        try:
            self.collection = collection or get_or_build_full_vector_store()
            self.embed_model = embed_model or load_embedding_model()
        except Exception as e:
            raise RuntimeError(f"[RAG INIT FAILED] {e}")

        self.k = k
        self.generation_backend = generation_backend
        self.hf_model = hf_model
        self.local_model = local_model

        if self.debug:
            print("\n[DEBUG RAG PIPELINE]")
            print("Backend:", self.generation_backend)
            print("HF Model:", self.hf_model)
            print("Local Model:", self.local_model)
            print("Top-K:", self.k)
            print("----------------------\n")

    def answer(
        self,
        question: str,
        k: Optional[int] = None,
        where: Optional[dict] = None,
    ) -> Dict[str, Any]:

        if not isinstance(question, str) or not question.strip():
            return {
                "question": question,
                "answer": "Error: Invalid question input.",
                "retrieved_chunks": [],
                "prompt": None,
            }

        try:
            # STEP 1: RETRIEVE
            retrieved_chunks = retrieve(
                question=question,
                collection=self.collection,
                model=self.embed_model,
                k=k or self.k,
                where=where,
            )

            # 🔥 FIX: ONLY REFUSE IF NOTHING IS RETRIEVED AT ALL
            if not retrieved_chunks:
                return {
                    "question": question,
                    "answer": (
                        "There is not enough information in the provided complaints "
                        "to answer this question."
                    ),
                    "retrieved_chunks": [],
                    "prompt": None,
                }

            # 🔥 OPTIONAL SAFETY: soft filter only (DO NOT BLOCK ALL CONTEXT)
            retrieved_chunks = [
                c for c in retrieved_chunks
                if c.get("similarity", 0) >= 0.15
            ]

            # fallback: if filtering removed everything, keep top 2
            if not retrieved_chunks:
                retrieved_chunks = retrieved_chunks[:2]

            # STEP 2: PROMPT
            prompt = build_prompt(question, retrieved_chunks)

            # STEP 3: MODEL SELECTION
            model_name = (
                self.hf_model
                if self.generation_backend == "hf_inference_api"
                else self.local_model
            )

            # STEP 4: GENERATION
            answer_text = generate_answer(
                prompt=prompt,
                backend=self.generation_backend,
                model_name=model_name,
            )

            return {
                "question": question,
                "answer": answer_text,
                "retrieved_chunks": retrieved_chunks,
                "prompt": prompt,
            }

        except Exception as e:
            return {
                "question": question,
                "answer": f"RAG pipeline error: {str(e)}",
                "retrieved_chunks": [],
                "prompt": None,
            }