"""
RAG Orchestration Layer (Production-grade FIXED + STREAMING)
"""

from typing import Optional, Dict, Any, Iterator

# -----------------------------
# CONFIG IMPORT (STRICT)
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
from src.generator import generate_answer, generate_answer_stream
from src.prompt import build_prompt
from src.retriever import retrieve
from src.vector_store_loader import get_or_build_full_vector_store


# -----------------------------
# VALID BACKENDS
# -----------------------------
VALID_BACKENDS = {"hf_inference_api", "local_pipeline"}

# The exact refusal sentence prompt.py instructs the model to use when
# context doesn't support an answer. If this appears anywhere in a
# generated answer, nothing after it is trustworthy - the model has stated
# it doesn't have enough information, so any further text is a
# prompt-following contradiction (observed in testing: the model would
# sometimes emit this sentence and then keep going and fill in the answer
# template anyway).
REFUSAL_SENTENCE = "There is not enough information in the provided complaints to answer this question."
NO_CONTEXT_MESSAGE = "There is not enough relevant information in the complaints dataset to answer this question."


# -----------------------------
# PIPELINE
# -----------------------------
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
            raise ValueError(
                f"Invalid backend: {generation_backend}. Must be {VALID_BACKENDS}"
            )

        self.debug = debug

        try:
            # IMPORTANT: ONLY LOAD EXISTING VECTOR DB
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

    def _model_name_for_backend(self) -> str:
        return self.hf_model if self.generation_backend == "hf_inference_api" else self.local_model

    # -----------------------------
    # MAIN FUNCTION (non-streaming, unchanged behavior + refusal guard)
    # -----------------------------
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
            retrieved_chunks = retrieve(
                question=question,
                collection=self.collection,
                model=self.embed_model,
                k=k or self.k,
                where=where,
            )

            if not retrieved_chunks:
                return {
                    "question": question,
                    "answer": NO_CONTEXT_MESSAGE,
                    "retrieved_chunks": [],
                    "prompt": None,
                }

            prompt = build_prompt(question, retrieved_chunks)

            answer_text = generate_answer(
                prompt=prompt,
                backend=self.generation_backend,
                model_name=self._model_name_for_backend(),
            )

            # Guard against the model emitting the refusal sentence and then
            # continuing to answer anyway - nothing after it is reliable.
            if REFUSAL_SENTENCE.lower() in answer_text.lower():
                answer_text = REFUSAL_SENTENCE

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

    # -----------------------------
    # STREAMING (NEW) - used by the Gradio UI
    # -----------------------------
    def answer_stream(
        self,
        question: str,
        k: Optional[int] = None,
        where: Optional[dict] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Generator version of answer(). Yields events as they happen so a UI
        can show retrieved sources immediately (retrieval is fast) and
        stream the answer token-by-token as it's generated, rather than
        blocking until the entire response is ready.

        Yields dicts shaped as one of:
            {"type": "sources", "retrieved_chunks": [...]}
            {"type": "token", "text": "..."}                  (zero or more)
            {"type": "done", "answer": "...", "question": question}
            {"type": "error", "answer": "..."}                 (instead of "done")
        """
        if not isinstance(question, str) or not question.strip():
            yield {"type": "error", "answer": "Error: Invalid question input."}
            return

        try:
            retrieved_chunks = retrieve(
                question=question,
                collection=self.collection,
                model=self.embed_model,
                k=k or self.k,
                where=where,
            )
            yield {"type": "sources", "retrieved_chunks": retrieved_chunks}

            if not retrieved_chunks:
                yield {"type": "token", "text": NO_CONTEXT_MESSAGE}
                yield {"type": "done", "answer": NO_CONTEXT_MESSAGE, "question": question}
                return

            prompt = build_prompt(question, retrieved_chunks)

            full_answer = ""
            refused = False
            for piece in generate_answer_stream(
                prompt=prompt,
                backend=self.generation_backend,
                model_name=self._model_name_for_backend(),
            ):
                if refused:
                    # Already hit the refusal sentence - stop accepting
                    # further tokens; anything after it is unreliable.
                    break

                full_answer += piece
                yield {"type": "token", "text": piece}

                lower = full_answer.lower()
                if REFUSAL_SENTENCE.lower() in lower:
                    refused = True
                    idx = lower.find(REFUSAL_SENTENCE.lower())
                    full_answer = full_answer[: idx + len(REFUSAL_SENTENCE)]

            yield {"type": "done", "answer": full_answer, "question": question}

        except Exception as e:
            yield {"type": "error", "answer": f"RAG pipeline error: {str(e)}"}