"""
Embedding Module
Generates vector embeddings for text chunks using a transformer model.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


def load_model(model_name: str = "all-MiniLM-L6-v2"):
    # Load sentence transformer embedding model.
    return SentenceTransformer(model_name)

def embed_texts(model, texts: list[str], batch_size: int = 64) -> np.ndarray:
    # Convert text chunks into vector embeddings.

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings