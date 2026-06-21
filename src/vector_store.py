"""
Vector Store Module
Stores embeddings + metadata into ChromaDB for semantic search.
"""

from __future__ import annotations

import chromadb
import numpy as np
import pandas as pd
from pathlib import Path


def build_vector_store(
    chunk_df: pd.DataFrame,
    embeddings: np.ndarray,
    persist_dir: str = "vector_store",
    collection_name: str = "complaint_chunks",
):
    # Save embeddings into ChromaDB.

    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_dir)

    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)

    collection = client.create_collection(name=collection_name)

    ids = chunk_df["chunk_id"].tolist()
    docs = chunk_df["chunk_text"].tolist()
    meta = chunk_df[["complaint_id", "product_category"]].to_dict("records")

    BATCH = 500

    for i in range(0, len(ids), BATCH):
        collection.add(
            ids=ids[i:i+BATCH],
            documents=docs[i:i+BATCH],
            embeddings=embeddings[i:i+BATCH].tolist(),
            metadatas=meta[i:i+BATCH],
        )

    return collection