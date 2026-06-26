import math
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd

try:
    from config import (
        EMBEDDING_MODEL_NAME,
        COLLECTION_NAME,
        TOP_K,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        PREBUILT_PARQUET_PATH,
        VECTOR_STORE_PATH,
    )
except Exception:
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    COLLECTION_NAME = "crediTrust_complaints"
    TOP_K = 5
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    PREBUILT_PARQUET_PATH = "data/raw/complaint_embeddings.parquet"
    VECTOR_STORE_PATH = "vector_store"



def load_embeddings_parquet(path: str = PREBUILT_PARQUET_PATH):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_parquet(path)

    print(f"Loaded {len(df):,} rows")
    print(f"Columns: {list(df.columns)}")

    return df


def clean_metadata(metadata: dict):
    if not isinstance(metadata, dict):
        return {}

    cleaned = {}
    for k, v in metadata.items():
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        cleaned[k] = v

    return cleaned


def build_full_vector_store(
    parquet_path: str = PREBUILT_PARQUET_PATH,
    persist_dir: str = str(VECTOR_STORE_PATH),
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 512,
    force_rebuild: bool = False,
):
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_dir)

    existing = {c.name: c for c in client.list_collections()}

    if collection_name in existing and not force_rebuild:
        print(f"[VECTOR STORE] Reusing existing collection: {collection_name}")
        return existing[collection_name]

    if collection_name in existing and force_rebuild:
        print("[VECTOR STORE] Deleting old collection (force rebuild)")
        client.delete_collection(collection_name)

    df = load_embeddings_parquet(parquet_path)

    # COLUMN MAPPING
    id_col = "id"
    text_col = "document"
    embedding_col = "embedding"
    metadata_col = "metadata"

    # CREATE COLLECTION
    collection = client.create_collection(
        name=collection_name,
        metadata={
            "embedding_model": EMBEDDING_MODEL_NAME,
            "hnsw:space": "cosine",  
        },
    )

    n = len(df)

    # INGESTION LOOP
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = df.iloc[start:end]

        ids = batch[id_col].astype(str).tolist()

        embeddings = np.array(
            batch[embedding_col].tolist(),
            dtype=np.float32
        ).tolist()

        documents = batch[text_col].astype(str).tolist()

        metadatas = [
            clean_metadata(m) if isinstance(m, dict) else {}
            for m in batch[metadata_col]
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        print(f"Ingested {end:,}/{n:,}")

    print(f"Done. Total chunks: {collection.count():,}")

    return collection


# PUBLIC API
def get_or_build_full_vector_store(**kwargs):
    
    return build_full_vector_store(**kwargs)