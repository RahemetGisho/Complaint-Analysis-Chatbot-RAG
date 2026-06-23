"""
Central configuration for CrediTrust RAG System.

This file controls:
- embedding models
- vector store settings
- retrieval behavior
- LLM generation settings
- file paths
"""

from pathlib import Path

# PATHS
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data"
VECTOR_STORE_PATH = BASE_DIR / "vector_store"
REPORTS_PATH = BASE_DIR / "reports"

PREBUILT_PARQUET_PATH = DATA_PATH / "raw" / "complaint_embeddings.parquet"
EVALUATION_OUTPUT_PATH = REPORTS_PATH / "task3_evaluation.md"


# EMBEDDING CONFIG
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


# VECTOR STORE CONFIG
COLLECTION_NAME = "crediTrust_complaints"
TOP_K = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# LLM GENERATION CONFIG

GENERATION_BACKEND = "hf_inference_api"

HF_GENERATION_MODEL = "Qwen/Qwen2.5-7B-Instruct"

LOCAL_GENERATION_MODEL = "google/flan-t5-base"

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2


# RETRIEVAL CONFIG
ENABLE_METADATA_FILTERING = False


# EVALUATION CONFIG
EVAL_QUESTIONS_COUNT = 8
EVAL_OUTPUT_FORMAT = "markdown"