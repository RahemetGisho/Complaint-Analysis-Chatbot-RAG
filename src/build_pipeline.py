"""
Full Pipeline Runner
Runs:
1. Sampling
2. Chunking
3. Embedding
4. Vector store indexing
"""

import pandas as pd

from sampling import stratified_sample
from chunking import chunk_dataframe
from embedding import load_model, embed_texts
from vector_store import build_vector_store


def main():
    # Load data
    df = pd.read_csv("data/processed/filtered_complaints.csv")

    # Step 1: Sampling
    sample = stratified_sample(df, target_n=12000)

    # Step 2: Chunking
    chunk_df = chunk_dataframe(sample, chunk_size=500, chunk_overlap=50)

    # Step 3: Embedding
    model = load_model()
    embeddings = embed_texts(model, chunk_df["chunk_text"].tolist())

    # Step 4: Vector store
    build_vector_store(chunk_df, embeddings)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()