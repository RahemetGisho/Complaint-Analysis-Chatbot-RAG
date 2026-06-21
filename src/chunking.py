"""
Chunking Module

Splits complaint narratives into smaller overlapping text chunks
for better embedding performance.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

TEXT_COL = "cleaned_narrative"
ID_COL = "Complaint ID"


def chunk_dataframe(
    df: pd.DataFrame,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> pd.DataFrame:
    """
    Convert each complaint into multiple overlapping chunks.
    Returns a dataframe containing one row per chunk.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    records = []

    for _, row in df.iterrows():

        text = str(row[TEXT_COL])
        chunks = splitter.split_text(text)

        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):

            records.append(
                {
                    "chunk_id": f"{row[ID_COL]}_{i}",
                    "complaint_id": row[ID_COL],
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                    "chunk_text": chunk,
                    "product_category": row.get("product_category"),
                }
            )

    chunk_df = pd.DataFrame(records)

    print(
        f"Chunked {len(df):,} complaints into "
        f"{len(chunk_df):,} chunks"
    )

    print(
        f"Average chunks per complaint: "
        f"{len(chunk_df) / len(df):.2f}"
    )

    return chunk_df


if __name__ == "__main__":

    INPUT_PATH = "data/processed/sampled_complaints.csv"
    OUTPUT_PATH = "data/processed/chunks_metadata.csv"

    df = pd.read_csv(INPUT_PATH)

    chunk_df = chunk_dataframe(
        df=df,
        chunk_size=500,
        chunk_overlap=50,
    )

    Path("data/processed").mkdir(
        parents=True,
        exist_ok=True,
    )

    chunk_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Saved chunk metadata to: {OUTPUT_PATH}")