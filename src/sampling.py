"""
Sampling Module
Responsible for creating a stratified sample from the cleaned dataset.
Ensures proportional representation across product categories.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CATEGORY_COL = "product_category"


def stratified_sample(
    df: pd.DataFrame,
    target_n: int,
    category_col: str = CATEGORY_COL,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Stratified sampling with proportional allocation across categories.
    """

    counts = df[category_col].value_counts()
    shares = counts / counts.sum()
    raw_alloc = shares * target_n

    floor_alloc = np.floor(raw_alloc).astype(int)
    remainder = target_n - floor_alloc.sum()

    fractional = (raw_alloc - floor_alloc).sort_values(ascending=False)

    alloc = floor_alloc.copy()
    for cat in fractional.index[:remainder]:
        alloc[cat] += 1

    sampled_parts = []

    print("Sampling allocation:")
    for cat, n_target in alloc.items():
        n_take = min(n_target, counts[cat])

        print(f"{cat}: {n_take}")

        sampled_parts.append(
            df[df[category_col] == cat].sample(
                n=n_take,
                random_state=random_state
            )
        )

    sample = pd.concat(sampled_parts, ignore_index=True)

    sample = (
        sample.sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    print(f"\nTotal sampled rows: {len(sample):,}")

    return sample


if __name__ == "__main__":

    INPUT_PATH = "data/processed/filtered_complaints.csv"
    OUTPUT_PATH = "data/processed/sampled_complaints.csv"

    df = pd.read_csv(INPUT_PATH)

    sample = stratified_sample(
        df=df,
        target_n=12000
    )

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    sample.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved sample to: {OUTPUT_PATH}")