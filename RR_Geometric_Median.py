#!/usr/bin/env python3
"""
Compute geometric median of each label column from `label_proportions.csv`.

Outputs:
 - geometric_median_label_proportions.csv         : geometric median of each label column
 - normalized_geometric_median_label_proportions.csv : normalized (sums exactly to 1) version
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("label_proportions.csv")
OUT_MEDIAN = Path("geometric_median_label_proportions.csv")
OUT_MEDIAN_NORM = Path("normalized_geometric_median_label_proportions.csv")

LABEL_COLS = [
    "Facts",
    "Argument",
    "Statute",
    "Precedent",
    "Ratio of the decision",
    "Ruling by Present Court",
    "Ruling by Lower Court",
]

# --- Weiszfeld’s algorithm for geometric median ---
def geometric_median(X, eps=1e-6, max_iter=500):
    """
    Compute geometric median of a set of points using Weiszfeld's algorithm.
    X: numpy array of shape (N, d)
    """
    y = np.mean(X, axis=0)  # initial guess = arithmetic mean
    for _ in range(max_iter):
        D = np.linalg.norm(X - y, axis=1)
        # if current estimate coincides with a data point
        if np.any(D == 0):
            return X[D == 0][0]
        W = 1 / D
        y1 = np.sum(W[:, np.newaxis] * X, axis=0) / W.sum()
        if np.linalg.norm(y - y1) < eps:
            return y1
        y = y1
    return y  # return after max_iter if not converged

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"{INPUT} not found. Please run this script in the folder containing the CSV.")

    df = pd.read_csv(INPUT)

    # Validate columns
    missing = [c for c in LABEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")

    data = df[LABEL_COLS].to_numpy()

    # Compute geometric median
    gm = geometric_median(data)

    # Normalize to sum=1
    gm_sum = gm.sum()
    gm_norm = gm / gm_sum if gm_sum != 0 else gm

    # Print results
    print("Geometric median (per label):")
    print(pd.Series(gm, index=LABEL_COLS).round(6))
    print(f"Sum(geometric median) = {gm_sum:.12f}")

    print("\nNormalized geometric median (sums to 1):")
    print(pd.Series(gm_norm, index=LABEL_COLS).round(6))
    print(f"Sum(normalized) = {gm_norm.sum():.12f}")

    # Save to CSVs
    def save_one_row(series, path):
        out_df = pd.DataFrame([series])
        out_df = out_df[LABEL_COLS]
        out_df.to_csv(path, index=False)
        print(f"Wrote: {path}")

    save_one_row(pd.Series(gm, index=LABEL_COLS), OUT_MEDIAN)
    save_one_row(pd.Series(gm_norm, index=LABEL_COLS), OUT_MEDIAN_NORM)

if __name__ == "__main__":
    main()
