#!/usr/bin/env python3
"""
Compute coordinate-wise median of each label column from `label_proportions.csv`.

Steps:
 - Take the median across rows for each label column
 - Normalize so the 7 values sum to 1
 - Save results to CSV
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("label_proportions.csv")
OUT_MEDIAN = Path("median_label_proportions.csv")
OUT_MEDIAN_NORM = Path("normalized_median_label_proportions.csv")

LABEL_COLS = [
    "Facts",
    "Argument",
    "Statute",
    "Precedent",
    "Ratio of the decision",
    "Ruling by Present Court",
    "Ruling by Lower Court",
]

# --- Load data ---
if not INPUT.exists():
    raise FileNotFoundError(f"{INPUT} not found. Please run this script in the folder containing the CSV.")

df = pd.read_csv(INPUT)

# --- Check columns ---
missing = [c for c in LABEL_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns in CSV: {missing}")

# --- Compute coordinate-wise median ---
median_series = df[LABEL_COLS].median(axis=0)

# --- Normalize so the medians sum to 1 ---
median_sum = median_series.sum()
normalized_median = median_series / median_sum if median_sum != 0 else median_series

# --- Print results ---
print("Coordinate-wise median (per label):")
print(median_series.round(6))
print(f"Sum(median) = {median_sum:.12f}")

print("\nNormalized coordinate-wise median (sums to 1):")
print(normalized_median.round(6))
print(f"Sum(normalized median) = {normalized_median.sum():.12f}")

# --- Save to CSV ---
def save_one_row(series, path):
    out_df = pd.DataFrame([series])
    out_df = out_df[LABEL_COLS]
    out_df.to_csv(path, index=False)
    print(f"Wrote: {path}")

save_one_row(median_series, OUT_MEDIAN)
save_one_row(normalized_median, OUT_MEDIAN_NORM)
