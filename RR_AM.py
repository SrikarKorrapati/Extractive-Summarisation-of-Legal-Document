#!/usr/bin/env python3
"""
Compute mean of each label column from `label_proportions.csv`.

Outputs:
 - mean_label_proportions.csv         : simple arithmetic mean of each label column
 - normalized_mean_label_proportions.csv : normalized (sums exactly to 1) version of the mean
 - weighted_mean_label_proportions.csv   : optional mean weighted by total_count (if present)
"""

import pandas as pd
from pathlib import Path

INPUT = Path("label_proportions.csv")
OUT_MEAN = Path("mean_label_proportions.csv")
OUT_NORM = Path("normalized_mean_label_proportions.csv")
OUT_WEIGHTED = Path("weighted_mean_label_proportions.csv")

LABEL_COLS = [
    "Facts",
    "Argument",
    "Statute",
    "Precedent",
    "Ratio of the decision",
    "Ruling by Present Court",
    "Ruling by Lower Court",
]

if not INPUT.exists():
    raise FileNotFoundError(f"{INPUT} not found. Please run this script in the folder containing the CSV.")

df = pd.read_csv(INPUT)

# Validate columns
missing = [c for c in LABEL_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns in CSV: {missing}")

# Simple arithmetic mean (mean across rows)
mean_series = df[LABEL_COLS].mean(axis=0)

# Normalize the mean so it sums exactly to 1 (useful to eliminate tiny rounding error)
mean_sum = mean_series.sum()
normalized_mean = mean_series / mean_sum if mean_sum != 0 else mean_series

# Weighted mean using total_count if available
if "total_count" in df.columns:
    total_counts = df["total_count"]
    # if any total_count is zero or NaN, pandas will handle; we avoid division by zero below
    if total_counts.sum() > 0:
        weighted_sum = (df[LABEL_COLS].multiply(total_counts, axis=0)).sum(axis=0)
        weighted_mean = weighted_sum / total_counts.sum()
        weighted_mean_sum = weighted_mean.sum()
        weighted_mean_normalized = weighted_mean / weighted_mean_sum if weighted_mean_sum != 0 else weighted_mean
    else:
        weighted_mean = None
        weighted_mean_normalized = None
else:
    weighted_mean = None
    weighted_mean_normalized = None

# Print results
print("Simple mean (per label):")
print(mean_series.round(6))
print(f"Sum(simple mean) = {mean_sum:.12f}")

print("\nNormalized mean (sums to 1):")
print(normalized_mean.round(6))
print(f"Sum(normalized) = {normalized_mean.sum():.12f}")

if weighted_mean is not None:
    print("\nWeighted mean by total_count (per label):")
    print(weighted_mean.round(6))
    print(f"Sum(weighted mean) = {weighted_mean_sum:.12f}")
    print("\nWeighted mean normalized (sums to 1):")
    print(weighted_mean_normalized.round(6))
    print(f"Sum(weighted normalized) = {weighted_mean_normalized.sum():.12f}")

# Save CSVs (single-row CSV, header = columns)
def save_one_row(series, path):
    out_df = pd.DataFrame([series])
    # ensure column order
    out_df = out_df[LABEL_COLS]
    out_df.to_csv(path, index=False)
    print(f"Wrote: {path}")

save_one_row(mean_series, OUT_MEAN)
save_one_row(normalized_mean, OUT_NORM)
if weighted_mean is not None:
    save_one_row(weighted_mean, OUT_WEIGHTED)
