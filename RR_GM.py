import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("label_proportions.csv")
OUT_GEOM = Path("geometric_mean_label_proportions.csv")
OUT_GEOM_NORM = Path("normalized_geometric_mean_label_proportions.csv")

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

# Convert to numpy array and ensure float dtype
data = df[LABEL_COLS].to_numpy(dtype=float)

# Epsilon to avoid log(0). Default is small: 1e-12.
# Increase eps (e.g., 1e-9 or 1e-6) if you have many exact zeros and want to
# give them slightly larger weight in the geometric mean.
eps = 1e-3

# Compute geometric mean in log-space for numerical stability:
# geom_mean_j = exp( mean_i( log(data_ij + eps) ) )
log_vals = np.log(data + eps)          # shape (N, d)
mean_log = np.mean(log_vals, axis=0)   # shape (d,)
geom_mean = np.exp(mean_log)           # shape (d,)

# Normalize so it sums to 1 (unless sum is zero)
geom_sum = geom_mean.sum()
if geom_sum == 0 or not np.isfinite(geom_sum):
    raise RuntimeError("Geometric mean sum is zero or non-finite. Consider increasing eps.")

normalized_geom_mean = geom_mean / geom_sum

# Print results
print("Geometric mean (per label):")
print(pd.Series(geom_mean, index=LABEL_COLS).round(6))
print(f"Sum(geometric mean) = {geom_sum:.12f}")

print("\nNormalized geometric mean (sums to 1):")
print(pd.Series(normalized_geom_mean, index=LABEL_COLS).round(6))
print(f"Sum(normalized) = {normalized_geom_mean.sum():.12f}")

# Save CSVs
def save_one_row(series, path):
    out_df = pd.DataFrame([series])
    out_df = out_df[LABEL_COLS]
    out_df.to_csv(path, index=False)
    print(f"Wrote: {path}")

save_one_row(pd.Series(geom_mean, index=LABEL_COLS), OUT_GEOM)
save_one_row(pd.Series(normalized_geom_mean, index=LABEL_COLS), OUT_GEOM_NORM)
