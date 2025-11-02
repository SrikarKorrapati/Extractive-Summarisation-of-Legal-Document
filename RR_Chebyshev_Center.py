#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import cvxpy as cp

INPUT = Path("label_proportions.csv")
OUTPUT = Path("chebyshev_center_label_proportions.csv")

LABEL_COLS = [
    "Facts",
    "Argument",
    "Statute",
    "Precedent",
    "Ratio of the decision",
    "Ruling by Present Court",
    "Ruling by Lower Court",
]

def chebyshev_center(data):
    """
    Compute Chebyshev center on the simplex.
    data: numpy array (N, d)
    """
    N, d = data.shape
    x = cp.Variable(d)
    t = cp.Variable()  # radius to minimize

    constraints = [
        x >= 0,            # simplex nonnegativity
        cp.sum(x) == 1,    # simplex constraint
    ]

    # Each data point must be within distance t
    for i in range(N):
        constraints.append(cp.norm2(data[i] - x) <= t)

    prob = cp.Problem(cp.Minimize(t), constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError("Chebyshev center optimization did not converge")

    return x.value

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"{INPUT} not found. Please run this script in the folder containing the CSV.")

    df = pd.read_csv(INPUT)

    missing = [c for c in LABEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")

    data = df[LABEL_COLS].to_numpy()

    center = chebyshev_center(data)
    center = np.maximum(center, 0)  # clip any tiny negatives
    center = center / center.sum()  # normalize to sum=1

    # Print results
    print("Chebyshev center (per label):")
    print(pd.Series(center, index=LABEL_COLS).round(6))
    print(f"Sum(Chebyshev center) = {center.sum():.12f}")

    # Save
    out_df = pd.DataFrame([center], columns=LABEL_COLS)
    out_df.to_csv(OUTPUT, index=False)
    print(f"Wrote: {OUTPUT.resolve()}")

if __name__ == "__main__":
    main()
