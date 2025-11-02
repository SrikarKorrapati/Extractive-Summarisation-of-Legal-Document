import csv
from pathlib import Path
import re

# CONFIG
INPUT_DIR = Path("Labelled_Data")
OUTPUT_CSV = Path("label_proportions.csv")
MAX_FILES = 40

# canonical labels in the desired output order
CANONICAL_LABELS = [
    "Facts",
    "Argument",
    "Statute",
    "Precedent",
    "Ratio of the decision",
    "Ruling by Present Court",
    "Ruling by Lower Court",
]

def normalize_label_text(s: str) -> str:
    """Normalize label for matching: lowercase, remove punctuation & extra spaces."""
    s = s.strip().lower()
    # collapse multiple spaces/tabs
    s = re.sub(r"\s+", " ", s)
    # remove punctuation except keep letters and digits and spaces
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s

# Build mapping from normalized label -> canonical label
_normal_map = {normalize_label_text(lbl): lbl for lbl in CANONICAL_LABELS}

def map_label(raw_label: str):
    """Map raw label text to canonical label. Return None if not recognized."""
    key = normalize_label_text(raw_label)
    # try exact normalized match
    if key in _normal_map:
        return _normal_map[key]
    # try some partial matching heuristics
    for k_norm, canon in _normal_map.items():
        if key == k_norm:
            return canon
        # allow short names like 'facts', 'argument', 'statute', 'precedent', 'ratio', 'ruling'
        if key in k_norm or k_norm in key:
            # be conservative: only accept if the short key maps unambiguously
            # (if multiple possible matches, we skip)
            candidates = [c for kn, c in _normal_map.items() if key in kn or kn in key]
            if len(set(candidates)) == 1:
                return candidates[0]
    return None

def process_file(path: Path):
    """
    Read file and count labels. Returns dict {canonical_label: count}, total_count.
    """
    counts = {lbl: 0 for lbl in CANONICAL_LABELS}
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n\r")
            if not line.strip():
                continue
            # split on last tab; this lets text contain tabs earlier if any
            if "\t" in line:
                parts = line.rsplit("\t", 1)
            else:
                # fallback: if no tab, try splitting by multiple spaces
                parts = re.split(r"\s{2,}", line, maxsplit=1)
            if len(parts) == 2:
                _, raw_label = parts
            else:
                # cannot find a label — skip this line
                continue
            mapped = map_label(raw_label)
            if mapped:
                counts[mapped] += 1
                total += 1
            else:
                # unknown label — ignore; you could log it if desired
                # print(f"[WARN] Unknown label in {path.name}: '{raw_label}'")
                continue
    return counts, total

def main():
    if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
        print(f"Input folder '{INPUT_DIR}' not found. Please create it and put files inside.")
        return

    # list files sorted by name, take first MAX_FILES
    all_files = sorted([p for p in INPUT_DIR.iterdir() if p.is_file()])
    files_to_process = all_files[:MAX_FILES]
    if not files_to_process:
        print(f"No files in '{INPUT_DIR}'.")
        return
    if len(all_files) < MAX_FILES:
        print(f"Only {len(all_files)} files found; will process all of them.")

    rows = []
    for p in files_to_process:
        counts, total = process_file(p)
        # convert counts -> proportions (if total==0 keep zeros)
        if total > 0:
            proportions = {lbl: counts[lbl] / total for lbl in CANONICAL_LABELS}
        else:
            proportions = {lbl: 0.0 for lbl in CANONICAL_LABELS}
        row = {"filename": p.name, "total_count": total}
        row.update(proportions)
        rows.append(row)

    # write CSV with ordered columns
    fieldnames = ["filename"] + CANONICAL_LABELS + ["total_count"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            # format floats to e.g. 6 decimal places
            out = {k: r[k] if k in ("filename", "total_count") else f"{r[k]:.6f}" for k in fieldnames}
            writer.writerow(out)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV.resolve()}")

if __name__ == "__main__":
    main()
