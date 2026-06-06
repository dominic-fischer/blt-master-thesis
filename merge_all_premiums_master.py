"""
merge_master.py

Merges the master CSV with all summary CSVs, removing old patch stats
and adding premium columns for all cases and thresholds.

Input:
    floresplus_MASTER_CSV.csv
    calibration/summary_{case}.csv  (6 files)

Output:
    floresplus_MASTER_CSV_updated.csv

Usage:
    python merge_master.py
"""

import csv
import os

# ── config ────────────────────────────────────────────────────────────────────
MASTER_CSV  = "floresplus_MASTER_CSV.csv"
OUTPUT_CSV  = "floresplus_MASTER_CSV_updated.csv"
CALIBRATION = "calibration"

# columns to remove from master
REMOVE_COLS = {"Total patches", "Total bytes", "Avg b/patch", "Avg patches/sent", "Premium vs EN"}

CASES = {
    "raw_entropy":       [1.1904, 1.3340, 1.4946, 1.7988],
    "raw_monotonicity":  [0.2359, 0.3662, 0.5928, 0.9496],
    "raw_combined":      [0.2359, 0.3662, 0.5928, 0.9496],
    "norm_entropy":      [0.4072, 0.5293, 0.7222, 1.0371],
    "norm_monotonicity": [0.2480, 0.3887, 0.6260, 1.0039],
    "norm_combined":     [0.2480, 0.3887, 0.6260, 1.0039],
}


def threshold_key(t: float) -> str:
    return f"t_{t:.4f}"


def col_name(case: str, t: float) -> str:
    return f"prem_{case}_{threshold_key(t)}"


# ── load summary CSVs ─────────────────────────────────────────────────────────
print("Loading summary CSVs...")
summary = {}  # {lang_code: {col_name: value}}

for case_name, thresholds in CASES.items():
    path = os.path.join(CALIBRATION, f"summary_{case_name}.csv")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping")
        continue
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row["lang_code"]
            if lang not in summary:
                summary[lang] = {}
            for t in thresholds:
                tk = threshold_key(t)
                cn = col_name(case_name, t)
                summary[lang][cn] = row.get(tk, "")
    print(f"  Loaded {path}")

# ── load and update master CSV ────────────────────────────────────────────────
print("\nLoading master CSV...")
with open(MASTER_CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    old_cols = reader.fieldnames
    rows = list(reader)

# build new column list
keep_cols = [c for c in old_cols if c not in REMOVE_COLS]
new_premium_cols = [
    col_name(case, t)
    for case in CASES
    for t in CASES[case]
]
new_cols = keep_cols + new_premium_cols

print(f"  {len(rows)} languages")
print(f"  Removing: {sorted(REMOVE_COLS)}")
print(f"  Adding: {len(new_premium_cols)} premium columns")

# merge
matched = 0
unmatched = []
for row in rows:
    lang_code = row.get("Code_orig", "")
    if lang_code in summary:
        for cn in new_premium_cols:
            row[cn] = summary[lang_code].get(cn, "")
        matched += 1
    else:
        for cn in new_premium_cols:
            row[cn] = ""
        unmatched.append(lang_code)

print(f"  Matched: {matched}, Unmatched: {len(unmatched)}")
if unmatched:
    print(f"  Unmatched codes: {unmatched}")

# write output
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=new_cols, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved → {OUTPUT_CSV}")
print("Done.")