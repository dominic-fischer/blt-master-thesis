"""
add_results_to_csv.py

For each language in floresplus_MASTER_CSV.csv, loads the corresponding
results/restructured/{Code_Orig}.json and computes mean pps and global bpp
(total_bytes / total_patches) for each case/threshold combination, adding
them as new columns to the CSV.

Also computes pps_premium columns: lang_pps / eng_pps for each case/threshold.

Column naming:
  {case}_{threshold_key}_pps         e.g. raw_entropy_t_1.1904_pps
  {case}_{threshold_key}_bpp         e.g. raw_entropy_t_1.1904_bpp
  {case}_{threshold_key}_pps_premium e.g. raw_entropy_t_1.1904_pps_premium

Usage:
    python add_results_to_csv.py
"""

import json
import pandas as pd
from pathlib import Path

RESULTS_DIR = "results/restructured"
CSV_PATH    = "floresplus_MASTER_CSV.csv"
ENGLISH     = "eng_Latn"


def load_results(code_orig: str) -> list[dict]:
    path = Path(RESULTS_DIR) / f"{code_orig}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def aggregate(sentences: list[dict]) -> dict[str, float]:
    """Compute mean pps and global bpp (total_bytes / total_patches) for every case/threshold."""
    totals = {}  # key -> (total_patches, total_bytes, n_sentences)

    for s in sentences:
        n_bytes = len(s["bytes_entropies"])
        for case_name, thresholds in s.get("eval_modes", {}).items():
            for t_key, vals in thresholds.items():
                col = f"{case_name}_{t_key}"
                if col not in totals:
                    totals[col] = [0, 0, 0]
                totals[col][0] += vals["n_patches"]
                totals[col][1] += n_bytes
                totals[col][2] += 1

    result = {}
    for col, (total_patches, total_bytes, n_sentences) in totals.items():
        result[f"{col}_pps"] = round(total_patches / n_sentences, 4)
        result[f"{col}_bpp"] = round(total_bytes / total_patches, 4)
    return result


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} languages from {CSV_PATH}")

    all_rows = []
    for _, row in df.iterrows():
        code_orig = row["Code_Orig"]
        sentences = load_results(code_orig)
        agg       = aggregate(sentences)
        all_rows.append({**row.to_dict(), **agg})
        print(f"  {code_orig}: {len(agg)} result columns")

    out = pd.DataFrame(all_rows)

    # compute pps premiums relative to English
    eng_row = out[out["Code_Orig"] == ENGLISH]
    if len(eng_row) == 0:
        raise ValueError(f"English row ({ENGLISH}) not found in CSV")
    pps_cols = [c for c in out.columns if c.endswith("_pps")]
    for col in pps_cols:
        eng_pps = eng_row[col].values[0]
        out[col.replace("_pps", "_pps_premium")] = (out[col] / eng_pps).round(4)
    print(f"  Added {len(pps_cols)} pps_premium columns")

    # keep original columns first, then result columns sorted
    orig_cols   = list(df.columns)
    result_cols = sorted(c for c in out.columns if c not in orig_cols)
    out = out[orig_cols + result_cols]

    out.to_csv(CSV_PATH, index=False)
    print(f"\nSaved → {CSV_PATH}  ({len(result_cols)} new columns added)")


if __name__ == "__main__":
    main()