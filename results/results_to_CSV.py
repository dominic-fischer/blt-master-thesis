"""
results_to_CSV.py

For each language in --csv-in-path, loads the corresponding
--results-dir/{Code_Orig}.json and computes mean pps and global bpp
(total_bytes / total_patches) for each case/threshold combination, adding
them as new columns to the CSV.

Also computes pps_premium columns: lang_pps / eng_pps for each case/threshold.

Column naming:
  {case}_{threshold_key}_pps         e.g. raw_entropy_t_1.1904_pps
  {case}_{threshold_key}_bpp         e.g. raw_entropy_t_1.1904_bpp
  {case}_{threshold_key}_pps_premium e.g. raw_entropy_t_1.1904_pps_premium

The INPUT csv is left exactly as-is; the new columns are written to a
SEPARATE output csv (--csv-out-path), which defaults to
<csv-in-path stem>_with_results.csv if not given explicitly, so the
input file is never silently overwritten.

Usage:
    python results_to_CSV.py --results-dir results/own_models/<run>/step_<step>
    python results_to_CSV.py --results-dir <dir> --csv-in-path floresplus_MASTER.csv --csv-out-path floresplus_MASTER_with_results.csv
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

ENGLISH = "eng_Latn"


def default_csv_out_path(csv_in_path: str) -> str:
    """<name>.csv -> <name>_with_results.csv, so the input CSV is never
    silently overwritten unless --csv-out-path is explicitly set to the
    same path."""
    p = Path(csv_in_path)
    return str(p.with_name(f"{p.stem}_with_results{p.suffix}"))


def load_results(results_dir: str, code_orig: str) -> list[dict]:
    path = Path(results_dir) / f"{code_orig}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def aggregate(sentences: list[dict]) -> dict[str, float]:
    """Compute mean pps and global bpp (total_bytes / total_patches) for every case/threshold."""
    totals = {}  # key -> [total_patches, total_bytes, n_sentences]

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


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", default="results/restructured",
                         help="Directory of per-language JSON files with eval_modes "
                              "populated (default: results/restructured).")
    parser.add_argument("--csv-in-path", default="floresplus_MASTER.csv",
                         help="Input CSV, read but never modified "
                              "(default: floresplus_MASTER.csv).")
    parser.add_argument("--csv-out-path", default=None,
                         help="Output CSV with the new columns added. If omitted, "
                              "derived from --csv-in-path as <name>_with_results.csv.")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_out_path = args.csv_out_path or default_csv_out_path(args.csv_in_path)

    df = pd.read_csv(args.csv_in_path)
    print(f"Loaded {len(df)} languages from {args.csv_in_path}")

    all_rows = []
    for _, row in df.iterrows():
        code_orig = row["Code_Orig"]
        sentences = load_results(args.results_dir, code_orig)
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

    out_dir = os.path.dirname(csv_out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(csv_out_path, index=False)
    print(f"\nSaved → {csv_out_path}  ({len(result_cols)} new columns added)")
    print(f"(input CSV {args.csv_in_path} left unmodified)")


if __name__ == "__main__":
    main()