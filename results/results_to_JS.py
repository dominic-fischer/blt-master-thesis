"""
results_to_JS.py

Converts a FLORES+ results CSV to a single-line JS file:
    const LANG_DATA = [{...}, {...}, ...];

Usage:
    python results_to_JS.py
    python results_to_JS.py --csv-in-path floresplus_MASTER_with_results.csv --out-path results/results_data.js
"""

import argparse
import json
import os

import pandas as pd

DEFAULT_CSV_IN_PATH = "floresplus_MASTER.csv"
DEFAULT_OUT_PATH = "results/results_data.js"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv-in-path", default=DEFAULT_CSV_IN_PATH,
                         help=f"Input CSV to convert (default: {DEFAULT_CSV_IN_PATH}).")
    parser.add_argument("--out-path", default=DEFAULT_OUT_PATH,
                         help=f"Output JS file path (default: {DEFAULT_OUT_PATH}).")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv_in_path)
    records = df.to_dict(orient="records")

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_path, "w", encoding="utf-8") as f:
        f.write("const LANG_DATA = ")
        json.dump(records, f, ensure_ascii=False)
        f.write(";")
    print(f"Saved → {args.out_path}  ({len(records)} languages)")


if __name__ == "__main__":
    main()