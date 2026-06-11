"""
csv_to_json.py

Converts floresplus_MASTER_CSV.csv to a single-line JS file:
    const LANG_DATA = [{...}, {...}, ...];

Usage:
    python csv_to_json.py
"""

import json
import pandas as pd

CSV_PATH = "floresplus_MASTER_CSV.csv"
OUT_PATH = "results/results_data.js"


def main():
    df = pd.read_csv(CSV_PATH)
    records = df.to_dict(orient="records")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("const LANG_DATA = ")
        json.dump(records, f, ensure_ascii=False)
        f.write(";")
    print(f"Saved → {OUT_PATH}  ({len(records)} languages)")


if __name__ == "__main__":
    main()