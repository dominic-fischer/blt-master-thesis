"""
summarize_patching.py

Reads restructured JSON files and generates one summary CSV per case,
with languages as rows and thresholds as columns, reporting premium vs English.

Output: calibration/summary_{case_name}.csv

Usage:
    python summarize_patching.py
"""

import json
import os
import csv
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results/restructured"
OUTPUT_DIR  = "calibration"
ENGLISH     = "eng_Latn"

CASES = {
    "raw_entropy":       [1.1904, 1.3340, 1.4946, 1.7988],
    "raw_monotonicity":  [0.2359, 0.3662, 0.5928, 0.9496],
    "norm_entropy":      [0.4072, 0.5293, 0.7222, 1.0371],
    "norm_monotonicity": [0.2480, 0.3887, 0.6260, 1.0039],
    "raw_combined":      [0.2359, 0.3662, 0.5928, 0.9496],  # threshold_add values
    "norm_combined":     [0.2480, 0.3887, 0.6260, 1.0039],
}


# ── helpers ───────────────────────────────────────────────────────────────────

def threshold_key(threshold: float) -> str:
    return f"t_{threshold:.4f}"


def mean_pps(sentences: list[dict], case_name: str, threshold: float) -> float:
    key = threshold_key(threshold)
    pps_list = [
        s["eval_modes"][case_name][key]["n_patches"]
        for s in sentences
        if case_name in s.get("eval_modes", {})
        and key in s["eval_modes"][case_name]
    ]
    return sum(pps_list) / len(pps_list) if pps_list else 0.0


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    paths = sorted(Path(RESULTS_DIR).glob("*.json"))
    print(f"Found {len(paths)} language files")

    # load all data
    all_data = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            all_data[path.stem] = json.load(f)
    lang_codes = sorted(all_data.keys())

    eng_sentences = all_data[ENGLISH]

    for case_name, thresholds in CASES.items():
        print(f"  Processing {case_name}...")

        # compute English pps at each threshold (reference)
        eng_pps = {t: mean_pps(eng_sentences, case_name, t) for t in thresholds}

        rows = []
        for lang_code in lang_codes:
            row = {"lang_code": lang_code}
            for t in thresholds:
                lang_mean = mean_pps(all_data[lang_code], case_name, t)
                premium = lang_mean / eng_pps[t] if eng_pps[t] > 0 else 0.0
                row[threshold_key(t)] = round(premium, 4)
            rows.append(row)

        # save CSV
        out = os.path.join(OUTPUT_DIR, f"summary_{case_name}.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["lang_code"] + [threshold_key(t) for t in thresholds])
            writer.writeheader()
            writer.writerows(rows)

        print(f"    Saved → {out}")
        print(f"    English pps at each threshold:")
        for t in thresholds:
            print(f"      {threshold_key(t)}: {eng_pps[t]:.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()