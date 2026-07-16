"""
run_patching.py

Computes patch lengths for all cases and thresholds and stores results
in the restructured JSON files under eval_modes.

Cases and thresholds are read from --summary-csv (default
calibrate_thresholds/thresholds_summary.csv). For standard cases, the
threshold column values (t_low, t_mid, t_high, t_anchor) are used
directly as the patching threshold. For the combined case, fixed_t is
the entropy threshold and t_low/t_mid/t_high/t_anchor are threshold_add
values.

Any eval_modes keys in the JSON that are NOT present in the CSV are removed.

Output: updates <results-dir>/{lang_code}.json in place (indented)

Usage (from repo root):
    python model_eval/run_patching.py --results-dir results/own_models/entropy_10M_..._lr4.5e-3/step_0000006000
    python model_eval/run_patching.py --results-dir <dir> --summary-csv calibrate_thresholds/thresholds_summary.csv
"""

import argparse
import csv
import json
import os
import torch
from pathlib import Path
from tqdm import tqdm
from bytelatent.data.patcher import (
    find_entropy_patch_start_ids,
    patch_lengths_from_start_ids,
)

DEFAULT_SUMMARY_CSV = "calibrate_thresholds/thresholds_summary.csv"

# maps case name → which score index to use
SCORE_IDX = {
    "raw_entropy":      1,
    "raw_monotonicity": 1,
    "norm_entropy":     2,
    "combined":         1,
}

# maps case name → monotonicity flag (combined uses threshold_add instead)
MONOTONICITY = {
    "raw_entropy":      False,
    "raw_monotonicity": True,
    "norm_entropy":     False,
    "combined":         False,
}


# ── load cases from CSV ───────────────────────────────────────────────────────

def load_cases(csv_path: str) -> dict:
    """
    Returns a dict keyed by case name. Each value is a dict with:
      - score_idx:       int
      - monotonicity:    bool
      - fixed_threshold: float or None  (only for combined)
      - thresholds:      list[float]    (t_add values for combined, else t values)
    """
    cases = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["case"]
            thresholds = sorted({
                float(row["t_low"]),
                float(row["t_mid"]),
                float(row["t_high"]),
                float(row["t_anchor"]),
            })
            fixed_t = float(row["fixed_t"]) if row["fixed_t"] else None
            cases[name] = {
                "score_idx":       SCORE_IDX[name],
                "monotonicity":    MONOTONICITY[name],
                "fixed_threshold": fixed_t,
                "thresholds":      thresholds,
            }
    return cases


# ── patch helper ──────────────────────────────────────────────────────────────

def compute_patch_lengths(
    scores: list[float],
    threshold: float,
    threshold_add: float | None,
    monotonicity: bool,
) -> list[int]:
    bos_score = 99.0
    scores_with_bos = torch.tensor([[bos_score] + scores])
    patch_start_ids = find_entropy_patch_start_ids(
        scores_with_bos,
        threshold=threshold,
        threshold_add=threshold_add,
        monotonicity=monotonicity,
    )
    n_bytes = len(scores)
    try:
        patch_lengths = patch_lengths_from_start_ids(patch_start_ids, n_bytes + 1)
        return [l for l in patch_lengths[0].tolist()[1:] if l > 0]
    except AssertionError:
        return [n_bytes]


def threshold_key(threshold: float) -> str:
    return f"t_{threshold:.4f}"


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute patch lengths for all cases/thresholds over an "
                    "existing run_eval.py results directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory of per-language JSON files produced by run_eval.py "
             "(e.g. results/own_models/<run>/step_<step>/). Updated in place.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=DEFAULT_SUMMARY_CSV,
        help=f"CSV defining cases and thresholds (default {DEFAULT_SUMMARY_CSV}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cases = load_cases(args.summary_csv)
    valid_case_names = set(cases.keys())

    print(f"Loaded {len(cases)} cases from {args.summary_csv}:")
    for name, case in cases.items():
        if case["fixed_threshold"] is not None:
            print(f"  {name}: fixed_t={case['fixed_threshold']:.4f}, "
                  f"t_add values={[f'{t:.4f}' for t in case['thresholds']]}")
        else:
            print(f"  {name}: thresholds={[f'{t:.4f}' for t in case['thresholds']]}")
    print()

    paths = sorted(Path(args.results_dir).glob("*.json"))
    print(f"Found {len(paths)} language files in {args.results_dir}\n")

    for path in paths:
        lang_code = path.stem
        with open(path, encoding="utf-8") as f:
            sentences = json.load(f)

        for sentence in tqdm(sentences, desc=lang_code, leave=False):
            if "eval_modes" not in sentence:
                sentence["eval_modes"] = {}

            # remove stale cases not present in the CSV
            stale = [k for k in sentence["eval_modes"] if k not in valid_case_names]
            for k in stale:
                del sentence["eval_modes"][k]

            for case_name, case in cases.items():
                if case_name not in sentence["eval_modes"]:
                    sentence["eval_modes"][case_name] = {}

                scores = [be[case["score_idx"]] for be in sentence["bytes_entropies"]]
                is_combined = case["fixed_threshold"] is not None

                for t in case["thresholds"]:
                    key = threshold_key(t)
                    if key in sentence["eval_modes"][case_name]:
                        continue  # already computed, skip

                    if is_combined:
                        threshold     = case["fixed_threshold"]
                        threshold_add = t
                    else:
                        threshold     = t
                        threshold_add = None

                    lengths = compute_patch_lengths(
                        scores,
                        threshold=threshold,
                        threshold_add=threshold_add,
                        monotonicity=case["monotonicity"],
                    )
                    n_patches = len(lengths)
                    n_bytes   = sum(lengths)
                    avg_bpp   = n_bytes / n_patches if n_patches > 0 else 0.0

                    sentence["eval_modes"][case_name][key] = {
                        "n_patches":           n_patches,
                        "avg_bytes_per_patch": round(avg_bpp, 4),
                        "patch_lengths":       lengths,
                    }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(sentences, f, ensure_ascii=False, indent=2)

        print(f"  {lang_code}: done")

    print("\nDone.")


if __name__ == "__main__":
    main()