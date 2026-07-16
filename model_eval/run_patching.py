"""
run_patching.py

Computes patch lengths for all cases and thresholds and stores results
in the restructured JSON files under eval_modes.

Cases and thresholds are read from --summary-csv (default
calibrated_thresholds/thresholds_summary.csv, or a per-checkpoint
recalibrated CSV from calibrate_thresholds.py). For standard cases, the
threshold column values (t_low, t_mid, t_high, t_anchor) are used
directly as the patching threshold. For the combined case, fixed_t is
the entropy threshold and t_low/t_mid/t_high/t_anchor are threshold_add
values.

KEYED BY EXACT NUMERIC THRESHOLD (as originally): eval_modes entries are
stored as eval_modes[case_name]["t_<value>"], e.g. "t_1.3340" -- the
literal calibrated threshold, not a bound label. Bound identity
(low/mid/high/anchor) is intentionally NOT baked in here; it's
reconstructed downstream by results_to_txt_premiums.py, which
cross-references the same --summary-csv this script used to figure out
which numeric value corresponds to which bound, per case. This keeps
--summary-csv as the single source of truth for that mapping instead of
duplicating it into the patching output. (Iterating the 4 named bounds
from the CSV, rather than blindly deduplicating via a set as an earlier
version of this script did, still avoids silently losing an entry if two
bounds happen to coincide numerically for a given case -- the second one
just correctly reuses the already-computed result instead of redoing it.)

Any eval_modes keys in the JSON that are NOT present in the CSV are removed.

Output: updates <results-dir>/{lang_code}.json in place (indented)

Usage (from repo root):
    python model_eval/run_patching.py --results-dir results/own_models/entropy_10M_..._lr4.5e-3/step_0000006000
    python model_eval/run_patching.py --results-dir <dir> --summary-csv calibrated_thresholds/thresholds_summary.csv
"""

import argparse
import csv
import json
import os
import torch
from pathlib import Path
from tqdm import tqdm
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))  # noqa: E402
from bytelatent.data.patcher import (
    find_entropy_patch_start_ids,
    patch_lengths_from_start_ids,
)

DEFAULT_SUMMARY_CSV = "calibrated_thresholds/base_model_thresholds_summary.csv"

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

BOUND_NAMES = ["low", "mid", "high", "anchor"]


def load_cases(csv_path: str) -> dict:
    """
    Returns a dict keyed by case name. Each value is a dict with:
      - score_idx:         int
      - monotonicity:      bool
      - fixed_threshold:   float or None  (only for combined)
      - named_thresholds:  dict[str, float], keys "low"/"mid"/"high"/"anchor"
                           (t_add values for combined, else t values)
    """
    cases = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["case"]
            named_thresholds = {
                "low":    float(row["t_low"]),
                "mid":    float(row["t_mid"]),
                "high":   float(row["t_high"]),
                "anchor": float(row["t_anchor"]),
            }
            fixed_t = float(row["fixed_t"]) if row["fixed_t"] else None
            cases[name] = {
                "score_idx":        SCORE_IDX[name],
                "monotonicity":     MONOTONICITY[name],
                "fixed_threshold":  fixed_t,
                "named_thresholds": named_thresholds,
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
        bounds_str = ", ".join(f"{b}={case['named_thresholds'][b]:.4f}" for b in BOUND_NAMES)
        if case["fixed_threshold"] is not None:
            print(f"  {name}: fixed_t={case['fixed_threshold']:.4f}, t_add: {bounds_str}")
        else:
            print(f"  {name}: {bounds_str}")
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

                for bound_name in BOUND_NAMES:
                    t = case["named_thresholds"][bound_name]
                    key = threshold_key(t)
                    if key in sentence["eval_modes"][case_name]:
                        continue  # already computed (this exact value), skip

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