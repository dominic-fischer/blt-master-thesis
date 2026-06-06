"""
find_english_bounds_all.py

Runs English sentences across a range of thresholds for all four cases:
  - raw entropy mode
  - raw monotonicity mode
  - normalised entropy mode
  - normalised monotonicity mode

For each case, finds:
  - The threshold at exactly 3.5 bpp (lower bound)
  - The threshold at exactly 5.5 bpp (upper bound)
  - The midpoint between them
  - The exact anchor threshold that gives 32.77 pps (binary search)

Output: printed to stdout + saved to calibration/english_bounds_{case}.csv

Usage:
    python find_english_bounds_all.py
"""

import json
import os
import csv
import torch
from pathlib import Path
from bytelatent.data.patcher import (
    find_entropy_patch_start_ids,
    patch_lengths_from_start_ids,
)

# ── config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results/restructured"
OUTPUT_DIR  = "calibration"
ENGLISH     = "eng_Latn"
TARGET_PPS  = 32.77
BPP_LOW     = 3.5
BPP_HIGH    = 5.5
TOLERANCE   = 0.05  # pps tolerance for binary search

CASES = {
    "raw_entropy": {
        "score_idx":    1,
        "monotonicity": False,
        "search_low":   0.5,
        "search_high":  4.0,
    },
    "raw_monotonicity": {
        "score_idx":    1,
        "monotonicity": True,
        "search_low":   0.1,
        "search_high":  3.0,
    },
    "norm_entropy": {
        "score_idx":    2,
        "monotonicity": False,
        "search_low":   -2.0,
        "search_high":  3.0,
    },
    "norm_monotonicity": {
        "score_idx":    2,
        "monotonicity": True,
        "search_low":   -1.0,
        "search_high":  2.0,
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def compute_patch_lengths(
    scores: list[float],
    threshold: float,
    monotonicity: bool,
) -> list[int]:
    bos_score = 99.0
    scores_with_bos = torch.tensor([[bos_score] + scores])
    patch_start_ids = find_entropy_patch_start_ids(
        scores_with_bos,
        threshold=threshold,
        threshold_add=None,
        monotonicity=monotonicity,
    )
    n_bytes = len(scores)
    try:
        patch_lengths = patch_lengths_from_start_ids(patch_start_ids, n_bytes + 1)
        return [l for l in patch_lengths[0].tolist()[1:] if l > 0]
    except AssertionError:
        return [n_bytes]


def eval_english(sentences, threshold, score_idx, monotonicity):
    """Returns (mean_pps, mean_bpp) for English at a given threshold."""
    all_pps, all_bpp = [], []
    for s in sentences:
        scores  = [be[score_idx] for be in s["bytes_entropies"]]
        lengths = compute_patch_lengths(scores, threshold, monotonicity)
        n_patches = len(lengths)
        n_bytes   = sum(lengths)
        all_pps.append(n_patches)
        all_bpp.append(n_bytes / n_patches if n_patches > 0 else 0.0)
    return sum(all_pps) / len(all_pps), sum(all_bpp) / len(all_bpp)


def binary_search_pps(sentences, target_pps, score_idx, monotonicity, low, high, tolerance=0.05):
    """Binary search for threshold giving target_pps. Higher threshold → lower pps."""
    for _ in range(50):
        mid = (low + high) / 2
        mean_pps, mean_bpp = eval_english(sentences, mid, score_idx, monotonicity)
        if abs(mean_pps - target_pps) < tolerance:
            return mid, mean_pps, mean_bpp
        if mean_pps > target_pps:
            low = mid   # need higher threshold to reduce pps
        else:
            high = mid  # need lower threshold to increase pps
        if high - low < 1e-6:
            break
    mean_pps, mean_bpp = eval_english(sentences, mid, score_idx, monotonicity)
    return mid, mean_pps, mean_bpp


def binary_search_bpp(sentences, target_bpp, score_idx, monotonicity, low, high, tolerance=0.01):
    """Binary search for threshold giving target_bpp. Higher threshold → higher bpp."""
    for _ in range(50):
        mid = (low + high) / 2
        mean_pps, mean_bpp = eval_english(sentences, mid, score_idx, monotonicity)
        if abs(mean_bpp - target_bpp) < tolerance:
            return mid, mean_pps, mean_bpp
        if mean_bpp < target_bpp:
            low = mid   # need higher threshold to increase bpp
        else:
            high = mid  # need lower threshold to decrease bpp
        if high - low < 1e-6:
            break
    mean_pps, mean_bpp = eval_english(sentences, mid, score_idx, monotonicity)
    return mid, mean_pps, mean_bpp


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(Path(RESULTS_DIR) / f"{ENGLISH}.json", encoding="utf-8") as f:
        sentences = json.load(f)

    print(f"Loaded {len(sentences)} English sentences")
    print(f"Target pps : {TARGET_PPS}  (English at Meta entropy threshold)")
    print(f"Target bpp : {BPP_LOW} - {BPP_HIGH}  (±1 around Meta's 4.5)\n")

    summary_rows = []

    for case_name, case in CASES.items():
        print(f"── {case_name} ──")
        idx  = case["score_idx"]
        mono = case["monotonicity"]
        slow = case["search_low"]
        shigh = case["search_high"]

        # binary search for lower bound (3.5 bpp)
        t_low, pps_low, bpp_low = binary_search_bpp(
            sentences, BPP_LOW, idx, mono, slow, shigh)
        print(f"  Lower bound  (bpp≈{BPP_LOW}): threshold={t_low:.4f}  pps={pps_low:.2f}  bpp={bpp_low:.4f}")

        # binary search for upper bound (5.5 bpp)
        t_high, pps_high, bpp_high = binary_search_bpp(
            sentences, BPP_HIGH, idx, mono, slow, shigh)
        print(f"  Upper bound  (bpp≈{BPP_HIGH}): threshold={t_high:.4f}  pps={pps_high:.2f}  bpp={bpp_high:.4f}")

        # midpoint
        t_mid = (t_low + t_high) / 2
        pps_mid, bpp_mid = eval_english(sentences, t_mid, idx, mono)
        print(f"  Midpoint                  : threshold={t_mid:.4f}  pps={pps_mid:.2f}  bpp={bpp_mid:.4f}")

        # exact anchor (32.77 pps)
        t_anchor, pps_anchor, bpp_anchor = binary_search_pps(
            sentences, TARGET_PPS, idx, mono, slow, shigh)
        print(f"  Anchor       (pps≈{TARGET_PPS}): threshold={t_anchor:.4f}  pps={pps_anchor:.2f}  bpp={bpp_anchor:.4f}")

        thresholds = sorted([t_low, t_mid, t_high, t_anchor])
        print(f"\n  Final 4 thresholds: {[f'{t:.4f}' for t in thresholds]}\n")

        summary_rows.append({
            "case": case_name,
            "t_low": t_low, "pps_low": pps_low, "bpp_low": bpp_low,
            "t_mid": t_mid, "pps_mid": pps_mid, "bpp_mid": bpp_mid,
            "t_high": t_high, "pps_high": pps_high, "bpp_high": bpp_high,
            "t_anchor": t_anchor, "pps_anchor": pps_anchor, "bpp_anchor": bpp_anchor,
            "final_thresholds": thresholds,
        })

    # save summary
    out = os.path.join(OUTPUT_DIR, "english_bounds_summary.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "t_low", "bpp_low", "t_mid", "bpp_mid",
                         "t_high", "bpp_high", "t_anchor", "pps_anchor", "bpp_anchor",
                         "final_thresholds"])
        for r in summary_rows:
            writer.writerow([
                r["case"],
                f"{r['t_low']:.4f}", f"{r['bpp_low']:.4f}",
                f"{r['t_mid']:.4f}", f"{r['bpp_mid']:.4f}",
                f"{r['t_high']:.4f}", f"{r['bpp_high']:.4f}",
                f"{r['t_anchor']:.4f}", f"{r['pps_anchor']:.2f}", f"{r['bpp_anchor']:.4f}",
                str([f"{t:.4f}" for t in r["final_thresholds"]]),
            ])
    print(f"Summary saved → {out}")


if __name__ == "__main__":
    main()