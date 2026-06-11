"""
find_english_bounds_all.py

Runs English sentences across a range of thresholds for all cases:
  - raw entropy mode
  - raw monotonicity mode
  - normalised entropy mode
  - combined mode (entropy threshold + monotonicity delta)

For each case, finds:
  - The threshold at exactly 3.5 bpp (lower bound)
  - The threshold at exactly 5.5 bpp (upper bound)
  - The midpoint between them
  - The exact anchor threshold that gives 32.77 pps (binary search)

For combined mode specifically:
  - t is fixed by finding the entropy threshold that gives bpp=3.5 at t_add=0
  - t_add is then swept to find the upper bound, midpoint, and anchor

Output: printed to stdout + saved to calibrate_thresholds/thresholds_summary.csv

Usage:
    python find_english_bounds_all.py
"""

import json
import os
import csv
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from bytelatent.data.patcher import (
    find_entropy_patch_start_ids,
    patch_lengths_from_start_ids,
)

# ── config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results/restructured"
OUTPUT_DIR  = "calibrate_thresholds"
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
}

COMBINED = {
    "score_idx":    1,
    "t_search_low":  0.5,
    "t_search_high": 4.0,
    "t_add_max":     3.0,
}


# ── patch helpers ─────────────────────────────────────────────────────────────

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


def compute_patch_lengths_combined(
    scores: list[float],
    t: float,
    t_add: float,
) -> list[int]:
    bos_score = 99.0
    scores_with_bos = torch.tensor([[bos_score] + scores])
    patch_start_ids = find_entropy_patch_start_ids(
        scores_with_bos,
        threshold=t,
        threshold_add=t_add,
        monotonicity=False,
    )
    n_bytes = len(scores)
    try:
        patch_lengths = patch_lengths_from_start_ids(patch_start_ids, n_bytes + 1)
        return [l for l in patch_lengths[0].tolist()[1:] if l > 0]
    except AssertionError:
        return [n_bytes]


# ── eval helpers ──────────────────────────────────────────────────────────────

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


def eval_english_combined(sentences, t, t_add, score_idx):
    """Returns (mean_pps, mean_bpp) for English in combined mode."""
    all_pps, all_bpp = [], []
    for s in sentences:
        scores  = [be[score_idx] for be in s["bytes_entropies"]]
        lengths = compute_patch_lengths_combined(scores, t, t_add)
        n_patches = len(lengths)
        n_bytes   = sum(lengths)
        all_pps.append(n_patches)
        all_bpp.append(n_bytes / n_patches if n_patches > 0 else 0.0)
    return sum(all_pps) / len(all_pps), sum(all_bpp) / len(all_bpp)


# ── binary search helpers ─────────────────────────────────────────────────────

def binary_search_pps(sentences, target_pps, score_idx, monotonicity, low, high, tolerance=0.05):
    """Binary search for threshold giving target_pps. Higher threshold → lower pps."""
    mid = (low + high) / 2
    for _ in range(50):
        mid = (low + high) / 2
        mean_pps, mean_bpp = eval_english(sentences, mid, score_idx, monotonicity)
        if abs(mean_pps - target_pps) < tolerance:
            return mid, mean_pps, mean_bpp
        if mean_pps > target_pps:
            low = mid
        else:
            high = mid
        if high - low < 1e-6:
            break
    mean_pps, mean_bpp = eval_english(sentences, mid, score_idx, monotonicity)
    return mid, mean_pps, mean_bpp


def binary_search_bpp(sentences, target_bpp, score_idx, monotonicity, low, high, tolerance=0.01):
    """Binary search for threshold giving target_bpp. Higher threshold → higher bpp."""
    mid = (low + high) / 2
    for _ in range(50):
        mid = (low + high) / 2
        mean_pps, mean_bpp = eval_english(sentences, mid, score_idx, monotonicity)
        if abs(mean_bpp - target_bpp) < tolerance:
            return mid, mean_pps, mean_bpp
        if mean_bpp < target_bpp:
            low = mid
        else:
            high = mid
        if high - low < 1e-6:
            break
    mean_pps, mean_bpp = eval_english(sentences, mid, score_idx, monotonicity)
    return mid, mean_pps, mean_bpp


def binary_search_t_for_combined(sentences, target_bpp, score_idx, t_add, low, high, tolerance=0.01):
    """Binary search over t (entropy threshold) in combined mode at fixed t_add."""
    mid = (low + high) / 2
    for _ in range(50):
        mid = (low + high) / 2
        mean_pps, mean_bpp = eval_english_combined(sentences, mid, t_add, score_idx)
        if abs(mean_bpp - target_bpp) < tolerance:
            return mid, mean_pps, mean_bpp
        if mean_bpp < target_bpp:
            low = mid
        else:
            high = mid
        if high - low < 1e-6:
            break
    mean_pps, mean_bpp = eval_english_combined(sentences, mid, t_add, score_idx)
    return mid, mean_pps, mean_bpp


def binary_search_t_add(sentences, target, score_idx, t, low, high, mode="bpp", tolerance=0.05):
    """Binary search over t_add at fixed t. mode='bpp' or 'pps'."""
    mid = (low + high) / 2
    for _ in range(50):
        mid = (low + high) / 2
        mean_pps, mean_bpp = eval_english_combined(sentences, t, mid, score_idx)
        value = mean_bpp if mode == "bpp" else mean_pps
        if abs(value - target) < tolerance:
            return mid, mean_pps, mean_bpp
        if mode == "bpp":
            if mean_bpp < target:
                low = mid   # higher t_add → fewer boundaries → higher bpp
            else:
                high = mid
        else:  # pps: higher t_add → fewer boundaries → lower pps
            if mean_pps > target:
                low = mid
            else:
                high = mid
        if high - low < 1e-6:
            break
    mean_pps, mean_bpp = eval_english_combined(sentences, t, mid, score_idx)
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

    # ── standard cases ────────────────────────────────────────────────────────
    for case_name, case in CASES.items():
        print(f"── {case_name} ──")
        idx   = case["score_idx"]
        mono  = case["monotonicity"]
        slow  = case["search_low"]
        shigh = case["search_high"]

        t_low, pps_low, bpp_low = binary_search_bpp(
            sentences, BPP_LOW, idx, mono, slow, shigh)
        print(f"  Lower bound  (bpp≈{BPP_LOW}): threshold={t_low:.4f}  pps={pps_low:.2f}  bpp={bpp_low:.4f}")

        t_high, pps_high, bpp_high = binary_search_bpp(
            sentences, BPP_HIGH, idx, mono, slow, shigh)
        print(f"  Upper bound  (bpp≈{BPP_HIGH}): threshold={t_high:.4f}  pps={pps_high:.2f}  bpp={bpp_high:.4f}")

        t_mid = (t_low + t_high) / 2
        pps_mid, bpp_mid = eval_english(sentences, t_mid, idx, mono)
        print(f"  Midpoint                  : threshold={t_mid:.4f}  pps={pps_mid:.2f}  bpp={bpp_mid:.4f}")

        t_anchor, pps_anchor, bpp_anchor = binary_search_pps(
            sentences, TARGET_PPS, idx, mono, slow, shigh)
        print(f"  Anchor       (pps≈{TARGET_PPS}): threshold={t_anchor:.4f}  pps={pps_anchor:.2f}  bpp={bpp_anchor:.4f}")

        thresholds = sorted([t_low, t_mid, t_high, t_anchor])
        print(f"\n  Final 4 thresholds: {[f'{t:.4f}' for t in thresholds]}\n")

        summary_rows.append({
            "case":            case_name,
            "fixed_t":         "",
            "t_low":           t_low,   "pps_low":    pps_low,    "bpp_low":    bpp_low,
            "t_mid":           t_mid,   "pps_mid":    pps_mid,    "bpp_mid":    bpp_mid,
            "t_high":          t_high,  "pps_high":   pps_high,   "bpp_high":   bpp_high,
            "t_anchor":        t_anchor,"pps_anchor":  pps_anchor, "bpp_anchor": bpp_anchor,
            "final_thresholds": thresholds,
        })

    # ── combined mode ─────────────────────────────────────────────────────────
    print("── combined (raw entropy + monotonicity delta) ──")
    idx       = COMBINED["score_idx"]
    t_sl      = COMBINED["t_search_low"]
    t_sh      = COMBINED["t_search_high"]
    t_add_max = COMBINED["t_add_max"]

    # step 1: fix t so that bpp=3.5 at t_add=0 (lower bound)
    t_fixed, pps_lb, bpp_lb = binary_search_t_for_combined(
        sentences, BPP_LOW, idx, t_add=0.0, low=t_sl, high=t_sh)
    print(f"  Fixed t (bpp≈{BPP_LOW} at t_add=0): t={t_fixed:.4f}  pps={pps_lb:.2f}  bpp={bpp_lb:.4f}")

    # step 2: upper bound — find t_add giving bpp=5.5
    t_add_high, pps_ub, bpp_ub = binary_search_t_add(
        sentences, BPP_HIGH, idx, t_fixed, low=0.0, high=t_add_max, mode="bpp")
    print(f"  Upper bound  (bpp≈{BPP_HIGH}): t_add={t_add_high:.4f}  pps={pps_ub:.2f}  bpp={bpp_ub:.4f}")

    # step 3: midpoint
    t_add_mid = t_add_high / 2
    pps_mid, bpp_mid = eval_english_combined(sentences, t_fixed, t_add_mid, idx)
    print(f"  Midpoint                  : t_add={t_add_mid:.4f}  pps={pps_mid:.2f}  bpp={bpp_mid:.4f}")

    # step 4: anchor — find t_add giving 32.77 pps
    t_add_anchor, pps_anchor, bpp_anchor = binary_search_t_add(
        sentences, TARGET_PPS, idx, t_fixed, low=0.0, high=t_add_max, mode="pps")
    print(f"  Anchor       (pps≈{TARGET_PPS}): t_add={t_add_anchor:.4f}  pps={pps_anchor:.2f}  bpp={bpp_anchor:.4f}")

    t_adds = sorted([0.0, t_add_mid, t_add_high, t_add_anchor])
    print(f"\n  Fixed t={t_fixed:.4f}, final 4 t_add values: {[f'{t:.4f}' for t in t_adds]}\n")

    summary_rows.append({
        "case":             "combined",
        "fixed_t":          t_fixed,
        "t_low":            0.0,         "pps_low":    pps_lb,      "bpp_low":    bpp_lb,
        "t_mid":            t_add_mid,   "pps_mid":    pps_mid,     "bpp_mid":    bpp_mid,
        "t_high":           t_add_high,  "pps_high":   pps_ub,      "bpp_high":   bpp_ub,
        "t_anchor":         t_add_anchor,"pps_anchor":  pps_anchor,  "bpp_anchor": bpp_anchor,
        "final_thresholds": t_adds,
    })

    # ── save summary ──────────────────────────────────────────────────────────
    out = os.path.join(OUTPUT_DIR, "thresholds_summary.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case", "fixed_t",
            "t_low",    "pps_low",    "bpp_low",
            "t_mid",    "pps_mid",    "bpp_mid",
            "t_high",   "pps_high",   "bpp_high",
            "t_anchor", "pps_anchor", "bpp_anchor",
            "final_thresholds",
        ])
        for r in summary_rows:
            writer.writerow([
                r["case"],
                f"{r['fixed_t']:.4f}" if r["fixed_t"] != "" else "",
                f"{r['t_low']:.4f}",    f"{r['pps_low']:.2f}",    f"{r['bpp_low']:.4f}",
                f"{r['t_mid']:.4f}",    f"{r['pps_mid']:.2f}",    f"{r['bpp_mid']:.4f}",
                f"{r['t_high']:.4f}",   f"{r['pps_high']:.2f}",   f"{r['bpp_high']:.4f}",
                f"{r['t_anchor']:.4f}", f"{r['pps_anchor']:.2f}", f"{r['bpp_anchor']:.4f}",
                str([f"{t:.4f}" for t in r["final_thresholds"]]),
            ])
    print(f"Summary saved → {out}")


if __name__ == "__main__":
    main()