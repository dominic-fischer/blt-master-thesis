"""
calibrate_threshold.py

Applies Meta's calibrated threshold (1.335) to all FLORES languages and reports:
  - bytes-per-patch (bpp) per language
  - patches-per-sentence (pps) per language

English pps is taken as the fairness reference target: since Meta calibrated
on English-dominant data and we observe ~3.9 bpp on English FLORES (close to
their 4.5 target), we assume their threshold is well-calibrated for English.
A fair threshold would produce the same pps across all languages.

Output:
    calibration/meta_threshold_results.csv   — per-language bpp and pps
    calibration/meta_threshold_summary.txt   — summary statistics

Usage:
    python calibrate_threshold.py
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
RESULTS_DIR      = "results/restructured"
OUTPUT_DIR       = "calibration"
META_THRESHOLD   = 1.335442066192627
REFERENCE_LANG   = "eng_Latn"


# ── helpers ───────────────────────────────────────────────────────────────────

def compute_patch_lengths(scores: list[float], threshold: float) -> list[int]:
    """
    Run entropy patching (global threshold, no monotonicity) with BOS sentinel.
    Returns patch lengths as a list of positive ints.
    """
    bos_score = 99.0
    scores_with_bos = torch.tensor([[bos_score] + scores])
    patch_start_ids = find_entropy_patch_start_ids(
        scores_with_bos,
        threshold=threshold,
        threshold_add=None,
        monotonicity=False,
    )
    n_bytes = len(scores)
    patch_lengths = patch_lengths_from_start_ids(patch_start_ids, n_bytes + 1)
    return [l for l in patch_lengths[0].tolist()[1:] if l > 0]


def load_results(results_dir: str) -> dict:
    data = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        lang_code = path.stem
        with open(path, encoding="utf-8") as f:
            data[lang_code] = json.load(f)
    return data


def compute_lang_stats(sentences: list[dict], threshold: float) -> dict:
    """
    Returns mean bpp and mean pps for one language.
    """
    all_bpp = []
    all_pps = []
    for s in sentences:
        scores = [be[1] for be in s["bytes_entropies"]]
        lengths = compute_patch_lengths(scores, threshold)
        n_patches = len(lengths)
        n_bytes = sum(lengths)
        all_pps.append(n_patches)
        all_bpp.append(n_bytes / n_patches if n_patches > 0 else 0.0)
    return {
        "mean_bpp": sum(all_bpp) / len(all_bpp) if all_bpp else 0.0,
        "mean_pps": sum(all_pps) / len(all_pps) if all_pps else 0.0,
    }


def variance(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading results from {RESULTS_DIR}...")
    all_data = load_results(RESULTS_DIR)
    lang_codes = sorted(all_data.keys())
    print(f"  {len(lang_codes)} languages found")
    print(f"  Threshold: {META_THRESHOLD}")
    print(f"  Reference language: {REFERENCE_LANG}\n")

    # ── compute stats per language ────────────────────────────────────────────
    results = {}
    for lang_code in lang_codes:
        stats = compute_lang_stats(all_data[lang_code], META_THRESHOLD)
        results[lang_code] = stats
        # print(f"  {lang_code:<30}  bpp={stats['mean_bpp']:.4f}  pps={stats['mean_pps']:.2f}")

    # ── reference: English PPS ────────────────────────────────────────────────
    ref_bpp = results[REFERENCE_LANG]["mean_bpp"]
    ref_pps = results[REFERENCE_LANG]["mean_pps"]
    print(f"\n  Reference ({REFERENCE_LANG}):")
    print(f"    mean bpp : {ref_bpp:.4f}  (Meta target: 4.5 on training data)")
    print(f"    mean pps : {ref_pps:.2f}  ← fairness target")

    # ── overall statistics ────────────────────────────────────────────────────
    all_bpp = [results[l]["mean_bpp"] for l in lang_codes]
    all_pps = [results[l]["mean_pps"] for l in lang_codes]

    mean_bpp_overall = sum(all_bpp) / len(all_bpp)
    mean_pps_overall = sum(all_pps) / len(all_pps)
    var_bpp = variance(all_bpp)
    var_pps = variance(all_pps)

    # pps deviation and premium vs English
    pps_deviations = {l: results[l]["mean_pps"] - ref_pps for l in lang_codes}
    pps_premium    = {l: results[l]["mean_pps"] / ref_pps for l in lang_codes}

    print(f"\n  Overall mean bpp : {mean_bpp_overall:.4f}  variance: {var_bpp:.4f}")
    print(f"  Overall mean pps : {mean_pps_overall:.2f}  variance: {var_pps:.4f}")

    # ── save CSV ──────────────────────────────────────────────────────────────
    out_csv = os.path.join(OUTPUT_DIR, "meta_threshold_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lang_code", "mean_bpp", "mean_pps", "pps_dev_from_english", "pps_premium_vs_english"])
        for lang_code in lang_codes:
            writer.writerow([
                lang_code,
                f"{results[lang_code]['mean_bpp']:.4f}",
                f"{results[lang_code]['mean_pps']:.4f}",
                f"{pps_deviations[lang_code]:.4f}",
                f"{pps_premium[lang_code]:.4f}",
            ])
        writer.writerow(["OVERALL_MEAN", f"{mean_bpp_overall:.4f}", f"{mean_pps_overall:.4f}", "", ""])
        writer.writerow(["OVERALL_VARIANCE", f"{var_bpp:.4f}", f"{var_pps:.4f}", "", ""])
        writer.writerow([f"REFERENCE ({REFERENCE_LANG})", f"{ref_bpp:.4f}", f"{ref_pps:.4f}", "0.0", "1.0"])
    print(f"\n  Saved → {out_csv}")

    # ── save summary ──────────────────────────────────────────────────────────
    out_txt = os.path.join(OUTPUT_DIR, "meta_threshold_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Meta threshold: {META_THRESHOLD}\n")
        f.write(f"Reference language: {REFERENCE_LANG}\n")
        f.write(f"Reference bpp: {ref_bpp:.4f} (Meta target: 4.5 on training data)\n")
        f.write(f"Reference pps: {ref_pps:.4f} (fairness target)\n\n")
        f.write(f"Overall mean bpp: {mean_bpp_overall:.4f}  variance: {var_bpp:.4f}\n")
        f.write(f"Overall mean pps: {mean_pps_overall:.4f}  variance: {var_pps:.4f}\n\n")
        f.write(f"Per-language pps premium vs English (sorted by premium):\n")
        for lang_code, premium in sorted(pps_premium.items(), key=lambda x: x[1]):
            f.write(f"  {lang_code:<30}  pps={results[lang_code]['mean_pps']:.2f}  premium={premium:.4f}x\n")
    print(f"  Saved → {out_txt}")

    print("\nDone.")


if __name__ == "__main__":
    main()