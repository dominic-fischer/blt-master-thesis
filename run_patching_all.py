"""
run_patching.py

Computes patch lengths for all cases and thresholds and stores results
in the restructured JSON files under eval_modes.

Cases and thresholds (derived from find_english_bounds_all.py):
  raw_entropy:       [1.1904, 1.3340, 1.4946, 1.7988]
  raw_monotonicity:  [0.2359, 0.3662, 0.5928, 0.9496]
  norm_entropy:      [0.4072, 0.5293, 0.7222, 1.0371]
  norm_monotonicity: [0.2480, 0.3887, 0.6260, 1.0039]
  raw_combined:      fixed threshold=1.3340, threshold_add in [0.2359, 0.3662, 0.5928, 0.9496]
  norm_combined:     fixed threshold=0.5293, threshold_add in [0.2480, 0.3887, 0.6260, 1.0039]

Output: updates results/restructured/{lang_code}.json in place

Usage:
    python run_patching.py
"""

import json
import os
import torch
from pathlib import Path
from tqdm import tqdm
from bytelatent.data.patcher import (
    find_entropy_patch_start_ids,
    patch_lengths_from_start_ids,
)

# ── config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results/restructured"

# only combined cases — the rest are already computed
CASES = {
    "raw_combined": {
        "score_idx":       1,
        "monotonicity":    False,
        "fixed_threshold": 1.3340,
        "thresholds":      [0.2359, 0.3662, 0.5928, 0.9496],  # these are threshold_add values
    },
    "norm_combined": {
        "score_idx":       2,
        "monotonicity":    False,
        "fixed_threshold": 0.5293,
        "thresholds":      [0.2480, 0.3887, 0.6260, 1.0039],
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

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

def main():
    paths = sorted(Path(RESULTS_DIR).glob("*.json"))
    print(f"Found {len(paths)} language files")
    print(f"Cases: {list(CASES.keys())} (combined only)")
    print()

    for path in paths:
        lang_code = path.stem
        with open(path, encoding="utf-8") as f:
            sentences = json.load(f)

        for sentence in tqdm(sentences, desc=lang_code, leave=False):
            if "eval_modes" not in sentence:
                sentence["eval_modes"] = {}

            for case_name, case in CASES.items():
                if case_name not in sentence["eval_modes"]:
                    sentence["eval_modes"][case_name] = {}

                scores = [be[case["score_idx"]] for be in sentence["bytes_entropies"]]

                for threshold_add in case["thresholds"]:
                    key = threshold_key(threshold_add)
                    if key in sentence["eval_modes"][case_name]:
                        continue  # already computed, skip

                    lengths = compute_patch_lengths(
                        scores,
                        threshold=case["fixed_threshold"],
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
            json.dump(sentences, f, ensure_ascii=False)

        print(f"  {lang_code}: done")

    print("\nDone.")


if __name__ == "__main__":
    main()