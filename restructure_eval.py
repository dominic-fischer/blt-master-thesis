"""
restructure_eval.py

Converts existing results/eval/{lang_code}.json files to the new structure:

{
  "id": 0,
  "text": "...",
  "text_en": "...",
  "n_bytes": 44,
  "bytes_entropies": [[byte, entropy, entropy_norm], ...],
  "eval_modes": {
    "entropy": {
      "t_1.75": {
        "n_patches": 12,
        "avg_bytes_per_patch": 4.3,
        "patch_lengths": [3, 5, 2, ...]
      }
    }
  }
}

Output: results/restructured/{lang_code}.json

Usage:
    python restructure_eval.py
"""

import json
import os
import torch
from pathlib import Path
from tqdm import tqdm


# ── config ────────────────────────────────────────────────────────────────────
INPUT_DIR  = "results/eval"
OUTPUT_DIR = "results/restructured"

EXISTING_MODE      = "entropy"
EXISTING_THRESHOLD = 1.75


# ── helpers ───────────────────────────────────────────────────────────────────

def normalize_scores(scores: list[float]) -> list[float]:
    arr = torch.tensor(scores)
    mean = arr.mean()
    std = arr.std()
    return ((arr - mean) / (std + 1e-8)).tolist()


def restructure_sentence(sentence: dict) -> dict:
    text_bytes = sentence["text_bytes"]
    scores     = sentence["scores"]
    norm       = normalize_scores(scores)

    bytes_entropies = [
        [b, round(e, 6), round(en, 6)]
        for b, e, en in zip(text_bytes, scores, norm)
    ]

    patch_lengths = [p["length"] for p in sentence["patches"]]
    n_patches     = len(patch_lengths)
    n_bytes       = sentence["n_bytes"]
    avg_bpp       = n_bytes / max(n_patches, 1)

    return {
        "id":              sentence["id"],
        "text":            sentence["text"],
        "text_en":         sentence["eng_text"],
        "n_bytes":         n_bytes,
        "bytes_entropies": bytes_entropies,
        "eval_modes": {
            EXISTING_MODE: {
                f"t_{EXISTING_THRESHOLD}": {
                    "n_patches":           n_patches,
                    "avg_bytes_per_patch": round(avg_bpp, 4),
                    "patch_lengths":       patch_lengths,
                }
            }
        }
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_files = sorted(Path(INPUT_DIR).glob("*.json"))
    print(f"Found {len(input_files)} language files in {INPUT_DIR}")

    for path in input_files:
        lang_code = path.stem
        with open(path, encoding="utf-8") as f:
            sentences = json.load(f)

        restructured = [
            restructure_sentence(s)
            for s in tqdm(sentences, desc=lang_code, leave=False)
        ]

        out_path = os.path.join(OUTPUT_DIR, f"{lang_code}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(restructured, f, ensure_ascii=False, indent=2)

        print(f"  {lang_code}: {len(restructured)} sentences → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()