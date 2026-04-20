"""
eval_flores.py
Run BLT patching over a diverse subset of FLORES+ languages and save results to JSON.

Usage:
    python eval_flores.py

Output: one JSON file per language at results/{lang_code}.json
"""

import json
import os
from datasets import load_dataset
from tqdm import tqdm

from blt_patcher import load_patcher, patch_text


# ── config ────────────────────────────────────────────────────────────────────
LIMIT = 500 # set to an int for quick testing

REPO = "facebook/blt-1b"
ENTROPY_REPO = "hf-weights/entropy_model"
SPLIT = "dev"
OUTPUT_DIR = "results"

FLORES_DATASET = "openlanguagedata/flores_plus"

LANGUAGES = {
    "eng_Latn": "English",
    "deu_Latn": "German",
    "nya_Latn": "Chichewa",
    "fra_Latn": "French",
    "ron_Latn": "Romanian",
    "rus_Cyrl": "Russian",
    "arb_Arab": "Arabic",
    "cmn_Hans": "Chinese (Simplified)",
    "cmn_Hant": "Chinese (Traditional)",
    "hin_Deva": "Hindi",
    "tha_Thai":  "Thai",
    "kor_Hang": "Korean",
    "tam_Taml": "Tamil",
    "shn_Mymr": "Shan",
    "kas_Arab": "Kashmiri (Arabic)",
    "kas_Deva": "Kashmiri (Devanagari)",
    "min_Arab": "Minangkabau (Arabic)",
    "min_Latn": "Minangkabau (Latin)",
    "hrv_Latn": "Croatian",
    "srp_Cyrl": "Serbian",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_language(lang_code: str):
    return load_dataset(FLORES_DATASET, lang_code, split=SPLIT)


def run_language(lang_code: str, lang_name: str, eng_dataset, tokenizer, patcher):
    print(f"\nProcessing {lang_name} ({lang_code})...")

    dataset = eng_dataset if lang_code == "eng_Latn" else load_language(lang_code)

    assert len(dataset) == len(eng_dataset), (
        f"Length mismatch for {lang_code}: {len(dataset)} vs {len(eng_dataset)}"
    )

    results = []
    for i, (row, eng_row) in enumerate(
        tqdm(zip(dataset, eng_dataset), total=len(dataset), desc=lang_name)
    ):
        if LIMIT is not None and i >= LIMIT:
            break

        text = row["text"]
        eng_text = eng_row["text"]
        result = patch_text(text, tokenizer, patcher)

        results.append({
            "id": i,
            "text": text,
            "text_bytes": result["text_bytes"],
            "eng_text": eng_text,
            "n_patches": result["n_patches"],
            "n_bytes": result["n_bytes"],
            "avg_bytes_per_patch": result["avg_bytes_per_patch"],
            "patches": [
                {"text": p[0], "bytes": p[1], "length": p[2]}
                for p in result["patches"]
            ],
            "scores": result["scores"],
        })

    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading patcher...")
    tokenizer, patcher = load_patcher(repo=REPO, entropy_repo=ENTROPY_REPO)

    print(f"Loading English ({SPLIT})...")
    eng_dataset = load_language("eng_Latn")

    for lang_code, lang_name in LANGUAGES.items():
        results = run_language(lang_code, lang_name, eng_dataset, tokenizer, patcher)

        out_path = os.path.join(OUTPUT_DIR, f"{lang_code}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(results)} entries → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()