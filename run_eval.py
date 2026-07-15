"""
run_eval.py
Run BLT patching over a diverse subset of FLORES+ languages and save results to JSON.
Usage:
    python run_eval.py --entropy_repo dumps/entropy_10M_20lang_4gpu_sourcesbalanced_steps3000_ckpt200_lr4.5e-3/checkpoints/0000003000/consolidated
Output: one JSON file per language at
    results/own_models/<run_name>/step_<step>/{lang_code}.json
where <run_name> is the path component immediately after "dumps/" in
--entropy_repo (e.g. "entropy_10M_20lang_4gpu_sourcesbalanced_steps3000_ckpt200_lr4.5e-3")
and <step> is the checkpoint number (from "checkpoints/<step>/..." in the
path, if present) -- so results from different runs AND different
checkpoints of the same run don't silently overwrite each other under one
flat directory.
"""
import argparse
import csv
import json
import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import subprocess
from blt_patcher import load_patcher, patch_text

# ── config ────────────────────────────────────────────────────────────────────
LIMIT = None  # set to an int for quick testing
REPO = "facebook/blt-1b"
SPLIT = "dev"
OUTPUT_BASE_DIR = "results/own_models"
FLORES_DATASET = "openlanguagedata/flores_plus"

# read the languages from floresplus_MASTER_CSV.csv, like this: Code_Orig : Name
with open("floresplus_MASTER_CSV.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    LANGUAGES = {row["Code_Orig"]: row["Name"] for row in reader}


# ── helpers ───────────────────────────────────────────────────────────────────
def derive_run_subdir(entropy_repo: str) -> str:
    """Uses the path component immediately after "dumps/" in entropy_repo
    as the results subdirectory (e.g.
    "dumps/entropy_10M_..._lr4.5e-3/checkpoints/0000003000/consolidated"
    -> "entropy_10M_..._lr4.5e-3"), so different runs' FLORES+ results
    land in their own folder instead of overwriting each other under one
    flat results/own_models/ directory.

    If a "checkpoints/<step>" component is also present, that step number
    becomes a further subdirectory (e.g. ".../entropy_10M_..._lr4.5e-3/
    step_0000003000/"), so different checkpoints of the SAME run don't
    overwrite each other either -- otherwise evaluating checkpoint 2000
    and then checkpoint 3000 of one run would land in the same folder.

    Falls back to a sanitized version of the whole repo string if "dumps"
    doesn't appear in the path at all -- e.g. a bare HF repo id like
    "hf_weights/entropy_model", which isn't tied to any particular local
    training run or checkpoint."""
    parts = os.path.normpath(entropy_repo).split(os.sep)

    if "dumps" not in parts:
        fallback = entropy_repo.strip(os.sep).replace(os.sep, "_")
        print(f"  NOTE: no 'dumps/<run_name>' component found in "
              f"--entropy_repo={entropy_repo!r} -- using sanitized fallback "
              f"subdirectory name {fallback!r} instead.")
        return fallback

    dumps_idx = parts.index("dumps")
    if dumps_idx + 1 >= len(parts) or not parts[dumps_idx + 1]:
        fallback = entropy_repo.strip(os.sep).replace(os.sep, "_")
        print(f"  NOTE: 'dumps' has no run-name component after it in "
              f"--entropy_repo={entropy_repo!r} -- using sanitized fallback "
              f"subdirectory name {fallback!r} instead.")
        return fallback
    run_name = parts[dumps_idx + 1]

    subdir = run_name
    if "checkpoints" in parts:
        ckpt_idx = parts.index("checkpoints")
        if ckpt_idx + 1 < len(parts) and parts[ckpt_idx + 1]:
            subdir = os.path.join(run_name, f"step_{parts[ckpt_idx + 1]}")

    return subdir


def normalize_scores(scores: list[float]) -> list[float]:
    arr = torch.tensor(scores, dtype=torch.float32)  # force fp16 here, not on the model
    mean = arr.mean()
    std = arr.std()
    return ((arr - mean) / (std + 1e-8)).tolist()


def load_language(lang_code: str):
    return load_dataset(FLORES_DATASET, lang_code, split=SPLIT)


def run_language(lang_code: str, lang_name: str, eng_dataset, tokenizer, patcher):
    print(f"\nProcessing {lang_name} ({lang_code})...")
    if lang_code == "eng_Latn":
        dataset = eng_dataset
    else:
        try:
            dataset = load_language(lang_code)
        except Exception as e:
            print(f"  Skipping: {e}")
            return None

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
        text_bytes = result["text_bytes"]
        scores = result["scores"]
        norm = normalize_scores(scores)
        bytes_entropies = [
            [b, round(e, 6), round(en, 6)]
            for b, e, en in zip(text_bytes, scores, norm)
        ]
        results.append({
            "id": i,
            "text": text,
            "text_en": eng_text,
            "n_bytes": result["n_bytes"],
            "bytes_entropies": bytes_entropies
        })
    return results


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BLT patching over FLORES+ languages."
    )
    parser.add_argument(
        "--entropy_repo",
        type=str,
        required=True,
        help="Repo id (or path) of the entropy model to use for patching, "
             "e.g. --entropy_repo dumps/entropy_10M_..._lr4.5e-3/checkpoints/0000003000/consolidated",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    run_subdir = derive_run_subdir(args.entropy_repo)
    output_dir = os.path.join(OUTPUT_BASE_DIR, run_subdir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results under: {output_dir}/")

    print("Loading patcher...")
    tokenizer, patcher = load_patcher(repo=REPO, entropy_repo=args.entropy_repo)
    print(f"Loading English ({SPLIT})...")
    eng_dataset = load_language("eng_Latn")

    for lang_code, lang_name in LANGUAGES.items():
        results = run_language(lang_code, lang_name, eng_dataset, tokenizer, patcher)
        if results is None:
            continue
        out_path = os.path.join(output_dir, f"{lang_code}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(results)} entries → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()