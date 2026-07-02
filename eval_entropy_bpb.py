"""
eval_entropy_bpb.py

Computes held-out bits-per-byte (bpb) for a trained entropy-model checkpoint,
per language and overall, using the *.val.jsonl files that
prepare_language_shards.py set aside (excluded from training).

Tokenization matches training exactly (see bytelatent/tokenizers/blt_tokenizer.py):
    token_id = byte_value + OFFSET(4), BOS_ID=1 prepended, EOS_ID=2 appended.

Usage:
    python eval_entropy_bpb.py \
        <consolidated_checkpoint_dir>  \
        <lang_shards_root_dir> \
        [--max-docs-per-lang 200] [--device cuda]

Example:
    python eval_entropy_bpb.py \
        /local/scratch/dfische/blt/dumps/entropy_medium_20lang/checkpoints/0000000782/consolidated \
        /local/scratch/dfische/blt/data/lang_shards
"""

import argparse
import glob
import json
import math
import os

import torch
import torch.nn.functional as F

from bytelatent.entropy_model import load_entropy_model

OFFSET = 4
BOS_ID = 1
EOS_ID = 2


def encode_bytes(text: str) -> list[int]:
    raw = text.encode("utf-8", errors="ignore")
    tokens = [b + OFFSET for b in raw]
    return [BOS_ID] + tokens + [EOS_ID]


@torch.no_grad()
def eval_language(model, val_path: str, device: str, max_docs: int) -> tuple[float, int]:
    """Returns (total_nats, total_bytes) across up to max_docs documents."""
    total_nats = 0.0
    total_bytes = 0
    with open(val_path) as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            doc = json.loads(line)
            text = doc.get("text", "")
            if not text:
                continue
            tokens = encode_bytes(text)
            if len(tokens) < 2:
                continue
            x = torch.tensor(tokens[:-1], device=device).unsqueeze(0)
            y = torch.tensor(tokens[1:], device=device).unsqueeze(0)

            logits = model(x)
            loss = F.cross_entropy(
                logits.float().flatten(0, 1), y.flatten(0, 1), reduction="sum"
            )
            total_nats += loss.item()
            # count only the original text bytes (exclude BOS/EOS from the
            # denominator, matching how training's n_bytes is computed)
            total_bytes += len(text.encode("utf-8", errors="ignore"))
    return total_nats, total_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir",
                         help="consolidated checkpoint dir (contains consolidated.pth + params.json)")
    parser.add_argument("lang_shards_root")
    parser.add_argument("--max-docs-per-lang", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    state_dict_path = os.path.join(args.checkpoint_dir, "consolidated.pth")
    model, model_args = load_entropy_model(
        args.checkpoint_dir, state_dict_path, device=args.device
    )
    print(f"Loaded model: dim={model_args.dim} n_layers={model_args.n_layers} "
          f"n_heads={model_args.n_heads}")

    val_files = sorted(glob.glob(os.path.join(args.lang_shards_root, "*", "*.val.jsonl")))
    if not val_files:
        print(f"No *.val.jsonl files found under {args.lang_shards_root}")
        return

    grand_total_nats = 0.0
    grand_total_bytes = 0
    results = []
    for val_path in val_files:
        language_code = os.path.basename(os.path.dirname(val_path))
        nats, n_bytes = eval_language(model, val_path, args.device, args.max_docs_per_lang)
        if n_bytes == 0:
            print(f"  {language_code}: no bytes evaluated, skipping")
            continue
        bpb = nats / math.log(2) / n_bytes
        results.append((language_code, bpb, n_bytes))
        grand_total_nats += nats
        grand_total_bytes += n_bytes

    print(f"\n{'language':<12} {'bpb':>8} {'val_bytes':>12}")
    print("-" * 34)
    for language_code, bpb, n_bytes in sorted(results, key=lambda r: r[1], reverse=True):
        print(f"{language_code:<12} {bpb:>8.4f} {n_bytes:>12,}")

    overall_bpb = grand_total_nats / math.log(2) / grand_total_bytes
    print("-" * 34)
    print(f"{'OVERALL':<12} {overall_bpb:>8.4f} {grand_total_bytes:>12,}")


if __name__ == "__main__":
    main()