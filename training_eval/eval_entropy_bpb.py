"""
eval_entropy_bpb.py

Computes held-out bits-per-byte (bpb) for a trained entropy-model checkpoint,
per language and overall, using the *.val.jsonl files that
prepare_language_shards.py set aside (excluded from training).

Also, if a metrics.jsonl path is given (or auto-found next to the
checkpoint's dump_dir), looks up the TRAINING loss/bpb logged at this exact
step and prints it alongside the held-out numbers -- this is the cheap way
to distinguish overfitting (train improving, held-out worsening) from an
optimization problem (both worsening together, e.g. an LR schedule mismatched
to a short run's total step count). No extra forward passes needed since
train.py already logs this at every step.

Evaluation depth: each language is evaluated on EXACTLY the same number of
bytes (--target-bytes-per-lang, default 400,000), not a fixed document
count. If the document that would cross the threshold is longer than the
remaining budget, it's truncated mid-document to hit the target exactly
(byte-level models have no notion of a "valid" truncation boundary, so
this is safe). This matters because the underlying training corpus is
intentionally byte-imbalanced across languages (equal *content* per
language, via add_language_allocations.py, means byte-heavy languages like
Tamil/Georgian have more raw bytes for the same content) -- a fixed
*document* count would let long-document languages dominate the aggregate
OVERALL bpb purely by accident, unrelated to how good the model actually
is at that language. With every language contributing exactly the same
byte count, OVERALL becomes a genuine equal-weight macro-average across
languages ("how good is the model at each language, treated as equally
important"), rather than an artifact of whatever happened to be in the
first N validation documents. Note this is a deliberate DIFFERENT question
than training-time bpb, which is weighted by data.sources (proportional to
shard byte-size, which itself already encodes equal content) -- so don't
expect this OVERALL to track TRAIN bpb as closely as a data.sources-
weighted aggregate would. It's a fairness / per-language-competence
metric, not a training-consistency check.

If a language's val.jsonl doesn't contain target_bytes worth of data at
all, it's flagged explicitly in the output (see short_languages) rather
than silently producing a smaller, non-comparable sample for that language
alone.

Tokenization matches training exactly (see bytelatent/tokenizers/blt_tokenizer.py):
    token_id = byte_value + OFFSET(4), BOS_ID=1 prepended, EOS_ID=2 appended.

Usage:
    python eval_entropy_bpb.py \
        <consolidated_checkpoint_dir>  \
        <lang_shards_root_dir> \
        [--target-bytes-per-lang 400000] [--device cuda] [--metrics-jsonl PATH]

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


def find_train_metrics_at_step(metrics_jsonl_path: str, target_step: int) -> dict | None:
    """Returns the metrics.jsonl entry at or nearest-below target_step,
    or None if the file doesn't exist / has no entries <= target_step."""
    if not os.path.exists(metrics_jsonl_path):
        return None
    best = None
    with open(metrics_jsonl_path) as f:
        for line in f:
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = m.get("global_step")
            if step is None or step > target_step:
                continue
            if best is None or step > best.get("global_step", -1):
                best = m
    return best


def infer_step_from_checkpoint_dir(checkpoint_dir: str) -> int | None:
    """checkpoint_dir is typically <dump_dir>/checkpoints/<step>/consolidated --
    extract the zero-padded step number from the parent directory name."""
    parent = os.path.basename(os.path.dirname(os.path.normpath(checkpoint_dir)))
    try:
        return int(parent)
    except ValueError:
        return None


@torch.no_grad()
def eval_language(model, val_path: str, device: str, target_bytes: int,
                   max_seqlen: int) -> tuple[float, int, bool]:
    """Returns (total_nats, total_bytes, hit_target) across documents read
    from val_path, stopping at EXACTLY target_bytes -- if the document that
    would cross the threshold is longer than the remaining budget, only the
    leading `remaining` bytes of it are fed to the model (byte-level models
    have no notion of "valid" byte boundaries, so slicing mid-document is
    fine). hit_target is False only if the file runs out of documents
    before reaching target_bytes -- the caller should flag this, since it
    breaks the equal-bytes-per-language comparison.
    Documents are also truncated to max_seqlen tokens (including BOS/EOS)
    since the model's RoPE table is only precomputed up to that length --
    whichever of the two limits (remaining byte budget vs. max_seqlen) is
    smaller wins for a given document."""
    total_nats = 0.0
    total_bytes = 0
    with open(val_path) as f:
        for line in f:
            if total_bytes >= target_bytes:
                return total_nats, total_bytes, True
            doc = json.loads(line)
            text = doc.get("text", "")
            if not text:
                continue
            raw = text.encode("utf-8", errors="ignore")
            remaining = target_bytes - total_bytes
            n = min(len(raw), remaining, max_seqlen - 2)
            if n < 1:
                continue
            raw = raw[:n]
            tokens = [BOS_ID] + [b + OFFSET for b in raw] + [EOS_ID]
            x = torch.tensor(tokens[:-1], device=device).unsqueeze(0)
            y = torch.tensor(tokens[1:], device=device).unsqueeze(0)

            logits = model(x)
            loss = F.cross_entropy(
                logits.float().flatten(0, 1), y.flatten(0, 1), reduction="sum"
            )
            total_nats += loss.item()
            total_bytes += n  # exact -- matches what was actually fed in
    # ran out of documents before reaching target_bytes
    return total_nats, total_bytes, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir",
                         help="consolidated checkpoint dir (contains consolidated.pth + params.json)")
    parser.add_argument("lang_shards_root")
    parser.add_argument("--target-bytes-per-lang", type=int, default=150_000,
                         help="Evaluate exactly this many bytes per language "
                              "(not a fixed document count), so every "
                              "language contributes equally to OVERALL -- "
                              "a genuine macro-average across languages, "
                              "rather than being skewed by whichever "
                              "language's validation documents happen to be "
                              "longest.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                         help="Some internal model buffers are hardcoded to cuda "
                              "(inherited from the repo's inference code), so cpu "
                              "eval fails with a device mismatch. Use cuda when no "
                              "training run is competing for GPU memory.")
    parser.add_argument("--metrics-jsonl", default=None,
                         help="Path to the run's metrics.jsonl, to look up "
                              "training loss/bpb at this checkpoint's exact "
                              "step for comparison. If omitted, auto-derived "
                              "as <dump_dir>/metrics.jsonl assuming the "
                              "standard checkpoint_dir layout "
                              "(<dump_dir>/checkpoints/<step>/consolidated).")
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
    short_languages = []  # languages that ran out of val data before target_bytes
    for val_path in val_files:
        language_code = os.path.basename(os.path.dirname(val_path))
        nats, n_bytes, hit_target = eval_language(
            model, val_path, args.device, args.target_bytes_per_lang, model_args.max_seqlen
        )
        if n_bytes == 0:
            print(f"  {language_code}: no bytes evaluated, skipping")
            continue
        if not hit_target:
            short_languages.append((language_code, n_bytes))
        bpb = nats / math.log(2) / n_bytes
        results.append((language_code, bpb, n_bytes))
        grand_total_nats += nats
        grand_total_bytes += n_bytes

    print(f"\n{'language':<12} {'val_bpb':>8} {'val_bytes':>12}")
    print("-" * 34)
    for language_code, bpb, n_bytes in sorted(results, key=lambda r: r[1], reverse=True):
        print(f"{language_code:<12} {bpb:>8.4f} {n_bytes:>12,}")

    overall_bpb = grand_total_nats / math.log(2) / grand_total_bytes
    print("-" * 34)
    print(f"{'OVERALL':<12} {overall_bpb:>8.4f} {grand_total_bytes:>12,}")
    print(f"  (equal-weight macro-average: every language above contributed "
          f"{args.target_bytes_per_lang:,} bytes"
          f"{'' if not short_languages else ', except where noted below'})")

    if short_languages:
        print(f"\nWARNING: {len(short_languages)} language(s) ran out of "
              f"validation data before reaching {args.target_bytes_per_lang:,} "
              f"bytes -- OVERALL above is not a perfectly equal-weight "
              f"average for these:")
        for language_code, n_bytes in short_languages:
            print(f"  {language_code}: only {n_bytes:,} bytes available "
                  f"(val.jsonl exhausted)")

    # Training-time loss/bpb at this exact step, for direct comparison --
    # pulled from train.py's own logging, not recomputed (cheap, no extra
    # forward passes over training data needed). NOTE: training bpb is
    # weighted by data.sources (~ shard byte-size, i.e. NOT equal-weight
    # across languages), so it will not necessarily track this script's
    # OVERALL closely -- they're intentionally answering different
    # questions (see module docstring).
    metrics_path = args.metrics_jsonl
    if metrics_path is None:
        checkpoints_dir = os.path.dirname(os.path.normpath(args.checkpoint_dir))
        dump_dir = os.path.dirname(os.path.dirname(os.path.normpath(checkpoints_dir)))
        metrics_path = os.path.join(dump_dir, "metrics.jsonl")

    step = infer_step_from_checkpoint_dir(args.checkpoint_dir)
    train_metrics = find_train_metrics_at_step(metrics_path, step) if step is not None else None

    if train_metrics is not None:
        train_loss = train_metrics.get("loss/interval_across_gpu")
        train_bpb = train_metrics.get("bpb/interval_across_gpus")
        matched_step = train_metrics.get("global_step")
        bpb_str = f"{train_bpb:.4f}" if train_bpb is not None else "n/a"
        loss_str = f"{train_loss:.4f}" if train_loss is not None else "n/a"
        print(f"\n{'TRAIN (step ' + str(matched_step) + ')':<20} "
              f"bpb={bpb_str}   loss={loss_str}")
        if matched_step != step:
            print(f"  (nearest logged step <= {step}; training logs at "
                  f"logging.freq intervals, not necessarily every step)")
    else:
        print(f"\n(no training metrics found at step {step} in {metrics_path} "
              f"-- pass --metrics-jsonl explicitly if it's elsewhere)")


if __name__ == "__main__":
    main()