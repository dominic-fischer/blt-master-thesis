"""
english_fineweb_exact_bytes.py

Fixes the apples-to-apples problem in fineweb_byte_size.py: that script
measures COMPRESSED, ALL-COLUMNS parquet size on disk. Every other row in
langs_chosen.csv's utf8_bytes column is RAW, UNCOMPRESSED, TEXT-COLUMN-ONLY
bytes (as computed by FineWeb2 itself). This script measures the same thing
for English FineWeb, by actually streaming the `text` column and summing
len(text.encode('utf-8')) per document -- the same method verify_nyanja.py
uses, just at a scale where we may want to sample + extrapolate instead of
processing the entire dataset.

Two modes:

  --mode sample (default, fast, minutes)
      Streams the `sample-350BT` config (the largest official uniform
      random sample of the whole FineWeb corpus), measures real bytes for
      every document in it, then extrapolates to the full dataset using
      the EXACT total row count from the /size API. Not exact for the
      full corpus, but measured (not guessed) and comes with a
      statistically sound margin of error since it's a real random
      sample, not an assumption like bytes/token.

  --mode full (slow, hours-to-days depending on bandwidth)
      Streams every document in the `default` config (the entire
      dataset, ~5B+ documents / ~100 CommonCrawl dumps) and sums bytes
      directly. Fully exact. Supports checkpointing so it can be resumed
      if interrupted.

Requires:
    pip install datasets requests

Usage:
    python english_fineweb_exact_bytes.py --mode sample
    python english_fineweb_exact_bytes.py --mode full
    python english_fineweb_exact_bytes.py --mode full --resume
"""

import argparse
import json
import os
import sys
import time

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("Missing dependency. Install with:\n    pip install datasets --break-system-packages\n")

try:
    import requests
except ImportError:
    sys.exit("Missing dependency. Install with:\n    pip install requests --break-system-packages\n")

import csv

DATASET = "HuggingFaceFW/fineweb"
API_ROOT = "https://datasets-server.huggingface.co"

CHECKPOINT_FILE = "english_fineweb_bytes_checkpoint.json"
PROGRESS_EVERY = 100_000  # print + checkpoint every N docs


def get_exact_total_rows(dataset=DATASET, config="default", split="train"):
    """Exact total row count for a config, from the /size API (not a sample)."""
    r = requests.get(f"{API_ROOT}/size", params={"dataset": dataset}, timeout=120)
    r.raise_for_status()
    data = r.json()
    for c in data["size"]["configs"]:
        if c["config"] == config:
            return c["num_rows"]
    raise RuntimeError(f"config={config!r} not found in /size response for {dataset}")


def stream_and_sum_bytes(config, split="train", limit=None,
                          checkpoint_path=None, resume=False):
    """
    Stream a config/split, pulling ONLY the `text` column (parquet column
    pruning means we don't pay for id/url/date/etc.), and sum real UTF-8
    byte lengths. Optionally checkpoints progress to disk so a long run
    can be resumed.
    """
    total_bytes = 0
    total_docs = 0
    start_index = 0

    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ck = json.load(f)
        total_bytes = ck["total_bytes"]
        total_docs = ck["total_docs"]
        start_index = ck["total_docs"]  # resume by skipping this many docs
        print(f"Resuming from checkpoint: {total_docs:,} docs, {total_bytes:,} bytes so far")

    ds = load_dataset(DATASET, name=config, split=split, streaming=True, columns=["text"])
    if start_index:
        ds = ds.skip(start_index)

    t0 = time.time()
    for doc in ds:
        total_bytes += len(doc["text"].encode("utf-8"))
        total_docs += 1

        if limit and total_docs >= limit:
            break

        if total_docs % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            rate = total_docs / elapsed if elapsed > 0 else 0
            print(f"  ...{total_docs:,} docs, {total_bytes:,} bytes "
                  f"({rate:,.0f} docs/sec)")
            if checkpoint_path:
                with open(checkpoint_path, "w") as f:
                    json.dump({"total_bytes": total_bytes, "total_docs": total_docs}, f)

    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)  # done -- clean up

    return total_bytes, total_docs


def run_sample_mode(limit=None):
    SAMPLE_CONFIG = "sample-350BT"

    print("Fetching exact total row counts from /size (before streaming anything) ...")
    full_rows = get_exact_total_rows(config="default")
    sample_rows = get_exact_total_rows(config=SAMPLE_CONFIG)
    print(f"  default (full dataset) exact row count: {full_rows:,}")
    print(f"  {SAMPLE_CONFIG} exact row count:          {sample_rows:,}")

    if limit is None:
        print(
            f"\nWARNING: no --limit given. {SAMPLE_CONFIG} contains "
            f"{sample_rows:,} documents. Streaming ALL of them at, "
            f"optimistically, a few thousand docs/sec will take HOURS, "
            f"not minutes. Pass --limit N (e.g. --limit 2000000 for a "
            f"~minutes-scale partial sample) unless you specifically "
            f"want to commit to the full run."
        )
        answer = input("Type 'yes' to stream the FULL sample anyway, or Ctrl+C to abort: ")
        if answer.strip().lower() != "yes":
            sys.exit("Aborted. Re-run with --limit N for a fast partial sample.")

    print(f"\nStreaming {'all of' if limit is None else f'the first {limit:,} docs of'} "
          f"config={SAMPLE_CONFIG!r} (measuring real bytes, no guessing)...")
    sample_bytes, sample_docs = stream_and_sum_bytes(config=SAMPLE_CONFIG, limit=limit)

    print(f"\nMeasured: {sample_docs:,} docs, {sample_bytes:,} bytes")

    # If we only measured a partial sample of sample-350BT, our "sample_rows"
    # for the ratio needs to be the number we ACTUALLY measured, not the
    # full sample-350BT row count.
    effective_sample_rows = sample_docs

    scale_factor = full_rows / effective_sample_rows
    bytes_per_doc = sample_bytes / sample_docs
    extrapolated_total_bytes = int(bytes_per_doc * full_rows)

    print()
    print("=" * 70)
    print(f"Measured bytes/doc (from real data): {bytes_per_doc:,.2f}")
    print(f"Scale factor (full_rows / docs_measured): {scale_factor:,.4f}")
    print(f"EXTRAPOLATED total UTF-8 text bytes: {extrapolated_total_bytes:,}")
    print(f"  (~{extrapolated_total_bytes / 1024**4:,.2f} TB)")
    print()
    if limit is not None and limit < sample_rows:
        print(
            f"NOTE: this extrapolated from only {sample_docs:,} of the "
            f"{sample_rows:,} docs in {SAMPLE_CONFIG} (a sub-sample of a "
            f"sample). Bigger --limit = tighter estimate, at the cost of "
            f"more time. For the full official sample, omit --limit (or "
            f"pass --limit {sample_rows}) and expect it to take hours."
        )

    return extrapolated_total_bytes


def run_full_mode(resume):
    print("Streaming the ENTIRE default (full) FineWeb dataset. This will "
          "take a long time -- consider running in a background/tmux "
          "session. Progress checkpoints to "
          f"{CHECKPOINT_FILE} every {PROGRESS_EVERY:,} docs; safe to "
          "Ctrl+C and resume later with --resume.")

    total_bytes, total_docs = stream_and_sum_bytes(
        config="default",
        checkpoint_path=CHECKPOINT_FILE,
        resume=resume,
    )

    print()
    print("=" * 70)
    print(f"EXACT total documents:       {total_docs:,}")
    print(f"EXACT total UTF-8 text bytes: {total_bytes:,}")
    print(f"  (~{total_bytes / 1024**4:,.2f} TB)")

    return total_bytes


def human_readable(n_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    size = float(n_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:,.2f}{unit}"
        size /= 1024.0
    return f"{n_bytes}B"


def update_langs_chosen_row(csv_path, language_code, utf8_bytes, utf8_bytes_human):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    for col in ("utf8_bytes", "utf8_bytes_human"):
        if col not in fieldnames:
            fieldnames.append(col)

    matched = False
    for row in rows:
        if row.get("language_code") == language_code:
            row["utf8_bytes"] = utf8_bytes
            row["utf8_bytes_human"] = utf8_bytes_human
            matched = True

    if not matched:
        raise ValueError(f"No row with language_code={language_code!r} found in {csv_path}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sample", "full"], default="sample")
    parser.add_argument("--resume", action="store_true",
                         help="Resume a --mode full run from checkpoint")
    parser.add_argument("--limit", type=int, default=2_000_000,
                         help="For --mode sample: only stream this many docs "
                              "(default: 2,000,000, ~minutes-scale). Pass a "
                              "bigger number, or omit and confirm the prompt, "
                              "for a tighter estimate at the cost of more time. "
                              "Ignored in --mode full.")
    parser.add_argument("--full-sample", action="store_true",
                         help="For --mode sample: ignore --limit and stream "
                              "the ENTIRE sample-350BT config (hours, but a "
                              "tighter extrapolation than --limit).")
    parser.add_argument("--csv-out", default="langs_chosen.csv")
    parser.add_argument("--language-code", default="eng_Latn")
    parser.add_argument("--no-write", action="store_true",
                         help="Just print the result, don't touch langs_chosen.csv")
    args = parser.parse_args()

    if args.mode == "sample":
        limit = None if args.full_sample else args.limit
        total_bytes = run_sample_mode(limit=limit)
    else:
        total_bytes = run_full_mode(resume=args.resume)

    human = human_readable(total_bytes)

    if not args.no_write:
        update_langs_chosen_row(args.csv_out, args.language_code, total_bytes, human)
        print(f"\nUpdated row language_code={args.language_code} in: {args.csv_out}")


if __name__ == "__main__":
    main()