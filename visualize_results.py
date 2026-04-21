"""
viz_results.py
Generate an HTML visualisation for specific sentences from the FLORES eval results.

Usage:
    # Visualise sentences 0-4 for English
    python viz_results.py --lang eng_Latn --sentences 0 1 2 3 4

    # Visualise sentences 0-4 for all languages (parallel view)
    python viz_results.py --sentences 0 1 2 3 4 --all-langs

    # Visualise a range
    python viz_results.py --lang eng_Latn --range 0 20

Output: viz/{lang_code}_s{start}-{end}.html  (or viz/all_s{...}.html for --all-langs)
"""

import argparse
import json
import os

from blt_visualize import BLTPatchVisualizer
from eval_flores import LANGUAGES

RESULTS_DIR = "results"
VIZ_DIR = "visualizations"


def load_results(lang_code: str) -> list:
    path = os.path.join(RESULTS_DIR, f"{lang_code}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No results found for {lang_code} at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def entries_to_patches(entry: dict) -> list:
    """Convert JSON patch dicts back to (chunk, length) tuples for the visualizer."""
    return [(p["bytes"], p["length"]) for p in entry["patches"]]


def build_viz(entries: list, lang_code: str) -> BLTPatchVisualizer:
    lang_name = LANGUAGES.get(lang_code, lang_code)
    viz = BLTPatchVisualizer()
    for entry in entries:
        viz.add(
            text=entry["text"],
            patches=entries_to_patches(entry),
            scores=entry.get("scores"),
            label=f"[{lang_name}] id={entry['id']}  |  {entry['n_patches']} patches, {entry['n_bytes']} bytes, avg {entry['avg_bytes_per_patch']:.2f} b/patch",
            threshold=None,  # not stored in JSON; set manually if needed
        )
    return viz


def resolve_indices(results: list, sentences: list[int] | None, range_: tuple | None) -> list[int]:
    if range_ is not None:
        start, end = range_
        indices = list(range(start, end))
    elif sentences is not None:
        indices = sentences
    else:
        raise ValueError("Provide either --sentences or --range")
    # Clamp to valid range
    max_id = len(results) - 1
    invalid = [i for i in indices if i > max_id]
    if invalid:
        print(f"  Warning: indices {invalid} out of range (max {max_id}), skipping")
    return [i for i in indices if i <= max_id]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default=None,
                        help="Language code, e.g. eng_Latn (omit with --all-langs)")
    parser.add_argument("--all-langs", action="store_true",
                        help="Run over all languages for the given sentences")
    parser.add_argument("--sentences", type=int, nargs="+", default=None,
                        help="Specific sentence indices to visualise")
    parser.add_argument("--range", type=int, nargs=2, default=None, metavar=("START", "END"),
                        help="Range of sentence indices [start, end)")
    parser.add_argument("--output-dir", type=str, default=VIZ_DIR)
    args = parser.parse_args()

    if args.lang is None and not args.all_langs:
        parser.error("Provide --lang or --all-langs")

    os.makedirs(args.output_dir, exist_ok=True)

    langs = list(LANGUAGES.keys()) if args.all_langs else [args.lang]

    for lang_code in langs:
        print(f"\nLoading {lang_code}...")
        try:
            results = load_results(lang_code)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        indices = resolve_indices(results, args.sentences, args.range)
        print(f"  Visualising {len(indices)} sentences: {indices}")

        entries = [results[i] for i in indices]
        viz = build_viz(entries, lang_code)

        # Build output filename
        if args.range:
            tag = f"s{args.range[0]}-{args.range[1]}"
        else:
            tag = "s" + "_".join(str(i) for i in indices)

        out_path = os.path.join(args.output_dir, f"{lang_code}_{tag}.html")
        viz.save(out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()