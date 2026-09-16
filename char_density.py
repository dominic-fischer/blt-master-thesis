#!/usr/bin/env python3
"""
char_density.py -- Information density (codepoints per language) from a
directory of BLT bytes_entropies JSON files under a FIXED, always-N-bytes
custom encoding.

Since the custom encoding maps every codepoint to exactly N bytes (no
UTF-8-style variable length), codepoint count can be read directly off
total byte count:

    n_codepoints = n_bytes_total / N

This is EXACT, not an approximation -- but only if the encoding really
is uniform-width with zero exceptions. The script checks this: if any
language's total byte count isn't evenly divisible by N, that's a red
flag that the "always N bytes" assumption doesn't hold for that
language's data (some character handled differently, a stray malformed
record, etc.), and it's reported as a warning rather than silently
truncated.

Total bytes per language is read directly from each record's "n_bytes"
field (falling back to counting bytes_entropies entries only if that
field happens to be missing from a given record).

Assumes every language's .json in the given directory covers the SAME
parallel corpus (i.e. n_docs should match across all files, and each
record is a translation of the same underlying sentence) -- so the
resulting codepoint counts are directly comparable: a language needing
FEWER codepoints to express the same content is more information-dense
per character, and vice versa.

Reports two density framings, both relative to a baseline language
(default: eng_Latn), so pick whichever reads more naturally for your
writeup:
  - codepoints_per_baseline_codepoint: how many of this language's
    codepoints it takes to match ONE baseline codepoint's worth of
    content. LOWER = denser than baseline.
  - density_index: baseline_codepoints / this_language_codepoints.
    HIGHER = denser than baseline (same direction as "premium" =1
    baseline convention used elsewhere in this project).

USAGE
    python3 char_density.py --bytes-per-char 2 --only-20 \\
        results/own_models/entropy_10M_20lang_4gpu_sourcesbalanced_steps10000_ckpt200_customenc_lr4.5e-3/step_0000006400/

    # different baseline language
    python3 char_density.py --bytes-per-char 2 --baseline deu_Latn --only-20 <dir>
"""

import argparse
import json
import os
import sys


OUR_20_LANGS = {
    "eng_Latn", "cmn_Hans", "deu_Latn", "jpn_Jpan", "spa_Latn", "fra_Latn",
    "ita_Latn", "vie_Latn", "arb_Arab", "tha_Thai", "kor_Hang", "ron_Latn",
    "fin_Latn", "heb_Hebr", "tam_Taml", "hrv_Latn", "srp_Cyrl", "kat_Geor",
    "amh_Ethi", "nya_Latn",
}


def collect_files(inputs, only_20):
    files = []
    for inp in inputs:
        if os.path.isdir(inp):
            for fn in sorted(os.listdir(inp)):
                if not fn.endswith(".json"):
                    continue
                stem = fn[:-5]
                if only_20 and stem not in OUR_20_LANGS:
                    continue
                files.append(os.path.join(inp, fn))
        else:
            files.append(inp)
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Compute codepoint counts and cross-language density from a fixed-width custom encoding.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="JSON file(s) and/or a directory of them")
    parser.add_argument("--bytes-per-char", type=int, required=True,
                         help="Fixed number of bytes per codepoint in this custom encoding (e.g. 2)")
    parser.add_argument("--only-20", action="store_true",
                         help="If an input is a directory, only process files whose name "
                              "(minus .json) is one of the 20 known language codes")
    parser.add_argument("--baseline", default="eng_Latn",
                         help="Language code used as the density=1.0 / density_index=1.0 "
                              "baseline (default: eng_Latn)")
    parser.add_argument("--out-json", default="char_density.json",
                         help="Path to write results as JSON. Empty string to skip writing.")
    args = parser.parse_args()

    files = collect_files(args.inputs, args.only_20)
    if not files:
        print("No matching .json files found.", file=sys.stderr)
        sys.exit(1)

    results = {}
    n_docs_seen = set()
    for path in files:
        label = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        n_docs = len(records)
        n_docs_seen.add(n_docs)
        # Use the "n_bytes" field directly when present (already computed
        # upstream, per record) -- falls back to counting bytes_entropies
        # entries only if a record happens to be missing that field.
        n_bytes_total = sum(
            rec["n_bytes"] if "n_bytes" in rec else len(rec.get("bytes_entropies", []))
            for rec in records
        )

        if n_bytes_total % args.bytes_per_char != 0:
            print(f"WARNING: {label} total bytes ({n_bytes_total}) is NOT evenly "
                  f"divisible by --bytes-per-char={args.bytes_per_char} -- the "
                  f"'always {args.bytes_per_char} bytes/codepoint' assumption may "
                  f"not hold for this language's data. Using integer division "
                  f"(remainder {n_bytes_total % args.bytes_per_char} bytes discarded).",
                  file=sys.stderr)

        n_codepoints = n_bytes_total // args.bytes_per_char
        results[label] = {
            "n_docs": n_docs,
            "n_bytes_total": n_bytes_total,
            "n_codepoints": n_codepoints,
        }

    if len(n_docs_seen) > 1:
        print(f"WARNING: languages have DIFFERENT numbers of documents ({sorted(n_docs_seen)}) -- "
              f"if this isn't a fully parallel corpus, codepoint-count comparisons across "
              f"languages may not be measuring the same underlying content.", file=sys.stderr)

    if args.baseline not in results:
        print(f"Warning: baseline '{args.baseline}' not found among {sorted(results)}; "
              f"density columns will be omitted.", file=sys.stderr)
        baseline_n = None
    else:
        baseline_n = results[args.baseline]["n_codepoints"]

    print(f"{'language':10s} {'n_docs':>7s} {'n_bytes':>10s} {'n_codepoints':>13s} "
          f"{'cp_per_baseline_cp':>19s} {'density_index':>14s}")
    for label in sorted(results):
        r = results[label]
        if baseline_n:
            cp_per_baseline = r["n_codepoints"] / baseline_n
            density_index = baseline_n / r["n_codepoints"]
        else:
            cp_per_baseline = density_index = float("nan")
        r["codepoints_per_baseline_codepoint"] = cp_per_baseline
        r["density_index"] = density_index
        print(f"{label:10s} {r['n_docs']:7d} {r['n_bytes_total']:10d} {r['n_codepoints']:13d} "
              f"{cp_per_baseline:19.4f} {density_index:14.4f}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()