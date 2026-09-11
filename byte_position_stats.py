#!/usr/bin/env python3
"""
byte_position_stats.py -- Per-position byte-VALUE distributions from BLT
`bytes_entropies` JSON dumps (results/base_model/{lang_script}.json).

Each JSON file is a list of records like:
    {
      "id": 0,
      "text": "...",
      "n_bytes": 447,
      "bytes_entropies": [
        [217, 2.212891, 0.812797],
        [138, 1.12207, -0.896113],
        ...
      ]
    }

Only the byte VALUE (element [0] of each entry) is used here -- this script
does not touch the model's entropy values at all. It answers a purely
structural question about the encoding itself:

    For each character byte-length (1, 2, 3, 4 bytes) found in a
    language's text, and for each POSITION within that length (byte[0] =
    lead byte, byte[1], byte[2], ... = continuation bytes), how many
    DISTINCT byte values occur there, and what share of occurrences does
    each one account for?

A position that's always the same single byte value (n_distinct=1, one
value at ~100% share) is structurally FIXED -- it carries no information
about character identity. A position with several values spread across
meaningful shares is genuinely variable -- it IS carrying identity
information. This directly tests, from real corpus occurrences (not an
idealized uniform-over-codepoints assumption), how much of a script's
per-character identity is resolved at each byte position.

Also computes, per language, a single-number SPREAD score in [0, 1]:
the normalized Shannon entropy of the per-position entropy SHARES for
the language's dominant (main) byte-length -- 0 = all identity entropy
concentrated in one byte, 1 = perfectly even across every position.
See spread_score() below for the exact definition. Undefined (null in
the JSON) for 1-byte languages, since there's no "across positions" to
measure with only one position.

HOW POSITION IS DETERMINED
    Every byte's own top bits say whether it starts a new character (and
    how many continuation bytes to expect) or continues one already in
    progress -- standard UTF-8 structure, not a heuristic:

        0xxxxxxx  (0x00-0x7F)  ASCII, 1-byte character
        110xxxxx  (0xC0-0xDF)  lead byte, 2-byte character
        1110xxxx  (0xE0-0xEF)  lead byte, 3-byte character
        11110xxx  (0xF0-0xF7)  lead byte, 4-byte character
        10xxxxxx  (0x80-0xBF)  continuation byte

    Walking the stream once, counting down from each lead byte's declared
    length, gives the exact position of every byte.

USAGE
    # single file
    python3 byte_position_stats.py results/base_model/amh_Ethi.json

    # a whole directory, restricted to our 20 languages (by filename stem)
    python3 byte_position_stats.py --only-20 results/base_model/

    # explicit file list
    python3 byte_position_stats.py results/base_model/kat_Geor.json results/base_model/srp_Cyrl.json

    # limit how many distinct values are PRINTED per position (full list
    # is always saved to the JSON regardless of this)
    python3 byte_position_stats.py --top-n 10 results/base_model/cmn_Hans.json

OUTPUT
    Prints a per-language, per-length, per-position breakdown, and saves
    the full results (every distinct value and its exact share, not just
    the printed top-N) to a JSON file -- see --out-json.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict, Counter

# The 20 languages used throughout this project. Filenames are expected as
# "<code>.json" (e.g. "amh_Ethi.json") directly inside results/base_model/.
OUR_20_LANGS = {
    "eng_Latn", "cmn_Hans", "deu_Latn", "jpn_Jpan", "spa_Latn", "fra_Latn",
    "ita_Latn", "vie_Latn", "arb_Arab", "tha_Thai", "kor_Hang", "ron_Latn",
    "fin_Latn", "heb_Hebr", "tam_Taml", "hrv_Latn", "srp_Cyrl", "kat_Geor",
    "amh_Ethi", "nya_Latn",
}


def spread_score(entropies):
    """Single-number 'how evenly is this character's identity entropy
    distributed across its byte positions' score, in [0, 1].

    Treats the per-position entropies as a distribution in their own
    right (what SHARE of the total identity entropy sits at each
    position) and computes the (normalized) Shannon entropy of THAT
    distribution:
        shares = [e / sum(entropies) for e in entropies]
        H = -sum(p * log2(p) for p in shares)
        spread = H / log2(len(entropies))   # normalize to [0, 1]

    0.0 = fully concentrated in a single byte (e.g. Georgian: ~0.05,
          Hebrew: ~0.01 -- everything deferred to one identity byte).
    1.0 = perfectly even across every position (e.g. Chinese: ~0.96).

    Returns None for 1-byte characters (a single position has no
    "distribution across positions" to measure -- this is a category
    difference, not a spread value of 0) or if all entropies are 0.
    """
    n = len(entropies)
    total = sum(entropies)
    if n <= 1 or total <= 0:
        return None
    shares = [e / total for e in entropies]
    h = -sum(p * math.log2(p) for p in shares if p > 0)
    return h / math.log2(n)


def utf8_lead_length(byte_val):
    """Return the character byte-length (1/2/3/4) this byte STARTS, based
    purely on its own bit pattern -- or None if it's a continuation byte
    (0x80-0xBF), which can't start a character on its own."""
    if byte_val <= 0x7F:
        return 1
    if 0xC0 <= byte_val <= 0xDF:
        return 2
    if 0xE0 <= byte_val <= 0xEF:
        return 3
    if 0xF0 <= byte_val <= 0xF7:
        return 4
    return None


def iter_positions(byte_vals):
    """Yield (char_length, position_within_char, index) for each byte in
    the sequence, tracking how many continuation bytes are still owed
    after each lead byte."""
    remaining = 0
    length = 1
    pos = 0
    for i, bv in enumerate(byte_vals):
        if remaining == 0:
            length = utf8_lead_length(bv)
            if length is None:
                # Malformed: a continuation byte where a lead byte was
                # expected (stream desync, truncation, etc). Resync by
                # treating this single byte as its own 1-byte unit rather
                # than silently mis-assigning positions for everything
                # after it.
                length = 1
            pos = 0
            remaining = length - 1
        else:
            pos += 1
            remaining -= 1
        yield length, pos, i


def iter_positions_fixed(byte_vals, fixed_length):
    """Like iter_positions(), but for a CUSTOM (non-UTF-8) encoding where
    every character is a known, fixed number of bytes -- e.g. a
    tokenizer that always emits 2 bytes per character regardless of
    codepoint. Bypasses utf8_lead_length() entirely, since a custom
    encoding's byte values have no reason to follow UTF-8's lead/
    continuation bit-pattern conventions (0xxxxxxx / 110xxxxx / 10xxxxxx
    / ...) -- using the UTF-8 classifier on such data would silently
    misclassify every byte. Simply walks the stream in fixed-size
    chunks: position 0, 1, ..., fixed_length-1, repeating.
    """
    for i, bv in enumerate(byte_vals):
        yield fixed_length, i % fixed_length, i


def analyze_file(path, label, top_n=15, fixed_length=None):
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    # (length, pos) -> Counter(byte_value -> count)               [marginal]
    value_buckets = defaultdict(Counter)
    # (length, pos) -> {prefix_tuple -> Counter(byte_value -> count)}  [conditional]
    cond_buckets = defaultdict(lambda: defaultdict(Counter))
    n_docs = len(records)
    total_bytes = 0

    for rec in records:
        triples = rec.get("bytes_entropies", [])
        byte_vals = [t[0] for t in triples]
        if fixed_length is not None:
            position_iter = iter_positions_fixed(byte_vals, fixed_length)
        else:
            position_iter = iter_positions(byte_vals)

        char_buffer = []
        for length, pos, i in position_iter:
            if pos == 0:
                char_buffer = []
            bv = byte_vals[i]
            char_buffer.append(bv)
            value_buckets[(length, pos)][bv] += 1
            # prefix = the bytes of THIS character seen before this
            # position (char_buffer minus the byte just appended)
            prefix = tuple(char_buffer[:pos])
            cond_buckets[(length, pos)][prefix][bv] += 1
            total_bytes += 1

    lengths_present = sorted(set(l for l, p in value_buckets))
    total_chars = sum(sum(value_buckets[(l, 0)].values()) for l in lengths_present)

    print(f"=== {label} (n={total_bytes} bytes across {n_docs} documents) ===")

    lengths_dict = {}  # built up here, attached to result LAST (see below)

    for length in lengths_present:
        n_chars = sum(value_buckets[(length, 0)].values())
        frac = n_chars / total_chars if total_chars else 0
        print(f"  {length}-byte characters: {n_chars} occurrences ({frac:.1%} of all characters)")

        length_result = {"n_chars": n_chars, "fraction": frac, "positions": {}}

        for pos in range(length):
            counter = value_buckets.get((length, pos), Counter())
            total_here = sum(counter.values())
            n_distinct = len(counter)
            marginal_ent = (
                -sum((c / total_here) * math.log2(c / total_here) for c in counter.values())
                if total_here else 0.0
            )

            # CONDITIONAL entropy H(byte[pos] | byte[0:pos]) -- the actual
            # chain-rule term: weighted average, over every distinct prefix
            # seen at this position, of the entropy of byte[pos] WITHIN
            # that prefix group. Equals the marginal entropy at pos=0
            # (nothing to condition on), but can be far LOWER at later
            # positions if earlier bytes predict this one -- e.g. a
            # custom encoding where byte[0] fully determines byte[1].
            prefix_groups = cond_buckets.get((length, pos), {})
            cond_ent = 0.0
            for prefix, pcounter in prefix_groups.items():
                group_total = sum(pcounter.values())
                w = group_total / total_here if total_here else 0
                group_ent = (
                    -sum((c / group_total) * math.log2(c / group_total) for c in pcounter.values())
                    if group_total else 0.0
                )
                cond_ent += w * group_ent

            sorted_items = counter.most_common()  # descending by count

            print(f"    byte[{pos}]: {n_distinct} distinct value(s)  |  "
                  f"marginal entropy={marginal_ent:.2f} bits  |  "
                  f"conditional entropy={cond_ent:.2f} bits")
            for bv, c in sorted_items[:top_n]:
                share = c / total_here
                print(f"        0x{bv:02x} ({bv:3d}): count={c:6d}  share={share:6.1%}")
            if top_n is not None and n_distinct > top_n:
                print(f"        ... and {n_distinct - top_n} more distinct value(s) "
                      f"(full list saved to JSON)")

            length_result["positions"][str(pos)] = {
                "n_distinct_values": n_distinct,
                "entropy_bits": marginal_ent,               # kept for backward compatibility
                "marginal_entropy_bits": marginal_ent,
                "conditional_entropy_bits": cond_ent,
                "values": [
                    {"byte": bv, "hex": f"0x{bv:02x}", "count": c, "share": c / total_here}
                    for bv, c in sorted_items
                ],
            }

        lengths_dict[str(length)] = length_result
        print()

    # main_length: whichever byte-length accounts for the most characters
    # in this language (its dominant encoding length). main_length_entropies
    # uses CONDITIONAL entropy per position -- the correct chain-rule
    # quantity, matching H(char) = sum_k H(byte_k | byte_0..k-1) exactly.
    # (Marginal entropy is still available per-position above, under
    # "marginal_entropy_bits", for the separate "how much does the raw
    # encoding vary here" question -- but summing MARGINAL entropies would
    # generally OVER-count the true joint/identity entropy whenever byte
    # positions are correlated, so it is not used for main_length_entropies.)
    main_length = max(lengths_present, key=lambda l: lengths_dict[str(l)]["n_chars"])
    main_length_entropies = [
        lengths_dict[str(main_length)]["positions"][str(p)]["conditional_entropy_bits"]
        for p in range(main_length)
    ]
    main_length_entropy_sum = sum(main_length_entropies)
    spread = spread_score(main_length_entropies)

    # Build the result dict with compact summary fields FIRST and the bulky
    # per-value "lengths" breakdown LAST -- in an editor with JSON folding
    # (e.g. VS Code), collapsing that final key hides the large nested
    # breakdown while leaving all the summary fields visible above it.
    result = {
        "n_docs": n_docs,
        "n_bytes_total": total_bytes,
        "main_length": main_length,
        "main_length_entropies": main_length_entropies,
        "main_length_entropy_sum": main_length_entropy_sum,
        "spread": spread,
        #"lengths": lengths_dict,
    }

    spread_str = f"{spread:.3f}" if spread is not None else "n/a (1-byte language)"
    print(f"  main_length: {main_length}  "
          f"main_length_entropies: {[round(e, 3) for e in main_length_entropies]}  "
          f"main_length_entropy_sum: {main_length_entropy_sum:.3f} bits  "
          f"spread: {spread_str}")
    print()

    return result


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
        description="Per-position byte-value distributions from BLT bytes_entropies JSON dumps."
    )
    parser.add_argument("inputs", nargs="+", help="JSON file(s) and/or a directory of them")
    parser.add_argument("--only-20", action="store_true",
                         help="If an input is a directory, only process files whose name "
                              "(minus .json) is one of the 20 known language codes")
    parser.add_argument("--top-n", type=int, default=15,
                         help="Max distinct values to PRINT per position (default: 15). "
                              "The saved JSON always has the full list regardless. "
                              "Use a large number (e.g. 999) to print everything.")
    parser.add_argument("--out-json", default="byte_position_stats.json",
                         help="Path to write combined results as JSON "
                              "(default: byte_position_stats.json in the current directory). "
                              "Pass an empty string to skip writing.")
    parser.add_argument("--fixed-length", type=int, default=None,
                         help="Use this for a CUSTOM (non-UTF-8) encoding where every "
                              "character is a known, fixed number of bytes (e.g. --fixed-length 2 "
                              "for an encoding that always emits 2 bytes/character). Bypasses "
                              "UTF-8 lead-byte detection entirely, since a custom encoding's byte "
                              "values have no reason to follow UTF-8's bit-pattern conventions -- "
                              "using the UTF-8 classifier on such data would silently misclassify "
                              "every byte. Omit this for ordinary UTF-8 text.")
    args = parser.parse_args()

    files = collect_files(args.inputs, args.only_20)
    if not files:
        print("No matching .json files found.", file=sys.stderr)
        sys.exit(1)

    all_results = {}
    for path in files:
        label = os.path.splitext(os.path.basename(path))[0]
        try:
            all_results[label] = analyze_file(path, label, top_n=args.top_n,
                                               fixed_length=args.fixed_length)
        except Exception as e:
            print(f"Skipping {path}: {e}", file=sys.stderr)

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"[byte_position_stats] Saved combined results -> {args.out_json}")


if __name__ == "__main__":
    main()