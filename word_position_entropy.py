#!/usr/bin/env python3
"""
word_position_entropy.py -- Where do patch boundaries fall relative to
words? Uses the same BLT `bytes_entropies` JSON dumps as
byte_position_stats.py, plus the word segmentation of word_length_stats.py
(which must sit in the same directory -- its segmenters are reused).

For every character of every text, the script knows
  - its byte span (fixed length with --fixed-length, e.g. 2 for the custom
    encoding; otherwise its UTF-8 length),
  - whether any of its bytes starts a patch under the chosen threshold
    rule (--mode global: entropy > t; --mode mono: entropy rise from the
    previous byte > t),
  - its position in its word: 1 = word onset, 2, 3, ... = word-internal;
    whitespace and punctuation-only characters are their own categories.

WORDS
    Same definition as word_length_stats.py: whitespace tokens, counting
    only letters/marks/numbers (Unicode L*, M*, N*). Languages without
    whitespace words (cmn_Hans, jpn_Jpan, tha_Thai) use every segmenter
    word_length_stats.py can load for them (e.g. jieba/pkuseg/icu,
    ginza_bunsetsu, pythainlp); results are reported per segmenter and
    averaged in "summary". Vietnamese and Korean use whitespace.

OUTPUT (JSON, per language)
    methods[m].categories[onset|internal|space|other]
        n_chars, n_boundaries (bytes that start a patch),
        n_boundaries_first_byte / n_boundaries_later_bytes,
        crossing_rate  (share of these chars with >= 1 boundary byte)
    methods[m].share_of_boundaries[category]
    methods[m].by_position  (word positions 1..MAX_POS, last one pooled)
        n_chars, crossing_rate, mean_char_entropy (sum over the char's
        bytes), mean_first_byte_entropy
    methods[m].segment_coverage  (share of word chars aligned to a
        segmenter token; 1.0 for whitespace)
    summary      = whitespace method, or the mean over segmenters
    rel_to_eng   (if eng_Latn was processed)
        premium            = boundaries / English boundaries
        premium_parts[cat] = boundaries in cat / English boundaries
                             (sums to premium -> stacked-bar decomposition)
    Compare rel_to_eng.premium with your premiums file: if they agree,
    the threshold rule and alignment match your patching code.

ALIGNMENT
    Entry i of bytes_entropies is the entropy of the model's prediction
    for the NEXT byte (i+1), as in BLT's patching code: a patch starts at
    byte i+1 when entry i (global) or its rise over entry i-1 (mono)
    exceeds t. Check: English boundaries should fall mostly on word
    onsets, and on characters' FIRST bytes under the custom encoding.

PLOT
    Boundary rate by position in the word, one line per language. A
    language's line only extends to positions that at least
    --min-word-share of its words reach (default 10%) and that have at
    least --min-chars characters (default 200), so the tail is not drawn
    from a handful of unusually long words. --from-json re-draws the plot
    from a saved results file without recomputing.

USAGE
    python3 word_position_entropy.py --only-20 --fixed-length 2 \\
        --threshold 2.0176 --mode global --plot word_positions.png \\
        results/<custom-encoding dumps>/
"""

import argparse
import json
import os
import sys
import unicodedata
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from word_length_stats import (OUR_20_LANGS, NO_WHITESPACE_WORDS,  # noqa: E402
                               load_segmenters, is_word_char)

MAX_POS = 8  # word positions >= MAX_POS are pooled into one bucket
CATEGORIES = ["onset", "internal", "space", "other"]

SCRIPT_TYPE = {
    "eng_Latn": "Alphabetic", "deu_Latn": "Alphabetic", "spa_Latn": "Alphabetic",
    "fra_Latn": "Alphabetic", "ita_Latn": "Alphabetic", "ron_Latn": "Alphabetic",
    "hrv_Latn": "Alphabetic", "srp_Cyrl": "Alphabetic", "nya_Latn": "Alphabetic",
    "fin_Latn": "Alphabetic", "vie_Latn": "Alphabetic", "kat_Geor": "Alphabetic",
    "arb_Arab": "Abjad", "heb_Hebr": "Abjad", "tam_Taml": "Abugida", "tha_Thai": "Abugida",
    "amh_Ethi": "Syllabary", "kor_Hang": "Syllabary", "jpn_Jpan": "Logosyllabary",
    "cmn_Hans": "Logographic",
}
DENSE_TYPES = {"Abjad", "Syllabary", "Logosyllabary", "Logographic"}
# Dominant UTF-8 bytes per character (Vietnamese: mostly 1, some 2-3;
# Georgian letters: 3).
BYTES_PER_CHAR = {
    "eng_Latn": 1, "deu_Latn": 1, "spa_Latn": 1, "fra_Latn": 1, "ita_Latn": 1,
    "ron_Latn": 1, "hrv_Latn": 1, "nya_Latn": 1, "fin_Latn": 1, "vie_Latn": 1,
    "arb_Arab": 2, "heb_Hebr": 2, "srp_Cyrl": 2,
    "kat_Geor": 3, "amh_Ethi": 3, "kor_Hang": 3, "jpn_Jpan": 3, "cmn_Hans": 3,
    "tam_Taml": 3, "tha_Thai": 3,
}
BYTES_PALETTE = {1: "#4C72B0", 2: "#55A868", 3: "#CCB974"}
PALETTE = {"Alphabetic": "#4C72B0", "Abjad": "#DD8452", "Abugida": "#55A868",
           "Syllabary": "#C44E52", "Logosyllabary": "#8172B3", "Logographic": "#937860"}


# ---------------------------------------------------------------------------
# Boundaries and word positions
# ---------------------------------------------------------------------------

def boundaries(ent, mode, t):
    """Boolean array: does a patch start at byte i?"""
    ent = np.asarray(ent, dtype=float)
    b = np.zeros(len(ent), dtype=bool)
    if mode == "global":
        b[1:] = ent[:-1] > t   # entropy at i predicts byte i+1
    else:
        b[2:] = (ent[1:-1] - ent[:-2]) > t
    if len(b):
        b[0] = True            # every document starts with a patch
    return b


def whitespace_positions(text):
    """Per character: position among the word chars of its whitespace
    token (1 = onset), or 0 for whitespace / non-word characters."""
    pos = [0] * len(text)
    k = 0
    for i, ch in enumerate(text):
        if ch.isspace():
            k = 0
        elif is_word_char(ch):
            k += 1
            pos[i] = k
    return pos


def segment_positions(text, tokens):
    """Like whitespace_positions, but word positions restart at each
    segmenter token. Tokens are aligned to the text by sequential search;
    characters not covered by any token keep their whitespace-based
    position. Returns (positions, n_word_chars_aligned)."""
    pos = whitespace_positions(text)
    cursor, aligned = 0, 0
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        idx = text.find(tok, cursor)
        if idx < 0:
            continue
        k = 0
        for j in range(idx, idx + len(tok)):
            if is_word_char(text[j]):
                k += 1
                pos[j] = k
                aligned += 1
        cursor = idx + len(tok)
    return pos, aligned


def category(ch, p):
    if p == 1:
        return "onset"
    if p >= 2:
        return "internal"
    return "space" if ch.isspace() else "other"


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------

def new_acc():
    return {
        "n_chars": 0, "n_boundaries": 0, "n_word_chars": 0, "n_aligned": 0,
        "cat": {c: defaultdict(float) for c in CATEGORIES},
        "pos": [defaultdict(float) for _ in range(MAX_POS)],
    }


def accumulate(acc, text, lengths, ent, bnd, positions):
    """ent must be the PREDICTION entropy per byte (entropy for byte i,
    i.e. the dump's entry i-1; NaN for the first byte)."""
    s = 0
    for ch, L, p in zip(text, lengths, positions):
        b = bnd[s:s + L]
        e = ent[s:s + L]
        nb = int(b.sum())
        c = acc["cat"][category(ch, p)]
        c["n_chars"] += 1
        c["n_boundaries"] += nb
        c["n_boundaries_first_byte"] += int(b[0])
        c["n_boundaries_later_bytes"] += nb - int(b[0])
        c["n_crossed"] += int(nb > 0)
        if p >= 1:
            q = acc["pos"][min(p, MAX_POS) - 1]
            q["n_chars"] += 1
            q["n_reach"] += int(p <= MAX_POS)   # words with >= p word chars
            q["n_crossed"] += int(nb > 0)
            if not np.isnan(e).any():   # skips a document's very first character
                q["n_ent"] += 1
                q["entropy_sum"] += float(e.sum())
                q["first_byte_entropy_sum"] += float(e[0])
        acc["n_chars"] += 1
        acc["n_boundaries"] += nb
        s += L


def finalize(acc):
    total_b = acc["n_boundaries"]
    cats = {}
    for c in CATEGORIES:
        d = acc["cat"][c]
        n = d["n_chars"]
        cats[c] = {
            "n_chars": int(n),
            "n_boundaries": int(d["n_boundaries"]),
            "n_boundaries_first_byte": int(d["n_boundaries_first_byte"]),
            "n_boundaries_later_bytes": int(d["n_boundaries_later_bytes"]),
            "crossing_rate": d["n_crossed"] / n if n else None,
        }
    by_pos = []
    n_words = acc["pos"][0]["n_reach"]
    for i, q in enumerate(acc["pos"]):
        n = q["n_chars"]
        by_pos.append({
            "position": f"{i + 1}+" if i + 1 == MAX_POS else str(i + 1),
            "n_chars": int(n),
            "n_words_reaching": int(q["n_reach"]),
            "share_of_words_reaching": q["n_reach"] / n_words if n_words else None,
            "crossing_rate": q["n_crossed"] / n if n else None,
            "mean_char_entropy": q["entropy_sum"] / q["n_ent"] if q["n_ent"] else None,
            "mean_first_byte_entropy": (q["first_byte_entropy_sum"] / q["n_ent"]
                                        if q["n_ent"] else None),
        })
    return {
        "n_chars": acc["n_chars"],
        "n_boundaries": total_b,
        "categories": cats,
        "share_of_boundaries": {c: (cats[c]["n_boundaries"] / total_b if total_b else None)
                                for c in CATEGORIES},
        "by_position": by_pos,
        "segment_coverage": (acc["n_aligned"] / acc["n_word_chars"]
                             if acc["n_word_chars"] else None),
    }


def average(objs):
    """Recursive mean over a list of equally structured results."""
    objs = [o for o in objs if o is not None]
    if not objs:
        return None
    first = objs[0]
    if isinstance(first, dict):
        return {k: average([o.get(k) for o in objs]) for k in first}
    if isinstance(first, list):
        return [average(list(items)) for items in zip(*objs)]
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        return float(np.mean(objs))
    return first


# ---------------------------------------------------------------------------
# Per-language analysis
# ---------------------------------------------------------------------------

def analyze_file(path, lang, args):
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    methods = {"whitespace": None}
    failed = {}
    if lang in NO_WHITESPACE_WORDS:
        segs, failed = load_segmenters(lang, use_stanza=False)
        methods = {name: fn for name, (fn, _vers) in segs.items()}
        if not methods:
            print(f"  [warn] {lang}: no segmenter available -> whitespace fallback",
                  file=sys.stderr)
            methods = {"whitespace": None}

    accs = {m: new_acc() for m in methods}
    n_skipped = 0
    for rec in records:
        text = rec.get("text", "")
        triples = rec.get("bytes_entropies", [])
        ent = np.array([t[1] for t in triples], dtype=float)
        lengths = ([args.fixed_length] * len(text) if args.fixed_length
                   else [len(ch.encode("utf-8")) for ch in text])
        if sum(lengths) != len(ent) or not text:
            n_skipped += 1
            continue
        bnd = boundaries(ent, args.mode, args.threshold)
        # entropy OF each byte's prediction = dump entry for the previous byte
        ent_pred = np.concatenate([[np.nan], ent[:-1]])
        n_word = sum(1 for ch in text if is_word_char(ch))
        for m, fn in methods.items():
            if fn is None:
                positions, aligned = whitespace_positions(text), n_word
            else:
                positions, aligned = segment_positions(text, fn(text))
            accs[m]["n_word_chars"] += n_word
            accs[m]["n_aligned"] += aligned
            accumulate(accs[m], text, lengths, ent_pred, bnd, positions)

    per_method = {m: finalize(a) for m, a in accs.items()}
    summary = per_method["whitespace"] if "whitespace" in per_method \
        else average(list(per_method.values()))

    s = summary
    print(f"=== {lang} ({len(records) - n_skipped} docs used, {n_skipped} skipped; "
          f"methods: {', '.join(methods)}) ===")
    print(f"  boundaries: {s['n_boundaries']:.0f} over {s['n_chars']:.0f} chars")
    for c in CATEGORIES:
        d = s["categories"][c]
        cr = d["crossing_rate"]
        print(f"  {c:9s} chars={d['n_chars']:8.0f}  share of boundaries="
              f"{s['share_of_boundaries'][c] or 0:6.1%}  crossing rate="
              f"{(cr if cr is not None else float('nan')):.3f}")
    rates = " ".join(f"{q['position']}:{q['crossing_rate']:.2f}" for q in s["by_position"]
                     if q["crossing_rate"] is not None)
    print(f"  crossing rate by word position: {rates}")
    for m, r in per_method.items():
        if r["segment_coverage"] is not None and r["segment_coverage"] < 0.99:
            print(f"  [warn] {m}: only {r['segment_coverage']:.1%} of word chars aligned "
                  f"to segmenter tokens", file=sys.stderr)
    print()

    return {
        "n_docs": len(records), "n_skipped_docs": n_skipped,
        "threshold": args.threshold, "mode": args.mode,
        "fixed_length": args.fixed_length,
        "script_type": SCRIPT_TYPE.get(lang),
        "summary": summary, "methods": per_method,
        "segmenters_failed": failed,
    }


def add_english_ratios(results):
    if "eng_Latn" not in results:
        print("[word_position_entropy] eng_Latn not processed -> no rel_to_eng.",
              file=sys.stderr)
        return
    eng_b = results["eng_Latn"]["summary"]["n_boundaries"]
    for res in results.values():
        s = res["summary"]
        res["rel_to_eng"] = {
            "premium": s["n_boundaries"] / eng_b,
            "premium_parts": {c: s["categories"][c]["n_boundaries"] / eng_b
                              for c in CATEGORIES},
        }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def last_position(by_pos, min_share, min_chars):
    """Number of leading word positions reached by at least min_share of
    the language's words and backed by at least min_chars characters."""
    k = 0
    for q in by_pos:
        share = q.get("share_of_words_reaching")
        if share is None or q["crossing_rate"] is None:
            break
        if share < min_share or q["n_chars"] < min_chars:
            break
        k += 1
    return k


def style_for(lang, script_type, color_by):
    """(colour, linestyle, legend key) for one language's line."""
    if color_by == "bytes":
        b = BYTES_PER_CHAR.get(lang, 1)
        return BYTES_PALETTE[b], ("--" if b == 1 else "-"), b
    st = script_type or "Alphabetic"
    return PALETTE.get(st, "#999999"), ("-" if st in DENSE_TYPES else "--"), st


def plot(results, out_path, threshold, mode, min_share, min_chars, color_by="script"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10, 7.5))
    xlabels = [str(i) for i in range(1, MAX_POS)] + [f"{MAX_POS}+"]
    print(f"Plotted positions (>= {min_share:.0%} of words, >= {min_chars} chars):")
    for l in sorted(results):
        col, ls, _key = style_for(l, results[l]["script_type"], color_by)
        by_pos = results[l]["summary"]["by_position"]
        k = last_position(by_pos, min_share, min_chars)
        print(f"  {l:10s} 1-{xlabels[k - 1] if k else '-'}")
        if k == 0:
            continue
        xs = np.arange(1, k + 1)
        ys = [by_pos[i]["crossing_rate"] for i in range(k)]
        ax.plot(xs, ys, marker="o", ms=4, lw=1.6, color=col, ls=ls, alpha=0.85)
    ax.set_xticks(np.arange(1, MAX_POS + 1))
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Position in the word (1 = onset)")
    ax.set_ylabel("Share of characters starting a patch")
    ax.set_title("Boundary rate by position in the word", fontweight="bold", pad=22)
    ax.text(0.5, 1.01, f"{mode} threshold (t = {threshold}); positions shown if reached by "
            f">= {min_share:.0%} of words", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=10, color="#555555")
    ax.grid(True, ls="--", alpha=0.35)
    if color_by == "bytes":
        present = {BYTES_PER_CHAR.get(l, 1) for l in results}
        handles = [Line2D([0], [0], color=BYTES_PALETTE[b], lw=2, ls="--" if b == 1 else "-",
                          label=f"{b} byte{'s' if b > 1 else ''}")
                   for b in sorted(present)]
        ax.legend(handles=handles, title="Bytes per char (solid = multi-byte)", fontsize=10)
    else:
        handles = [Line2D([0], [0], color=c, lw=2, ls="-" if t in DENSE_TYPES else "--", label=t)
                   for t, c in PALETTE.items()
                   if any(results[l]["script_type"] == t for l in results)]
        ax.legend(handles=handles, title="Script type (solid = dense)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[word_position_entropy] Saved plot -> {out_path}")


# ---------------------------------------------------------------------------

def collect_files(inputs, only_20):
    files = []
    for inp in inputs:
        if os.path.isdir(inp):
            for fn in sorted(os.listdir(inp)):
                if fn.endswith(".json") and (not only_20 or fn[:-5] in OUR_20_LANGS):
                    files.append(os.path.join(inp, fn))
        else:
            files.append(inp)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="bytes_entropies JSON file(s) and/or directories")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Calibrated threshold t of the run (e.g. 2.0176)")
    ap.add_argument("--mode", choices=["global", "mono"], default="global",
                    help="global: entropy > t; mono: entropy rise over previous byte > t")
    ap.add_argument("--fixed-length", type=int, default=None,
                    help="Bytes per character for a fixed-length custom encoding "
                         "(2 for Balanced-Custom); omit for UTF-8")
    ap.add_argument("--only-20", action="store_true",
                    help="For directories, only the 20 project languages")
    ap.add_argument("--out-json", default="word_position_entropy.json")
    ap.add_argument("--plot", default=None, help="Optional PNG path for the figure")
    ap.add_argument("--min-word-share", type=float, default=0.10,
                    help="Plot a position only if at least this share of the language's "
                         "words reach it (default 0.10)")
    ap.add_argument("--min-chars", type=int, default=200,
                    help="... and it is backed by at least this many characters (default 200)")
    ap.add_argument("--color-by", choices=["script", "bytes"], default="script",
                    help="Colour lines by script type (dense vs. not; default) or by "
                         "dominant UTF-8 bytes per character")
    ap.add_argument("--from-json", default=None,
                    help="Re-draw --plot from a saved results JSON instead of recomputing")
    args = ap.parse_args()

    if args.from_json:
        with open(args.from_json, encoding="utf-8") as f:
            results = json.load(f)
        any_res = next(iter(results.values()))
        if "share_of_words_reaching" not in any_res["summary"]["by_position"][0]:
            print("This JSON predates the word-share counts -- rerun the analysis once.",
                  file=sys.stderr)
            sys.exit(1)
        plot(results, args.plot or "word_positions.png", any_res["threshold"],
             any_res["mode"], args.min_word_share, args.min_chars, args.color_by)
        return
    if args.threshold is None or not args.inputs:
        ap.error("inputs and --threshold are required unless --from-json is given")

    files = collect_files(args.inputs, args.only_20)
    if not files:
        print("No matching .json files found.", file=sys.stderr)
        sys.exit(1)

    results = {}
    for path in files:
        lang = os.path.splitext(os.path.basename(path))[0]
        try:
            results[lang] = analyze_file(path, lang, args)
        except Exception as e:
            print(f"Skipping {path}: {type(e).__name__}: {e}", file=sys.stderr)

    add_english_ratios(results)
    if "eng_Latn" in results:
        print("Premium from boundary counts (check against your premiums file):")
        for l in sorted(results, key=lambda l: -results[l]["rel_to_eng"]["premium"]):
            r = results[l]["rel_to_eng"]
            parts = r["premium_parts"]
            print(f"  {l:10s} {r['premium']:.3f}  = onset {parts['onset']:.3f} + internal "
                  f"{parts['internal']:.3f} + space {parts['space']:.3f} + other {parts['other']:.3f}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[word_position_entropy] Saved results -> {args.out_json}")
    if args.plot:
        plot(results, args.plot, args.threshold, args.mode,
             args.min_word_share, args.min_chars, args.color_by)


if __name__ == "__main__":
    main()