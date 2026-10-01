#!/usr/bin/env python3
"""
Plot the distribution of per-byte (default) or per-character (--char-level)
entropies, one line per language,
coloured by script type (solid = dense script, dashed = alphabetic/abugida).

Reads the per-language JSON files (list of documents, each with
'bytes_entropies' = [[byte, entropy, ...], ...] and
'chars_entropies' = [[char, entropy, n_bytes, ...], ...]).

Note: byte- and char-level runs use different global thresholds; pass the
matching one via --threshold.

Plots the raw entropy distribution with the patching threshold as a vertical line.
Optional second panel (--with-norm): entropy divided by each language's own mean.

Example:
  python results/plot_entropy_distributions.py \
      results/own_models/entropy_10M_20lang_4gpu_sourcesbalanced_steps10000_ckpt200_lr4.5e-3/step_0000007200 \
      --char-level --threshold 1.9448 --out results/entropy_distributions_char.png
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SCRIPT_TYPE = {
    "amh_Ethi": "Syllabary",
    "arb_Arab": "Abjad",
    "cmn_Hans": "Logographic",
    "deu_Latn": "Alphabetic",
    "eng_Latn": "Alphabetic",
    "fin_Latn": "Alphabetic",
    "fra_Latn": "Alphabetic",
    "heb_Hebr": "Abjad",
    "hrv_Latn": "Alphabetic",
    "ita_Latn": "Alphabetic",
    "jpn_Jpan": "Logosyllabary",
    "kat_Geor": "Alphabetic",
    "kor_Hang": "Syllabary",
    "nya_Latn": "Alphabetic",
    "ron_Latn": "Alphabetic",
    "spa_Latn": "Alphabetic",
    "srp_Cyrl": "Alphabetic",
    "tam_Taml": "Abugida",
    "tha_Thai": "Abugida",
    "vie_Latn": "Alphabetic",
}

SCRIPT_ORDER = ["Alphabetic", "Abjad", "Abugida", "Syllabary", "Logosyllabary", "Logographic"]
SCRIPT_COLORS = {
    "Alphabetic": "#4C72B0",
    "Abjad": "#DD8452",
    "Abugida": "#55A868",
    "Syllabary": "#C44E52",
    "Logosyllabary": "#8172B3",
    "Logographic": "#937860",
}
DENSE_SCRIPTS = {"Abjad", "Syllabary", "Logosyllabary", "Logographic"}


WHITESPACE_BYTES = {9, 10, 11, 12, 13, 32}


def load_entropies(path, char_level=False, exclude_whitespace=False, byte_entropy_index=1,
                   diff=False):
    """Per-character entropies ('chars_entropies', summed over UTF-8 bytes) if
    char_level, otherwise per-byte entropies ('bytes_entropies') as stored.
    With diff=True, returns jumps H(x_t) - H(x_{t-1}) between consecutive
    positions, computed within each document (never across document borders)."""
    with open(path, encoding="utf-8") as f:
        docs = json.load(f)
    vals = []
    for doc in docs:
        doc_vals = []
        if char_level:
            for entry in doc.get("chars_entropies", []):
                if exclude_whitespace and str(entry[0]).isspace():
                    continue
                doc_vals.append(float(entry[1]))
        else:
            for entry in doc.get("bytes_entropies", []):
                if exclude_whitespace and int(entry[0]) in WHITESPACE_BYTES:
                    continue
                doc_vals.append(float(entry[byte_entropy_index]))
        if diff:
            doc_vals = np.diff(doc_vals).tolist()
        vals.extend(doc_vals)
    return np.asarray(vals)


def skewness(x):
    d = x - x.mean()
    return (d ** 3).mean() / (d ** 2).mean() ** 1.5


def plot_panel(ax, data, bins, normalize, logy, threshold=None, highlight=None):
    centers = (bins[:-1] + bins[1:]) / 2
    for lang, ents in data.items():
        script = SCRIPT_TYPE.get(lang, "Alphabetic")
        vals = ents / ents.mean() if normalize else ents
        hist, _ = np.histogram(vals, bins=bins, density=True)
        if logy:
            hist = np.where(hist > 0, hist, np.nan)
        faded = highlight is not None and lang not in highlight
        ax.plot(
            centers, hist,
            color="lightgray" if faded else SCRIPT_COLORS[script],
            linestyle="-" if script in DENSE_SCRIPTS else "--",
            linewidth=1.2 if faded else 1.8,
            alpha=0.6 if faded else 0.85,
            zorder=1 if faded else 2,
        )
        if highlight and lang in highlight:
            i = np.nanargmax(hist)
            ax.annotate(lang.split("_")[0], (centers[i], hist[i]),
                        textcoords="offset points", xytext=(4, 4), fontsize=9)
    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle=":", linewidth=1.2)
        ax.text(threshold, 0.98, f" t = {threshold}", transform=ax.get_xaxis_transform(),
                va="top", fontsize=9)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylabel("Density")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", help="Directory with one <lang>.json per language")
    p.add_argument("--threshold", type=float, default=None, help="Draw patching threshold on raw panel")
    p.add_argument("--bins", type=int, default=60)
    p.add_argument("--logy", action="store_true", help="Log-scale y-axis")
    p.add_argument("--exclude-whitespace", action="store_true", help="Drop whitespace characters")
    p.add_argument("--highlight", nargs="*", default=None,
                   help="Language codes to highlight (others in gray), e.g. eng_Latn heb_Hebr cmn_Hans")
    p.add_argument("--char-level", action="store_true",
                   help="Use per-character entropies (summed over UTF-8 bytes). "
                        "Default: per-byte entropies as stored in the file")
    p.add_argument("--byte-entropy-index", type=int, default=1,
                   help="Index of the entropy value in each 'bytes_entropies' entry (default 1)")
    p.add_argument("--diff", action="store_true",
                   help="Plot jumps H(x_t) - H(x_{t-1}) instead of levels "
                        "(what the monotonicity criterion thresholds)")
    p.add_argument("--with-norm", action="store_true",
                   help="Also plot a second panel normalized by each language's mean")
    p.add_argument("--xmax-pct", type=float, default=99.5,
                   help="x-axis ends at the max over languages of this per-language percentile")
    p.add_argument("--out", default="entropy_distributions.png")
    args = p.parse_args()

    unit = "character" if args.char_level else "byte"
    if args.diff and args.with_norm:
        raise SystemExit("--with-norm is not meaningful with --diff (mean jump is ~0)")

    files = sorted(Path(args.results_dir).glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files found in {args.results_dir}")

    data = {}
    for f in files:
        ents = load_entropies(f, args.char_level, args.exclude_whitespace,
                              args.byte_entropy_index, diff=args.diff)
        if ents.size == 0:
            print(f"Skipping {f.name}: no {'chars' if args.char_level else 'bytes'}_entropies")
            continue
        data[f.stem] = ents
        if f.stem not in SCRIPT_TYPE:
            print(f"Warning: no script type for {f.stem}, treating as Alphabetic")

    # Cross-check against the EntropySkew numbers
    if args.diff:
        thr = args.threshold
        print(f"{'lang':<10} {'script':<14} {'n':>9} {'std':>7} {'skew':>7} {'>thr':>7}")
        key = (lambda kv: -(kv[1] > thr).mean()) if thr is not None else (lambda kv: -kv[1].std())
        for lang, e in sorted(data.items(), key=key):
            above = f"{(e > thr).mean():>7.3f}" if thr is not None else f"{'-':>7}"
            print(f"{lang:<10} {SCRIPT_TYPE.get(lang, '?'):<14} {e.size:>9} {e.std():>7.3f} "
                  f"{skewness(e):>7.3f} {above}")
    else:
        print(f"{'lang':<10} {'script':<14} {'n':>9} {'mean':>7} {'skew':>7} {'cv':>6} {'<0.1 bit':>9}")
        for lang, e in sorted(data.items(), key=lambda kv: -skewness(kv[1])):
            print(f"{lang:<10} {SCRIPT_TYPE.get(lang, '?'):<14} {e.size:>9} {e.mean():>7.3f} "
                  f"{skewness(e):>7.3f} {e.std() / e.mean():>6.3f} {(e < 0.1).mean():>9.3f}")

    # Per-language percentile, so dense scripts with long tails are not cut off
    raw_max = max(np.percentile(e, args.xmax_pct) for e in data.values())
    raw_min = min(np.percentile(e, 100 - args.xmax_pct) for e in data.values()) if args.diff else 0.0
    raw_bins = np.linspace(raw_min, raw_max, args.bins + 1)
    panels = [("raw", raw_bins)]
    if args.with_norm:
        norm_max = max(np.percentile(e / e.mean(), args.xmax_pct) for e in data.values())
        panels.append(("norm", np.linspace(0, norm_max, args.bins + 1)))

    fig, axes = plt.subplots(1, len(panels), figsize=(9 * len(panels), 6), squeeze=False)
    for ax, (kind, bins) in zip(axes[0], panels):
        normalize = kind == "norm"
        plot_panel(ax, data, bins, normalize, args.logy,
                   threshold=None if normalize else args.threshold,
                   highlight=set(args.highlight) if args.highlight else None)
        if normalize:
            ax.set_xlabel(f"Per-{unit} entropy / language mean")
            ax.set_title("Shape (normalized by mean)")
        else:
            if args.diff:
                ax.set_xlabel(f"Entropy jump between consecutive {unit}s (bits)")
                ax.axvline(0, color="gray", linewidth=0.8, alpha=0.6)
            else:
                ax.set_xlabel(f"Per-{unit} entropy (bits)")
            if len(panels) > 1:
                ax.set_title(f"Raw per-{unit} entropy")

    present = [s for s in SCRIPT_ORDER if s in {SCRIPT_TYPE.get(l, "Alphabetic") for l in data}]
    handles = [Line2D([0], [0], color=SCRIPT_COLORS[s],
                      linestyle="-" if s in DENSE_SCRIPTS else "--", linewidth=2, label=s)
               for s in present]
    legend_loc = "lower left" if args.with_norm else "upper right"
    axes[0][-1].legend(handles=handles, title="Script type (solid = dense)", loc=legend_loc)

    title = (f"Distribution of entropy jumps between consecutive {unit}s" if args.diff
             else f"Distribution of per-{unit} entropy")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()