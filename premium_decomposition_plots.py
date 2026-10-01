#!/usr/bin/env python3
"""
premium_decomposition_plots.py -- Three separate figures from the JSON of
word_position_entropy.py:

    1. Premium (total)          vs. word ratio
    2. Onset part of premium    vs. word ratio
    3. Word-internal part       vs. word ratio

Each part = (boundaries of the language in that category) / (English total
boundaries), so  premium = onset + internal + rest (spaces, punctuation).
Every figure gets the same group fit (common slope + dense offset) as the
original premium plot; because OLS is linear, the slopes and offsets of
onset + internal + rest add up exactly to those of the total.

Word ratio = word onsets of the language / word onsets of English
(for cmn/jpn/tha: mean over segmenters, as in the JSON's "summary").

The plotting functions (label_points, scatter_with_fit, warn_legend_overlap)
are copied unchanged from the main plotting script. The STYLE block below
re-creates its constants; if you can import them from that script instead,
replace the STYLE block with that import.

USAGE
    python3 premium_decomposition_plots.py word_position_entropy.json \
        --subtitle "Balanced-Custom, global threshold (t = 2.0176)" --out-dir plots/
"""
import argparse, json, math, os, sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

# ---------------------------------------------------------------------------
# STYLE (re-created; replace with an import from the main script if possible)
# ---------------------------------------------------------------------------
FS_TITLE, FS_SUBTITLE = 16, 12
FS_AXIS_LABEL, FS_TICKS = 15, 12
FS_LEGEND, FS_LEGEND_TITLE = 11.5, 12
FS_POINT_LABEL = 11
POINT_SIZE = 160
LEGEND_LOC = "lower right"
DEFAULT_POINT_COLOR = "#4C72B0"
IGNORED_FILL_COLOR = "#000000"
IGNORED_LABEL_COLOR = "#888888"
GROUP_LINE_COLORS = ["#2f4b7c", "#b5473a"]          # fit lines, all figures
DENSITY_COLORS = ["#4C72B0", "#C44E52"]             # points, onset/internal figures

SCRIPT_PALETTE = {"Alphabetic": "#4C72B0", "Abjad": "#DD8452", "Abugida": "#55A868",
                  "Syllabary": "#C44E52", "Logosyllabary": "#8172B3",
                  "Logographic": "#937860"}
DENSE_TYPES = {"Abjad", "Syllabary", "Logosyllabary", "Logographic"}

CODE_TO_LANG_NAME = {
    "amh_Ethi": "Amharic", "arb_Arab": "Arabic", "cmn_Hans": "Mandarin Chinese",
    "deu_Latn": "German", "eng_Latn": "English", "fin_Latn": "Finnish",
    "fra_Latn": "French", "heb_Hebr": "Hebrew", "hrv_Latn": "Croatian",
    "ita_Latn": "Italian", "jpn_Jpan": "Japanese", "kat_Geor": "Georgian",
    "kor_Hang": "Korean", "nya_Latn": "Chichewa", "ron_Latn": "Romanian",
    "spa_Latn": "Spanish", "srp_Cyrl": "Serbian", "tam_Taml": "Tamil",
    "tha_Thai": "Thai", "vie_Latn": "Vietnamese"}


def is_split_color(col):
    return isinstance(col, (tuple, list)) and len(col) == 2 and all(
        isinstance(c, str) for c in col)


def p_str(p):
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def pearson_r_p(x, y, extra_df_used=0):
    """Pearson r; p from a t-test with n - 2 - extra_df_used degrees of freedom."""
    r = float(np.corrcoef(x, y)[0, 1])
    df = len(x) - 2 - extra_df_used
    t = r * math.sqrt(df / max(1e-12, 1 - r * r))
    return r, float(2 * stats.t.sf(abs(t), df))


def category_legend_handles(coloring):
    return [Line2D([], [], marker="o", linestyle="none", markersize=math.sqrt(POINT_SIZE),
                   markerfacecolor=col, markeredgecolor="#333333", markeredgewidth=0.8,
                   label=f"{name} (n={n})")
            for name, col, n in coloring["categories"]]


def ignored_legend_handle(n):
    return Line2D([], [], marker="o", linestyle="none", label=f"excluded (n={n})",
                  markerfacecolor="#cccccc", markeredgecolor="#333333")


def breakdown_legend_lines(breakdown):
    return list(breakdown)


def group_fit(x, y, is_dense, labels=("non-dense", "dense")):
    """Common slope + dense offset (OLS), in the dict format scatter_with_fit expects."""
    x, y, g = (np.asarray(a, dtype=float) for a in (x, y, is_dense))
    X = np.column_stack([np.ones_like(x), x, g])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    r2 = 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))
    # partial r of x and y, controlling for group membership
    rx = x - np.column_stack([np.ones_like(g), g]) @ np.linalg.lstsq(
        np.column_stack([np.ones_like(g), g]), x, rcond=None)[0]
    ry = y - np.column_stack([np.ones_like(g), g]) @ np.linalg.lstsq(
        np.column_stack([np.ones_like(g), g]), y, rcond=None)[0]
    pr, pp = pearson_r_p(rx, ry, extra_df_used=1)
    groups = []
    for val, lab in zip((0.0, 1.0), labels):
        m = g == val
        groups.append(dict(label=lab, n=int(m.sum()), intercept=b[0] + b[2] * val,
                           x_min=float(x[m].min()), x_max=float(x[m].max())))
    return dict(slope=float(b[1]), offset=float(b[2]), r2=float(r2),
                partial_r=pr, partial_p=pp, groups=groups)


# ---------------------------------------------------------------------------
# Plotting functions -- copied unchanged from the main plotting script
# ---------------------------------------------------------------------------

def label_points(ax, x, y, langs, avoid=None, muted=None):
    """Language-name labels next to each point (adjustText if installed).
    muted: optional bool per point -- drawn grey italic (excluded points)."""
    names = [CODE_TO_LANG_NAME.get(l, l) for l in langs] if langs else []
    if not names:
        return []
    muted = list(muted) if muted is not None else [False] * len(names)
    style = lambda i: (dict(color=IGNORED_LABEL_COLOR, fontstyle="italic") if muted[i]
                       else dict(color="#222222"))
    try:
        from adjustText import adjust_text
    except ImportError:
        adjust_text = None

    if adjust_text is not None:
        texts = [ax.text(xv, yv, nm, fontsize=FS_POINT_LABEL, zorder=4, **style(i))
                 for i, (xv, yv, nm) in enumerate(zip(x, y, names))]
        arrows = dict(arrowstyle="-", color="#999999", lw=0.6)
        objs = [a for a in (avoid or []) if a is not None]
        try:
            adjust_text(texts, x=list(x), y=list(y), ax=ax, expand=(1.4, 1.8),
                        objects=objs or None, arrowprops=arrows)
        except TypeError:
            adjust_text(texts, x=list(x), y=list(y), ax=ax, arrowprops=arrows)
        return texts

    x_groups = defaultdict(list)
    for i, xv in enumerate(x):
        x_groups[round(float(xv), 2)].append(i)
    texts = []
    for idxs in x_groups.values():
        idxs.sort(key=lambda i: y[i], reverse=True)
        for rank, i in enumerate(idxs):
            texts.append(ax.annotate(
                names[i], (x[i], y[i]), textcoords="offset points",
                xytext=(8, 6 + rank * 14), fontsize=FS_POINT_LABEL, **style(i),
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5, shrinkA=0, shrinkB=4)))
    return texts


def scatter_with_fit(ax, x, y, langs, xlabel, ylabel, title, extra_df_used=0,
                     coloring=None, show_category_legend=True, subtitle=None,
                     breakdown=None, invert_x=False, invert_y=False, compact=False,
                     groupfit=None, ignore_mask=None):
    """Scatter plot with regression line and one combined legend.
    groupfit: optional dict from group_fit -- draws one parallel line per
    group (common slope) and lists them in the legend; the joint line is
    then drawn lighter for reference.
    ignore_mask: optional bool per point -- those points are drawn (half
    black) but left out of the fit and of r / p."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ignored = (np.zeros(len(x), dtype=bool) if ignore_mask is None
               else np.asarray(ignore_mask, dtype=bool))
    keep = ~ignored
    xk, yk = x[keep], y[keep]
    r, p = pearson_r_p(xk, yk, extra_df_used=extra_df_used)
    scale = 0.85 if compact else 1.0

    cols = coloring["colors"] if coloring is not None else [DEFAULT_POINT_COLOR] * len(x)
    solid = [i for i, col in enumerate(cols) if keep[i] and not is_split_color(col)]
    if solid:
        ax.scatter(x[solid], y[solid], c=[cols[i] for i in solid], s=POINT_SIZE,
                   edgecolors="#333333", linewidths=0.8, zorder=3)
    for i, col in enumerate(cols):
        if keep[i] and not is_split_color(col):
            continue
        if ignored[i]:
            # Excluded point: own colour on the left, black on the right.
            left, right = (col[0] if is_split_color(col) else col), IGNORED_FILL_COLOR
        else:
            left, right = col
        ax.plot(x[i], y[i], marker="o", linestyle="", markersize=math.sqrt(POINT_SIZE),
                fillstyle="left", markerfacecolor=left, markerfacecoloralt=right,
                markeredgecolor="#333333", markeredgewidth=0.8, zorder=3)

    handles = []
    if coloring is not None and show_category_legend:
        handles += category_legend_handles(coloring)
    if ignored.any() and show_category_legend:
        handles.append(ignored_legend_handle(int(ignored.sum())))
    fit_line = None
    if len(xk) >= 2:
        coefs = np.polyfit(xk, yk, 1)
        x_line = np.linspace(xk.min(), xk.max(), 100)
        joint_kw = dict(color="red", linewidth=1.8, alpha=1.0)
        if groupfit is not None:
            joint_kw = dict(color="#999999", linewidth=1.4, alpha=0.8)
        fit_handle, = ax.plot(x_line, np.polyval(coefs, x_line), linestyle="--", zorder=2,
                               label=f"{'Joint fit' if groupfit else 'Linear fit'}: "
                                     f"r = {0 if round(r, 2) == 0 else r:.2f}, "
                                     f"r$^2$ = {r * r:.2f}, p = {p_str(p)}",
                               **joint_kw)
        handles.append(fit_handle)
        fit_line = fit_handle
    if groupfit is not None:
        for g, col in zip(groupfit["groups"], GROUP_LINE_COLORS):
            xs = np.linspace(g["x_min"], g["x_max"], 50)
            h, = ax.plot(xs, groupfit["slope"] * xs + g["intercept"], linestyle="-",
                         color=col, linewidth=2.2, zorder=2,
                         label=f"{g['label']} (n={g['n']}): intercept {g['intercept']:.2f}")
            handles.append(h)
        rp = groupfit["partial_r"]
        handles.append(Line2D([], [], linestyle="none", marker="none",
                              label=f"Common slope {groupfit['slope']:.2f}, offset "
                                    f"{groupfit['offset']:+.2f}: R$^2$ = {groupfit['r2']:.2f}"))
        handles.append(Line2D([], [], linestyle="none", marker="none",
                              label=f"Partial r (controlling for group) = {rp:.2f}, "
                                    f"p = {p_str(groupfit['partial_p'])}"))
    if breakdown is not None:
        handles += [Line2D([], [], linestyle="none", marker="none", label=line)
                    for line in breakdown_legend_lines(breakdown)]

    ax.set_xlabel(xlabel, fontsize=FS_AXIS_LABEL * scale)
    ax.set_ylabel(ylabel, fontsize=FS_AXIS_LABEL * scale)
    ax.tick_params(labelsize=FS_TICKS * scale)
    if subtitle:
        ax.set_title(title, fontsize=FS_TITLE * scale, fontweight="bold", pad=26)
        ax.text(0.5, 1.012, subtitle, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=FS_SUBTITLE * scale, color="#555555")
    else:
        ax.set_title(title, fontsize=FS_TITLE * scale, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35, zorder=0)

    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()

    legend = None
    if handles:
        title_kw = {}
        if coloring is not None and show_category_legend:
            title_kw = dict(title=coloring["title"], title_fontsize=FS_LEGEND_TITLE * scale)
        legend_kw = dict(handles=handles, loc=LEGEND_LOC, fontsize=FS_LEGEND * scale,
                         framealpha=0.92, **title_kw)
        try:
            legend = ax.legend(alignment="left", **legend_kw)  # matplotlib >= 3.6
        except TypeError:
            legend = ax.legend(**legend_kw)
        legend.set_zorder(5)

    texts = label_points(ax, x, y, langs, avoid=[legend], muted=ignored)

    ax._overlap_check = dict(legend=legend, x=x, y=y, langs=list(langs or []),
                             texts=texts, fit_line=fit_line, title=title)

    return r, p


def warn_legend_overlap(fig, ax, where=""):
    """Warns on stderr if the legend covers points, labels or the fit line."""
    info = getattr(ax, "_overlap_check", None)
    if not info or info["legend"] is None:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = info["legend"].get_window_extent(renderer)
    name = lambda c: CODE_TO_LANG_NAME.get(c, c)

    radius = math.sqrt(POINT_SIZE) / 2 * fig.dpi / 72
    pts = ax.transData.transform(np.column_stack([info["x"], info["y"]]))
    hit_points = [name(l) for (px, py), l in zip(pts, info["langs"])
                  if box.x0 - radius <= px <= box.x1 + radius
                  and box.y0 - radius <= py <= box.y1 + radius]
    hit_labels = [t.get_text() for t in info["texts"]
                  if t.get_window_extent(renderer).overlaps(box)]
    hit_line = False
    if info["fit_line"] is not None:
        lx, ly = info["fit_line"].get_data()
        lp = ax.transData.transform(np.column_stack([lx, ly]))
        hit_line = any(box.contains(px, py) for px, py in lp)

    parts = []
    if hit_points:
        parts.append(f"points ({', '.join(hit_points)})")
    if hit_labels:
        parts.append(f"labels ({', '.join(hit_labels)})")
    if hit_line:
        parts.append("the regression line")
    if parts:
        where_str = f" [{where}]" if where else ""
        print(f"Warning{where_str}: legend ({LEGEND_LOC}) overlaps {'; '.join(parts)}. "
              f"Try --legend-loc to move it.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

def load_parts(path):
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    if "eng_Latn" not in res:
        sys.exit("eng_Latn missing from the JSON -- cannot normalise to English.")
    eng = res["eng_Latn"]["summary"]
    E_b = eng["n_boundaries"]
    E_w = eng["categories"]["onset"]["n_chars"]
    rows = []
    for lang, r in sorted(res.items()):
        s, c = r["summary"], r["summary"]["categories"]
        rows.append(dict(
            lang=lang, script=r.get("script_type") or "Alphabetic",
            word_ratio=c["onset"]["n_chars"] / E_w,
            total=s["n_boundaries"] / E_b,
            onset=c["onset"]["n_boundaries"] / E_b,
            internal=c["internal"]["n_boundaries"] / E_b,
            rest=(c["space"]["n_boundaries"] + c["other"]["n_boundaries"]) / E_b))
    return rows


def make_coloring(rows):
    order = list(SCRIPT_PALETTE)
    counts = defaultdict(int)
    for r in rows:
        counts[r["script"]] += 1
    return dict(title="Script type",
                colors=[SCRIPT_PALETTE.get(r["script"], DEFAULT_POINT_COLOR) for r in rows],
                categories=[(t, SCRIPT_PALETTE[t], counts[t]) for t in order if counts[t]])


def make_density_coloring(rows):
    """Points coloured like the group fit lines: non-dense vs. dense."""
    dense = [r["script"] in DENSE_TYPES for r in rows]
    n_d = sum(dense)
    return dict(title="Script density",
                colors=[DENSITY_COLORS[int(d)] for d in dense],
                categories=[("non-dense", DENSITY_COLORS[0], len(rows) - n_d),
                            ("dense", DENSITY_COLORS[1], n_d)])


FIGURES = [
    ("total", "Premium", "Premium vs. Word ratio (rel. to English)", "premium_total"),
    ("onset", "Onset part of premium",
     "Onset part of premium vs. Word ratio (rel. to English)", "premium_onset_part"),
    ("internal", "Word-internal part of premium",
     "Word-internal part of premium vs. Word ratio (rel. to English)",
     "premium_internal_part"),
]


def main():
    global LEGEND_LOC
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--subtitle", default="Balanced-Custom, global threshold (t = 2.0176)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--prefix", default="Balanced-Custom_")
    ap.add_argument("--legend-loc", nargs=3, default=["lower right", "upper left", "upper right"],
                    metavar=("TOTAL", "ONSET", "INTERNAL"),
                    help="legend location per figure (matplotlib loc strings)")
    ap.add_argument("--script-colors", action="store_true",
                    help="colour the onset/internal figures by script type too "
                         "(default: by density, matching the group fit lines)")
    ap.add_argument("--free-y", action="store_true",
                    help="let each figure choose its own y-range (default: all three "
                         "use the same y-span so slopes/offsets are visually comparable)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_parts(args.json)
    langs = [r["lang"] for r in rows]
    x = np.array([r["word_ratio"] for r in rows])
    dense = np.array([r["script"] in DENSE_TYPES for r in rows])
    coloring = make_coloring(rows)
    density_coloring = make_density_coloring(rows)

    # Check: the parts must add up to the premium.
    gap = max(abs(r["total"] - r["onset"] - r["internal"] - r["rest"]) for r in rows)
    assert gap < 1e-9, f"parts do not sum to the premium (max gap {gap})"

    # Shared y-span: the largest data range of the three, plus margin.
    ys = {k: np.array([r[k] for r in rows]) for k in ("total", "onset", "internal", "rest")}
    span = max(np.ptp(ys[k]) for k, *_ in FIGURES) * 1.18

    fits = {k: group_fit(x, ys[k], dense) for k in ys}
    print("Group fits (common slope, dense offset):")
    for k, f in fits.items():
        print(f"  {k:9s} slope {f['slope']:+.3f}   offset {f['offset']:+.3f}   R2 {f['r2']:.2f}")
    print(f"  onset + internal + rest: slope "
          f"{sum(fits[k]['slope'] for k in ('onset', 'internal', 'rest')):+.3f}, offset "
          f"{sum(fits[k]['offset'] for k in ('onset', 'internal', 'rest')):+.3f}")

    for (key, ylabel, title, stem), loc in zip(FIGURES, args.legend_loc):
        LEGEND_LOC = loc
        by_density = key != "total" and not args.script_colors
        fig, ax = plt.subplots(figsize=(10.5, 8))
        y = ys[key]
        if not args.free_y:
            mid = (y.max() + y.min()) / 2
            # leave room for the labels above the highest point
            ax.set_ylim(mid - span / 2, mid + span / 2)
            ax.set_autoscaley_on(False)
        scatter_with_fit(ax, x, y, langs, "Word ratio (rel. to English)", ylabel, title,
                         coloring=density_coloring if by_density else coloring,
                         subtitle=args.subtitle, groupfit=fits[key])
        plt.tight_layout()
        out = os.path.join(args.out_dir, f"{args.prefix}{stem}.png")
        warn_legend_overlap(fig, ax, os.path.basename(out))
        plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()