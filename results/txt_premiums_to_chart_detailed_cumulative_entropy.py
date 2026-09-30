#!/usr/bin/env python3
"""
Cumulative entropy premium color chart.

Draws ONE chart with a row per language and a column per
(setup, level, cumulative_metric) combination:

                          imbalanced                                    balanced ...
                 normal               char-level                normal ...
           nomono mono_0 rel0p5 rel0p25  ...
    English
    German
    ...
    Δ (max-min)

Examples:
  python txt_premiums_to_chart_cumulative.py
  python txt_premiums_to_chart_cumulative.py --suppress '*_nomono'
  python txt_premiums_to_chart_cumulative.py --latex cumulative_premiums.tex
"""

import argparse
import fnmatch
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SETUPS = ["imbalanced", "balanced", "balanced-custom"]
LEVELS = ["normal", "char-level"]

# Key identifiers mapping to substring/patterns in filenames
METRICS = ["cumulative_nomono", "cumulative_mono_rel0p25"]

SETUP_LABELS = {"imbalanced": "Imbalanced", "balanced": "Balanced", "balanced-custom": "Balanced-Custom"}
LEVEL_LABELS = {"normal": "normal", "char-level": "char-level"}
METRIC_LABELS = {
    "cumulative_nomono": "pure",
    "cumulative_mono_0": "mono_0",
    "cumulative_mono_rel0p5": "mono_rel0p5",
    "cumulative_mono_rel0p25": "mono: 0.25t"
}

SETTING_DIRS = {
    ("imbalanced", "normal"): "results/txt_premiums/t_anchor/Imbalanced/step_0000002600",
    ("balanced", "normal"): "results/txt_premiums/t_anchor/Balanced/step_0000007200",
    ("balanced-custom", "normal"): "results/txt_premiums/t_anchor/Balanced-Custom/step_0000006400",
    ("imbalanced", "char-level"): "results/txt_premiums/char_level/t_anchor/Imbalanced/step_0000002600",
    ("balanced", "char-level"): "results/txt_premiums/char_level/t_anchor/Balanced/step_0000007200",
    ("balanced-custom", "char-level"): "results/txt_premiums/char_level/t_anchor/Balanced-Custom/step_0000006400",
}

LEVEL_ALIASES = {
    "normal": "normal", "byte": "normal", "byte-level": "normal", "bytes": "normal",
    "char-level": "char-level", "char-lvl": "char-level", "char": "char-level",
    "charlevel": "char-level", "char_level": "char-level",
}

METRIC_ALIASES = {
    "nomono": "cumulative_nomono", "cumulative_nomono": "cumulative_nomono",
    "mono_0": "cumulative_mono_0", "cumulative_mono_0": "cumulative_mono_0",
    "rel0p5": "cumulative_mono_rel0p5", "cumulative_mono_rel0p5": "cumulative_mono_rel0p5",
    "rel0p25": "cumulative_mono_rel0p25", "cumulative_mono_rel0p25": "cumulative_mono_rel0p25",
}

CODE_TO_LANG_NAME = {
    "eng_Latn": "English", "cmn_Hans": "Mandarin Chinese", "deu_Latn": "German",
    "jpn_Jpan": "Japanese", "spa_Latn": "Spanish", "fra_Latn": "French",
    "ita_Latn": "Italian", "vie_Latn": "Vietnamese", "arb_Arab": "Arabic",
    "tha_Thai": "Thai", "kor_Hang": "Korean", "ron_Latn": "Romanian",
    "fin_Latn": "Finnish", "heb_Hebr": "Hebrew", "tam_Taml": "Tamil",
    "hrv_Latn": "Croatian", "srp_Cyrl": "Serbian", "kat_Geor": "Georgian",
    "amh_Ethi": "Amharic", "nya_Latn": "Chichewa",
}

KNOWN_HEADERS = ["Language", "Premium", "PPS", "BPP", "EntropyMean",
                 "EntropyVar", "EntropySkew", "EntropyKurtosis",
                 "EntropyAutocorr1", "EntropyVolatility"]


def col_key(setup, level, metric):
    return f"{setup}_{level}_{metric}"


def parse_setting_key(key):
    key = key.strip().lower()
    for setup in sorted(SETUPS, key=len, reverse=True):
        if key.startswith(setup + "_"):
            level = LEVEL_ALIASES.get(key[len(setup) + 1:])
            if level:
                return setup, level
    raise ValueError(f"Unknown setting '{key}'. Expected <setup>_<level>.")


def normalize_pattern(pat):
    parts = pat.strip().lower().split("_")
    parts = [LEVEL_ALIASES.get(p, METRIC_ALIASES.get(p, p)) for p in parts]
    return "_".join(parts)


# --------------------------------------------------------------------------- #
# Input parsing
# --------------------------------------------------------------------------- #

def find_cumulative_files(folder):
    """Finds matching cumulative files for each metric in the target folder."""
    files = [f for f in os.listdir(folder) if f.endswith("_premiums_sorted.txt")]
    metric_files = {}

    for metric in METRICS:
        matches = [f for f in files if f"_{metric}_" in f]
        if len(matches) == 1:
            metric_files[metric] = os.path.join(folder, matches[0])
        elif len(matches) > 1:
            # Fallback exact match if multiple exist
            matches_sorted = sorted(matches, key=len)
            metric_files[metric] = os.path.join(folder, matches_sorted[0])

    return metric_files


def _to_float(raw):
    try:
        return float(str(raw).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def read_premiums(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    header_idx = next((i for i, l in enumerate(lines) if l.strip() and not l.lstrip().startswith("#")), None)
    if header_idx is None:
        raise ValueError(f"No header line found in {path}")
    header_line = lines[header_idx].rstrip("\n")

    present, positions, search_from = [], [], 0
    for h in KNOWN_HEADERS:
        idx = header_line.find(h, search_from)
        if idx == -1:
            continue
        present.append(h)
        positions.append(idx)
        search_from = idx + len(h)
    if "Premium" not in present:
        raise ValueError(f"No 'Premium' column found in header of {path}")
    bounds = dict(zip(present, zip(positions, positions[1:] + [None])))

    def cell(line, h):
        a, b = bounds[h]
        return (line[a:b] if b is not None else line[a:]).strip()

    data = {}
    for line in lines[header_idx + 1:]:
        if not line.strip() or line.lstrip().startswith("#"):
            break
        line = line.rstrip("\n")
        lang, prem = cell(line, "Language"), _to_float(cell(line, "Premium"))
        if lang and prem is not None:
            data[lang] = prem
    if not data:
        raise ValueError(f"No valid data rows parsed from {path}")
    return data


def build_columns(setting_dirs):
    columns, skipped = [], []
    for setup in SETUPS:
        for level in LEVELS:
            folder = setting_dirs.get((setup, level))
            if not folder or not os.path.isdir(folder):
                skipped.append((f"{setup}_{level}", "not configured" if not folder else f"missing dir {folder}"))
                continue
            
            found_files = find_cumulative_files(folder)
            for metric in METRICS:
                path = found_files.get(metric)
                if not path:
                    skipped.append((col_key(setup, level, metric), f"missing file in {folder}"))
                    continue
                columns.append(dict(key=col_key(setup, level, metric), setup=setup, level=level,
                                    metric=metric, path=path, data=read_premiums(path)))
    return columns, skipped


def apply_suppression(columns, patterns):
    patterns = [normalize_pattern(p) for p in patterns]
    all_keys = [c["key"] for c in columns]
    for p in patterns:
        if not fnmatch.filter(all_keys, p):
            print(f"warning: --suppress '{p}' matched no column (available: {', '.join(all_keys)})")
    return [c for c in columns if not any(fnmatch.fnmatch(c["key"], p) for p in patterns)]


def order_languages(columns, sort_by, reverse=False):
    langs = []
    for c in columns:
        for l in c["data"]:
            if l not in langs:
                langs.append(l)

    if sort_by == "file":
        return langs[::-1] if reverse else langs
    if sort_by == "name":
        return sorted(langs, key=lambda l: CODE_TO_LANG_NAME.get(l, l), reverse=reverse)
    if sort_by == "mean":
        def mean(l):
            vals = [c["data"][l] for c in columns if l in c["data"]]
            return sum(vals) / len(vals)
        return sorted(langs, key=mean, reverse=reverse)

    col = next((c for c in columns if c["key"] == normalize_pattern(sort_by)), None)
    if col is None:
        raise ValueError(f"--sort-by '{sort_by}' is not one of: file, name, mean, "
                         f"{', '.join(c['key'] for c in columns)}")
    present = sorted((l for l in langs if l in col["data"]), key=col["data"].get, reverse=reverse)
    return present + [l for l in langs if l not in col["data"]]


# --------------------------------------------------------------------------- #
# Colors
# --------------------------------------------------------------------------- #

def green_red_gradient(t):
    t = np.clip(t, 0, 1)
    if t < 0.5:
        tt = t / 0.5
        r = 0.13 + tt * (0.95 - 0.13)
        g = 0.62 + tt * (0.75 - 0.62)
        b = 0.20 + tt * (0.15 - 0.20)
    else:
        tt = (t - 0.5) / 0.5
        r = 0.95 + tt * (0.75 - 0.95)
        g = 0.75 + tt * (0.10 - 0.75)
        b = 0.15 + tt * (0.10 - 0.15)
    return (r, g, b)


def premium_color(p, p_min, p_max):
    t = (p - p_min) / (p_max - p_min) if p_max > p_min else 0
    return green_red_gradient(t)


def text_color_for(bg):
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return "black" if lum > 0.55 else "white"


def scale_group(col, mode):
    return {"column": col["key"], "metric": col["metric"], "setup": col["setup"],
            "level": col["level"], "all": "all"}[mode]


def compute_scales(columns, langs, scale):
    groups = {}
    for c in columns:
        groups.setdefault(scale_group(c, scale), []).extend(c["data"][l] for l in langs if l in c["data"])
    group_range = {g: (min(v), max(v)) for g, v in groups.items()}
    col_delta = {}
    for c in columns:
        v = [c["data"][l] for l in langs if l in c["data"]]
        col_delta[c["key"]] = max(v) - min(v) if v else 0.0
    return group_range, col_delta


# --------------------------------------------------------------------------- #
# LaTeX Table
# --------------------------------------------------------------------------- #

BOX_W = "2.6em"

def write_latex(columns, langs, out_path, scale="column", delta_pos="bottom",
                caption=None, label=None, bare=False, fit=False, bold=False, boxes=False):
    group_range, col_delta = compute_scales(columns, langs, scale)

    setups = list(dict.fromkeys(c["setup"] for c in columns))
    levels_of = {s: list(dict.fromkeys(c["level"] for c in columns if c["setup"] == s)) for s in setups}
    n_of = lambda pred: sum(1 for c in columns if pred(c))

    spec = ["l"]
    prev = None
    for c in columns:
        if prev is not None and c["setup"] != prev["setup"]:
            spec.append(r"!{\hspace{1em}}")
        elif prev is not None and c["level"] != prev["level"]:
            spec.append(r"!{\hspace{0.3em}}")
        spec.append("c")
        prev = c
    spec = "".join(spec)

    setup_cells, setup_rules, level_cells, level_rules = [""], [], [""], []
    col = 2
    for s in setups:
        n_s = n_of(lambda c: c["setup"] == s)
        setup_cells.append(rf"\multicolumn{{{n_s}}}{{c}}{{\textbf{{{SETUP_LABELS.get(s, s)}}}}}")
        setup_rules.append(rf"\cmidrule(lr){{{col}-{col + n_s - 1}}}")
        for l in levels_of[s]:
            n_l = n_of(lambda c: c["setup"] == s and c["level"] == l)
            level_cells.append(rf"\multicolumn{{{n_l}}}{{c}}{{{LEVEL_LABELS.get(l, l)}}}")
            level_rules.append(rf"\cmidrule(lr){{{col}-{col + n_l - 1}}}")
            col += n_l
    metric_cells = [r"\textbf{Language}"] + [METRIC_LABELS.get(c["metric"], c["metric"]) for c in columns]

    def cell(c, lang):
        val = c["data"].get(lang)
        if val is None:
            dash = r"\textcolor{gray}{--}"
            return rf"\makebox[{BOX_W}]{{{dash}}}" if boxes else dash
        rgb = premium_color(val, *group_range[scale_group(c, scale)])
        r, g, b = (int(round(255 * x)) for x in rgb)
        txt = f"{val:.2f}"
        if bold:
            txt = rf"\textbf{{{txt}}}"
        if text_color_for(rgb) == "white":
            txt = rf"\textcolor{{white}}{{{txt}}}"
        if boxes:
            return rf"\colorbox[RGB]{{{r},{g},{b}}}{{\makebox[{BOX_W}]{{{txt}}}}}"
        return rf"\cellcolor[RGB]{{{r},{g},{b}}}{txt}"

    delta_row = " & ".join([r"$\Delta$ (max$-$min)"] + [f"{col_delta[c['key']]:.2f}" for c in columns]) + r" \\"

    lines = [r"\begin{tabular}{" + spec + "}", r"\toprule",
             " & ".join(setup_cells) + r" \\", " ".join(setup_rules),
             " & ".join(level_cells) + r" \\", " ".join(level_rules),
             " & ".join(metric_cells) + r" \\", r"\midrule"]
    if delta_pos == "top":
        lines += [delta_row, r"\midrule"]
    for lang in langs:
        name = CODE_TO_LANG_NAME.get(lang, lang)
        lines.append(" & ".join([name] + [cell(c, lang) for c in columns]) + r" \\")
    if delta_pos == "bottom":
        lines += [r"\midrule", delta_row]
    lines += [r"\bottomrule", r"\end{tabular}"]

    if boxes:
        lines = [r"\begingroup", r"\setlength{\fboxsep}{1.5pt}"] + lines + [r"\endgroup"]
    body = "\n".join(lines)
    if fit:
        body = r"\resizebox{\textwidth}{!}{%" + "\n" + body + "\n}"

    if not bare:
        if caption is None:
            caption = "Cumulative entropy premiums per language across settings."
        body = "\n".join([r"\begin{table*}[t]", r"\centering", r"\small", body,
                          rf"\caption{{{caption}}}"] +
                         ([rf"\label{{{label}}}"] if label else []) + [r"\end{table*}"])

    header = "% Generated by txt_premiums_to_chart_cumulative.py\n"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + body + "\n")
    print(f"saved {out_path}")


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

FS_TITLE, FS_SETUP, FS_LEVEL, FS_METRIC, FS_LANG, FS_CELL, FS_DELTA, FS_LEGEND = 13, 10.5, 9.5, 8.0, 9.5, 8.5, 8.5, 8.5

def text_width_in(text, fontsize, fontweight="normal"):
    fig = plt.figure()
    t = fig.text(0, 0, text, fontsize=fontsize, fontweight=fontweight)
    fig.canvas.draw()
    w = t.get_window_extent().width / fig.dpi
    plt.close(fig)
    return w


def draw_chart(columns, langs, out_path, title=None, scale="column", delta_pos="bottom",
               col_w=0.68, level_gap=0.18, setup_gap=0.4):
    labels = {l: CODE_TO_LANG_NAME.get(l, l) for l in langs}
    delta_label = "\u0394 (max\u2212min)"

    margin = 0.25
    lang_w = max(text_width_in(s, FS_LANG, "medium") for s in list(labels.values()) + [delta_label]) + 0.25
    box_w = col_w - 0.08

    setups = list(dict.fromkeys(c["setup"] for c in columns))
    levels_of = {s: list(dict.fromkeys(c["level"] for c in columns if c["setup"] == s)) for s in setups}
    cols_of = {(s, l): [c for c in columns if c["setup"] == s and c["level"] == l]
               for s in setups for l in levels_of[s]}

    def level_width(s, l):
        return max(len(cols_of[(s, l)]) * col_w,
                   text_width_in(LEVEL_LABELS.get(l, l), FS_LEVEL, "semibold") + 0.15)

    def setup_inner_width(s):
        return sum(level_width(s, l) for l in levels_of[s]) + level_gap * (len(levels_of[s]) - 1)

    col_x, group_span = {}, {}
    x = margin + lang_w
    for i, s in enumerate(setups):
        if i:
            x += setup_gap
        inner = setup_inner_width(s)
        s_w = max(inner, text_width_in(SETUP_LABELS.get(s, s), FS_SETUP, "bold") + 0.2)
        group_span[s] = (x, x + s_w)
        lx = x + (s_w - inner) / 2
        for j, l in enumerate(levels_of[s]):
            if j:
                lx += level_gap
            l_w = level_width(s, l)
            group_span[(s, l)] = (lx, lx + l_w)
            cx = lx + (l_w - len(cols_of[(s, l)]) * col_w) / 2
            for c in cols_of[(s, l)]:
                col_x[c["key"]] = cx
                cx += col_w
            lx += l_w
        x += s_w
    fig_w = x + margin

    row_h, box_h = 0.28, 0.21
    title_h = 0.45 if title else 0.0
    setup_h, level_h, metric_h = 0.32, 0.28, 0.26
    delta_h, legend_h = 0.34, 0.75
    n = len(langs)
    fig_h = margin + title_h + setup_h + level_h + metric_h + n * row_h + delta_h + legend_h + margin

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    y = fig_h - margin
    if title:
        ax.text(fig_w / 2, y - title_h / 2, title, fontsize=FS_TITLE, fontweight="bold",
                ha="center", va="center", color="#333333")
        y -= title_h

    def bracket(x0, x1, yb, color):
        ax.plot([x0 + 0.05, x1 - 0.05], [yb, yb], color=color, lw=1.0, solid_capstyle="butt")

    setup_y = y - setup_h / 2
    level_y = y - setup_h - level_h / 2
    metric_y = y - setup_h - level_h - metric_h / 2
    for setup in setups:
        x0, x1 = group_span[setup]
        ax.text((x0 + x1) / 2, setup_y, SETUP_LABELS.get(setup, setup), fontsize=FS_SETUP,
                fontweight="bold", ha="center", va="center")
        bracket(x0, x1, y - setup_h + 0.02, "#555555")
        for level in levels_of[setup]:
            x0, x1 = group_span[(setup, level)]
            ax.text((x0 + x1) / 2, level_y, LEVEL_LABELS.get(level, level), fontsize=FS_LEVEL,
                    fontweight="semibold", ha="center", va="center", color="#333333")
            bracket(x0, x1, y - setup_h - level_h + 0.03, "#aaaaaa")
    for c in columns:
        ax.text(col_x[c["key"]] + col_w / 2, metric_y, METRIC_LABELS.get(c["metric"], c["metric"]),
                fontsize=FS_METRIC, ha="center", va="center", color="#444444")
    ax.text(margin, metric_y, "Language", fontsize=FS_LEVEL, fontweight="bold", va="center")
    y -= setup_h + level_h + metric_h

    group_range, col_delta = compute_scales(columns, langs, scale)

    def draw_delta_row(y_center, sep_y):
        ax.text(margin, y_center, delta_label, fontsize=FS_LANG, va="center", fontstyle="italic", color="#333333")
        for c in columns:
            ax.text(col_x[c["key"]] + col_w / 2, y_center, f"{col_delta[c['key']]:.2f}",
                    fontsize=FS_DELTA, ha="center", va="center", fontweight="bold", color="#333333")
        ax.plot([margin, fig_w - margin], [sep_y, sep_y], color="#cccccc", lw=0.8)

    if delta_pos == "top":
        draw_delta_row(y - delta_h / 2, y - delta_h + 0.02)
        y -= delta_h

    for r, lang in enumerate(langs):
        yc = y - r * row_h - row_h / 2
        if r % 2 == 1:
            ax.add_patch(patches.Rectangle((margin - 0.05, yc - row_h / 2), fig_w - 2 * margin + 0.1, row_h,
                                           facecolor="#f5f5f5", edgecolor="none", zorder=0))
        ax.text(margin, yc, labels[lang], fontsize=FS_LANG, va="center", fontweight="medium")
        for c in columns:
            bx, by = col_x[c["key"]] + (col_w - box_w) / 2, yc - box_h / 2
            val = c["data"].get(lang)
            if val is None:
                ax.add_patch(patches.FancyBboxPatch(
                    (bx, by), box_w, box_h, boxstyle="round,pad=0.01,rounding_size=0.04",
                    linewidth=0.6, linestyle="--", edgecolor="#cccccc", facecolor="none"))
                ax.text(bx + box_w / 2, yc, "N/A", ha="center", va="center",
                        fontsize=FS_CELL, color="#aaaaaa", fontweight="bold")
                continue
            color = premium_color(val, *group_range[scale_group(c, scale)])
            ax.add_patch(patches.FancyBboxPatch(
                (bx, by), box_w, box_h, boxstyle="round,pad=0.01,rounding_size=0.04",
                linewidth=0.6, edgecolor="#333333", facecolor=color))
            ax.text(bx + box_w / 2, yc, f"{val:.2f}", ha="center", va="center",
                    fontsize=FS_CELL, color=text_color_for(color), fontweight="bold")
    y -= n * row_h

    if delta_pos == "bottom":
        ax.plot([margin, fig_w - margin], [y - 0.02, y - 0.02], color="#cccccc", lw=0.8)
        draw_delta_row(y - delta_h / 2, y - 0.02)
        y -= delta_h

    leg_y = y - 0.35
    entries = [("Cumulative Premium (per-column scale)", None)] if scale == "column" else [(f"Premium — {g}", group_range[g]) for g in group_range]
    n_leg, leg_gap = len(entries), 0.5
    avail = fig_w - 2 * margin
    grad_w = min(3.0, (avail - (n_leg - 1) * leg_gap) / n_leg)
    total = n_leg * grad_w + (n_leg - 1) * leg_gap
    lx = (fig_w - total) / 2
    steps = 200
    for label, rng in entries:
        for k in range(steps):
            ax.add_patch(patches.Rectangle((lx + k / steps * grad_w, leg_y), grad_w / steps + 0.001, 0.12,
                                           color=green_red_gradient(k / steps), linewidth=0))
        ax.text(lx + grad_w / 2, leg_y + 0.18, label, fontsize=FS_LEGEND, ha="center", va="bottom")
        lo, hi = ("column min", "column max") if rng is None else (f"{rng[0]:.2f}", f"{rng[1]:.2f}")
        ax.text(lx, leg_y - 0.05, lo, fontsize=FS_LEGEND - 0.5, ha="left", va="top")
        ax.text(lx + grad_w, leg_y - 0.05, hi, fontsize=FS_LEGEND - 0.5, ha="right", va="top")
        lx += grad_w + leg_gap

    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_kv(items, what):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--{what} expects KEY=VALUE, got '{item}'")
        k, v = item.split("=", 1)
        out[parse_setting_key(k)] = v
    return out


def main():
    parser = argparse.ArgumentParser(description="Cumulative entropy premium colour chart across settings.")
    parser.add_argument("--suppress", nargs="+", action="extend", default=[], metavar="PATTERN",
                        help="Column keys to hide (e.g. '*_nomono', 'balanced_*').")
    parser.add_argument("--dir", action="append", metavar="SETTING=PATH",
                        help="Override directory for a setting (e.g. imbalanced_char-level=results/...)")
    parser.add_argument("--sort-by", default="file",
                        help="Row order: 'file', 'mean', 'name', or exact column key.")
    parser.add_argument("--reverse", action="store_true", help="Reverse language row order.")
    parser.add_argument("--scale", choices=["column", "metric", "setup", "level", "all"], default="column")
    parser.add_argument("--delta-pos", choices=["top", "bottom"], default="bottom")
    parser.add_argument("--title", default=None, help="Chart title.")
    parser.add_argument("--col-width", type=float, default=0.68)
    parser.add_argument("--out", default="all_settings_cumulative_chart.png")
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--latex", metavar="PATH", default=None)
    parser.add_argument("--latex-bare", action="store_true")
    parser.add_argument("--latex-fit", action="store_true")
    parser.add_argument("--latex-boxes", action="store_true")
    parser.add_argument("--latex-bold", action="store_true")
    parser.add_argument("--caption", default=None)
    parser.add_argument("--label", default="tab:cumulative_premiums")
    parser.add_argument("--list-columns", action="store_true")
    args = parser.parse_args()

    setting_dirs = dict(SETTING_DIRS)
    setting_dirs.update(parse_kv(args.dir, "dir"))

    columns, skipped = build_columns(setting_dirs)
    for key, why in skipped:
        print(f"note: skipping {key} ({why})")

    if args.list_columns:
        for c in columns:
            print(f"{c['key']:50s} {c['path']}")
        return

    suppress = [p for item in args.suppress for p in item.split(",") if p.strip()]
    columns = apply_suppression(columns, suppress)
    if not columns:
        raise SystemExit("Nothing to draw: no columns left.")

    langs = order_languages(columns, args.sort_by, reverse=args.reverse)
    if not args.no_png:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        draw_chart(columns, langs, args.out, title=args.title, scale=args.scale,
                   delta_pos=args.delta_pos, col_w=args.col_width)
    if args.latex:
        write_latex(columns, langs, args.latex, scale=args.scale, delta_pos=args.delta_pos,
                   caption=args.caption, label=args.label, bare=args.latex_bare,
                   fit=args.latex_fit, bold=args.latex_bold, boxes=args.latex_boxes)


if __name__ == "__main__":
    main()