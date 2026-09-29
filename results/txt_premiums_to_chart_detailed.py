#!/usr/bin/env python3
"""
All-settings premium color chart.

Draws ONE chart with a row per language and a column per
(setup, level, metric) combination, showing only the colour-coded premium:

                     imbalanced              balanced          balanced-custom
                 normal      char-level   normal   char-level   normal  char-level
               global mono  global mono    ...
    English
    German
    ...
    Δ (max-min)   ...                                   <- per-column spread

Column keys look like  <setup>_<level>_<metric>, e.g.
    imbalanced_normal_global
    balanced-custom_char-level_mono

Settings whose directory is not configured / doesn't exist are skipped with a
warning, so the chart works with whatever results you currently have.

Examples:
  python txt_premiums_to_chart_all_settings.py
  python txt_premiums_to_chart_all_settings.py --suppress imbalanced_char-level_mono
  python txt_premiums_to_chart_all_settings.py --suppress '*_mono' 'balanced-custom_*'
  python txt_premiums_to_chart_all_settings.py --dir imbalanced_char-level=results/.../Imbalanced-Char/step_0000003000
  python txt_premiums_to_chart_all_settings.py --scale all --sort-by mean --delta-pos top
  python txt_premiums_to_chart_all_settings.py --sort-by mean --reverse     # worst -> best
  python txt_premiums_to_chart_all_settings.py --latex premiums_table.tex
  python txt_premiums_to_chart_all_settings.py --no-png --latex tab.tex --latex-bare --latex-fit
  python txt_premiums_to_chart_all_settings.py --list-columns

LaTeX output needs  \\usepackage{booktabs}  and  \\usepackage[table]{xcolor}
(plus \\usepackage{graphicx} with --latex-fit).
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

# Order here = order in the chart (left to right).
SETUPS = ["imbalanced", "balanced", "balanced-custom"]
LEVELS = ["normal", "char-level"]
METRICS = ["global", "mono"]

SETUP_LABELS = {"imbalanced": "Imbalanced", "balanced": "Balanced", "balanced-custom": "Balanced-Custom"}
LEVEL_LABELS = {"normal": "normal", "char-level": "char-level"}
METRIC_LABELS = {"global": "global", "mono": "mono"}

# Where the *_premiums_sorted.txt files for each (setup, level) live.
# Value: a directory, or (directory, filename_substring) if one folder holds
# files for several settings and you need to pick the right ones.
# None = not available yet (columns skipped). Override from the CLI with --dir / --match.
SETTING_DIRS = {
    ("imbalanced", "normal"): "results/txt_premiums/t_anchor/Imbalanced/step_0000002600",
    ("balanced", "normal"): "results/txt_premiums/t_anchor/Balanced/step_0000007200",
    ("balanced-custom", "normal"): "results/txt_premiums/t_anchor/Balanced-Custom/step_0000006400",
    ("imbalanced", "char-level"): "results/txt_premiums/char_level/t_anchor/Imbalanced/step_0000002600",
    ("balanced", "char-level"): "results/txt_premiums/char_level/t_anchor/Balanced/step_0000007200",
    ("balanced-custom", "char-level"): "results/txt_premiums/char_level/t_anchor/Balanced-Custom/step_0000006400",
}

# Accepted spellings on the CLI -> canonical name
LEVEL_ALIASES = {
    "normal": "normal", "byte": "normal", "byte-level": "normal", "bytes": "normal",
    "char-level": "char-level", "char-lvl": "char-level", "char": "char-level",
    "charlevel": "char-level", "char_level": "char-level",
}
METRIC_ALIASES = {"global": "global", "mono": "mono", "monotonicity": "mono"}

CODE_TO_LANG_NAME = {
    "eng_Latn": "English", "cmn_Hans": "Mandarin Chinese", "deu_Latn": "German",
    "jpn_Jpan": "Japanese", "spa_Latn": "Spanish", "fra_Latn": "French",
    "ita_Latn": "Italian", "vie_Latn": "Vietnamese", "arb_Arab": "Arabic",
    "tha_Thai": "Thai", "kor_Hang": "Korean", "ron_Latn": "Romanian",
    "fin_Latn": "Finnish", "heb_Hebr": "Hebrew", "tam_Taml": "Tamil",
    "hrv_Latn": "Croatian", "srp_Cyrl": "Serbian", "kat_Geor": "Georgian",
    "amh_Ethi": "Amharic", "nya_Latn": "Chichewa",
}

# Header names in the order results_to_txt_premiums.py writes them. The
# premiums file is FIXED-WIDTH, so rows are sliced at these header offsets.
KNOWN_HEADERS = ["Language", "Premium", "PPS", "BPP", "EntropyMean",
                 "EntropyVar", "EntropySkew", "EntropyKurtosis",
                 "EntropyAutocorr1", "EntropyVolatility"]


def col_key(setup, level, metric):
    return f"{setup}_{level}_{metric}"


def parse_setting_key(key):
    """'imbalanced_char-lvl' -> ('imbalanced', 'char-level')."""
    key = key.strip().lower()
    for setup in sorted(SETUPS, key=len, reverse=True):   # 'balanced-custom' before 'balanced'
        if key.startswith(setup + "_"):
            level = LEVEL_ALIASES.get(key[len(setup) + 1:])
            if level:
                return setup, level
    raise ValueError(f"Unknown setting '{key}'. Expected <setup>_<level>, "
                     f"setup in {SETUPS}, level in {LEVELS}.")


def normalize_pattern(pat):
    """Lets '--suppress imbalanced_char-lvl_monotonicity' work too."""
    parts = pat.strip().lower().split("_")
    parts = [LEVEL_ALIASES.get(p, METRIC_ALIASES.get(p, p)) for p in parts]
    return "_".join(parts)


# --------------------------------------------------------------------------- #
# Input parsing
# --------------------------------------------------------------------------- #

def find_pair(folder, match=None):
    """Returns (global_file, mono_file) inside folder."""
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
             if f.endswith("_premiums_sorted.txt") and (not match or match in f)]
    base = lambda f: os.path.basename(f).lower()

    monos = [f for f in files if "mono" in base(f)]
    globs = [f for f in files if f not in monos and ("global" in base(f) or "entropy" in base(f))]
    if len(monos) == 1 and len(globs) == 1:
        return globs[0], monos[0]
    if not monos and not globs and len(files) == 2:
        return files[0], files[1]
    raise ValueError(
        f"Could not pick exactly one global and one mono file in {folder}"
        + (f" (match='{match}')" if match else "")
        + f"; candidates: {[os.path.basename(f) for f in files]}. Use --match to disambiguate."
    )


def _to_float(raw):
    try:
        return float(str(raw).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def read_premiums(path):
    """Returns an ordered dict {lang_code: premium} in file order."""
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


def build_columns(setting_dirs, matches):
    """Returns (columns, skipped) where columns is an ordered list of dicts
    {key, setup, level, metric, path, data}."""
    columns, skipped = [], []
    for setup in SETUPS:
        for level in LEVELS:
            spec = setting_dirs.get((setup, level))
            match = matches.get((setup, level))
            if isinstance(spec, tuple):
                spec, match = spec[0], match or spec[1]
            if not spec or not os.path.isdir(spec):
                skipped.append((f"{setup}_{level}", "not configured" if not spec else f"missing dir {spec}"))
                continue
            gfile, mfile = find_pair(spec, match)
            for metric, path in (("global", gfile), ("mono", mfile)):
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
    """Row order. reverse=True flips it (e.g. highest premium first); languages
    missing from the sort column always stay at the bottom."""
    langs = []
    for c in columns:
        for l in c["data"]:
            if l not in langs:
                langs.append(l)

    if sort_by == "file":
        return langs[::-1] if reverse else langs   # union in file order, first column first
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
    """Columns in the same group share one colour scale."""
    return {"column": col["key"], "metric": col["metric"], "setup": col["setup"],
            "level": col["level"], "all": "all"}[mode]


def compute_scales(columns, langs, scale):
    """Returns ({scale_group: (min, max)}, {column_key: max - min})."""
    groups = {}
    for c in columns:
        groups.setdefault(scale_group(c, scale), []).extend(c["data"][l] for l in langs if l in c["data"])
    group_range = {g: (min(v), max(v)) for g, v in groups.items()}
    col_delta = {}
    for c in columns:
        v = [c["data"][l] for l in langs if l in c["data"]]
        col_delta[c["key"]] = max(v) - min(v)
    return group_range, col_delta


# --------------------------------------------------------------------------- #
# LaTeX table
# --------------------------------------------------------------------------- #

def _tex_escape(s):
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
             .replace("_", r"\_").replace("#", r"\#").replace("$", r"\$"))


BOX_W = "2.6em"   # width of the coloured boxes in --latex-boxes mode


def write_latex(columns, langs, out_path, scale="column", delta_pos="bottom",
                caption=None, label=None, bare=False, fit=False, bold=False, boxes=False):
    """Writes a booktabs table with \\cellcolor-coded premiums.
    Needs in the preamble:  \\usepackage{booktabs}  \\usepackage[table]{xcolor}"""
    group_range, col_delta = compute_scales(columns, langs, scale)

    setups = list(dict.fromkeys(c["setup"] for c in columns))
    levels_of = {s: list(dict.fromkeys(c["level"] for c in columns if c["setup"] == s)) for s in setups}
    n_of = lambda pred: sum(1 for c in columns if pred(c))

    # Column spec: extra space between setup groups (left uncoloured, so the
    # groups are visibly separated), a little between level groups.
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

    # Header rows (column index 1 = Language, data columns start at 2)
    setup_cells, setup_rules, level_cells, level_rules = [""], [], [""], []
    col = 2
    for s in setups:
        n_s = n_of(lambda c: c["setup"] == s)
        setup_cells.append(rf"\multicolumn{{{n_s}}}{{c}}{{\textbf{{{_tex_escape(SETUP_LABELS.get(s, s))}}}}}")
        setup_rules.append(rf"\cmidrule(lr){{{col}-{col + n_s - 1}}}")
        for l in levels_of[s]:
            n_l = n_of(lambda c: c["setup"] == s and c["level"] == l)
            level_cells.append(rf"\multicolumn{{{n_l}}}{{c}}{{{_tex_escape(LEVEL_LABELS.get(l, l))}}}")
            level_rules.append(rf"\cmidrule(lr){{{col}-{col + n_l - 1}}}")
            col += n_l
    metric_cells = [r"\textbf{Language}"] + [_tex_escape(METRIC_LABELS.get(c["metric"], c["metric"]))
                                            for c in columns]

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
        if boxes:   # \colorbox works even where colortbl's \cellcolor is disabled
            return rf"\colorbox[RGB]{{{r},{g},{b}}}{{\makebox[{BOX_W}]{{{txt}}}}}"
        return rf"\cellcolor[RGB]{{{r},{g},{b}}}{txt}"

    delta_row = " & ".join([r"$\Delta$ (max$-$min)"] +
                           [f"{col_delta[c['key']]:.2f}" for c in columns]) + r" \\"

    lines = [r"\begin{tabular}{" + spec + "}", r"\toprule",
             " & ".join(setup_cells) + r" \\", " ".join(setup_rules),
             " & ".join(level_cells) + r" \\", " ".join(level_rules),
             " & ".join(metric_cells) + r" \\", r"\midrule"]
    if delta_pos == "top":
        lines += [delta_row, r"\midrule"]
    for lang in langs:
        name = _tex_escape(CODE_TO_LANG_NAME.get(lang, lang))
        lines.append(" & ".join([name] + [cell(c, lang) for c in columns]) + r" \\")
    if delta_pos == "bottom":
        lines += [r"\midrule", delta_row]
    lines += [r"\bottomrule", r"\end{tabular}"]

    if boxes:   # smaller box padding, kept local to the table
        lines = [r"\begingroup", r"\setlength{\fboxsep}{1.5pt}"] + lines + [r"\endgroup"]
    body = "\n".join(lines)
    if fit:
        body = r"\resizebox{\textwidth}{!}{%" + "\n" + body + "\n}"

    if not bare:
        if caption is None:
            how = {"column": "per column", "metric": "per metric (global / mono)", "setup": "per setup",
                   "level": "per level", "all": "across the whole table"}[scale]
            caption = (f"Premiums per language and setting. Cell colours are scaled {how} "
                       f"(green = lowest, red = highest premium). "
                       f"$\\Delta$ is the max$-$min spread of each column.")
        body = "\n".join([r"\begin{table*}[t]", r"\centering", r"\small", body,
                          rf"\caption{{{caption}}}"] +
                         ([rf"\label{{{label}}}"] if label else []) + [r"\end{table*}"])

    header = ("% Generated by txt_premiums_to_chart_all_settings.py\n"
              "% Preamble needs: \\usepackage{booktabs}, \\usepackage[table]{xcolor}"
              + (", \\usepackage{graphicx}" if fit else "") + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + body + "\n")
    print(f"saved {out_path}")


# --------------------------------------------------------------------------- #
# Drawing  (all coordinates in inches)
# --------------------------------------------------------------------------- #

FS_TITLE = 13
FS_SETUP = 10.5
FS_LEVEL = 9.5
FS_METRIC = 8.5
FS_LANG = 9.5
FS_CELL = 8.8
FS_DELTA = 8.8
FS_LEGEND = 8.5


def text_width_in(text, fontsize, fontweight="normal"):
    fig = plt.figure()
    t = fig.text(0, 0, text, fontsize=fontsize, fontweight=fontweight)
    fig.canvas.draw()
    w = t.get_window_extent().width / fig.dpi
    plt.close(fig)
    return w


def draw_chart(columns, langs, out_path, title=None, scale="column", delta_pos="bottom",
               col_w=0.72, level_gap=0.18, setup_gap=0.4):
    labels = {l: CODE_TO_LANG_NAME.get(l, l) for l in langs}
    delta_label = "\u0394 (max\u2212min)"

    # --- Horizontal layout -------------------------------------------------
    margin = 0.25
    lang_w = max(text_width_in(s, FS_LANG, "medium") for s in list(labels.values()) + [delta_label]) + 0.25
    box_w = col_w - 0.1

    # Groups are made at least as wide as their header label (matters when
    # suppression leaves a single column under a long label); columns are
    # centered inside a group that had to be widened.
    setups = list(dict.fromkeys(c["setup"] for c in columns))
    levels_of = {s: list(dict.fromkeys(c["level"] for c in columns if c["setup"] == s)) for s in setups}
    cols_of = {(s, l): [c for c in columns if c["setup"] == s and c["level"] == l]
               for s in setups for l in levels_of[s]}

    def level_width(s, l):
        return max(len(cols_of[(s, l)]) * col_w,
                   text_width_in(LEVEL_LABELS.get(l, l), FS_LEVEL, "semibold") + 0.15)

    def setup_inner_width(s):
        return sum(level_width(s, l) for l in levels_of[s]) + level_gap * (len(levels_of[s]) - 1)

    col_x = {}                            # key -> left edge of column slot
    group_span = {}                       # setup or (setup, level) -> (x0, x1)
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

    # --- Vertical layout (top-down) ----------------------------------------
    row_h, box_h = 0.28, 0.21
    title_h = 0.45 if title else 0.0
    setup_h, level_h, metric_h = 0.32, 0.28, 0.26
    delta_h = 0.34
    legend_h = 0.75
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

    # --- Group headers ------------------------------------------------------
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

    # --- Scales -------------------------------------------------------------
    group_range, col_delta = compute_scales(columns, langs, scale)
    groups = group_range
    # --- Delta row ----------------------------------------------------------
    def draw_delta_row(y_center, sep_y):
        ax.text(margin, y_center, delta_label, fontsize=FS_LANG, va="center",
                fontstyle="italic", color="#333333")
        for c in columns:
            ax.text(col_x[c["key"]] + col_w / 2, y_center, f"{col_delta[c['key']]:.2f}",
                    fontsize=FS_DELTA, ha="center", va="center", fontweight="bold", color="#333333")
        ax.plot([margin, fig_w - margin], [sep_y, sep_y], color="#cccccc", lw=0.8)

    if delta_pos == "top":
        draw_delta_row(y - delta_h / 2, y - delta_h + 0.02)
        y -= delta_h

    # --- Rows ---------------------------------------------------------------
    for r, lang in enumerate(langs):
        yc = y - r * row_h - row_h / 2
        if r % 2 == 1:   # subtle zebra striping to help reading across
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

    # --- Legend -------------------------------------------------------------
    leg_y = y - 0.35
    if scale == "column":
        entries = [("Premium (per-column scale)", None)]
    else:
        entries = [(f"Premium \u2014 {g}" if g != "all" else "Premium", group_range[g]) for g in groups]
    n_leg = len(entries)
    leg_gap = 0.5
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
    parser = argparse.ArgumentParser(description="One premium colour chart across all settings.")
    parser.add_argument("--suppress", nargs="+", action="extend", default=[], metavar="PATTERN",
                        help="Column keys (or glob patterns) to hide, e.g. imbalanced_char-level_mono, "
                             "'*_mono', 'balanced-custom_*'. Comma-separated also works. Repeatable.")
    parser.add_argument("--dir", action="append", metavar="SETTING=PATH",
                        help="Override the folder for a setting, e.g. imbalanced_char-level=results/.../step_X")
    parser.add_argument("--match", action="append", metavar="SETTING=SUBSTR",
                        help="Only use premium files whose name contains SUBSTR for that setting "
                             "(when one folder holds files for several settings).")
    parser.add_argument("--sort-by", default="file",
                        help="Row order: 'file' (first shown column's ranking, default), 'mean', 'name', "
                             "or a column key.")
    parser.add_argument("--reverse", action="store_true",
                        help="Reverse the row order (e.g. with --sort-by mean: highest premium first).")
    parser.add_argument("--scale", choices=["column", "metric", "setup", "level", "all"], default="column",
                        help="Which columns share a colour scale (default: each column its own).")
    parser.add_argument("--delta-pos", choices=["top", "bottom"], default="bottom",
                        help="Where to put the per-column max-min row.")
    parser.add_argument("--title", default=None, help="Optional chart title.")
    parser.add_argument("--col-width", type=float, default=0.72, help="Column width in inches.")
    parser.add_argument("--out", default="all_settings_premium_chart.png", help="Output PNG path.")
    parser.add_argument("--no-png", action="store_true", help="Don't write the PNG (e.g. when only --latex is wanted).")
    parser.add_argument("--latex", metavar="PATH", default=None,
                        help="Also write a colour-coded LaTeX (booktabs) table to PATH.")
    parser.add_argument("--latex-bare", action="store_true",
                        help="Write only the tabular (no table*/caption), for \\input inside your own float.")
    parser.add_argument("--latex-fit", action="store_true",
                        help="Wrap the tabular in \\resizebox{\\textwidth}{!}{...}.")
    parser.add_argument("--latex-boxes", action="store_true",
                        help="Colour cells with \\colorbox instead of \\cellcolor (use this if the table "
                             "compiles but cells stay white, e.g. when a thesis class breaks colortbl).")
    parser.add_argument("--latex-bold", action="store_true", help="Bold the numbers in the table cells.")
    parser.add_argument("--caption", default=None, help="LaTeX caption (a sensible default is generated).")
    parser.add_argument("--label", default="tab:premiums_all_settings", help="LaTeX label.")
    parser.add_argument("--list-columns", action="store_true", help="Print available column keys and exit.")
    args = parser.parse_args()

    setting_dirs = dict(SETTING_DIRS)
    setting_dirs.update(parse_kv(args.dir, "dir"))
    matches = parse_kv(args.match, "match")

    columns, skipped = build_columns(setting_dirs, matches)
    for key, why in skipped:
        print(f"note: skipping {key} ({why})")

    if args.list_columns:
        for c in columns:
            print(f"{c['key']:40s} {c['path']}")
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