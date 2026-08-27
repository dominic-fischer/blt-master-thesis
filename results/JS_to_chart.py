#!/usr/bin/env python3
"""
JS_to_chart.py

For each of the three own-model results files (bare JSON arrays, possibly
wrapped in a "const LANG_DATA = [...];" JS assignment) living in a given
results directory, this script:

  1. Loads the array of per-language records (tolerates either a bare
     JSON array or one wrapped in a JS variable assignment).
  2. Filters it down to the 20 languages listed in
     training_setup/langs/langs_chosen.csv (column "language_code",
     e.g. "eng_Latn"), matched against each record's "Code_Orig" field.
  3. Derives MODES_META (labels + actual thresholds + anchor index) for
     the raw_entropy / raw_monotonicity modes directly from whatever
     "<mode>_t_<threshold>_pps_premium" columns are actually present in
     this model's data -- thresholds are run-specific, so they can't be
     hardcoded from the reference template.
  4. Computes GROUPS / BOUNDS bin edges spanning the observed min/max
     patching-premium values across those same columns.
  5. Injects LANG_DATA / MODES_META / GROUPS / BOUNDS into
     charts/eval_results_chart_own_models_template.html and writes the
     result to charts/eval_results_chart_own_models_<abbrev>.html (or
     charts/eval_results_chart_own_models_<subfolder>_<abbrev>.html if a
     non-default results directory was given -- see --help).

Expected layout (paths are resolved relative to this file):

    <repo_root>/
        charts/
            eval_results_chart_own_models_template.html
        results/
            JS_to_chart.py          <- this file
            results_JS/
                <the three results_*.js files, directly here by default>
        training_setup/
            langs/
                langs_chosen.csv

Usage:
    python results/JS_to_chart.py                          # default dir
    python results/JS_to_chart.py results/results_JS/own_models  # subfolder
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent            # <repo>/results
REPO_ROOT = SCRIPT_DIR.parent                             # <repo>
DEFAULT_RESULTS_JS_DIR = SCRIPT_DIR / "results_JS"
CHARTS_DIR = REPO_ROOT / "charts"
TEMPLATE_PATH = CHARTS_DIR / "eval_results_chart_own_models_template.html"
LANGS_CSV = REPO_ROOT / "training_setup" / "langs" / "langs_chosen.csv"

# ── Model file -> abbreviated name used in the output filename ─────────────
MODEL_FILES = {
    "entropy_10M_20lang_4gpu_sourcesbalanced_steps10000_ckpt200_customenc_lr4.5e-3_step_0000006400_results.js": "balanced-custom",
    "entropy_10M_20lang_4gpu_sourcesbalanced_steps10000_ckpt200_lr4.5e-3_step_0000007200_results.js": "balanced",
    "entropy_10M_20lang_4gpu_sourcesimbalanced_steps10000_ckpt200_lr4.5e-3_step_0000002600_results.js": "imbalanced",
}

# Only these two modes remain in the trimmed template (combined/normalised
# were removed), so these are the only premium columns we scan for min/max.
PREMIUM_COL_PATTERN = re.compile(r"^(raw_entropy|raw_monotonicity)_t_[\d.]+_pps_premium$")

# Same two modes, but capturing the mode name and threshold string so we can
# rebuild MODES_META (thresholds differ per model/run, so this can't be
# copied from the reference template -- it must come from the data itself).
THRESH_COL_PATTERN = re.compile(r"^(raw_entropy|raw_monotonicity)_t_([\d.]+)_pps_premium$")
MODE_LABELS = {"raw_entropy": "Entropy", "raw_monotonicity": "Monotonicity"}

# Field in the results records holding the per-language code (matches the
# "language_code" column in langs_chosen.csv, e.g. "eng_Latn").
CODE_FIELD = "Code_Orig"


def find_results_file(results_dir: Path, filename: str) -> Path:
    """Locate a results_JS file directly inside results_dir (no recursion
    into subfolders)."""
    candidate = results_dir / filename
    if not candidate.exists():
        raise FileNotFoundError(f"Could not find {filename!r} in {results_dir}")
    return candidate


def load_chosen_codes() -> set[str]:
    with open(LANGS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "language_code" not in (reader.fieldnames or []):
            raise ValueError(
                f"'language_code' column not found in {LANGS_CSV}; got {reader.fieldnames}"
            )
        codes = {row["language_code"].strip() for row in reader if row["language_code"].strip()}
    if not codes:
        raise ValueError(f"No language codes found in {LANGS_CSV}")
    return codes


def load_lang_data(js_path: Path, chosen_codes: set[str]) -> list[dict]:
    text = js_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{js_path} is empty")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Not a bare JSON array after all -- most likely it's wrapped in a
        # JS assignment like "const LANG_DATA = [...];". Pull out the
        # outermost [...] and parse that instead.
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                f"{js_path} does not look like JSON or a JS array literal.\n"
                f"First 200 chars: {text[:200]!r}"
            )
        data = json.loads(text[start:end + 1])

    if not isinstance(data, list):
        raise ValueError(f"{js_path} does not contain a JSON array")

    filtered = [rec for rec in data if rec.get(CODE_FIELD) in chosen_codes]

    found = {rec.get(CODE_FIELD) for rec in filtered}
    missing = chosen_codes - found
    if missing:
        print(
            f"  warning: {js_path.name}: {len(missing)} chosen language(s) "
            f"not found via field {CODE_FIELD!r}: {sorted(missing)}",
            file=sys.stderr,
        )

    # Drop any "char_" prefix (added by results_to_CSV.py --score-source=chars)
    # right here, so every downstream step (regex matching, MODES_META,
    # GROUPS/BOUNDS, and the template's own column-name lookups) only ever
    # has to deal with the plain byte-mode naming, regardless of whether
    # this particular results file came from a bytes- or chars-mode run.
    stripped = []
    for rec in filtered:
        new_rec = dict(rec)
        for key in list(new_rec):
            if key.startswith("char_"):
                target = key[len("char_"):]
                if target in new_rec:
                    print(
                        f"  warning: {js_path.name}: both {key!r} and {target!r} "
                        "present; keeping unprefixed, dropping char_ version",
                        file=sys.stderr,
                    )
                    del new_rec[key]
                else:
                    new_rec[target] = new_rec.pop(key)
        stripped.append(new_rec)

    return stripped


def compute_modes_meta(lang_data: list[dict]) -> dict:
    """Derive MODES_META {mode: {label, score, thresholds, anchor_idx}} from
    whatever raw_entropy_t_*/raw_monotonicity_t_* columns actually exist in
    this model's data (these thresholds are run-specific, so they can't be
    copied from the reference template)."""
    thresholds_by_mode: dict[str, set[float]] = {}
    for rec in lang_data:
        for key in rec:
            m = THRESH_COL_PATTERN.match(key)
            if m:
                mode, t_str = m.groups()
                # Round to 4 dp (matching the .toFixed(4) used to build the
                # column name in JS) so any float-parsing noise doesn't
                # create spurious near-duplicate threshold values.
                thresholds_by_mode.setdefault(mode, set()).add(round(float(t_str), 4))

    if not thresholds_by_mode:
        raise ValueError(
            "No raw_entropy_t_*/raw_monotonicity_t_*_pps_premium columns "
            "found in the filtered data -- check the results file's column names."
        )

    modes_meta = {}
    for mode, tset in thresholds_by_mode.items():
        thresholds = sorted(tset)
        print(f"  {mode}: {len(thresholds)} thresholds found: {thresholds}")
        # No fixed convention for which threshold is "the" anchor for a
        # given run -- default to the middle of the range.
        anchor_idx = len(thresholds) // 2
        modes_meta[mode] = {
            "label": MODE_LABELS.get(mode, mode),
            "score": "Raw",
            "thresholds": thresholds,
            "anchor_idx": anchor_idx,
        }
    return modes_meta


CANDIDATE_STEPS = (0.1, 0.25, 0.5, 1.0)


def compute_groups_bounds(lang_data: list[dict]) -> tuple[list[str], list[float]]:
    """Scan every raw_entropy_*/raw_monotonicity_*_pps_premium column across
    all thresholds and all records (both modes combined, since the
    template shares one GROUPS/BOUNDS array across whichever mode is
    selected), find the overall min/max, and build bin edges as follows:

      For each candidate step in CANDIDATE_STEPS (0.1, 0.25, 0.5, 1.0):
        - snap vmin DOWN to the nearest multiple of that step (bin start)
        - snap vmax UP to the nearest multiple of that step (bin end)
        - count how many equal-width bins of that size span start->end
      Keep whichever candidate's bin count is closest to 10.

    Because every candidate step evenly divides 1.0, and English's own
    premium is always exactly 1.0 for every column, 1.0 always lands
    exactly on a bin edge with no special-casing needed. Because start
    and end are snapped outward to *contain* vmin/vmax, every bin --
    including the first and last -- necessarily contains at least one
    real value, so there's no leftover empty leading/trailing bins to
    trim."""
    values = []
    for rec in lang_data:
        for key, val in rec.items():
            if PREMIUM_COL_PATTERN.match(key) and val is not None:
                try:
                    values.append(float(val))
                except (TypeError, ValueError):
                    pass

    if not values:
        raise ValueError("No premium values found to compute GROUPS/BOUNDS from")

    vmin, vmax = min(values), max(values)

    best_step, best_n, best_diff = None, None, None
    for step in CANDIDATE_STEPS:
        cand_start = math.floor(vmin / step) * step
        cand_end = math.ceil(vmax / step) * step
        n_bins = round((cand_end - cand_start) / step)
        diff = abs(n_bins - 10)
        print(f"  candidate step={step}: {n_bins} bins (|diff from 10|={diff})")
        if best_diff is None or diff < best_diff:
            best_step, best_n, best_diff = step, n_bins, diff

    step = best_step
    print(f"  chosen step={step} -> {best_n} bins")

    start = round(math.floor(vmin / step) * step, 10)

    bounds = [start]
    while bounds[-1] < vmax:
        bounds.append(round(bounds[-1] + step, 10))
    bounds.append(float("inf"))

    def fmt(x: float) -> str:
        return str(int(x)) if x == int(x) else f"{x:g}"

    groups = [f"{fmt(bounds[i])}–{fmt(bounds[i + 1])}" for i in range(len(bounds) - 2)]
    groups.append(f"{fmt(bounds[-2])}+")

    return groups, bounds


def inject_into_template(
    template_html: str, lang_data: list, groups: list, bounds: list, modes_meta: dict
) -> str:
    lang_data_js = json.dumps(lang_data, ensure_ascii=False)
    groups_js = json.dumps(groups, ensure_ascii=False)
    modes_meta_js = json.dumps(modes_meta, ensure_ascii=False)
    bounds_js = "[" + ",".join(
        "Infinity" if b == float("inf") else repr(b) for b in bounds
    ) + "]"

    out = re.sub(
        r"const LANG_DATA = \[.*?\];",
        lambda _: f"const LANG_DATA = {lang_data_js};",
        template_html, count=1, flags=re.DOTALL,
    )
    out = re.sub(
        r"const MODES_META = \{.*?\};",
        lambda _: f"const MODES_META = {modes_meta_js};",
        out, count=1, flags=re.DOTALL,
    )
    out = re.sub(
        r"const GROUPS = \[.*?\];",
        lambda _: f"const GROUPS = {groups_js};",
        out, count=1, flags=re.DOTALL,
    )
    out = re.sub(
        r"const BOUNDS = \[.*?\];",
        lambda _: f"const BOUNDS = {bounds_js};",
        out, count=1, flags=re.DOTALL,
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=str(DEFAULT_RESULTS_JS_DIR),
        help=(
            "Folder containing the three results_*.js files directly "
            "(not recursed into). Default: results/results_JS/"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    if not results_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {results_dir}")

    # If a subfolder of the default results_JS dir (or any other directory)
    # was explicitly given, fold its name into the output chart filenames,
    # e.g. eval_results_chart_own_models_<subfolder>_balanced.html
    subfolder_label = None
    if results_dir != DEFAULT_RESULTS_JS_DIR.resolve():
        subfolder_label = results_dir.name

    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE_PATH}")
    template_html = TEMPLATE_PATH.read_text(encoding="utf-8")

    chosen_codes = load_chosen_codes()
    print(f"Loaded {len(chosen_codes)} chosen language codes from {LANGS_CSV}")
    print(f"Looking for results in: {results_dir}")
    if subfolder_label:
        print(f"Subfolder label added to output names: {subfolder_label!r}")

    for filename, abbrev in MODEL_FILES.items():
        print(f"\n[{abbrev}]")
        js_path = find_results_file(results_dir, filename)
        print(f"  source: {js_path}")

        lang_data = load_lang_data(js_path, chosen_codes)
        print(f"  filtered to {len(lang_data)} / {len(chosen_codes)} languages")

        groups, bounds = compute_groups_bounds(lang_data)
        print(f"  GROUPS = {groups}")
        print(f"  BOUNDS = {bounds}")

        modes_meta = compute_modes_meta(lang_data)
        print(f"  MODES_META = {modes_meta}")

        out_html = inject_into_template(template_html, lang_data, groups, bounds, modes_meta)

        suffix = f"{subfolder_label}_{abbrev}" if subfolder_label else abbrev
        out_path = CHARTS_DIR / TEMPLATE_PATH.name.replace("_template", f"_{suffix}")
        out_path.write_text(out_html, encoding="utf-8")
        print(f"  wrote {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()