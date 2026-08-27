#!/usr/bin/env python3
"""
txt_premiums_to_chart.py

Reads results/txt_premiums/{,char_level/}t_anchor/<Model>/[step_*/]
<Model>_raw_<mode>_t_<threshold>_premiums_sorted.txt files and produces a single
grouped boxplot figure comparing byte-level and char-level patching premiums.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe; no display needed to write PNGs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

# Set global hatch line width so plot patches and legend patches match
matplotlib.rcParams["hatch.linewidth"] = 4.5

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent            # <repo>/results
REPO_ROOT = SCRIPT_DIR.parent                             # <repo>
TXT_PREMIUMS_DIR = SCRIPT_DIR / "txt_premiums"
BYTE_T_ANCHOR_DIR = TXT_PREMIUMS_DIR / "t_anchor"
CHAR_T_ANCHOR_DIR = TXT_PREMIUMS_DIR / "char_level" / "t_anchor"
DEFAULT_OUT_DIR = REPO_ROOT / "charts"

# ── Colors & Constants ──────────────────────────────────────────────────
MODES = ("raw_entropy", "raw_monotonicity")
MODE_LABELS = {"raw_entropy": "Entropy", "raw_monotonicity": "Monotonicity"}
MODE_COLORS = {"raw_entropy": "#4987EB", "raw_monotonicity": "#6DC973"}
CHAR_HATCH_COLOR = "#9C52CC"  # Purple accent for char-level stripes

MODEL_DISPLAY = {
    "_Base-Model": "Base",
    "Imbalanced": "Imbalanced",
    "Balanced": "Balanced",
    "Balanced-Custom": "Balanced-Custom",
}

# Updated model definitions grouped as:
# (model_dir_name, display_label, [(mode, t_anchor_dir, is_char), ...])
MERGED_MODELS = [
    (
        "_Base-Model",
        "Base",
        [
            ("raw_entropy", BYTE_T_ANCHOR_DIR, False),
            ("raw_monotonicity", BYTE_T_ANCHOR_DIR, False),
        ],
    ),
    (
        "Imbalanced",
        "Imbalanced",
        [
            ("raw_entropy", BYTE_T_ANCHOR_DIR, False),
            ("raw_monotonicity", BYTE_T_ANCHOR_DIR, False),
        ],
    ),
    (
        "Balanced",
        "Balanced",
        [
            ("raw_entropy", BYTE_T_ANCHOR_DIR, False),
            ("raw_entropy", CHAR_T_ANCHOR_DIR, True),
            ("raw_monotonicity", BYTE_T_ANCHOR_DIR, False),
            ("raw_monotonicity", CHAR_T_ANCHOR_DIR, True),
        ],
    ),
    (
        "Balanced-Custom",
        "Balanced-Custom",
        [
            ("raw_entropy", BYTE_T_ANCHOR_DIR, False),
            ("raw_entropy", CHAR_T_ANCHOR_DIR, True),
            ("raw_monotonicity", BYTE_T_ANCHOR_DIR, False),
            ("raw_monotonicity", CHAR_T_ANCHOR_DIR, True),
        ],
    ),
]


def parse_premiums_file(path: Path) -> list[float]:
    """Extract the Premium column from a *_premiums_sorted.txt file."""
    premiums = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            premiums.append(float(parts[1]))
        except ValueError:
            continue
    if not premiums:
        raise ValueError(f"No premium values parsed from {path}")
    return premiums


def find_premiums_file(t_anchor_dir: Path, model_dir_name: str, mode: str) -> Path:
    """Locate <model_dir_name>_<mode>_t_<threshold>_premiums_sorted.txt."""
    model_dir = t_anchor_dir / model_dir_name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    pattern = f"{model_dir_name}_{mode}_t_*_premiums_sorted.txt"
    matches = sorted(model_dir.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} found under {model_dir}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple files matching {pattern!r} found under {model_dir}: {matches}"
        )
    return matches[0]


def make_single_boxplot(out_path: Path) -> None:
    box_width = 0.55
    intra_gap = 0.70
    group_gap = 1.2

    positions: list[float] = []
    box_data: list[list[float]] = []
    box_modes: list[str] = []
    box_is_char: list[bool] = []
    tick_positions: list[float] = []
    tick_labels: list[str] = []

    x = 0.0
    for model_dir_name, label, boxes_config in MERGED_MODELS:
        group_start = x
        for mode, t_anchor_dir, is_char in boxes_config:
            path = find_premiums_file(t_anchor_dir, model_dir_name, mode)
            values = parse_premiums_file(path)
            
            positions.append(x)
            box_data.append(values)
            box_modes.append(mode)
            box_is_char.append(is_char)
            x += intra_gap

        group_end = positions[-1]
        tick_positions.append((group_start + group_end) / 2)
        tick_labels.append(label)
        x += group_gap

    fig, ax = plt.subplots(figsize=(14, 6.0))

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
    )

    for patch, mode, is_char in zip(bp["boxes"], box_modes, box_is_char):
        patch.set_facecolor(MODE_COLORS[mode])
        patch.set_alpha(0.65)
        
        if is_char:
            patch.set_hatch("//")
            patch.set_edgecolor(CHAR_HATCH_COLOR)
            patch.set_linewidth(0)

    # English reference line at y=1.0
    ax.axhline(1.0, color="#BD5B51", linestyle="--", linewidth=1.2, zorder=0)

    # Vertical separator line between Base and the rest of the models
    base_separator_x = (tick_positions[0] + tick_positions[1]) / 2
    ax.axvline(
        base_separator_x,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        zorder=1,
    )

    # Y-axis scaling with 0.5 step sizes
    ax.set_ylim(0.2, 6.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Patching premium relative to English")
    ax.set_title("Patching premium by model & level (anchor threshold)")

    # Legend elements inheriting global matplotlib.rcParams["hatch.linewidth"] = 4.5
    legend_handles = [
        Patch(facecolor=MODE_COLORS[m], alpha=0.65, label=MODE_LABELS[m]) for m in MODES
    ]
    legend_handles.append(
        Patch(
            facecolor="white",
            edgecolor=CHAR_HATCH_COLOR,
            hatch="//",
            linewidth=0,
            label="Char-level (Striped)",
        )
    )
    legend_handles.append(
        Line2D([0], [0], color="#BD5B51", linestyle="--", label="English premium (= 1.0)")
    )
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Directory to write output PNG into (default: {DEFAULT_OUT_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    make_single_boxplot(out_dir / "premium_boxplot_merged.png")
    print("Done.")


if __name__ == "__main__":
    main()