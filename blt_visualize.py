"""
blt_visualize.py — Drop-in visualization for BLT entropy patcher output.

Usage:
    from blt_visualize import BLTPatchVisualizer

    viz = BLTPatchVisualizer()
    viz.add(text, patches, scores=scores)   # call once per text
    viz.save("blt_output.html")             # write self-contained HTML
    # or: html_str = viz.render()

Note: patches should be (chunk_str, chunk_bytes_list, byte_length) tuples
      as produced by blt_patcher.patch_text().
"""

import html as html_lib
import json
from dataclasses import dataclass
from typing import List, Optional


# ─── colour palette ───────────────────────────────────────────────────────────
PATCH_COLORS = [
    "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c",
    "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00",
    "#cab2d6", "#6a3d9a", "#ffff99", "#b15928",
]
PATCH_TEXT = [
    "#1a4a6b", "#e8f4fb", "#2e5a0e", "#e8f7e8",
    "#7a1010", "#ffe8e8", "#7a4400", "#fff4e0",
    "#3a1a5a", "#f0e8ff", "#7a7a00", "#7a3010",
]


@dataclass
class PatchResult:
    text: str
    label: str
    patches: list
    scores: Optional[List[float]] = None
    threshold: Optional[float] = None


class BLTPatchVisualizer:

    def __init__(self):
        self._results: List[PatchResult] = []

    def add(self, text, patches, scores=None, label="", threshold=None):
        self._results.append(PatchResult(text, label, patches, scores, threshold))

    def render(self) -> str:
        sections = "\n".join(self._render_section(r, i) for i, r in enumerate(self._results))
        return _HTML_TEMPLATE.format(sections=sections)

    def save(self, path="blt_output.html"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.render())
        print(f"[blt_visualize] Saved → {path}")

    # ── normalise patch tuple ─────────────────────────────────────────────────

    @staticmethod
    def _norm(p):
        """(chunk, bytes, length) or (chunk, length) → (chunk, bytes_list, length)"""
        if len(p) == 3:
            return p[0], list(p[1]), p[2]
        chunk, length = p
        return chunk, list(chunk.encode("utf-8", errors="replace")), length

    # ── section ───────────────────────────────────────────────────────────────

    def _render_section(self, r: PatchResult, idx: int) -> str:
        patches = [self._norm(p) for p in r.patches]

        patch_html   = self._build_patch_html(patches)
        combined_svg = self._build_combined_svg(patches, r.scores, r.threshold)

        n_patches = len(patches)
        n_bytes   = sum(ln for _, _, ln in patches)
        avg       = n_bytes / max(n_patches, 1)

        label_text   = html_lib.escape(r.label) if r.label else f"sample {idx+1}"
        text_preview = html_lib.escape(r.text[:120] + ("…" if len(r.text) > 120 else ""))

        return f"""
        <section class="result-card">
          <div class="card-header">
            <span class="label-badge">{label_text}</span>
            <span class="stats">
              <b>{n_patches}</b> patches &middot; <b>{n_bytes}</b> bytes &middot; avg <b>{avg:.2f}</b> b/patch
            </span>
          </div>
          <p class="text-preview">{text_preview}</p>
          <div class="patch-display">{patch_html}</div>
          <div class="svg-wrap">{combined_svg}</div>
        </section>"""

    # ── coloured patch tokens ─────────────────────────────────────────────────

    @staticmethod
    def _build_patch_html(patches) -> str:
        parts = []
        for i, (chunk, _, _) in enumerate(patches):
            bg = PATCH_COLORS[i % len(PATCH_COLORS)]
            fg = PATCH_TEXT[i % len(PATCH_TEXT)]
            display = html_lib.escape(chunk).replace(" ", "&#95;") or \
                      "<span style='opacity:.35'>∅</span>"
            parts.append(
                f'<span class="patch-token" style="background:{bg};color:{fg}" '
                f'title="patch {i+1}">{display}</span>'
            )
        return "".join(parts)

    # ── combined SVG: entropy line chart + aligned byte/char table ────────────

    def _build_combined_svg(self, patches, scores, threshold) -> str:
        # ── layout constants ──────────────────────────────────────────────────
        CELL_W      = 28       # px per byte column
        MARGIN_L    = 52       # left margin (y-axis labels)
        MARGIN_R    = 16
        CHART_H     = 180      # height of the entropy chart area
        AXIS_H      = 18       # x-axis tick label row
        ROW_IDX_H   = 18       # byte index row
        ROW_VAL_H   = 22       # byte value row
        ROW_CHR_H   = 24       # character row
        PAD_TOP     = 16
        PAD_BOT     = 8

        # Flatten bytes
        flat_bytes = []   # (byte_val, patch_idx)
        for pi, (_, blist, _) in enumerate(patches):
            for bv in blist:
                flat_bytes.append((bv, pi))

        N = len(flat_bytes)
        if N == 0:
            return ""

        W = MARGIN_L + N * CELL_W + MARGIN_R
        TABLE_TOP = PAD_TOP + CHART_H + AXIS_H
        H = TABLE_TOP + ROW_IDX_H + ROW_VAL_H + ROW_CHR_H + PAD_BOT

        # x coordinate of byte i (centre of column)
        def cx(i):
            return MARGIN_L + i * CELL_W + CELL_W / 2

        # ── entropy chart ─────────────────────────────────────────────────────
        elements = []

        # Background
        elements.append(
            f'<rect x="0" y="0" width="{W}" height="{H}" fill="#f8f9fc" rx="0"/>'
        )

        # Chart area background
        chart_x = MARGIN_L
        chart_y = PAD_TOP
        chart_w = N * CELL_W
        elements.append(
            f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{CHART_H}" '
            f'fill="#ffffff" stroke="#d0d5e8" stroke-width="1"/>'
        )

        if scores:
            visible_scores = scores[:N]
            s_min = min(visible_scores)
            s_max = max(visible_scores)
            s_range = max(s_max - s_min, 0.01)

            def sy(v):
                norm = (v - s_min) / s_range
                return chart_y + CHART_H - norm * (CHART_H - 10) - 5

            # Gridlines (5 levels)
            for k in range(6):
                frac = k / 5
                gv = s_min + frac * s_range
                gy = sy(gv)
                elements.append(
                    f'<line x1="{chart_x}" y1="{gy:.1f}" x2="{chart_x+chart_w}" y2="{gy:.1f}" '
                    f'stroke="#e0e4f0" stroke-width="0.8"/>'
                )
                elements.append(
                    f'<text x="{chart_x - 4}" y="{gy + 4:.1f}" text-anchor="end" '
                    f'font-size="9" fill="#888">{gv:.2f}</text>'
                )

            # Threshold line
            if threshold is not None:
                ty = sy(threshold)
                elements.append(
                    f'<line x1="{chart_x}" y1="{ty:.1f}" x2="{chart_x+chart_w}" y2="{ty:.1f}" '
                    f'stroke="#c0392b" stroke-width="1.2" stroke-dasharray="5,4"/>'
                )
                elements.append(
                    f'<text x="{chart_x + chart_w + 3}" y="{ty + 4:.1f}" '
                    f'font-size="9" fill="#c0392b">t={threshold:.2f}</text>'
                )

            # Patch boundary vertical lines
            cursor = 0
            for pi, (_, _, length) in enumerate(patches[:-1]):
                cursor += length
                bx = MARGIN_L + cursor * CELL_W
                elements.append(
                    f'<line x1="{bx}" y1="{chart_y}" x2="{bx}" y2="{chart_y + CHART_H}" '
                    f'stroke="rgba(180,60,60,0.35)" stroke-width="1" stroke-dasharray="3,3"/>'
                )

            # Filled area under curve
            pts = " ".join(f"{cx(i):.1f},{sy(v):.1f}" for i, v in enumerate(visible_scores))
            first_x = cx(0)
            last_x  = cx(len(visible_scores) - 1)
            base_y  = chart_y + CHART_H
            elements.append(
                f'<polygon points="{first_x:.1f},{base_y} {pts} {last_x:.1f},{base_y}" '
                f'fill="rgba(42,109,181,0.08)"/>'
            )

            # Line
            polyline_pts = " ".join(f"{cx(i):.1f},{sy(v):.1f}" for i, v in enumerate(visible_scores))
            elements.append(
                f'<polyline points="{polyline_pts}" fill="none" '
                f'stroke="#2a6db5" stroke-width="1.5" stroke-linejoin="round"/>'
            )

            # Dots
            for i, v in enumerate(visible_scores):
                elements.append(
                    f'<circle cx="{cx(i):.1f}" cy="{sy(v):.1f}" r="2" '
                    f'fill="#2a6db5" opacity="0.7"/>'
                )

        # Y-axis label
        elements.append(
            f'<text x="10" y="{chart_y + CHART_H//2}" '
            f'text-anchor="middle" font-size="10" fill="#444" '
            f'transform="rotate(-90, 10, {chart_y + CHART_H//2})">Entropy</text>'
        )

        # X-axis tick labels (every N bytes to avoid crowding)
        tick_every = max(1, N // 40)
        axis_y = chart_y + CHART_H + 13
        for i in range(0, N, tick_every):
            elements.append(
                f'<text x="{cx(i):.1f}" y="{axis_y}" text-anchor="middle" '
                f'font-size="8" fill="#555">{i}</text>'
            )

        # ── byte table ────────────────────────────────────────────────────────
        row_idx_y = TABLE_TOP
        row_val_y = row_idx_y + ROW_IDX_H
        row_chr_y = row_val_y + ROW_VAL_H

        # Column backgrounds (alternating subtle stripe per patch)
        cursor = 0
        for pi, (_, blist, length) in enumerate(patches):
            bg = PATCH_COLORS[pi % len(PATCH_COLORS)]
            col_x = MARGIN_L + cursor * CELL_W
            # byte value row background
            elements.append(
                f'<rect x="{col_x}" y="{row_val_y}" width="{length * CELL_W}" '
                f'height="{ROW_VAL_H}" fill="{bg}" opacity="0.85"/>'
            )
            cursor += length

        # Row separator lines
        for row_y in [row_idx_y, row_val_y, row_chr_y, row_chr_y + ROW_CHR_H]:
            elements.append(
                f'<line x1="{MARGIN_L}" y1="{row_y}" x2="{MARGIN_L + N*CELL_W}" y2="{row_y}" '
                f'stroke="#d0d5e8" stroke-width="0.8"/>'
            )

        # Row labels
        for label, row_y, row_h in [
            ("idx",  row_idx_y,  ROW_IDX_H),
            ("byte", row_val_y,  ROW_VAL_H),
            ("char", row_chr_y,  ROW_CHR_H),
        ]:
            mid_y = row_y + row_h / 2 + 4
            elements.append(
                f'<text x="{MARGIN_L - 5}" y="{mid_y:.1f}" text-anchor="end" '
                f'font-size="9" fill="#888" font-weight="bold">{label}</text>'
            )

        # Byte index row
        for i, (_, _) in enumerate(flat_bytes):
            elements.append(
                f'<text x="{cx(i):.1f}" y="{row_idx_y + 13}" text-anchor="middle" '
                f'font-size="8" fill="#aaa">{i}</text>'
            )

        # Byte value row
        for i, (bv, pi) in enumerate(flat_bytes):
            fg = PATCH_TEXT[pi % len(PATCH_TEXT)]
            elements.append(
                f'<text x="{cx(i):.1f}" y="{row_val_y + 15}" text-anchor="middle" '
                f'font-size="9" font-weight="600" fill="{fg}">{bv}</text>'
            )

        # Character row — each char spans its UTF-8 byte width
        cursor = 0
        for pi, (chunk, _, length) in enumerate(patches):
            bg = PATCH_COLORS[pi % len(PATCH_COLORS)]
            fg = PATCH_TEXT[pi % len(PATCH_TEXT)]
            # background for char row
            col_x = MARGIN_L + cursor * CELL_W
            elements.append(
                f'<rect x="{col_x}" y="{row_chr_y}" width="{length * CELL_W}" '
                f'height="{ROW_CHR_H}" fill="{bg}" opacity="0.6"/>'
            )
            if chunk:
                byte_offset = 0
                for ch in chunk:
                    ch_bytes = len(ch.encode("utf-8"))
                    char_cx = MARGIN_L + (cursor + byte_offset) * CELL_W + ch_bytes * CELL_W / 2
                    display = html_lib.escape(ch) if ch != " " else "·"
                    elements.append(
                        f'<text x="{char_cx:.1f}" y="{row_chr_y + 17}" text-anchor="middle" '
                        f'font-size="11" font-weight="600" fill="{fg}">{display}</text>'
                    )
                    byte_offset += ch_bytes
            else:
                # empty chunk (partial byte boundary)
                char_cx = MARGIN_L + cursor * CELL_W + length * CELL_W / 2
                elements.append(
                    f'<text x="{char_cx:.1f}" y="{row_chr_y + 17}" text-anchor="middle" '
                    f'font-size="10" fill="{fg}" opacity="0.5">∅</text>'
                )
            cursor += length

        # Column dividers between patches
        cursor = 0
        for pi, (_, _, length) in enumerate(patches[:-1]):
            cursor += length
            div_x = MARGIN_L + cursor * CELL_W
            elements.append(
                f'<line x1="{div_x}" y1="{row_idx_y}" x2="{div_x}" y2="{row_chr_y + ROW_CHR_H}" '
                f'stroke="rgba(0,0,0,0.2)" stroke-width="1"/>'
            )

        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'style="display:block;overflow:visible;">'
            + "\n".join(elements)
            + "</svg>"
        )
        return svg


# ─── HTML shell ───────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>BLT Entropy Patcher — Visualisation</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:      #f0f2f8;
    --surface: #ffffff;
    --border:  #d0d5e8;
    --text:    #1a1d2e;
    --muted:   #5a5f7a;
    --accent:  #2a6db5;
    --font:    'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
  }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 13px;
    line-height: 1.6;
    padding: 2rem 1rem 4rem;
  }}

  .page-header {{
    max-width: 98vw;
    margin: 0 auto 2.5rem;
    border-bottom: 2px solid var(--border);
    padding-bottom: 1.2rem;
  }}
  .page-header h1 {{
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text);
  }}
  .page-header .subtitle {{
    color: var(--muted);
    font-size: 0.8rem;
    margin-top: 0.25rem;
  }}

  .result-card {{
    max-width: 98vw;
    margin: 0 auto 2rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }}

  .card-header {{
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.65rem 1rem;
    background: #e8ecf7;
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
  }}

  .label-badge {{
    font-size: 0.72rem;
    font-weight: 700;
    background: rgba(42,109,181,0.12);
    color: var(--accent);
    border: 1px solid rgba(42,109,181,0.3);
    border-radius: 4px;
    padding: 2px 8px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }}

  .stats {{
    font-size: 0.78rem;
    color: var(--muted);
  }}
  .stats b {{ color: var(--text); font-weight: 600; }}

  .text-preview {{
    padding: 0.55rem 1rem;
    font-size: 0.78rem;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}

  .patch-display {{
    padding: 0.85rem 1rem;
    line-height: 2.4;
    border-bottom: 1px solid var(--border);
    word-break: break-all;
  }}

  .patch-token {{
    display: inline-block;
    padding: 2px 5px;
    margin: 2px 1px;
    border-radius: 3px;
    font-size: 0.85rem;
    font-weight: 500;
    cursor: default;
    transition: transform 0.1s, box-shadow 0.1s;
    white-space: pre;
  }}
  .patch-token:hover {{
    transform: translateY(-2px);
    box-shadow: 0 3px 10px rgba(0,0,0,0.18);
    z-index: 1;
    position: relative;
  }}

  .svg-wrap {{
    overflow-x: auto;
    padding: 1rem;
  }}
</style>
</head>
<body>
<header class="page-header">
  <h1>BLT Entropy Patcher &mdash; Visualisation</h1>
  <p class="subtitle">Per-byte entropy &middot; patch boundaries &middot; byte↔character alignment</p>
</header>

{sections}

</body>
</html>
"""


# ─── convenience ─────────────────────────────────────────────────────────────

def visualize_patch_results(results, output_path="blt_output.html"):
    viz = BLTPatchVisualizer()
    for r in results:
        viz.add(
            text=r["text"],
            patches=r["patches"],
            scores=r.get("scores"),
            label=r.get("label", ""),
            threshold=r.get("threshold"),
        )
    viz.save(output_path)