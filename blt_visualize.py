"""
blt_visualize.py — Drop-in visualization for BLT entropy patcher output.

Usage:
    from blt_visualize import BLTPatchVisualizer

    viz = BLTPatchVisualizer()
    viz.add(text, patches, scores=scores)   # call once per text
    viz.save("blt_output.html")             # write self-contained HTML
    # or: html_str = viz.render()
"""

import json
import math
import html as html_lib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ─── colour palette (12 hues, evenly spaced; matches lucalp's HF space) ──────
PATCH_COLORS = [
    "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c",
    "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00",
    "#cab2d6", "#6a3d9a", "#ffff99", "#b15928",
]

# Readable dark text colours for each background above
PATCH_TEXT = [
    "#1a4a6b", "#e8f4fb", "#2e5a0e", "#e8f7e8",
    "#7a1010", "#ffe8e8", "#7a4400", "#fff4e0",
    "#3a1a5a", "#f0e8ff", "#7a7a00", "#7a3010",
]


@dataclass
class PatchResult:
    text: str
    label: str
    patches: List[Tuple[str, int]]           # (chunk_text, byte_length)
    scores: Optional[List[float]] = None     # per-patch entropy (may be None)
    threshold: Optional[float] = None


class BLTPatchVisualizer:
    """Accumulate patch results and render them as a single HTML page."""

    def __init__(self):
        self._results: List[PatchResult] = []

    # ── public API ────────────────────────────────────────────────────────────

    def add(
        self,
        text: str,
        patches: List[Tuple[str, int]],
        scores: Optional[List[float]] = None,
        label: str = "",
        threshold: Optional[float] = None,
    ) -> None:
        """Register one text + its patch decomposition for later rendering."""
        self._results.append(PatchResult(text, label, patches, scores, threshold))

    def render(self) -> str:
        """Return a self-contained HTML string."""
        sections = "\n".join(self._render_section(r, i) for i, r in enumerate(self._results))
        return _HTML_TEMPLATE.format(sections=sections)

    def save(self, path: str = "blt_output.html") -> None:
        """Write the HTML to *path* and print a confirmation."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.render())
        print(f"[blt_visualize] Saved → {path}")

    # ── internals ─────────────────────────────────────────────────────────────

    def _render_section(self, r: PatchResult, idx: int) -> str:
        patch_html = self._build_patch_html(r.patches)
        chart_json = self._build_chart_json(r.patches, r.scores, r.threshold)
        n_patches = len(r.patches)
        n_bytes = sum(l for _, l in r.patches)
        avg = n_bytes / max(n_patches, 1)
        label_text = html_lib.escape(r.label) if r.label else f"sample {idx+1}"
        text_preview = html_lib.escape(r.text[:90] + ("…" if len(r.text) > 90 else ""))

        chart_block = ""
        if chart_json:
            chart_block = f"""
            <div class="chart-wrap">
              <canvas id="chart-{idx}" height="180"></canvas>
            </div>
            <script>
              (function(){{
                var ctx = document.getElementById('chart-{idx}').getContext('2d');
                var cfg = {chart_json};
                new Chart(ctx, cfg);
              }})();
            </script>"""

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
          {chart_block}
        </section>"""

    @staticmethod
    def _build_patch_html(patches: List[Tuple[str, int]]) -> str:
        parts = []
        for i, (chunk, _) in enumerate(patches):
            color_idx = i % len(PATCH_COLORS)
            bg = PATCH_COLORS[color_idx]
            fg = PATCH_TEXT[color_idx]
            # Replace spaces with visible underscore (matches reference app)
            display = html_lib.escape(chunk).replace(" ", "&#95;")
            parts.append(
                f'<span class="patch-token" '
                f'style="background:{bg};color:{fg}" '
                f'title="patch {i+1}: {len(chunk)} chars">'
                f'{display}</span>'
            )
        return "".join(parts)

    @staticmethod
    def _build_chart_json(patches, scores, threshold):
        if not scores:
            return None

        MAX_BYTES = 100
        x_labels = []  # BOS
        cursor = 0

        for (chunk, length) in patches:
            for i in range(length):
                if cursor >= MAX_BYTES - 1:
                    break
                if i == 0:
                    char = chunk[0] if chunk else '?'
                    char = '_' if char == ' ' else char
                else:
                    char = '·'
                x_labels.append(char)
                cursor += 1
            if cursor >= MAX_BYTES - 1:
                break

        y_vals = [round(s, 4) for s in scores[:len(x_labels)]]

        datasets = [
            {
                "label": "Entropy",
                "data": y_vals,
                "borderColor": "#1f78b4",
                "backgroundColor": "rgba(31,120,180,0.15)",
                "pointRadius": 2,
                "pointHoverRadius": 4,
                "tension": 0,
                "fill": True,
                "borderWidth": 1.5,
            }
        ]

        if threshold is not None:
            thr = round(threshold, 4)
            datasets.append({
                "label": f"Threshold ({thr})",
                "data": [thr] * len(y_vals),
                "borderColor": "#e31a1c",
                "borderDash": [5, 4],
                "borderWidth": 1,
                "pointRadius": 0,
                "fill": False,
            })

        cfg = {
            "type": "line",
            "data": {"labels": x_labels, "datasets": datasets},
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "animation": {"duration": 400},
                "plugins": {
                    "legend": {"display": True, "position": "top",
                            "labels": {"font": {"size": 11}, "boxWidth": 12}},
                    "tooltip": {"mode": "index", "intersect": False},
                },
                "scales": {
                    "x": {
                        "title": {"display": True, "text": "Character",
                                "font": {"size": 11}},
                        "ticks": {"autoSkip": False, "maxRotation": 0,
                                "font": {"size": 8}},
                        "grid": {"color": "rgba(128,128,128,0.15)"},
                    },
                    "y": {
                        "title": {"display": True, "text": "Entropy",
                                "font": {"size": 11}},
                        "ticks": {"font": {"size": 10}},
                        "grid": {"color": "rgba(128,128,128,0.15)"},
                    },
                },
            },
        }
        return json.dumps(cfg)

# ─── HTML shell ───────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>BLT Entropy Patcher — Visualisation</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:      #0f1117;
    --surface: #1a1d27;
    --border:  #2e3144;
    --text:    #d4d6e4;
    --muted:   #7b7f9a;
    --accent:  #4a90d9;
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
    max-width: 860px;
    margin: 0 auto 2.5rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.2rem;
  }}
  .page-header h1 {{
    font-size: 1.3rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #e8eaf0;
  }}
  .page-header .subtitle {{
    color: var(--muted);
    font-size: 0.8rem;
    margin-top: 0.25rem;
  }}

  .result-card {{
    max-width: 860px;
    margin: 0 auto 2rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }}

  .card-header {{
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.65rem 1rem;
    background: rgba(255,255,255,0.03);
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
  }}

  .label-badge {{
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(74,144,217,0.18);
    color: var(--accent);
    border: 1px solid rgba(74,144,217,0.35);
    border-radius: 4px;
    padding: 2px 8px;
    letter-spacing: 0.03em;
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
    line-height: 2.2;
    border-bottom: 1px solid var(--border);
    word-break: break-all;
  }}

  .patch-token {{
    display: inline-block;
    padding: 2px 6px;
    margin: 2px 2px;
    border-radius: 4px;
    font-size: 0.85rem;
    font-weight: 500;
    cursor: default;
    transition: transform 0.1s, box-shadow 0.1s;
    white-space: pre;
  }}
  .patch-token:hover {{
    transform: translateY(-2px);
    box-shadow: 0 3px 10px rgba(0,0,0,0.45);
    z-index: 1;
    position: relative;
  }}

  .chart-wrap {{
    padding: 0.75rem 1rem 1rem;
    height: 220px;
    position: relative;
  }}
  .chart-wrap canvas {{
    width: 100% !important;
  }}
</style>
</head>
<body>
<header class="page-header">
  <h1>BLT Entropy Patcher &mdash; Output Visualisation</h1>
  <p class="subtitle">Colour-coded patch boundaries &amp; per-patch entropy &middot; hover a token for details</p>
</header>

{sections}

</body>
</html>
"""


# ─── convenience: build from your patch_text() output ────────────────────────

def visualize_patch_results(results, output_path="blt_output.html"):
    """
    results: list of dicts, each with keys:
        text      str
        patches   list of (chunk_str, byte_len)  ← your patches list
        scores    list of float | None
        label     str
        threshold float | None
    """
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