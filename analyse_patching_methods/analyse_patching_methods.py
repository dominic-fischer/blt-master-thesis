import os
import json
import pandas as pd
from collections import defaultdict

MASTER_CSV  = "floresplus_MASTER.csv"
RESULTS_DIR = "results/base_model"
RAW_T       = "t_1.3340"
MONO_T      = "t_0.3662"
NORM_T      = "t_0.5293"

MODES = [
    {"key": "raw_monotonicity", "t": MONO_T, "label": "mono"},
    {"key": "norm_entropy",     "t": NORM_T, "label": "norm"},
]


def boundary_delta_line(rt, cts, modes):
    """One-line summary of boundary count change per mode, relative to raw entropy.
    Negative delta = fewer boundaries than raw (coarser patching, 'improvement' in
    boundary count terms). Positive delta = more boundaries than raw ('worse')."""
    parts = [f"raw: {rt:,}"]
    for m in modes:
        lbl = m["label"]
        ct = cts[lbl]
        if rt == 0:
            delta_pct = float("nan")
        else:
            delta_pct = (ct - rt) / rt * 100
        direction = "fewer" if delta_pct < 0 else "more" if delta_pct > 0 else "same"
        parts.append(f"{lbl}: {ct:,} ({delta_pct:+.2f}%, {direction})")
    return "  |  ".join(parts)


# ── Load master ───────────────────────────────────────────────────────────────
df = pd.read_csv(MASTER_CSV)

# ── Accumulators ──────────────────────────────────────────────────────────────
raw_byte_counts = defaultdict(lambda: defaultdict(int))
raw_totals      = defaultdict(int)
code_to_meta    = {}

# per-mode accumulators, keyed by mode label
mode_byte_counts = {m["label"]: defaultdict(lambda: defaultdict(int)) for m in MODES}
mode_totals      = {m["label"]: defaultdict(int) for m in MODES}

# ── Build metadata lookup ─────────────────────────────────────────────────────
for _, row in df.iterrows():
    code      = row["Code_Orig"]
    stype     = row["Script_Type"]
    bpc       = row["Bytes_Per_Char"]
    group_key = (stype, bpc)
    code_to_meta[code] = group_key

# ── Single pass over files ────────────────────────────────────────────────────
all_codes = df["Code_Orig"].tolist()

for i, code in enumerate(all_codes):
    fpath = os.path.join(RESULTS_DIR, f"{code}.json")
    if not os.path.exists(fpath):
        print(f"  MISSING: {fpath}")
        continue

    group_key = code_to_meta[code]
    print(f"  [{i+1}/{len(all_codes)}] {code}    ", end="\r", flush=True)

    with open(fpath) as f:
        sentences = json.load(f)

    for sent in sentences:
        byte_vals = [b[0] for b in sent["bytes_entropies"]]

        # raw entropy (shared baseline)
        try:
            raw_patches = sent["eval_modes"]["raw_entropy"][RAW_T]["patch_lengths"]
        except KeyError:
            raw_patches = []
        pos = 0
        for plen in raw_patches:
            if pos > 0:
                b = byte_vals[pos - 1]
                raw_byte_counts[group_key][b] += 1
                raw_totals[group_key] += 1
            pos += plen

        # each comparison mode
        for m in MODES:
            lbl = m["label"]
            try:
                patches = sent["eval_modes"][m["key"]][m["t"]]["patch_lengths"]
            except KeyError:
                patches = []
            pos = 0
            for plen in patches:
                if pos > 0:
                    b = byte_vals[pos - 1]
                    mode_byte_counts[lbl][group_key][b] += 1
                    mode_totals[lbl][group_key] += 1
                pos += plen

print()


# ── column-pasting helper ───────────────────────────────────────────────────
def paste_columns(block_a, block_b, gap=6):
    """Paste two multi-line strings side by side, padding the narrower column."""
    lines_a = block_a.split("\n")
    lines_b = block_b.split("\n")
    width_a = max((len(l) for l in lines_a), default=0)
    n = max(len(lines_a), len(lines_b))
    lines_a += [""] * (n - len(lines_a))
    lines_b += [""] * (n - len(lines_b))
    out_lines = []
    for la, lb in zip(lines_a, lines_b):
        out_lines.append(f"{la.ljust(width_a)}{' ' * gap}{lb}")
    return "\n".join(out_lines)


def gainers_losers_block(label, raw_counts, cmp_counts, rt, ct, reverse):
    all_bytes = set(raw_counts) | set(cmp_counts)
    ranked = sorted(
        all_bytes,
        key=lambda b: (cmp_counts.get(b, 0) / ct if ct else 0) -
                      (raw_counts.get(b, 0) / rt if rt else 0),
        reverse=reverse
    )[:5]
    title = "gainers" if reverse else "losers"
    lines = [f"Top {title} under {label}:",
             f"{'byte':>5}  {'char':>6}  {'raw_%':>7}  {label[:4]+'_%':>7}  {'delta_%':>8}",
             "-" * 44]
    for b in ranked:
        raw_pct = raw_counts.get(b, 0) / rt * 100 if rt else 0
        cmp_pct = cmp_counts.get(b, 0) / ct * 100 if ct else 0
        char    = chr(b) if 32 <= b < 127 else f"x{b:02x}"
        lines.append(f"{b:>5}  {char:>6}  {raw_pct:>7.2f}  {cmp_pct:>7.2f}  {cmp_pct - raw_pct:>+8.2f}")
    return "\n".join(lines)


# ── Write report ──────────────────────────────────────────────────────────────
BPC_LABEL   = {"1 byte": "1-byte", "2 bytes": "2-byte", "3 bytes": "3-byte"}
STYPE_ORDER = ["Alphabet", "Abjad", "Abugida", "Syllabary", "Logosyllabary","Logographic"]

os.makedirs("analyse_patching_methods", exist_ok=True)
out_path = "analyse_patching_methods/analyse_patching_methods.txt"

with open(out_path, "w") as out:

    for stype in STYPE_ORDER:
        out.write(f"\n{'═'*70}\n")
        out.write(f"  {stype.upper()}\n")
        out.write(f"{'═'*70}\n")

        for bpc in ["1 byte", "2 bytes", "3 bytes"]:
            group_key = (stype, bpc)

            sub = df[(df["Script_Type"] == stype) & (df["Bytes_Per_Char"] == bpc)]
            if sub.empty:
                continue

            rt = raw_totals.get(group_key, 0)
            cts = {m["label"]: mode_totals[m["label"]].get(group_key, 0) for m in MODES}

            boundary_line = f"boundary counts — {boundary_delta_line(rt, cts, MODES)}"
            out.write(f"\n  ── {BPC_LABEL[bpc]}  ({boundary_line}) ──\n\n")

            raw_counts = raw_byte_counts.get(group_key, {})

            # gainers side by side across modes
            gainer_blocks = []
            loser_blocks = []
            for m in MODES:
                lbl = m["label"]
                cmp_counts = mode_byte_counts[lbl].get(group_key, {})
                ct = cts[lbl]
                gainer_blocks.append(gainers_losers_block(lbl, raw_counts, cmp_counts, rt, ct, reverse=True))
                loser_blocks.append(gainers_losers_block(lbl, raw_counts, cmp_counts, rt, ct, reverse=False))

            pasted_gainers = gainer_blocks[0]
            for blk in gainer_blocks[1:]:
                pasted_gainers = paste_columns(pasted_gainers, blk)
            pasted_losers = loser_blocks[0]
            for blk in loser_blocks[1:]:
                pasted_losers = paste_columns(pasted_losers, blk)

            for line in pasted_gainers.split("\n"):
                out.write(f"  {line}\n")
            out.write("\n")
            for line in pasted_losers.split("\n"):
                out.write(f"  {line}\n")

print(f"Written: {out_path}")