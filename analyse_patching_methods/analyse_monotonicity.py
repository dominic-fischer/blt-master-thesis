"""
analyse_monotonicity_plateaus.py

Compares raw entropy vs raw monotonicity patching by computing
within-patch entropy statistics (mean and variance) per patch.

For each (script_type, bytes_per_char) group, reports:
  - mean within-patch entropy variance  (lower = flatter patches)
  - mean within-patch entropy mean      (higher = high-entropy plateaus)

A plateau signature = high mean entropy + low variance, more common
in monotonicity than raw entropy.

Output: analyse_patching_methods/analyse_monotonicity.txt
"""

import os
import json
import statistics
import pandas as pd
from collections import defaultdict

MASTER_CSV  = "floresplus_MASTER_CSV.csv"
RESULTS_DIR = "results/restructured"
RAW_T       = "t_1.3340"   # anchor threshold for raw entropy
MONO_T      = "t_0.3662"   # anchor threshold for raw monotonicity
ENTROPY_IDX = 1            # index in bytes_entropies triple

OUTPUT_DIR  = "analyse_patching_methods"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load master ───────────────────────────────────────────────────────────────
df = pd.read_csv(MASTER_CSV)

# ── Accumulators ──────────────────────────────────────────────────────────────
# per group: lists of (within_patch_mean, within_patch_variance) for each patch
raw_patch_stats  = defaultdict(lambda: {"var": [], "mean": []})
mono_patch_stats = defaultdict(lambda: {"var": [], "mean": []})

code_to_meta = {}
for _, row in df.iterrows():
    code_to_meta[row["Code_Orig"]] = (row["Script_Type"], row["Bytes_Per_Char"])

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
        entropies = [b[ENTROPY_IDX] for b in sent["bytes_entropies"]]

        def collect(patch_lengths, accum):
            pos = 0
            for plen in patch_lengths:
                seg = entropies[pos:pos + plen]
                if len(seg) >= 2:
                    accum["var"].append(statistics.variance(seg))
                    accum["mean"].append(statistics.mean(seg))
                elif len(seg) == 1:
                    accum["var"].append(0.0)
                    accum["mean"].append(seg[0])
                pos += plen

        try:
            raw_pl = sent["eval_modes"]["raw_entropy"][RAW_T]["patch_lengths"]
        except KeyError:
            raw_pl = []
        collect(raw_pl, raw_patch_stats[group_key])

        try:
            mono_pl = sent["eval_modes"]["raw_monotonicity"][MONO_T]["patch_lengths"]
        except KeyError:
            mono_pl = []
        collect(mono_pl, mono_patch_stats[group_key])

print()

# ── Helper ────────────────────────────────────────────────────────────────────
def summarise(stats):
    if not stats["var"]:
        return None
    n           = len(stats["var"])
    mean_var    = sum(stats["var"])  / n
    mean_mean   = sum(stats["mean"]) / n
    # plateau patches: mean entropy > 2.0 AND variance < 0.5
    plateau_n   = sum(1 for v, m in zip(stats["var"], stats["mean"])
                      if m > 2.0 and v < 0.5)
    plateau_pct = plateau_n / n * 100
    return dict(n=n, mean_var=mean_var, mean_mean=mean_mean,
                plateau_n=plateau_n, plateau_pct=plateau_pct)

# ── Write report ──────────────────────────────────────────────────────────────
STYPE_ORDER = ["Alphabet", "Abjad", "Abugida", "Logographic"]
BPC_ORDER   = ["1 byte", "2 bytes", "3 bytes"]

out_path = os.path.join(OUTPUT_DIR, "analyse_monotonicity.txt")
with open(out_path, "w") as out:

    for stype in STYPE_ORDER:
        out.write(f"\n{'═'*70}\n")
        out.write(f"  {stype.upper()}\n")
        out.write(f"{'═'*70}\n")

        for bpc in BPC_ORDER:
            group_key = (stype, bpc)
            raw_s  = summarise(raw_patch_stats[group_key])
            mono_s = summarise(mono_patch_stats[group_key])
            if raw_s is None and mono_s is None:
                continue

            bpc_label = bpc.replace(" ", "-")
            out.write(f"\n  ── {bpc_label} ──\n\n")
            out.write(f"  {'metric':<30} {'raw_entropy':>14} {'raw_mono':>14} {'delta':>10}\n")
            out.write(f"  {'-'*70}\n")

            metrics = [
                ("patches (n)",        f"{raw_s['n']:,}"     if raw_s else "—",
                                        f"{mono_s['n']:,}"    if mono_s else "—", None),
                ("mean within-patch var",
                 raw_s["mean_var"]  if raw_s  else None,
                 mono_s["mean_var"] if mono_s else None, True),
                ("mean within-patch entropy",
                 raw_s["mean_mean"]  if raw_s  else None,
                 mono_s["mean_mean"] if mono_s else None, True),
                ("plateau patches %",
                 raw_s["plateau_pct"]  if raw_s  else None,
                 mono_s["plateau_pct"] if mono_s else None, True),
            ]

            for label, rv, mv, numeric in metrics:
                if numeric:
                    r_str = f"{rv:.4f}" if rv is not None else "—"
                    m_str = f"{mv:.4f}" if mv is not None else "—"
                    d_str = f"{mv - rv:+.4f}" if (rv is not None and mv is not None) else "—"
                else:
                    r_str, m_str, d_str = str(rv), str(mv), ""
                out.write(f"  {label:<30} {r_str:>14} {m_str:>14} {d_str:>10}\n")

            # Per-language breakdown (sorted by plateau_pct delta)
            sub = df[(df["Script_Type"] == stype) & (df["Bytes_Per_Char"] == bpc)]
            if not sub.empty:
                lang_rows = []
                for _, row in sub.iterrows():
                    code = row["Code_Orig"]
                    raw_pps  = row.get(f"raw_entropy_{RAW_T}_pps",  None)
                    mono_pps = row.get(f"raw_monotonicity_{MONO_T}_pps", None)
                    raw_prem  = row.get(f"raw_entropy_{RAW_T}_pps_premium",  None)
                    mono_prem = row.get(f"raw_monotonicity_{MONO_T}_pps_premium", None)
                    if raw_prem is not None and mono_prem is not None:
                        lang_rows.append((row["Name"], code, raw_prem, mono_prem,
                                          mono_prem - raw_prem))

                if lang_rows:
                    lang_rows.sort(key=lambda x: x[4])  # sorted by mono-raw delta
                    out.write(f"\n  Per-language pps premium  (sorted by mono−raw delta):\n")
                    out.write(f"  {'name':<35} {'code':<22} "
                              f"{'raw_prem':>9} {'mono_prem':>10} {'delta':>8}\n")
                    out.write(f"  {'-'*88}\n")
                    for name, code, rp, mp, d in lang_rows:
                        out.write(f"  {str(name):<35} {code:<22} "
                                  f"{rp:>9.3f} {mp:>10.3f} {d:>+8.3f}\n")

print(f"Written: {out_path}")