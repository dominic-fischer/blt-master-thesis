import os
import json
import pandas as pd
from collections import defaultdict

MASTER_CSV  = "floresplus_MASTER_CSV.csv"
RESULTS_DIR = "results/restructured"
RAW_T       = "t_1.3340"
NORM_T      = "t_0.5293"

# ── Load master ───────────────────────────────────────────────────────────────
df = pd.read_csv(MASTER_CSV)

df["prem_delta"] = df["prem_norm_entropy_t_0.5293"] - df["prem_raw_entropy_t_1.3340"]

# ── Accumulators ──────────────────────────────────────────────────────────────
raw_byte_counts  = defaultdict(lambda: defaultdict(int))
norm_byte_counts = defaultdict(lambda: defaultdict(int))
raw_totals       = defaultdict(int)
norm_totals      = defaultdict(int)
code_to_meta     = {}

# ── Build metadata lookup ─────────────────────────────────────────────────────
for _, row in df.iterrows():
    code      = row["Code_orig"]
    stype     = row["Script_Type"]
    bpc       = row["Bytes_Per_Char"]
    group_key = (stype, bpc)
    code_to_meta[code] = group_key

# ── Single pass over files ────────────────────────────────────────────────────
all_codes = df["Code_orig"].tolist()

for i, code in enumerate(all_codes):
    fpath = os.path.join(RESULTS_DIR, f"{code}.json")
    if not os.path.exists(fpath):
        print(f"  MISSING: {fpath}")
        continue

    group_key = code_to_meta[code]
    print(f"  [{i+1}/{len(all_codes)}] {code}", end="\r", flush=True)

    with open(fpath) as f:
        sentences = json.load(f)

    for sent in sentences:
        byte_vals = [b[0] for b in sent["bytes_entropies"]]

        try:
            raw_patches = sent["eval_modes"]["raw_entropy"][RAW_T]["patch_lengths"]
        except KeyError:
            raw_patches = []
        pos = 0
        for plen in raw_patches:
            if pos > 0:
                raw_byte_counts[group_key][byte_vals[pos - 1]] += 1
                raw_totals[group_key] += 1
            pos += plen

        try:
            norm_patches = sent["eval_modes"]["norm_entropy"][NORM_T]["patch_lengths"]
        except KeyError:
            norm_patches = []
        pos = 0
        for plen in norm_patches:
            if pos > 0:
                norm_byte_counts[group_key][byte_vals[pos - 1]] += 1
                norm_totals[group_key] += 1
            pos += plen

print()

# ── Write report ──────────────────────────────────────────────────────────────
BPC_LABEL = {"1 byte": "1-byte", "2 bytes": "2-byte", "3 bytes": "3-byte"}
STYPE_ORDER = ["Alphabet", "Abjad", "Abugida", "Logographic"]

with open("analyse_patching_methods/analyse_normalisation.txt", "w") as out:

    for stype in STYPE_ORDER:
        out.write(f"\n{'═'*70}\n")
        out.write(f"  {stype.upper()}\n")
        out.write(f"{'═'*70}\n")

        for bpc in ["1 byte", "2 bytes", "3 bytes"]:
            group_key = (stype, bpc)

            # Languages in this group, sorted by improvement descending (most negative first)
            sub = df[(df["Script_Type"] == stype) & (df["Bytes_Per_Char"] == bpc)]
            if sub.empty:
                continue

            rt = raw_totals.get(group_key, 0)
            nt = norm_totals.get(group_key, 0)

            out.write(f"\n  ── {BPC_LABEL[bpc]}  "
                      f"(raw boundaries: {rt:,}  |  norm boundaries: {nt:,}) ──\n\n")

            # Languages sorted by prem_delta ascending (most improvement first)
            # out.write(f"  {'code':<30} {'name':<35} {'script':<20} "
            #           f"{'raw':>7} {'norm':>7} {'delta':>8}\n")
            # out.write(f"  {'-'*105}\n")
            # for _, row in sub.sort_values("prem_delta").iterrows():
            #     out.write(f"  {row['Code_orig']:<30} {str(row['Language']):<35} "
            #               f"{str(row['Script_Name']):<20} "
            #               f"{row['prem_raw_entropy_t_1.3340']:>7.3f} "
            #               f"{row['prem_norm_entropy_t_0.5293']:>7.3f} "
            #               f"{row['prem_delta']:>8.3f}\n")

            # Top 10 bytes by raw count
            raw_counts  = raw_byte_counts.get(group_key, {})
            norm_counts = norm_byte_counts.get(group_key, {})
            top_bytes   = sorted(raw_counts, key=lambda b: raw_counts[b], reverse=True)[:10]

            # out.write(f"\n  Top 10 bytes by raw frequency:\n")
            # out.write(f"  {'byte':>5}  {'char':>6}  {'raw_n':>9}  {'raw_%':>7}  "
            #           f"{'norm_n':>9}  {'norm_%':>7}  {'delta_%':>8}\n")
            # out.write(f"  {'-'*62}\n")
            # for b in top_bytes:
            #     raw_n    = raw_counts.get(b, 0)
            #     norm_n   = norm_counts.get(b, 0)
            #     raw_pct  = raw_n  / rt  * 100 if rt  else 0
            #     norm_pct = norm_n / nt  * 100 if nt  else 0
            #     char     = chr(b) if 32 <= b < 127 else f"x{b:02x}"
            #     out.write(f"  {b:>5}  {char:>6}  {raw_n:>9,}  {raw_pct:>7.2f}  "
            #               f"{norm_n:>9,}  {norm_pct:>7.2f}  {norm_pct-raw_pct:>+8.2f}\n")
            # out.write(f"\n")

            # Top gainers
            all_bytes = set(raw_counts) | set(norm_counts)
            gainers = sorted(all_bytes,
                             key=lambda b: (norm_counts.get(b, 0) / nt if nt else 0) -
                                           (raw_counts.get(b, 0) / rt if rt else 0),
                             reverse=True)[:5]
            out.write(f"  Top gainers under normalisation:\n")
            out.write(f"  {'byte':>5}  {'char':>6}  {'raw_%':>7}  {'norm_%':>7}  {'delta_%':>8}\n")
            out.write(f"  {'-'*44}\n")
            for b in gainers:
                raw_pct  = raw_counts.get(b, 0) / rt  * 100 if rt  else 0
                norm_pct = norm_counts.get(b, 0) / nt * 100 if nt  else 0
                char     = chr(b) if 32 <= b < 127 else f"x{b:02x}"
                out.write(f"  {b:>5}  {char:>6}  {raw_pct:>7.2f}  {norm_pct:>7.2f}  "
                          f"{norm_pct-raw_pct:>+8.2f}\n")

print("Written: analyse_patching_methods/analyse_normalisation.txt")