"""
Merges columns from results/summary.txt into floresplus_MASTER_CSV.csv.

Join key logic:
  - Default: Code + "_" + Script  (e.g. eng_Latn)
  - When Code_Script is duplicated in MASTER, append the glottocode suffix
    if that suffixed key exists in the summary (e.g. twi_Latn_akua1239).
  - Special case: Norwegian Bokmål radical variety uses "_radical" suffix.
"""

import re
import pandas as pd
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
SUMMARY_PATH = Path("results/summary.txt")
MASTER_PATH  = Path("floresplus_MASTER_CSV.csv")
OUTPUT_PATH  = Path("floresplus_MASTER_CSV_merged.csv")

# ── 1. load summary.txt ───────────────────────────────────────────────────────
# Parse by anchoring on the Code column (word_underscore pattern) rather than
# whitespace splitting, which breaks when language names contain parentheses
# followed by 2+ spaces (e.g. "Modern Standard Arabic (Latin)  arb_Latn").
rows = []
with open(SUMMARY_PATH) as f:
    for line in f:
        line = line.rstrip()
        if not line or line.startswith("-"):
            continue
        m = re.search(r"(\S+_\S+)\s+(\d.*)", line)
        if m:
            lang = line[:m.start()].strip()
            code = m.group(1)
            rest = m.group(2).split()
            rows.append([lang, code] + rest)

cols = ["Language", "Code", "Sentences", "Total patches", "Total bytes",
        "Avg b/patch", "Avg patches/sent", "Premium vs EN"]
summary = pd.DataFrame(rows, columns=cols)
for c in cols[2:]:
    summary[c] = pd.to_numeric(summary[c])

summary["join_key"] = summary["Code"]
summary_glotto_keys = set(summary.loc[summary["Code"].str.count("_") > 1, "Code"])

print(f"Summary rows : {len(summary)}")

# ── 2. load MASTER CSV ────────────────────────────────────────────────────────
master = pd.read_csv(MASTER_PATH)
master["Code_Script"] = master["Code"] + "_" + master["Script"]
dupes = master.duplicated(subset="Code_Script", keep=False)

def make_join_key(row):
    cs = row["Code_Script"]
    if not dupes[row.name]:
        return cs
    glotto_raw = str(row["Glottocode"])
    if "radical" in glotto_raw:
        return cs + "_radical"
    glotto = glotto_raw.split("(")[0].strip()
    candidate = cs + "_" + glotto
    return candidate if candidate in summary_glotto_keys else cs

master["join_key"] = master.apply(make_join_key, axis=1)

print(f"MASTER rows  : {len(master)}")

# ── 3. merge ──────────────────────────────────────────────────────────────────
summary_m = summary.rename(columns={"Code": "Code_orig"})
merged = master.merge(summary_m, on="join_key", how="left")
merged = merged.drop(columns=["Code_Script", "join_key"])

# ── 4. report ─────────────────────────────────────────────────────────────────
n_missing = merged["Premium vs EN"].isna().sum()
print(f"Merged rows  : {len(merged)}")
print(f"Missing premium: {n_missing}")
if n_missing:
    print(merged[merged["Premium vs EN"].isna()][["Code", "Script", "Name"]])

# ── 5. save ───────────────────────────────────────────────────────────────────
merged.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved → {OUTPUT_PATH}")