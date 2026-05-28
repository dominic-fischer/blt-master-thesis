"""
Adds a "Minimal Full Classification" column to floresplus_langs_enriched.csv.

A node is REDUNDANT when its parent has the exact same number of members
(i.e. the parent has exactly one child in the dataset, so the intermediate
node adds no discriminating information).

Also prints the resulting family tree in an indented format.
"""

import pandas as pd
from collections import defaultdict

INPUT  = "floresplus_langs_enriched.csv"
OUTPUT = "floresplus_langs_enriched_minimal.csv"

SEP = " > "

# ── Load ────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT)

# ── Count members per node ───────────────────────────────────────────────────
# For every full classification path, every prefix is a node.
# We count how many *leaf rows* (languages) fall under each node.
# A row is identified by its unique (Glottocode, Code, Script) combination
# so duplicates (e.g. same language with two scripts) are kept separate.

node_count: dict[str, int] = defaultdict(int)

for path in df["Full_Classification"].dropna():
    parts = [p.strip() for p in path.split(">")]
    for depth in range(1, len(parts) + 1):
        node = SEP.join(parts[:depth])
        node_count[node] += 1

# ── Find redundant nodes ─────────────────────────────────────────────────────
# A node N is redundant iff its parent has the same member count as N.
# (parent count == N count  →  every language in the parent is also in N
#  →  they are the same partition  →  the intermediate node adds no info)

def parent_of(node: str) -> str | None:
    parts = node.split(SEP)
    if len(parts) <= 1:
        return None
    return SEP.join(parts[:-1])

redundant: set[str] = set()
for node, cnt in node_count.items():
    par = parent_of(node)
    if par is not None and node_count.get(par) == cnt:
        redundant.add(node)

# ── Build minimal path ───────────────────────────────────────────────────────

def minimal_path(full_path: str) -> str:
    parts = [p.strip() for p in full_path.split(">")]
    kept = []
    for depth, part in enumerate(parts, start=1):
        node = SEP.join(parts[:depth])
        if node not in redundant:
            kept.append(part)
    return SEP.join(kept)

df["Minimal Full Classification"] = df["Full_Classification"].apply(
    lambda x: minimal_path(x) if pd.notna(x) else x
)

# ── Save ─────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT, index=False)
print(f"Saved → {OUTPUT}\n")

# ── Print indented tree ───────────────────────────────────────────────────────
# Build tree from minimal paths only
tree: dict = {}  # nested dict: node_label -> children dict

for path in df["Minimal Full Classification"].dropna().unique():
    parts = [p.strip() for p in path.split(">")]
    cursor = tree
    for part in parts:
        cursor = cursor.setdefault(part, {})

def print_tree(node: dict, indent: int = 0) -> None:
    for label in sorted(node):
        children = node[label]
        prefix = "    " * indent + ("└─ " if indent > 0 else "")
        print(f"{prefix}{label}")
        print_tree(children, indent + 1)

print("=" * 60)
print("Minimal Classification Tree")
print("=" * 60)
print_tree(tree)