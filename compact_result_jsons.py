"""
compact_result_jsons.py

One-off: rewrites every results JSON under results/own_models/ (and
results/base_model/, if present) without indentation. Content is unchanged
-- only whitespace is removed -- and each file is verified by re-reading
it and comparing to the original before moving on.

Usage (from repo root):
    python compact_result_jsons.py
"""
import glob, json, os

paths = sorted(glob.glob("results/own_models/**/*.json", recursive=True)
               + glob.glob("results/base_model/*.json"))
before = after = 0
for p in paths:
    size0 = os.path.getsize(p)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    with open(tmp, encoding="utf-8") as f:
        assert json.load(f) == data, f"content mismatch for {p}"
    os.replace(tmp, p)
    size1 = os.path.getsize(p)
    before += size0; after += size1
    print(f"{size0/1e6:7.1f} MB -> {size1/1e6:6.1f} MB  {p}")
print(f"\nTotal: {before/1e9:.2f} GB -> {after/1e9:.2f} GB")
big = [p for p in paths if os.path.getsize(p) > 100e6]
print("Files still over 100 MB:", big if big else "none")