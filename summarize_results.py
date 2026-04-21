"""
summarize_results.py
Aggregate patching statistics across languages from the FLORES eval results.

Output: results/summary.json  +  printed table

Usage:
    python summarize_results.py
"""

import json
import os
from eval_flores import LANGUAGES

RESULTS_DIR = "results"

def summarize_language(lang_code: str) -> dict | None:
    path = os.path.join(RESULTS_DIR, f"{lang_code}.json")
    if not os.path.exists(path):
        print(f"  Missing: {path}, skipping")
        return None

    with open(path, encoding="utf-8") as f:
        results = json.load(f)

    if not results:
        return None

    total_patches = sum(r["n_patches"] for r in results)
    total_bytes = sum(r["n_bytes"] for r in results)
    avg_bytes_per_patch = total_bytes / max(total_patches, 1)
    avg_patches_per_sentence = total_patches / len(results)

    return {
        "lang_code": lang_code,
        "lang_name": LANGUAGES.get(lang_code, lang_code),
        "n_sentences": len(results),
        "total_patches": total_patches,
        "total_bytes": total_bytes,
        "avg_bytes_per_patch": round(avg_bytes_per_patch, 4),
        "avg_patches_per_sentence": round(avg_patches_per_sentence, 4),
    }


def main():
    summary = []

    for lang_code in LANGUAGES:
        result = summarize_language(lang_code)
        if result is not None:
            summary.append(result)

    summary.sort(key=lambda x: x["total_patches"])

    # Print table
    # Compute premium vs English
    eng = next((s for s in summary if s["lang_code"] == "eng_Latn"), None)
    for s in summary:
        if eng and eng["avg_patches_per_sentence"] > 0:
            s["patch_premium"] = round(s["avg_patches_per_sentence"] / eng["avg_patches_per_sentence"], 4)
        else:
            s["patch_premium"] = None

    # Print table
    max_name_len = max(len(s["lang_name"]) for s in summary)
    max_code_len = max(len(s["lang_code"]) for s in summary)
    header = (
        f"{'Language':<{max_name_len}} {'Code':<{max_code_len}} {'Sentences':>10} "
        f"{'Total patches':>14} {'Total bytes':>12} "
        f"{'Avg b/patch':>12} {'Avg patches/sent':>17} {'Premium vs EN':>14}"
    )
    lines = [header, "-" * len(header)]
    for s in summary:
        premium_str = f"{s['patch_premium']:.4f}" if s["patch_premium"] is not None else "N/A"
        lines.append(
            f"{s['lang_name']:<{max_name_len}} {s['lang_code']:<{max_code_len}} {s['n_sentences']:>10} "
            f"{s['total_patches']:>14} {s['total_bytes']:>12} "
            f"{s['avg_bytes_per_patch']:>12.4f} {s['avg_patches_per_sentence']:>17.4f} "
            f"{premium_str:>14}"
        )

    table = "\n".join(lines)
    print("\n" + table)

    # Save JSON
    out_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {out_path}")

    # Save TXT
    txt_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"Saved → {txt_path}")


if __name__ == "__main__":
    main()