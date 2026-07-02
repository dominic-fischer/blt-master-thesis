"""
check_fineweb_availability.py

For each language in langs_chosen.csv (already enriched with one
"<config_name>_bytes" column per model size via add_language_allocations.py),
checks whether FineWeb2 (or FineWeb, for English) actually has that many
bytes available for that language.

Special cases:
  - eng_Latn routes to HuggingFaceFW/fineweb instead of fineweb-2
    (fineweb-2 does not cover English).
  - cmn_Hans is queried in fineweb-2 under the config name cmn_Hani.

Because byte requirements are monotonically increasing across model configs
(same ratio_vs_english, scaled by a growing total budget), each language's
data is streamed ONCE, accumulating bytes, and every config threshold is
marked OK the moment cumulative bytes cross it. This avoids re-scanning the
same language once per config.

Output: a CSV matrix, one row per language, one column per model config,
cell value "OK" or "--".

NOTE: this script needs network access to huggingface.co (via `datasets`),
run it in the same environment as your original fineweb2_data_amounts.py.

Usage:
    python check_fineweb_availability.py \
        training_setup/langs/langs_chosen.csv \
        [output_csv] \
        [--max-docs N]
"""

import csv
import sys
import argparse

from datasets import load_dataset


FINEWEB2_DATASET = "HuggingFaceFW/fineweb-2"
FINEWEB_EN_DATASET = "HuggingFaceFW/fineweb"
FINEWEB_EN_CONFIG = "default"  # adjust if your original script used a
                                 # specific dump/sample config for English

# fineweb-2 language-code overrides (our code -> fineweb-2's actual config name)
LANG_CODE_OVERRIDES = {
    "cmn_Hans": "cmn_Hani",
}

DEFAULT_MAX_DOCS = 200_000  # safety cap so a language that never reaches its
                             # largest threshold doesn't stream forever


def get_config_columns(fieldnames: list[str]) -> list[str]:
    return [f for f in fieldnames if f.endswith("_bytes")]


def resolve_dataset(language_code: str) -> tuple[str, str | None]:
    """Returns (dataset_name, dataset_config) for a given language code."""
    if language_code == "eng_Latn":
        return FINEWEB_EN_DATASET, FINEWEB_EN_CONFIG
    fineweb2_config = LANG_CODE_OVERRIDES.get(language_code, language_code)
    return FINEWEB2_DATASET, fineweb2_config


def check_language(language_code: str, thresholds: dict[str, int],
                    max_docs: int) -> dict[str, str]:
    """Streams a language's data, returns {config_name: "OK"/"--"}."""
    dataset_name, dataset_config = resolve_dataset(language_code)
    result = {cfg: "--" for cfg in thresholds}
    remaining = dict(sorted(thresholds.items(), key=lambda kv: kv[1]))

    print(f"--- {language_code} -> {dataset_name} ({dataset_config}) ---")
    try:
        ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
    except Exception as e:
        print(f"  ERROR loading dataset: {e}")
        return result

    cumulative_bytes = 0
    docs_seen = 0
    try:
        for doc in ds:
            text = doc.get("text", "")
            cumulative_bytes += len(text.encode("utf-8"))
            docs_seen += 1

            # mark any thresholds we've now crossed
            crossed = [cfg for cfg, target in remaining.items()
                       if cumulative_bytes >= target]
            for cfg in crossed:
                result[cfg] = "OK"
                del remaining[cfg]

            if not remaining:
                print(f"  all {len(thresholds)} thresholds OK after "
                      f"{docs_seen} docs ({cumulative_bytes:,} bytes)")
                break
            if docs_seen >= max_docs:
                print(f"  hit max_docs={max_docs} cap, "
                      f"{cumulative_bytes:,} bytes seen, "
                      f"{len(remaining)} threshold(s) unmet: {list(remaining)}")
                break
    except Exception as e:
        print(f"  ERROR while streaming: {e} "
              f"(after {docs_seen} docs, {cumulative_bytes:,} bytes)")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("langs_csv", nargs="?",
                         default="training_setup/langs/langs_chosen.csv")
    parser.add_argument("output_csv", nargs="?",
                         default="training_setup/langs/fineweb_availability.csv")
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS)
    args = parser.parse_args()

    with open(args.langs_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        lang_rows = list(reader)

    config_columns = get_config_columns(fieldnames)
    if not config_columns:
        print("No '<config>_bytes' columns found — run add_language_allocations.py first.")
        sys.exit(1)
    config_names = [c[: -len("_bytes")] for c in config_columns]

    matrix_rows = []
    for row in lang_rows:
        language_code = row["FLORES_Code"]
        thresholds = {
            cfg_name: int(row[col])
            for cfg_name, col in zip(config_names, config_columns)
        }
        availability = check_language(language_code, thresholds, args.max_docs)
        matrix_rows.append({"language_code": language_code, **availability})

    out_fieldnames = ["language_code"] + config_names
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(matrix_rows)

    print(f"\nWrote availability matrix to {args.output_csv}")


if __name__ == "__main__":
    main()