"""
add_language_allocations.py

Reads:
  - langs_chosen.csv        : must contain a 'ratio_vs_english' column
                               (and some language identifier column, e.g. 'language_code')
  - model_configs_computed.csv : must contain 'config_name' and
                               'advisable_training_data_bytes' columns

For EACH model config found in model_configs_computed.csv, adds one new
column to langs_chosen.csv named "<config_name>_bytes", containing the
fairness-adjusted (ratio-corrected) byte allocation for that language under
that model's total training-data budget.

Math (per config):
    content_unit = total_budget / sum(ratio_vs_english over all languages)
    bytes_allocated(lang) = content_unit * ratio_vs_english(lang)

This guarantees, independently for every config:
    sum(bytes_allocated over all languages) == total_budget
    bytes_allocated(lang) / ratio_vs_english(lang) == content_unit  (constant)
  -> i.e. every language gets the same amount of English-equivalent content,
     just spent as more or fewer raw bytes depending on script efficiency.

Usage:
    python add_language_allocations.py [langs_chosen.csv] [model_configs_computed.csv] [output_path]

Defaults to the project's standard paths (see DEFAULT_LANGS_PATH /
DEFAULT_CONFIGS_PATH below), so it can be run with no arguments from the
repo root:

    python add_language_allocations.py

If output_path is omitted, langs_chosen.csv is overwritten in place
(a .bak backup of the original is written alongside it first).
"""

import csv
import shutil
import sys
from pathlib import Path


DEFAULT_LANGS_PATH = "training_setup/langs/langs_chosen.csv"
DEFAULT_CONFIGS_PATH = "training_setup/model_configs_computed.csv"

RATIO_COLUMN = "ratio_vs_english"
CONFIG_NAME_COLUMN = "config_name"
BUDGET_COLUMN = "advisable_training_data_bytes"


def load_configs(configs_path: str) -> list[dict]:
    with open(configs_path, newline="") as f:
        rows = list(csv.DictReader(f))
    # Skip a DOCUMENTATION row if present (non-numeric budget field)
    configs = []
    for row in rows:
        try:
            row[BUDGET_COLUMN] = int(row[BUDGET_COLUMN])
        except (ValueError, KeyError):
            continue
        configs.append(row)
    if not configs:
        raise ValueError(f"No valid config rows found in {configs_path}")
    return configs


def load_languages(langs_path: str) -> tuple[list[dict], list[str]]:
    with open(langs_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if RATIO_COLUMN not in fieldnames:
        raise ValueError(f"'{RATIO_COLUMN}' column not found in {langs_path}")
    return rows, fieldnames


def add_allocations(lang_rows: list[dict], configs: list[dict]) -> list[str]:
    """Mutates lang_rows in place, adding '<config_name>_bytes' keys.
    Returns the list of new column names added, in config order."""
    ratio_sum = sum(float(r[RATIO_COLUMN]) for r in lang_rows)

    new_columns = []
    for cfg in configs:
        cfg_name = cfg[CONFIG_NAME_COLUMN]
        total_budget = cfg[BUDGET_COLUMN]
        content_unit = total_budget / ratio_sum

        col_name = f"{cfg_name}_bytes"
        new_columns.append(col_name)

        for row in lang_rows:
            ratio = float(row[RATIO_COLUMN])
            row[col_name] = round(content_unit * ratio)

    return new_columns


def main():
    langs_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LANGS_PATH
    configs_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CONFIGS_PATH
    output_path = sys.argv[3] if len(sys.argv) > 3 else langs_path

    if not Path(langs_path).exists():
        print(f"Error: langs file not found at '{langs_path}'")
        sys.exit(1)
    if not Path(configs_path).exists():
        print(f"Error: configs file not found at '{configs_path}'")
        sys.exit(1)

    configs = load_configs(configs_path)
    lang_rows, fieldnames = load_languages(langs_path)
    new_columns = add_allocations(lang_rows, configs)

    # Back up original file if we're overwriting in place
    if Path(output_path) == Path(langs_path):
        backup_path = langs_path + ".bak"
        shutil.copy(langs_path, backup_path)
        print(f"Backed up original to {backup_path}")

    out_fieldnames = fieldnames + new_columns
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(lang_rows)

    print(f"Added columns: {new_columns}")
    for cfg in configs:
        cfg_name = cfg[CONFIG_NAME_COLUMN]
        col = f"{cfg_name}_bytes"
        check_sum = sum(row[col] for row in lang_rows)
        print(f"  {cfg_name}: total_budget={cfg[BUDGET_COLUMN]:,}  "
              f"sum_allocated={check_sum:,}")
    print(f"Wrote {len(lang_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()