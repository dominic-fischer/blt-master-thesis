"""
compute_model_sizes.py

Reads a CSV of transformer config specs (dim, n_heads, head_dim, n_layers,
ffn_dim_multiplier, vocab_size) and computes, per row:

  - ffn_hidden_dim   : SwiGLU FFN hidden size
  - per_layer_params : params in one transformer block (attn + FFN)
  - total_params     : full model param count (untied embed + output head)
  - aspect_ratio     : dim / n_layers, a rough width-vs-depth shorthand

Usage:
    python compute_model_sizes.py blt_configs.csv blt_configs_computed.csv
"""

import csv
import sys
import math


def round_up_to_multiple(value: float, multiple: int = 256) -> int:
    return math.ceil(value / multiple) * multiple


def compute_ffn_hidden_dim(dim: int, ffn_dim_multiplier: float) -> int:
    # Llama-style SwiGLU sizing: default FFN width = 2/3 * 4*dim, then scaled
    # by ffn_dim_multiplier, then rounded up to the nearest multiple of 256.
    raw = int((2 / 3) * 4 * dim)
    scaled = raw * ffn_dim_multiplier
    return round_up_to_multiple(scaled, 256)


def compute_per_layer_params(dim: int, ffn_hidden_dim: int) -> int:
    attn_params = 4 * dim ** 2             # Q, K, V, O projections
    ffn_params = 3 * dim * ffn_hidden_dim  # SwiGLU gate, up, down projections
    return attn_params + ffn_params


def compute_total_params(per_layer_params: int, n_layers: int, vocab_size: int, dim: int) -> int:
    # untied embedding + output head: 2 * vocab_size * dim
    return per_layer_params * n_layers + 2 * vocab_size * dim


def compute_aspect_ratio(dim: int, n_layers: int) -> float:
    # width-to-depth ratio; a rough shorthand for how "wide" vs "deep" the
    # model is. Real transformer families (GPT-2, etc.) tend to sit
    # somewhere around ~30-65 -- much lower means unusually deep/narrow,
    # much higher means unusually wide/shallow.
    return dim / n_layers


def main(in_path: str, out_path: str) -> None:
    with open(in_path, newline="") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)
        fieldnames = list(dict.fromkeys(
            reader.fieldnames + ["ffn_hidden_dim", "per_layer_params", "total_params", "aspect_ratio"]
        ))

    computed_rows = []
    for row in rows:
        dim = int(row["dim"])
        n_heads = int(row["n_heads"])
        head_dim = int(row["head_dim"])
        n_layers = int(row["n_layers"])
        ffn_dim_multiplier = float(row["ffn_dim_multiplier"])
        vocab_size = int(row["vocab_size"])

        assert dim == n_heads * head_dim, (
            f"{row['config_name']}: dim ({dim}) != n_heads*head_dim "
            f"({n_heads}*{head_dim}={n_heads * head_dim})"
        )

        ffn_hidden_dim = compute_ffn_hidden_dim(dim, ffn_dim_multiplier)
        per_layer_params = compute_per_layer_params(dim, ffn_hidden_dim)
        total_params = compute_total_params(per_layer_params, n_layers, vocab_size, dim)
        aspect_ratio = compute_aspect_ratio(dim, n_layers)

        row["ffn_hidden_dim"] = ffn_hidden_dim
        row["per_layer_params"] = per_layer_params
        row["total_params"] = total_params
        row["aspect_ratio"] = round(aspect_ratio, 2)
        computed_rows.append(row)

    with open(out_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(computed_rows)

    print(f"Wrote {len(computed_rows)} rows to {out_path}\n")

    # Print a quick summary table (dim/n_layers = "aspect ratio") to stdout
    header = f"{'config_name':12}{'dim':>6}{'n_heads':>9}{'n_layers':>10}{'aspect_ratio':>14}{'total_params':>16}"
    print(header)
    print("-" * len(header))
    for row in computed_rows:
        print(
            f"{row['config_name']:12}{int(row['dim']):6}{int(row['n_heads']):9}"
            f"{int(row['n_layers']):10}{row['aspect_ratio']:14}{int(row['total_params']):16,}"
        )


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "training_setup/model_configs.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "training_setup/model_configs_computed.csv"
    main(in_path, out_path)