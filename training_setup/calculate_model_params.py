"""
compute_model_sizes.py

Reads a base CSV of transformer config specs (dim, n_heads, head_dim, n_layers,
ffn_dim_multiplier, vocab_size) and computes, per row:

  - ffn_hidden_dim        : SwiGLU FFN hidden size
  - per_layer_params      : params in one transformer block
  - total_params          : full model param count (tied embed/output head)
  - advisable_training_data_bytes : Chinchilla-style ~20 bytes/param budget

Skips the DOCUMENTATION row (copied through unchanged, computed columns left blank).

Usage:
    python compute_model_sizes.py blt_configs_base.csv blt_configs_computed.csv
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
    attn_params = 4 * dim ** 2                # Q, K, V, O projections
    ffn_params = 3 * dim * ffn_hidden_dim     # SwiGLU gate, up, down projections
    return attn_params + ffn_params


def compute_total_params(per_layer_params: int, n_layers: int, vocab_size: int, dim: int) -> int:
    return per_layer_params * n_layers + vocab_size * dim


def compute_advisable_training_data_bytes(total_params: int, ratio: int = 20) -> int:
    return total_params * ratio


def main(in_path: str, out_path: str) -> None:
    with open(in_path, newline="") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)
        fieldnames = reader.fieldnames + [
            "ffn_hidden_dim",
            "per_layer_params",
            "total_params",
            "advisable_training_data_bytes",
        ]

    computed_rows = []
    for row in rows:
        if row["config_name"] == "DOCUMENTATION":
            row["ffn_hidden_dim"] = "FFN hidden size after SwiGLU sizing + rounding to 256"
            row["per_layer_params"] = "params in one transformer block (attn + FFN)"
            row["total_params"] = "total trainable parameters, approximate"
            row["advisable_training_data_bytes"] = "Chinchilla-style ~20 bytes per parameter"
            computed_rows.append(row)
            continue

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
        advisable_bytes = compute_advisable_training_data_bytes(total_params)

        row["ffn_hidden_dim"] = ffn_hidden_dim
        row["per_layer_params"] = per_layer_params
        row["total_params"] = total_params
        row["advisable_training_data_bytes"] = advisable_bytes
        computed_rows.append(row)

    with open(out_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(computed_rows)

    print(f"Wrote {len(computed_rows)} rows to {out_path}")


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "training_setup/model_configs_base.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "training_setup/model_configs_computed.csv"
    main(in_path, out_path)