import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

MODEL_OUTPUT_DIR = Path(
    "results/own_models/entropy_10M_20lang_4gpu_sourcesbalanced_steps10000_ckpt200_lr4.5e-3/step_0000007200"
)


def byte_to_character_entropy(bytes_entropies):
    """Aggregates byte-level entropies into character-level entropies by inspecting

    UTF-8 byte headers (lead bytes vs continuation bytes).
    """
    char_entropies = []
    current_char_entropy = 0.0

    for item in bytes_entropies:
        byte_val = item[0]
        entropy_val = item[1]

        # UTF-8 Continuation bytes start with binary '10xxxxxx' (128 to 191 in decimal)
        is_continuation = 128 <= byte_val <= 191

        if is_continuation and len(char_entropies) > 0:
            # Accumulate entropy into the current character unit
            current_char_entropy += entropy_val
        else:
            # If previous character exists, store its aggregated entropy
            if current_char_entropy > 0:
                char_entropies.append(current_char_entropy)
            # Start new character unit
            current_char_entropy = entropy_val

    if current_char_entropy > 0:
        char_entropies.append(current_char_entropy)

    return char_entropies


def compute_char_level_shape(file_path: Path) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    char_entropies = []
    for sample in data:
        if "bytes_entropies" in sample:
            # Convert UTF-8 bytes -> aggregated character entropy
            c_entropies = byte_to_character_entropy(sample["bytes_entropies"])
            char_entropies.extend(c_entropies)

    char_entropies = np.array(char_entropies, dtype=np.float64)

    if len(char_entropies) == 0:
        return {"language": file_path.stem, "kurtosis": np.nan}

    skew_val = skew(char_entropies)
    kurt_val = kurtosis(char_entropies, fisher=True)

    return {
        "language": file_path.stem.replace(".json", ""),
        "char_spikiness_kurtosis": round(kurt_val, 4),
        "char_skewness": round(skew_val, 4),
        "mean_char_H": round(np.mean(char_entropies), 4),
        "var_char_H": round(np.var(char_entropies, ddof=1), 4),
        "profile": (
            "Flat with Spikes"
            if kurt_val > 0.5
            else ("Wavy" if kurt_val < -0.2 else "Moderate")
        ),
    }


def analyze_directory(directory_path: Path):
    json_files = sorted(list(directory_path.glob("*.json")))
    if not json_files:
        json_files = sorted(list(directory_path.rglob("*.json")))

    results = [compute_char_level_shape(fp) for fp in json_files]
    df = pd.DataFrame(results)
    return df.sort_values(by="char_spikiness_kurtosis", ascending=False)


if __name__ == "__main__":
    df_shapes = analyze_directory(MODEL_OUTPUT_DIR)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print("=== CHARACTER-LEVEL DISTRIBUTION SHAPE METRICS ===")
    print(df_shapes.to_string(index=False))