"""
inspect_sentence.py

Given a lang_code (e.g. "eng_Latn") and a sentence index, reads the
pre-computed restructured results and:

  A) Saves HTML visualisations for every mode × threshold combination
     that exists in the restructured JSON, using BLTPatchVisualizer.

  B) Saves a .txt file with per-byte entropy, binary breakdown, top-k
     next-byte predictions (re-run through the entropy model).

Usage:
    python inspect_sentence.py eng_Latn 42
    python inspect_sentence.py hin_Deva 0 --top_k 3
    python inspect_sentence.py khm_Khmr 17 --out_dir my_output
"""

import argparse
import json
import os
import sys
import torch
from pathlib import Path

from blt_patcher import load_patcher, patch_text
from blt_visualize import BLTPatchVisualizer

# ── config ────────────────────────────────────────────────────────────────────
REPO           = "facebook/blt-1b"
ENTROPY_REPO   = "hf-weights/entropy_model"
RESTRUCTURED   = "results/restructured"

W_BITS   = 33
W_SCRIPT = 50

# ── Unicode / byte helpers (unchanged from original) ─────────────────────────
SCRIPT_RANGES = [
    (0,     127,   "ASCII"),
    (128,   591,   "Latin-Ext"),
    (592,   687,   "IPA"),
    (688,   879,   "Other_a"),
    (880,   1023,  "Greek"),
    (1024,  1327,  "Cyrillic"),
    (1328,  1423,  "Armenian"),
    (1424,  1535,  "Hebrew"),
    (1536,  1791,  "Arabic"),
    (1792,  1871,  "Syriac"),
    (1872,  2303,  "Other_b"),
    (2304,  2431,  "Devanagari"),
    (2432,  2559,  "Bengali"),
    (2560,  2687,  "Gurmukhi"),
    (2688,  2815,  "Gujarati"),
    (2816,  2943,  "Oriya"),
    (2944,  3071,  "Tamil"),
    (3072,  3199,  "Telugu"),
    (3200,  3327,  "Kannada"),
    (3328,  3455,  "Malayalam"),
    (3456,  3583,  "Sinhala"),
    (3584,  3711,  "Thai"),
    (3712,  3839,  "Lao"),
    (3840,  4095,  "Tibetan"),
    (4096,  4255,  "Myanmar"),
    (4256,  4351,  "Georgian"),
    (4352,  4607,  "Hangul-Jamo"),
    (4608,  5119,  "Ethiopic"),
    (5120,  6015,  "Other_c"),
    (6016,  6143,  "Khmer"),
    (6144,  6319,  "Mongolian"),
    (6320,  11903, "Other_d"),
    (11904, 12031, "CJK-Rad"),
    (12032, 12287, "Other_e"),
    (12288, 12351, "CJK-Sym"),
    (12352, 12447, "Hiragana"),
    (12448, 12543, "Katakana"),
    (12544, 13311, "Other_f"),
    (13312, 19903, "CJK-ExtA"),
    (19904, 19967, "Other_g"),
    (19968, 40959, "CJK"),
    (40960, 44031, "Other_h"),
    (44032, 55215, "Hangul"),
    (55216, 63743, "Other_i"),
    (63744, 64255, "CJK-Compat"),
    (64256, 65535, "Other_j"),
]

SPECIAL_CHARS = {0x00: "\\0", 0x09: "\\t", 0x0A: "\\n", 0x0D: "\\r"}


def get_script(codepoint):
    for start, end, name in SCRIPT_RANGES:
        if start <= codepoint <= end:
            return name, start, end
    return "Other", None, None


def seq_length(b):
    if b < 0x80: return 1
    if b < 0xE0: return 2
    if b < 0xF0: return 3
    return 4


def format_byte_binary(b):
    s = f"{b:08b}"
    if b < 0x80:  return f"0|{s[1:]}"
    if b < 0xC0:  return f"10|{s[2:]}"
    if b < 0xE0:  return f"110|{s[3:]}"
    if b < 0xF0:  return f"1110|{s[4:]}"
    return              f"11110|{s[5:]}"


def format_char(b):
    if b in SPECIAL_CHARS: return SPECIAL_CHARS[b]
    if 0x20 <= b <= 0x7E:  return f"'{chr(b)}'"
    return f"0x{b:02x}"


def build_char_map(context_bytes):
    char_map = []
    i = 0
    while i < len(context_bytes):
        b = context_bytes[i]
        length = seq_length(b)
        for j in range(length):
            if i + j < len(context_bytes):
                char_map.append((j, length, b))
        i += length
    return char_map


def make_script_str(cp_low, cp_high=None):
    if cp_high is None:
        name, start, end = get_script(cp_low)
        range_str = f"({start}-{end})" if start is not None else ""
        return f"{name}{range_str}"
    scripts = []
    cp = cp_low
    while cp <= cp_high:
        name, start, end = get_script(cp)
        if not scripts or scripts[-1][0] != name:
            scripts.append((name, start, end))
        cp = (end + 1) if end is not None else cp_high + 1
    low   = scripts[0][1]  if scripts[0][1]  is not None else cp_low
    high  = scripts[-1][2] if scripts[-1][2] is not None else cp_high
    names = "/".join(s[0] for s in scripts)
    return f"{names}({low}-{high})"


def format_bits(byte_val, off, total, lead, context_bytes, pos):
    pc = byte_val & 0x3F
    if byte_val < 0x80:
        name, start, end = get_script(byte_val)
        range_str = f"({start}-{end})" if start is not None else ""
        return f"{byte_val:07b}={byte_val}", f"{name}{range_str}"
    if byte_val >= 0xC0:
        if byte_val < 0xE0:
            cp_low  = (byte_val & 0x1F) << 6
            cp_high = cp_low | 0x3F
            lb = f"{byte_val & 0x1F:05b}"
            return f"{lb}+bbbbbb={cp_low}-{cp_high}", make_script_str(cp_low, cp_high)
        if byte_val < 0xF0:
            cp_low  = max((byte_val & 0x0F) << 12, 0x800)
            cp_high = cp_low | 0xFFF
            lb = f"{byte_val & 0x0F:04b}"
            return f"{lb}+1bbbbb+bbbbbb={cp_low}-{cp_high}", make_script_str(cp_low, cp_high)
        cp_low  = (byte_val & 0x07) << 18
        cp_high = cp_low | 0x3FFFF
        lb = f"{byte_val & 0x07:03b}"
        return f"{lb}+bbbbbb+bbbbbb+bbbbbb={cp_low}-{cp_high}", make_script_str(cp_low, cp_high)
    # continuation byte
    if total == 2:
        if off == 0:
            lb = f"{lead & 0x1F:05b}"
            pb = f"{pc:06b}"
            cp = ((lead & 0x1F) << 6) | pc
            return f"{lb}+{pb}={cp}", make_script_str(cp)
    if total == 3:
        if off == 0:
            lb = f"{lead & 0x0F:04b}"; pb = f"{pc:06b}"
            cp_low = ((lead & 0x0F) << 12) | (pc << 6); cp_high = cp_low | 0x3F
            return f"{lb}+{pb}+bbbbbb={cp_low}-{cp_high}", make_script_str(cp_low, cp_high)
        if off == 1:
            lb = f"{lead & 0x0F:04b}"; c1 = context_bytes[pos] & 0x3F
            c1b = f"{c1:06b}"; pb = f"{pc:06b}"
            cp = ((lead & 0x0F) << 12) | (c1 << 6) | pc
            return f"{lb}+{c1b}+{pb}={cp}", make_script_str(cp)
    if total == 4:
        if off == 0:
            lb = f"{lead & 0x07:03b}"; pb = f"{pc:06b}"
            cp_low = ((lead & 0x07) << 18) | (pc << 12); cp_high = cp_low | 0xFFF
            return f"{lb}+{pb}+bbbbbb+bbbbbb={cp_low}-{cp_high}", make_script_str(cp_low, cp_high)
        if off == 1:
            lb = f"{lead & 0x07:03b}"; c1 = context_bytes[pos] & 0x3F
            c1b = f"{c1:06b}"; pb = f"{pc:06b}"
            cp_low = ((lead & 0x07) << 18) | (c1 << 12) | (pc << 6); cp_high = cp_low | 0x3F
            return f"{lb}+{c1b}+{pb}+bbbbbb={cp_low}-{cp_high}", make_script_str(cp_low, cp_high)
        if off == 2:
            lb = f"{lead & 0x07:03b}"; c1 = context_bytes[pos - 1] & 0x3F
            c2 = context_bytes[pos] & 0x3F
            c1b = f"{c1:06b}"; c2b = f"{c2:06b}"; pb = f"{pc:06b}"
            cp = ((lead & 0x07) << 18) | (c1 << 12) | (c2 << 6) | pc
            return f"{lb}+{c1b}+{c2b}+{pb}={cp}", make_script_str(cp)
    return "?", "?"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lang_code",  help="e.g. eng_Latn, hin_Deva, khm_Khmr")
    parser.add_argument("index",      type=int, help="Sentence index (0-based)")
    parser.add_argument("--top_k",    type=int, default=1,
                        help="Top-k next-byte predictions to show (default: 1)")
    parser.add_argument("--out_dir",  default=None,
                        help="Output directory (default: inspect_results/inspect_{lang_code}_{index}/)")
    parser.add_argument("--restructured_dir", default=RESTRUCTURED,
                        help=f"Path to restructured results (default: {RESTRUCTURED})")
    args = parser.parse_args()

    lang_code = args.lang_code
    idx       = args.index
    top_k     = max(1, min(args.top_k, 10))
    out_dir   = args.out_dir or f"inspect_results/inspect_{lang_code}_{idx}"
    os.makedirs(out_dir, exist_ok=True)

    # ── load restructured sentence ────────────────────────────────────────────
    json_path = Path(args.restructured_dir) / f"{lang_code}.json"
    if not json_path.exists():
        sys.exit(f"Error: {json_path} not found. "
                 f"Run restructure_eval.py / run_patching.py first.")

    with open(json_path, encoding="utf-8") as f:
        sentences = json.load(f)

    if idx < 0 or idx >= len(sentences):
        sys.exit(f"Error: index {idx} out of range (0–{len(sentences) - 1}).")

    sentence = sentences[idx]
    text     = sentence["text"]
    text_en  = sentence.get("text_en", "")
    print(f"\nLanguage : {lang_code}")
    print(f"Index    : {idx}")
    print(f"Text     : {text}")
    if text_en and text_en != text:
        print(f"English  : {text_en}")

    # ── load patcher + re-run entropy model for predictions ──────────────────
    print("\nLoading patcher...")
    tokenizer, patcher = load_patcher(repo=REPO, entropy_repo=ENTROPY_REPO)
    offset = tokenizer.offsetting_special_char

    result        = patch_text(text, tokenizer, patcher)
    context_bytes = result["text_bytes"]
    scores        = result["scores"]    # fresh from model (should match stored)
    preds         = result["preds"]     # logits for top-k display
    char_map      = build_char_map(context_bytes)

    # ── A) visualisations ─────────────────────────────────────────────────────
    # Iterate every mode × threshold stored in the restructured JSON
    eval_modes = sentence.get("eval_modes", {})

    if not eval_modes:
        print("Warning: no eval_modes found in restructured JSON for this sentence.")
    else:
        print(f"\nGenerating visualisations → {out_dir}/")

    for mode_name, thresholds_dict in eval_modes.items():
        for t_key, mode_data in thresholds_dict.items():
            # t_key is like "t_1.3340"
            threshold = float(t_key[2:])  # strip leading "t_"
            patch_lengths = mode_data["patch_lengths"]

            # reconstruct patches list for the visualiser
            patches = []
            cursor  = 0
            for length in patch_lengths:
                if length == 0:
                    break
                patches.append((context_bytes[cursor:cursor + length], length))
                cursor += length

            # inside the mode × threshold loop, replace the scores= line:

            # get the right scores for this mode
            if "norm" in mode_name:
                viz_scores = [be[2] for be in sentence["bytes_entropies"]]  # entropy_norm
            else:
                viz_scores = [be[1] for be in sentence["bytes_entropies"]]  # entropy_raw

            viz = BLTPatchVisualizer()
            viz.add(
                text=text,
                patches=patches,
                scores=viz_scores,          # <── was just `scores` before
                label=f"{lang_code} [{idx}] — {mode_name} {t_key}",
                threshold=threshold,
            )
            fname = f"{lang_code}_{idx}_{mode_name}_{t_key}.html"
            viz.save(os.path.join(out_dir, fname))
            n_p = mode_data["n_patches"]
            bpp = mode_data["avg_bytes_per_patch"]
            print(f"  {fname}  ({n_p} patches, {bpp:.2f} bytes/patch)")

    # ── B) txt file with per-byte analysis ───────────────────────────────────
    lines = []
    lines.append("=" * 80)
    lines.append(f"  Language : {lang_code}")
    lines.append(f"  Index    : {idx}")
    lines.append(f"  Text     : {text}")
    if text_en and text_en != text:
        lines.append(f"  English  : {text_en}")
    lines.append(f"  Bytes    : {len(context_bytes)}")
    lines.append("")

    # Also show stored entropies alongside fresh ones for comparison
    stored_entropies = {i: be[1] for i, be in enumerate(sentence.get("bytes_entropies", []))}
    stored_norm      = {i: be[2] for i, be in enumerate(sentence.get("bytes_entropies", []))}

    for i, (byte_val, score, pred) in enumerate(zip(context_bytes, scores, preds)):
        off, total, lead = char_map[i]
        char_disp = format_char(byte_val)
        byte_bin  = format_byte_binary(byte_val)
        cur_bits, cur_script = format_bits(byte_val, off - 1, total, lead, context_bytes, i - 1)

        stored_e    = stored_entropies.get(i, float("nan"))
        stored_en   = stored_norm.get(i, float("nan"))

        lines.append(
            f"  [{i:4d}]  {char_disp:<6}  {byte_bin:<10}  "
            f"entropy={score:.3f}  "
            f"{cur_bits:<{W_BITS}}  {cur_script:<{W_SCRIPT}}"
        )

        probs = torch.softmax(torch.tensor(pred), dim=-1)
        topk  = probs.topk(min(top_k, 5))

        for pred_idx, p in zip(topk.indices, topk.values):
            b     = pred_idx.item() - offset
            pchar = format_char(b) if 0 <= b <= 255 else f"[{pred_idx.item()}]"
            pbin  = format_byte_binary(b) if 0 <= b <= 255 else "?"
            cp_bits, cp_script = format_bits(b, off, total, lead, context_bytes, i)
            lines.append(
                f"       →  {pchar:<6}  {pbin:<10}  prob=   {p.item():.3f}  "
                f"{cp_bits:<{W_BITS}}  {cp_script:<{W_SCRIPT}}"
            )

        # separator: solid line at end of a full codepoint, blank between bytes
        lines.append("  ---" if off == total - 1 else "")

    lines.append("")

    # also add a compact patch summary per mode/threshold at the end
    if eval_modes:
        lines.append("─" * 80)
        lines.append("  Patch summary from restructured results:")
        lines.append("")
        for mode_name, thresholds_dict in eval_modes.items():
            for t_key, mode_data in thresholds_dict.items():
                lines.append(
                    f"  {mode_name:<20} {t_key:<12}  "
                    f"n_patches={mode_data['n_patches']}  "
                    f"avg_bpp={mode_data['avg_bytes_per_patch']:.4f}  "
                    f"lengths={mode_data['patch_lengths']}"
                )
        lines.append("")

    txt_path = os.path.join(out_dir, f"{lang_code}_{idx}_analysis.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nText analysis → {txt_path}")
    print("Done.")


if __name__ == "__main__":
    main()