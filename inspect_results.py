"""
inspect_results.py

Given a lang_code (e.g. "eng_Latn") and a sentence index, reads the
pre-computed restructured results and:

  A) Saves HTML visualisations for every mode x threshold combination
     that exists in the restructured JSON, using BLTPatchVisualizer.
     This now covers BOTH granularities: byte-mode ("eval_modes",
     patch boundaries thresholded over per-byte scores, patch_lengths
     already in bytes) AND char-mode ("char_eval_modes", patch
     boundaries thresholded over per-CHARACTER summed scores --
     reconstructed against context_bytes using patch_lengths_bytes,
     since that's what run_patching.py's char-mode output already
     stores precisely so downstream tools like this one don't have to
     re-derive byte spans from character spans themselves). Char-mode
     HTML files are written to a "char_level" subfolder under out_dir
     (mirroring the results/*/char_level/ convention used elsewhere in
     this pipeline), so they never collide with the byte-mode files for
     the same mode/threshold, and the folder itself makes the
     granularity distinction clear without needing a filename prefix.
     Per-byte coloring (viz_scores) is the SAME bytes_entropies array
     either way -- only the patch BOUNDARIES differ between
     granularities, not the underlying per-byte scores being visualised.

  B) Saves a .txt file with per-byte entropy, binary breakdown, top-k
     next-byte predictions (re-run through the entropy model), plus a
     compact patch summary block covering both eval_modes and
     char_eval_modes.

MODEL SELECTION: --entropy_repo is the primary argument -- it's used to
reload the actual model and recompute fresh scores/predictions for
comparison against the stored ones. --restructured_dir (where the STORED
JSON lives) now defaults to:
    results/own_models/<model_name>/<step>/
which is auto-derived by parsing the --entropy_repo path structure.

Usage:
    python inspect_sentence.py amh_Ethi 0 --entropy_repo results/entropy_10M_20lang_4gpu_sourcesbalanced_steps6000_ckpt200_lr4.5e-3/checkpoints/0000006000/consolidated
"""

import argparse
import json
import os
import sys
import re
from os import path
import torch
from pathlib import Path

sys.path.append(path.join(path.dirname(path.abspath(__file__)), "model_eval"))
from results_paths import results_dir_for, derive_filename_stem

from blt_patcher import load_patcher, patch_text
from blt_visualize import BLTPatchVisualizer

# -- config --------------------------------------------------------------------
DEFAULT_REPO = "facebook/blt-1b"
DEFAULT_ENTROPY_REPO = "hf-weights/entropy_model"

W_BITS   = 33
W_SCRIPT = 50

# -- Auto-derive Restructured Directory Helper ----------------------------------
def derive_own_results_dir(entropy_repo_path: str) -> str:
    norm_path = os.path.normpath(entropy_repo_path)
    parts = norm_path.split(os.sep)
    
    # We look for the common training output pattern: <model_dir>/checkpoints/<step>/consolidated
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        if idx > 0 and idx + 1 < len(parts):
            # Extract the model directory name
            model_name = parts[idx - 1]
            
            # Extract and format the step directory (e.g., step_0000006000)
            raw_step = parts[idx + 1]
            step = raw_step if raw_step.startswith("step_") else f"step_{raw_step}"
            
            return os.path.join("results", "own_models", model_name, step)

    # Fallback pattern matching with regex
    match = re.search(r"([^/]+)/checkpoints/(\d+)", norm_path.replace("\\", "/"))
    if match:
        model_name, step = match.groups()
        return os.path.join("results", "own_models", model_name, f"step_{step}")

    print(f"Warning: Could not extract checkpoints/step pattern from '{entropy_repo_path}'.")
    return results_dir_for(entropy_repo_path)

# -- Unicode / byte helpers (unchanged from original) --------------------------
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
    (3328,  3583,  "Sinhala"),
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


def build_char_lengths_fn(custom_encoding):
    """
    Returns a function text -> List[int] giving the byte-length of each
    character in `text`, according to custom_encoding (or UTF-8 if None
    / character not found in the mapping).
    """
    if custom_encoding is None:
        return lambda text: [len(ch.encode("utf-8")) for ch in text]

    hex_to_char = custom_encoding["bytes_hex_to_char"]
    char_to_bytelen = {ch: len(hex_code) // 2 for hex_code, ch in hex_to_char.items()}

    def char_lengths(text):
        lengths = []
        for ch in text:
            if ch in char_to_bytelen:
                lengths.append(char_to_bytelen[ch])
            else:
                # fallback for any char outside the custom mapping
                lengths.append(len(ch.encode("utf-8")))
        return lengths

    return char_lengths


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


def reconstruct_patches(context_bytes, patch_lengths_in_bytes):
    """Given a list of patch lengths ALREADY IN BYTES (either
    eval_modes' native "patch_lengths", or char_eval_modes'
    "patch_lengths_bytes" -- both are byte-length-per-patch lists, just
    derived over different granularities), slices context_bytes into
    (patch_bytes, length) tuples for the visualiser. Shared by both the
    byte-mode and char-mode visualisation loops below."""
    patches = []
    cursor = 0
    for length in patch_lengths_in_bytes:
        if length == 0:
            break
        patches.append((context_bytes[cursor:cursor + length], length))
        cursor += length
    return patches


# -- main ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lang_code",  help="e.g. eng_Latn, hin_Deva, khm_Khmr")
    parser.add_argument("index",      type=int, help="Sentence index (0-based)")
    parser.add_argument("--entropy_repo", default=DEFAULT_ENTROPY_REPO,
                        help=f"Repo id (or path) of the entropy model to reload for "
                             f"fresh scores/predictions -- e.g. a dumps/.../checkpoints/"
                             f"<step>/consolidated path, or an HF repo id (default "
                             f"{DEFAULT_ENTROPY_REPO}). Also used to auto-derive "
                             f"--restructured_dir and the output subfolder, unless "
                             f"overridden explicitly.")
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help=f"Repo id of the full BLT model (default {DEFAULT_REPO}).")
    parser.add_argument("--top_k",    type=int, default=1,
                        help="Top-k next-byte predictions to show (default: 1)")
    parser.add_argument("--out_dir",  default=None,
                        help="Output directory. If omitted, derived as "
                             "inspect_results/<stem>/inspect_{lang_code}_{index}/, "
                             "where <stem> comes from --entropy_repo -- so different "
                             "models/checkpoints don't overwrite each other's output.")
    parser.add_argument("--custom_encoding_path", default=None,
                        help="Path to a custom per-character byte encoding JSON "
                             "(see training_setup/build_custom_encoding.py). MUST "
                             "match whatever the entropy model at --entropy_repo was "
                             "actually trained with, or entropy scores/patch "
                             "boundaries will be meaningless. If omitted, plain "
                             "UTF-8 is used.")
    parser.add_argument("--restructured_dir", default=None,
                        help="Path to restructured results. If omitted, derived from "
                             "--entropy_repo -- now auto-resolving to "
                             "results/own_models/<model_name>/<step>/ "
                             "if checkpoints exist in the path.")
    args = parser.parse_args()

    lang_code = args.lang_code
    idx       = args.index
    top_k     = max(1, min(args.top_k, 10))

    # >>> ADAPTED SECTION: Custom derivation mapping to results/own_models/
    restructured_dir = args.restructured_dir or derive_own_results_dir(args.entropy_repo)
    
    stem = derive_filename_stem(args.entropy_repo)
    out_dir = args.out_dir or os.path.join("inspect_results", stem, f"inspect_{lang_code}_{idx}")
    os.makedirs(out_dir, exist_ok=True)

    # -- load restructured sentence ---------------------------------------------
    json_path = Path(restructured_dir) / f"{lang_code}.json"
    if not json_path.exists():
        sys.exit(f"Error: {json_path} not found. "
                 f"Run restructure_eval.py / run_patching.py first, or check that "
                 f"--entropy_repo (or an explicit --restructured_dir) points at the "
                 f"right model.")

    with open(json_path, encoding="utf-8") as f:
        sentences = json.load(f)

    if idx < 0 or idx >= len(sentences):
        sys.exit(f"Error: index {idx} out of range (0-{len(sentences) - 1}).")

    sentence = sentences[idx]
    text     = sentence["text"]
    text_en  = sentence.get("text_en", "")
    print(f"\nLanguage : {lang_code}")
    print(f"Index    : {idx}")
    print(f"Restructured dir: {restructured_dir}")
    print(f"Entropy repo    : {args.entropy_repo}")
    print(f"Text     : {text}")
    if text_en and text_en != text:
        print(f"English  : {text_en}")

    # -- load patcher + re-run entropy model for predictions --------------------
    tokenizer, patcher, custom_encoding = load_patcher(
        repo=args.repo,
        entropy_repo=args.entropy_repo,
        custom_encoding_path=args.custom_encoding_path,
    )
    print(f"Entropy model device: {next(patcher.entropy_model.parameters()).device}")
    get_char_lengths = build_char_lengths_fn(custom_encoding)
    offset = tokenizer.offsetting_special_char

    # after
    result = patch_text(text, tokenizer, patcher, custom_encoding=custom_encoding)
    context_bytes = result["text_bytes"]
    scores        = result["scores"]    # fresh from model (should match stored)
    preds         = result["preds"]     # logits for top-k display
    char_map      = build_char_map(context_bytes)


    # -- A) visualisations --------------------------------------------------------
    # Iterate every mode x threshold stored in the restructured JSON, for
    # BOTH granularities: byte-mode (eval_modes) and char-mode
    # (char_eval_modes, if present -- e.g. only after run_patching.py has
    # been run with --score-source chars for this language/checkpoint).
    eval_modes      = sentence.get("eval_modes", {})
    char_eval_modes = sentence.get("char_eval_modes", {})


    print(f"len(context_bytes) = {len(context_bytes)}")
    print(f"len(stored bytes_entropies) = {len(sentence.get('bytes_entropies', []))}")
    for mode_name, thresholds_dict in eval_modes.items():
        for t_key, mode_data in thresholds_dict.items():
            total = sum(mode_data["patch_lengths"])
            print(f"  [bytes] {mode_name} {t_key}: sum(patch_lengths)={total}")
    if char_eval_modes:
        n_chars = len(sentence.get("chars_entropies", []))
        print(f"len(stored chars_entropies) = {n_chars}")
        for mode_name, thresholds_dict in char_eval_modes.items():
            for t_key, mode_data in thresholds_dict.items():
                total_chars = sum(mode_data["patch_lengths_chars"])
                total_bytes = sum(mode_data["patch_lengths_bytes"])
                print(f"  [chars] {mode_name} {t_key}: sum(patch_lengths_chars)={total_chars}  "
                      f"sum(patch_lengths_bytes)={total_bytes}")

    if not eval_modes and not char_eval_modes:
        print("Warning: no eval_modes or char_eval_modes found in restructured JSON for this sentence.")
    else:
        print(f"\nGenerating visualisations -> {out_dir}/")

    # -- byte-mode visualisations (unchanged behavior) --
    for mode_name, thresholds_dict in eval_modes.items():
        for t_key, mode_data in thresholds_dict.items():
            # t_key is like "t_1.3340"
            threshold = float(t_key[2:])  # strip leading "t_"
            patches = reconstruct_patches(context_bytes, mode_data["patch_lengths"])

            # get the right scores for this mode
            if "norm" in mode_name:
                viz_scores = [be[2] for be in sentence["bytes_entropies"]]  # entropy_norm
            else:
                viz_scores = [be[1] for be in sentence["bytes_entropies"]]  # entropy_raw

            viz = BLTPatchVisualizer()
            viz.add(
                text=text,
                patches=patches,
                scores=viz_scores,
                label=f"{lang_code} [{idx}] - {mode_name} {t_key}",
                threshold=threshold,
                char_lengths=get_char_lengths(text),
            )
            fname = f"{lang_code}_{idx}_{mode_name}_{t_key}.html"
            viz.save(os.path.join(out_dir, fname))
            n_p = mode_data["n_patches"]
            bpp = mode_data["avg_bytes_per_patch"]
            print(f"  {fname}  ({n_p} patches, {bpp:.2f} bytes/patch)")

    # -- char-mode visualisations (new) --
    # Patch BOUNDARIES were determined by thresholding per-character
    # SUMMED scores, so the chart should plot THAT quantity (one point
    # per character, at that character's center x-position) rather than
    # per-byte scores -- otherwise the chart just reproduces the same
    # byte-by-byte zigzag as byte-mode with different boundary lines
    # overlaid, which doesn't show what was actually thresholded. We pass
    # char_scores=[summed raw entropy per character] from chars_entropies
    # (index 1 -- see add_char_entropies.py); blt_visualize.py's
    # _build_combined_svg switches to character-granularity plotting
    # whenever char_scores is given, while the byte grid/table below is
    # unaffected. char_eval_modes only ever contains raw_entropy/
    # raw_monotonicity (no norm_entropy/combined equivalent -- see
    # calibrate_thresholds.py and run_patching.py), and chars_entropies
    # only ever stores the raw (not normalized) summed score, so there's
    # only ever one char_scores column to use here, unconditionally.
    # Written to a "char_level" subfolder under out_dir, mirroring the
    # results/*/char_level/ convention used elsewhere in this pipeline --
    # so no filename prefix is needed to disambiguate from byte-mode.
    for mode_name, thresholds_dict in char_eval_modes.items():
        for t_key, mode_data in thresholds_dict.items():
            threshold = float(t_key[2:])  # strip leading "t_"
            patches = reconstruct_patches(context_bytes, mode_data["patch_lengths_bytes"])

            char_scores = [ce[1] for ce in sentence["chars_entropies"]]  # summed raw entropy, per character

            viz = BLTPatchVisualizer()
            viz.add(
                text=text,
                patches=patches,
                char_scores=char_scores,
                label=f"{lang_code} [{idx}] - char:{mode_name} {t_key}",
                threshold=threshold,
                char_lengths=get_char_lengths(text),
            )
            char_out_dir = os.path.join(out_dir, "char_level")
            os.makedirs(char_out_dir, exist_ok=True)
            fname = f"{lang_code}_{idx}_{mode_name}_{t_key}.html"
            viz.save(os.path.join(char_out_dir, fname))
            n_p = mode_data["n_patches"]
            bpp = mode_data["avg_bytes_per_patch"]
            print(f"  char_level/{fname}  ({n_p} patches [char-level boundaries], {bpp:.2f} bytes/patch)")

    # -- B) txt file with per-byte analysis ----------------------------------------
    lines = []
    lines.append("=" * 80)
    lines.append(f"  Language : {lang_code}")
    lines.append(f"  Index    : {idx}")
    lines.append(f"  Entropy repo: {args.entropy_repo}")
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
                f"       ->  {pchar:<6}  {pbin:<10}  prob=   {p.item():.3f}  "
                f"{cp_bits:<{W_BITS}}  {cp_script:<{W_SCRIPT}}"
            )

        # separator: solid line at end of a full codepoint, blank between bytes
        lines.append("  ---" if off == total - 1 else "")

    lines.append("")

    # also add a compact patch summary per mode/threshold at the end,
    # for both byte-mode and char-mode
    if eval_modes or char_eval_modes:
        lines.append("-" * 80)
        lines.append("  Patch summary from restructured results:")
        lines.append("")
        for mode_name, thresholds_dict in eval_modes.items():
            for t_key, mode_data in thresholds_dict.items():
                lines.append(
                    f"  [bytes] {mode_name:<20} {t_key:<12}  "
                    f"n_patches={mode_data['n_patches']}  "
                    f"avg_bpp={mode_data['avg_bytes_per_patch']:.4f}  "
                    f"lengths={mode_data['patch_lengths']}"
                )
        for mode_name, thresholds_dict in char_eval_modes.items():
            for t_key, mode_data in thresholds_dict.items():
                lines.append(
                    f"  [chars] {mode_name:<20} {t_key:<12}  "
                    f"n_patches={mode_data['n_patches']}  "
                    f"avg_bpp={mode_data['avg_bytes_per_patch']:.4f}  "
                    f"lengths_chars={mode_data['patch_lengths_chars']}  "
                    f"lengths_bytes={mode_data['patch_lengths_bytes']}"
                )
        lines.append("")

    txt_path = os.path.join(out_dir, f"{lang_code}_{idx}_analysis.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nText analysis -> {txt_path}")
    print("Done.")


if __name__ == "__main__":
    main()