"""
blt_patcher.py
Core BLT patching logic, decoupled from I/O.

Space: A patch is created at every whitespace byte.

Static: A patch is created every N bytes (e.g. N=4).

Byte: A patch is created at every byte (i.e. no patching).

Entropy:
    # Their "Entropy" mode in the paper
    1. monotonicity=False + threshold_add is None
    → entropies > threshold (plain global threshold)
    # New patch if current_entropy > threshold

    2. monotonicity=True 
    → patch_start_mask_from_entropy_with_monotonicity(entropies, threshold)
    # New patch if current_entropy - previous_entropy > threshold

    # Presumably their "Entropy + Monotonicity" in the paper
    3. monotonicity=False + threshold_add is not None 
    → patch_start_mask_global_and_monotonicity(entropies, threshold, threshold_add)
    # New patch if current_entropy > t                   (global threshold)
    #          AND H(t) - H(t-1) > t_add                 (locally rising)
    #          AND previous byte was NOT a boundary      (no consecutive boundaries)
"""

import json
import torch
from bytelatent.hf import BltTokenizerAndPatcher
from bytelatent.data.patcher import PatchingModeEnum


PATCHING_MODES = {
    "entropy":     ("entropy",     True),
    "static":      ("static",      False),
    "space":       ("space",       False),
    "byte":        ("byte",        False),
}

NEEDS_BOS = {PatchingModeEnum.entropy}


def load_patcher(
    repo: str = "facebook/blt-1b",
    entropy_repo: str = "hf-weights/entropy_model",
    patching_mode: str = "entropy",
    threshold: float = 1.75,
    threshold_add: float = None,
    monotonicity: bool = False,
    patch_size: int = 4,
    custom_encoding_path: str | None = None,
):
    """Load and configure BLT tokenizer and patcher.

    custom_encoding_path: if given, patch_text() will encode input text
    using this custom per-character byte mapping (see
    training_setup/build_custom_encoding.py / bytelatent_tokenizer.py's
    custom_encoding_path support) instead of plain UTF-8 -- MUST match
    whatever the entropy model at entropy_repo was actually trained with,
    or entropy scores (and therefore patch boundaries) will be
    meaningless. The returned tokenizer is unused for encoding either way
    (only its offsetting_special_char/bos_id are read) -- the custom
    encoding, if any, is loaded here and returned as a third value for
    patch_text() to use directly.
    """
    tok_and_patcher = BltTokenizerAndPatcher.from_pretrained(repo)

    tok_and_patcher.patcher_args.entropy_model_checkpoint_dir = entropy_repo
    mode_str, realtime = PATCHING_MODES[patching_mode]
    tok_and_patcher.patcher_args.patching_mode = mode_str
    tok_and_patcher.patcher_args.realtime_patching = realtime
    tok_and_patcher.patcher_args.threshold = threshold
    tok_and_patcher.patcher_args.threshold_add = threshold_add
    tok_and_patcher.patcher_args.monotonicity = monotonicity
    tok_and_patcher.patcher_args.patch_size = patch_size

    tokenizer = tok_and_patcher.tokenizer_args.build()
    patcher = tok_and_patcher.patcher_args.build()

    custom_encoding = None
    if custom_encoding_path is not None:
        with open(custom_encoding_path, encoding="utf-8") as f:
            custom_encoding = json.load(f)

    return tokenizer, patcher, custom_encoding


def _text_to_raw_bytes(text: str, custom_encoding: dict | None) -> bytes:
    """Encodes text to raw bytes using the custom per-character mapping if
    custom_encoding is given, otherwise plain UTF-8. Must match whatever
    encoding the entropy model being used was actually trained with (see
    bytelatent/tokenizers/blt_tokenizer.py's _custom_encode_to_bytes,
    which this mirrors) -- a mismatch silently produces meaningless
    entropy scores and patch boundaries."""
    if custom_encoding is None:
        return text.encode("utf-8")
    char_to_bytes_hex = custom_encoding["char_to_bytes_hex"]
    raw = bytearray()
    for ch in text:
        hex_key = char_to_bytes_hex.get(ch)
        if hex_key is None:
            raise ValueError(
                f"Character {ch!r} (U+{ord(ch):04X}) has no entry in the custom "
                f"encoding -- rebuild it with build_custom_encoding.py to include "
                f"this character."
            )
        raw.extend(bytes.fromhex(hex_key))
    return bytes(raw)


def _byte_to_char_map(text: str, custom_encoding: dict | None) -> dict:
    """Build a mapping from byte offset → character index, for either
    plain UTF-8 (variable bytes/char) or a custom encoding (fixed
    bytes_per_char)."""
    byte_to_char = {}
    char_idx = 0
    byte_idx = 0
    if custom_encoding is None:
        for char in text:
            char_byte_len = len(char.encode("utf-8"))
            for b in range(char_byte_len):
                byte_to_char[byte_idx + b] = char_idx
            byte_idx += char_byte_len
            char_idx += 1
    else:
        bytes_per_char = custom_encoding["bytes_per_char"]
        for char in text:
            for b in range(bytes_per_char):
                byte_to_char[byte_idx + b] = char_idx
            byte_idx += bytes_per_char
            char_idx += 1
    # Sentinel: end of string
    byte_to_char[byte_idx] = char_idx
    return byte_to_char


def patch_text(text: str, tokenizer, patcher, device: str = "cuda",
                custom_encoding: dict | None = None) -> dict:
    """
    Patch a single text string.

    custom_encoding: the parsed custom_encoding.json (from load_patcher's
    third return value), or None for plain UTF-8. Must match whatever the
    entropy model was trained with.

    Returns a dict with:
        - patches:             list of (chunk_str, chunk_bytes, byte_length) tuples
                               chunk_str is sliced at proper character boundaries
        - scores:              list of per-byte entropy scores (or None)
        - n_patches:           int
        - n_bytes:             int
        - text_bytes:          list of ints (raw bytes of the input text,
                                under whichever encoding was used)
        - avg_bytes_per_patch: float
        - threshold:           float
    """
    offset = tokenizer.offsetting_special_char
    bos_id = tokenizer.bos_id

    byte_seq = _text_to_raw_bytes(text, custom_encoding)
    ids = [b + offset for b in byte_seq]
    tokens = torch.tensor([ids], dtype=torch.long, device=device)

    if patcher.patching_mode in NEEDS_BOS:
        bos = torch.tensor([[bos_id]], dtype=torch.long, device=tokens.device)
        tokens_input = torch.cat([bos, tokens], dim=1)
    else:
        tokens_input = tokens

    patch_lengths, scores, preds = patcher.patch(tokens_input)

    if patcher.patching_mode in NEEDS_BOS:
        patch_lengths = patch_lengths[:, 1:]
        if scores is not None:
            scores = scores[:, 1:]
        if preds is not None:
            preds = preds[:, 1:]

    # Build byte→char map directly from text + encoding (no decode-and-verify
    # round trip needed under a custom encoding, unlike UTF-8's original
    # decode-back-from-bytes approach -- we already know exactly how many
    # bytes each character consumed).
    b2c = _byte_to_char_map(text, custom_encoding)
    bytes_list = list(byte_seq)

    # Build patch list as (chunk_str, chunk_bytes, byte_length) tuples
    # chunk_str is sliced at character boundaries to avoid broken characters
    patches = []
    byte_cursor = 0
    for length in patch_lengths[0].tolist():
        raw = bytes(bytes_list[byte_cursor: byte_cursor + length])
        start_char = b2c.get(byte_cursor, 0)
        end_char = b2c.get(byte_cursor + length, len(text))
        patches.append((list(raw), length))
        byte_cursor += length

    score_list = scores[0].tolist() if scores is not None else None
    pred_list = preds[0].cpu().tolist() if preds is not None else None
    n = len(patches)
    b = len(byte_seq)

    return {
        "patches": patches,
        "scores": score_list,
        "preds": pred_list,
        "n_patches": n,
        "n_bytes": b,
        "text_bytes": list(byte_seq),
        "avg_bytes_per_patch": b / max(n, 1),
        "threshold": patcher.threshold,
    }