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
):
    """Load and configure BLT tokenizer and patcher."""
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

    return tokenizer, patcher


def _byte_to_char_map(text: str) -> dict:
    """Build a mapping from byte offset → character index for a UTF-8 string."""
    byte_to_char = {}
    char_idx = 0
    byte_idx = 0
    for char in text:
        char_byte_len = len(char.encode("utf-8"))
        for b in range(char_byte_len):
            byte_to_char[byte_idx + b] = char_idx
        byte_idx += char_byte_len
        char_idx += 1
    # Sentinel: end of string
    byte_to_char[byte_idx] = char_idx
    return byte_to_char


def patch_text(text: str, tokenizer, patcher, device: str = "cuda") -> dict:
    """
    Patch a single text string.

    Returns a dict with:
        - patches:             list of (chunk_str, chunk_bytes, byte_length) tuples
                               chunk_str is sliced at proper character boundaries
        - scores:              list of per-byte entropy scores (or None)
        - n_patches:           int
        - n_bytes:             int
        - text_bytes:          list of ints (raw UTF-8 bytes of the input text)
        - avg_bytes_per_patch: float
        - threshold:           float
    """
    offset = tokenizer.offsetting_special_char
    bos_id = tokenizer.bos_id

    byte_seq = text.encode("utf-8")
    ids = [b + offset for b in byte_seq]
    tokens = torch.tensor([ids], dtype=torch.long, device=device)

    if patcher.patching_mode in NEEDS_BOS:
        bos = torch.tensor([[bos_id]], dtype=torch.long, device=tokens.device)
        tokens_input = torch.cat([bos, tokens], dim=1)
    else:
        tokens_input = tokens

    patch_lengths, scores = patcher.patch(tokens_input)

    if patcher.patching_mode in NEEDS_BOS:
        patch_lengths = patch_lengths[:, 1:]
        if scores is not None:
            scores = scores[:, 1:]

    # Decode full sequence once (guaranteed valid) and build byte→char map
    full_text = byte_seq.decode("utf-8")
    b2c = _byte_to_char_map(full_text)
    bytes_list = list(byte_seq)

    # Build patch list as (chunk_str, chunk_bytes, byte_length) tuples
    # chunk_str is sliced at character boundaries to avoid broken characters
    patches = []
    byte_cursor = 0
    for length in patch_lengths[0].tolist():
        raw = bytes(bytes_list[byte_cursor: byte_cursor + length])
        start_char = b2c.get(byte_cursor, 0)
        end_char = b2c.get(byte_cursor + length, len(full_text))
        chunk = full_text[start_char:end_char]
        patches.append((chunk, list(raw), length))
        byte_cursor += length

    score_list = scores[0].tolist() if scores is not None else None
    n = len(patches)
    b = len(byte_seq)

    return {
        "patches": patches,
        "scores": score_list,
        "n_patches": n,
        "n_bytes": b,
        "text_bytes": list(byte_seq),
        "avg_bytes_per_patch": b / max(n, 1),
        "threshold": patcher.threshold,
    }