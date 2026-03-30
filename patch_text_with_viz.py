import os
import torch
from bytelatent.hf import BltTokenizerAndPatcher
from bytelatent.data.patcher import PatchingModeEnum
from blt_visualize import BLTPatchVisualizer

repo = "facebook/blt-1b"
tok_and_patcher = BltTokenizerAndPatcher.from_pretrained(repo)
entropy_repo = "hf-weights/entropy_model"

patching_modes = [
    ("entropy", True),
    ("bpe_patcher", True),
    ("static", False),
    ("space", False),
    ("byte", False),
]

tokenizer = tok_and_patcher.tokenizer_args.build()
tok_and_patcher.patcher_args.entropy_model_checkpoint_dir = entropy_repo
tok_and_patcher.patcher_args.patching_mode, tok_and_patcher.patcher_args.realtime_patching = patching_modes[0]
tok_and_patcher.patcher_args.monotonicity = False
tok_and_patcher.patcher_args.threshold = 1.75

if tok_and_patcher.patcher_args.patching_mode == "static":
    tok_and_patcher.patcher_args.patch_size = 4

patcher = tok_and_patcher.patcher_args.build()

OFFSET  = tokenizer.offsetting_special_char  # 4
BOS_ID  = tokenizer.bos_id                   # 1

print(f"Patching mode: {tok_and_patcher.patcher_args.patching_mode}")
print(f"BOS_ID: {BOS_ID}, OFFSET: {OFFSET}")
print(f"Threshold: {patcher.threshold}")
print()

NEEDS_BOS = {PatchingModeEnum.entropy, PatchingModeEnum.bpe_patcher}
NEEDS_CPU = {PatchingModeEnum.bpe_patcher}

# ── shared visualizer ─────────────────────────────────────────────────────────
viz = BLTPatchVisualizer()

def text_to_tokens(text, device="cuda"):
    byte_seq = text.encode("utf-8")
    ids = [b + OFFSET for b in byte_seq]
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    return tokens, byte_seq


def patch_text(text, label=""):
    tokens, byte_seq = text_to_tokens(text)

    if patcher.patching_mode in NEEDS_BOS:
        bos = torch.tensor([[BOS_ID]], dtype=torch.long, device=tokens.device)
        tokens_input = torch.cat([bos, tokens], dim=1)
    else:
        tokens_input = tokens

    if patcher.patching_mode in NEEDS_CPU:
        tokens_input = tokens_input.cpu()

    patch_lengths, scores = patcher.patch(tokens_input)

    if patcher.patching_mode in NEEDS_BOS:
        patch_lengths = patch_lengths[:, 1:]
        if scores is not None:
            scores = scores[:, 1:]

    bytes_list = list(byte_seq)
    patches = []
    idx = 0
    for length in patch_lengths[0].tolist():
        chunk = bytes(bytes_list[idx : idx + length]).decode("utf-8", errors="replace")
        patches.append((chunk, length))
        idx += length

    # ── console output (unchanged) ────────────────────────────────────────────
    raw = scores[0].tolist()

    print(f"\n{'='*60}")
    print(f"TEXT: {text!r}  [{label}]")
    print(f"{'='*60}")
    byte_cursor = 0
    for i, (p, l) in enumerate(patches):
        print(f"  {i:02d}: {p!r:20s}  len={l}")
        for j in range(l):
            score_idx = byte_cursor + j
            if scores is not None and score_idx < len(raw):
                char = p[j] if j < len(p) else '·'
                char_display = '_' if char == ' ' else repr(char)
                print(f"        byte {score_idx:03d}: {char_display:6s}  entropy={raw[score_idx]:.4f}")
        byte_cursor += l

    n = len(patches)
    b = len(byte_seq)
    print(f"  → {n} patches for {b} bytes  (avg {b/max(n,1):.2f} bytes/patch)")

    # ── register with visualizer ──────────────────────────────────────────────

    score_list = raw if raw else None
    viz.add(
        text=text,
        patches=patches,
        scores=score_list,
        label=label,
        threshold=patcher.threshold,
    )


# ── run all samples ───────────────────────────────────────────────────────────
patch_text("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",  label="repetitive_sanity_check")
patch_text("I have potato blood in my veins. My life is potato. In your working life and in your living, it's always potatoes. I have potato blood in my veins. My life is potato. In your working life and in your living, it's always potatoes.",  label="repetitive_sanity_check_2")
patch_text("The quick brown fox jumps over the lazy dog.",  label="common_english")
patch_text("Daenerys Targaryen is in Game of Thrones, a fantasy epic by George R.R. Martin.", label="proper_nouns")
patch_text("x7Kq#mP2$nL9@wR4!vB6&jF0^hD3*cN8%",  label="high_entropy_garbage")
patch_text(
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    label="code",
)

print("\nDone.")

# ── write HTML visualisation ──────────────────────────────────────────────────
viz.save("blt_output.html")
print("Open blt_output.html in your browser to explore the results.")