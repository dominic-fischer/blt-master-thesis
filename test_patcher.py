import os
import torch
from bytelatent.hf import BltTokenizerAndPatcher

repo = "facebook/blt-1b"
tok_and_patcher = BltTokenizerAndPatcher.from_pretrained(repo)
entropy_repo = "hf-weights/entropy_model"

tokenizer = tok_and_patcher.tokenizer_args.build()
tok_and_patcher.patcher_args.entropy_model_checkpoint_dir = entropy_repo
tok_and_patcher.patcher_args.patching_mode = "entropy"
tok_and_patcher.patcher_args.realtime_patching = True
tok_and_patcher.patcher_args.monotonicity = False
patcher = tok_and_patcher.patcher_args.build()

OFFSET = tokenizer.offsetting_special_char  # 4
BOS_ID = tokenizer.bos_id                   # 1

print(f"BOS_ID: {BOS_ID}, OFFSET: {OFFSET}")
print(f"Threshold: {patcher.threshold}")
print()

def text_to_tokens(text, device='cuda'):
    byte_seq = text.encode('utf-8')
    ids = [b + OFFSET for b in byte_seq]
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    return tokens, byte_seq

def patch_text(text, label=""):
    tokens, byte_seq = text_to_tokens(text)

    bos = torch.tensor([[BOS_ID]], dtype=torch.long, device=tokens.device)
    tokens_input = torch.cat([bos, tokens], dim=1)

    patch_lengths, scores = patcher.patch(tokens_input)

    # drop the BOS patch
    patch_lengths = patch_lengths[:, 1:]
    if scores is not None:
        scores = scores[:, 1:]

    bytes_list = list(byte_seq)
    patches = []
    idx = 0
    for length in patch_lengths[0].tolist():
        chunk = bytes(bytes_list[idx:idx + length]).decode('utf-8', errors='replace')
        patches.append((chunk, length))
        idx += length

    print(f"\n{'='*60}")
    print(f"TEXT: {text!r}  [{label}]")
    print(f"{'='*60}")
    if scores is not None:
        for i, ((p, l), s) in enumerate(zip(patches, scores[0].tolist())):
            print(f"  {i:02d}: {p!r:20s}  len={l}  entropy={s:.4f}")
    else:
        for i, (p, l) in enumerate(patches):
            print(f"  {i:02d}: {p!r:20s}  len={l}")
    n = len(patches)
    b = len(byte_seq)
    print(f"  → {n} patches for {b} bytes  (avg {b/max(n,1):.2f} bytes/patch)")


patch_text("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", label="repetitive_sanity_check")
patch_text("The quick brown fox jumps over the lazy dog.", label="common_english")
patch_text("Daenerys Targaryen is in Game of Thrones, a fantasy epic by George R.R. Martin.", label="proper_nouns")
patch_text("x7Kq#mP2$nL9@wR4!vB6&jF0^hD3*cN8%", label="high_entropy_garbage")
patch_text("def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)", label="code")

print("\nDone.")