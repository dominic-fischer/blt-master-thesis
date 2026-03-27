
import os

from bytelatent.transformer import LMTransformer
from bytelatent.model.blt import ByteLatentTransformer
from bytelatent.hf import BltTokenizerAndPatcher

# Load tokenizer and patcher without model
repo = "facebook/blt-1b"  # Adjust if needed; assumes HF access
tok_and_patcher = BltTokenizerAndPatcher.from_pretrained(repo)

entropy_repo = "hf-weights/entropy_model"

# Get tokenizer and patcher
tokenizer = tok_and_patcher.tokenizer_args.build()

tok_and_patcher.patcher_args.entropy_model_checkpoint_dir = entropy_repo
tok_and_patcher.patcher_args.realtime_patching = True
patcher = tok_and_patcher.patcher_args.build()

print("Patching mode:", patcher.patcher_args.patching_mode)
print("Realtime:", patcher.patcher_args.realtime_patching)
print("Entropy model:", patcher.entropy_model)

# Test with a simple byte sequence
text = "A BLT is a bacon, lettuce and tomato sandwich."
byte_seq = text.encode('utf-8')
print(f"Input bytes: {byte_seq}")

# Convert bytes to torch tensor first (expects 1D uint8 tensor)
import torch
tokens = torch.tensor(list(byte_seq), dtype=torch.long).unsqueeze(0)

# Apply patcher
patch_lengths, scores = patcher.patch(tokens)

# --- Decode patches into readable chunks ---
bytes_list = list(byte_seq)

patches = []
idx = 0

for length in patch_lengths[0].tolist():
    patch_bytes = bytes_list[idx:idx + length]
    patch_text = bytes(patch_bytes).decode("utf-8", errors="replace")
    patches.append(patch_text)
    idx += length

print("\n--- PATCHES ---")
for i, p in enumerate(patches):
    print(f"{i:02d}: '{p}'")

# Optional: show scores alongside
if scores is not None:
    print("\n--- PATCH SCORES ---")
    for i, (p, s) in enumerate(zip(patches, scores[0].tolist())):
        print(f"{i:02d}: '{p}'  score={s:.4f}")

print(f"Patch lengths shape: {patch_lengths.shape}")
print(f"Patch lengths: {patch_lengths}")

if scores is not None:
    print(f"Scores shape: {scores.shape}")

print("Patcher test completed successfully!")