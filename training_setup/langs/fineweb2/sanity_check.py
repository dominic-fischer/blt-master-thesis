"""
Sanity check: stream the FULL Nyanja (nya_Latn) train split of FineWeb2
ourselves and sum the raw UTF-8 byte length of the `text` column,
then compare against the official figure from
fineweb2-language-distribution.csv (nya_Latn / train).

Nyanja's train split is small (~103K docs, ~418MB), so streaming the whole
thing is fast and cheap -- good candidate for a full, exact sanity check
(as opposed to a big language where we'd only sample).

Requires: pip install datasets
"""

from datasets import load_dataset

DATASET = "HuggingFaceFW/fineweb-2"
CONFIG = "nya_Latn"
SPLIT = "train"

# From fineweb2-language-distribution.csv (nya_Latn, split=train)
EXPECTED_BYTES = 417_963_769
EXPECTED_DOCS = 103_045


def main():
    print(f"Streaming {DATASET} / config={CONFIG} / split={SPLIT} ...")
    ds = load_dataset(DATASET, name=CONFIG, split=SPLIT, streaming=True)

    total_bytes = 0
    total_docs = 0

    for doc in ds:
        total_bytes += len(doc["text"].encode("utf-8"))
        total_docs += 1
        if total_docs % 10_000 == 0:
            print(f"  ...{total_docs:,} docs, {total_bytes:,} bytes so far")

    print()
    print("=" * 60)
    print(f"Documents streamed:  {total_docs:,}   (expected: {EXPECTED_DOCS:,})")
    print(f"Measured utf8 bytes: {total_bytes:,}")
    print(f"Expected utf8 bytes: {EXPECTED_BYTES:,}")

    diff = total_bytes - EXPECTED_BYTES
    pct = (diff / EXPECTED_BYTES * 100) if EXPECTED_BYTES else float("nan")
    print(f"Difference:          {diff:,} bytes ({pct:+.4f}%)")

    if total_bytes == EXPECTED_BYTES:
        print("EXACT MATCH.")
    elif abs(pct) < 0.01:
        print("Match within rounding/measurement noise.")
    else:
        print("Mismatch larger than expected -- worth investigating "
              "(e.g. does len(text.encode('utf-8')) include a trailing "
              "newline or normalization difference vs. how the CSV was computed?).")


if __name__ == "__main__":
    main()