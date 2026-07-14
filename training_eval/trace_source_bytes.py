"""
trace_source_bytes.py

Wraps bytelatent.train's real entrypoint to additionally log how many
bytes are actually drawn from each data.sources.<language> during
training -- without modifying any bytelatent source file, and without
changing training behavior otherwise.

WHY THIS HOOK POINT: SamplingIterator holds one COMPLETELY SEPARATE
SequenceIterator per source (source_to_iterator: dict[str, ...]) and only
ever draws from whichever source gets chosen each iteration -- packing/
buffering happens INSIDE each source's own SequenceIterator, never across
sources. So per-source attribution is fully resolved right here, before
anything downstream (MultiprocessIterator, PackingIterator, batch
collation in train.py) has a chance to mix sequences from different
sources together. Patching anywhere downstream of SamplingIterator would
require unpacking that mixing back out; patching here needs none of that.

WHAT THIS COUNTS: len(sequence.tokens) per yielded BltSequence, attributed
to whichever source it was drawn from -- at the byte-level tokenizer this
codebase uses (see eval_entropy_bpb.py's OFFSET/BOS_ID/EOS_ID), token
count IS byte count (including the BOS/EOS tokens per sequence -- matches
train.py's own train-time byte counting via batch.mask.sum(), so the two
totals are directly comparable; see the cross-check note below).

MULTI-RANK: torchrun runs one process per GPU (rank), and
args.data.build_from_rank(dp_rank, dp_degree) gives each rank its own
independent SamplingIterator over a DIFFERENT shard of the data -- so
this writes one log file PER RANK (tagged by torch.distributed.get_rank()
once distributed is initialized), not one combined file. Run
summarize_source_bytes.py <dump_dir> afterward to sum across all of them.

ASSUMPTION WORTH VERIFYING: if bytelatent's data loading additionally
uses its own internal multiprocessing (separate from torchrun's per-GPU
ranks) via Python's multiprocessing with a 'spawn' start method, this
patch -- applied only in the main process before those workers exist --
would NOT propagate into them (spawned workers re-import everything
fresh, unlike forked ones, which inherit the parent's already-patched
memory). This has NOT been confirmed against MultiprocessIterator's
actual implementation.
The cross-check in summarize_source_bytes.py tells you directly whether
this happened: summed per-source bytes should land close to
metrics.jsonl's own total n_bytes for the same run. If it's far off
(especially much smaller), some data is likely being drawn through a
worker path this patch never reached, and the per-language breakdown
below should be treated as incomplete until that's resolved.

Usage: launch_training.py --trace-source-bytes swaps this module in as
the torchrun entrypoint automatically (and sets SOURCE_BYTES_LOG_DIR);
you shouldn't normally need to invoke this file directly.
"""
import atexit
import json
import os
import time
from collections import defaultdict

import numpy as np

from bytelatent.data.iterators.sampling_iterator import SamplingIterator


def _get_rank() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def _get_log_path() -> str:
    log_dir = os.environ.get("SOURCE_BYTES_LOG_DIR")
    if not log_dir:
        raise SystemExit(
            "SOURCE_BYTES_LOG_DIR is not set -- trace_source_bytes.py must be "
            "launched via launch_training.py --trace-source-bytes, which sets it "
            "automatically."
        )
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"source_bytes.rank{_get_rank()}.jsonl")


def _patched_create_iter(self):
    """Reimplements SamplingIterator.create_iter's source-selection loop
    (same rng.choice logic, unchanged) with added per-source byte/draw
    counting and periodic append-only logging. Behaviorally identical to
    the original for training purposes -- yields the exact same sequence
    of BltSequence objects in the exact same order."""
    counts = defaultdict(int)
    draws = defaultdict(int)
    log_path = _get_log_path()
    flush_every = int(os.environ.get("SOURCE_BYTES_FLUSH_EVERY", "200"))
    since_flush = 0

    def flush():
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "time": time.time(),
                "bytes_by_source": dict(counts),
                "draws_by_source": dict(draws),
            }) + "\n")

    atexit.register(flush)  # best-effort final flush even on abrupt exit

    n_sources = len(self.source_to_weight)
    possible_sources = list(self.source_to_weight.keys())
    weights = np.array([self.source_to_weight[s] for s in possible_sources])
    source_to_python_iter = {
        source: self.source_to_iterator[source].create_iter()
        for source in possible_sources
    }
    while True:
        norm_weights = weights / weights.sum()
        source_choice = possible_sources[self.rng.choice(n_sources, p=norm_weights)]
        sequence = next(source_to_python_iter[source_choice])

        counts[source_choice] += len(sequence.tokens)
        draws[source_choice] += 1
        since_flush += 1
        if since_flush >= flush_every:
            flush()
            since_flush = 0

        yield sequence


SamplingIterator.create_iter = _patched_create_iter

# Defer to the real entrypoint, completely unmodified otherwise -- this
# import must come AFTER the patch above so anything bytelatent.train
# does at import time still sees the patched class.
from bytelatent.train import main as _bytelatent_main  # noqa: E402

if __name__ == "__main__":
    _bytelatent_main()