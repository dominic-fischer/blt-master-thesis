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

MULTI-RANK / MULTI-PROCESS: torchrun runs one process per GPU (rank), and
args.data.build_from_rank(dp_rank, dp_degree) gives each rank its own
independent SamplingIterator over a DIFFERENT shard of the data. A real
run confirmed that SamplingIterator.create_iter() actually executes
inside MultiprocessIterator's own internal data-loading worker
process(es), not directly in the main per-GPU-rank torchrun process --
evidenced by multiple concurrent, independently-growing byte counts
showing up interleaved in a single log file when this used
torch.distributed.get_rank() for naming (that call was silently failing
in every worker and falling back to rank 0, so all workers collided on
one file). Log files are now named by os.getpid() instead, which has no
such dependency and is always correct regardless of where the process
sits relative to torch.distributed. Run summarize_source_bytes.py
<dump_dir> afterward to sum across all of them.

CONFIRMED (previously an open assumption): SamplingIterator.create_iter()
runs inside a data-loading worker process distinct from the main
per-GPU-rank process, not in that main process itself. Naming log files
by os.getpid() (see _get_log_path) sidesteps this cleanly, since PIDs
don't depend on torch.distributed being valid in whichever process
happens to run the patched code. As a residual sanity check,
summarize_source_bytes.py's cross-check against metrics.jsonl's own
n_bytes logging still tells you directly whether the traced total is
capturing everything -- keep an eye on it after any change to this file
or to bytelatent's data-loading internals.

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
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from bytelatent.data.iterators.sampling_iterator import SamplingIterator


def _get_log_path() -> str:
    """Uses the OS process ID, not torch.distributed.get_rank(), to name
    each log file. This was originally rank-based, but real runs showed
    multiple concurrent series interleaved into a single 'rank0' file --
    strong evidence that SamplingIterator.create_iter() actually executes
    inside MultiprocessIterator's own data-loading worker process(es),
    not the main per-GPU-rank torchrun process itself. torch.distributed's
    process-group state (especially NCCL-backed) generally isn't valid in
    a forked/spawned worker, so dist.get_rank() was silently throwing and
    falling back to 0 in every worker -- causing them all to collide on
    the same file. os.getpid() has no such dependency: it's always
    correct and unique per real OS process regardless of where that
    process sits relative to torch.distributed or how it was created."""
    log_dir = os.environ.get("SOURCE_BYTES_LOG_DIR")
    if not log_dir:
        raise SystemExit(
            "SOURCE_BYTES_LOG_DIR is not set -- trace_source_bytes.py must be "
            "launched via launch_training.py --trace-source-bytes, which sets it "
            "automatically."
        )
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"source_bytes.pid{os.getpid()}.jsonl")


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