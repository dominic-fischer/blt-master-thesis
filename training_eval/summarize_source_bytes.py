"""
summarize_source_bytes.py

Aggregates the per-rank source_bytes.rank*.jsonl logs written by
trace_source_bytes.py into one total-bytes-per-language table, and
cross-checks the grand total against metrics.jsonl's own n_bytes logging
as a sanity check that no data went uncounted (e.g. because the
fork/spawn assumption noted in trace_source_bytes.py's docstring doesn't
hold for this bytelatent installation).

Usage:
    python summarize_source_bytes.py <dump_dir>
"""
import argparse
import glob
import json
import os
from collections import defaultdict


def read_last_json_line(path: str) -> dict | None:
    """Counts are cumulative within a rank's log, so only the LAST valid
    line is needed -- earlier lines are just earlier snapshots of the
    same running totals."""
    last = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir")
    args = parser.parse_args()

    log_dir = os.path.join(args.dump_dir, "source_bytes")
    rank_files = sorted(glob.glob(os.path.join(log_dir, "source_bytes.rank*.jsonl")))
    if not rank_files:
        print(f"No source_bytes.rank*.jsonl files found under {log_dir} -- "
              f"was this run launched with --trace-source-bytes?")
        return

    total_bytes = defaultdict(int)
    total_draws = defaultdict(int)
    for path in rank_files:
        last = read_last_json_line(path)
        if last is None:
            print(f"  WARNING: {path} has no valid JSON lines, skipping.")
            continue
        for source, n in last.get("bytes_by_source", {}).items():
            total_bytes[source] += n
        for source, n in last.get("draws_by_source", {}).items():
            total_draws[source] += n

    if not total_bytes:
        print("No usable data found in any rank log -- nothing to summarize.")
        return

    grand_total = sum(total_bytes.values())
    print(f"({len(rank_files)} rank file(s) found: "
          f"{[os.path.basename(p) for p in rank_files]})\n")
    print(f"{'language':<12} {'bytes':>16} {'share':>8} {'draws':>10}")
    print("-" * 50)
    for source, n in sorted(total_bytes.items(), key=lambda kv: kv[1], reverse=True):
        share = n / grand_total if grand_total else 0
        print(f"{source:<12} {n:>16,} {share:>7.2%} {total_draws[source]:>10,}")
    print("-" * 50)
    print(f"{'TOTAL':<12} {grand_total:>16,}")

    # Cross-check against metrics.jsonl's own n_bytes logging -- this is
    # the direct empirical check for whether the trace captured
    # everything (see trace_source_bytes.py's docstring on the
    # fork/spawn assumption).
    metrics_path = os.path.join(args.dump_dir, "metrics.jsonl")
    metrics_total = 0
    found_any = False
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n = row.get("n_bytes/interval_across_gpus")
                if n is not None:
                    metrics_total += n
                    found_any = True

    if found_any:
        diff_pct = (abs(grand_total - metrics_total) / metrics_total * 100
                    if metrics_total else float("nan"))
        print(f"\nCross-check vs metrics.jsonl's summed n_bytes/interval_across_gpus: "
              f"{metrics_total:,}  (traced total differs by {diff_pct:.2f}%)")
        if diff_pct > 5:
            print("  WARNING: >5% mismatch -- some data may be going through a data-"
                  "loading path this trace didn't capture (see trace_source_bytes.py's "
                  "docstring on the fork/spawn assumption). Treat the per-language "
                  "breakdown above with caution until this is resolved.")
        else:
            print("  Close match -- the per-language breakdown above is very likely "
                  "capturing the full picture.")
    else:
        print(f"\n(no n_bytes rows found in {metrics_path} to cross-check against)")


if __name__ == "__main__":
    main()