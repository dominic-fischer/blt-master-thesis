"""
summarize_source_bytes.py

Aggregates the per-process source_bytes.pid*.jsonl logs written by
trace_source_bytes.py into one total-bytes-per-language table, and
cross-checks the grand total against metrics.jsonl's own n_bytes logging
as a sanity check that no data went uncounted.

RATIO-VS-ENGLISH COMPARISON: also computes each language's OBSERVED byte
ratio relative to eng_Latn (total_bytes[source] / total_bytes[eng_Latn])
and compares it against the EXPECTED ratio_vs_english column in
--langs-csv (default training_setup/langs/langs_chosen.csv) -- the same
ratio that determined data.sources' sampling weights in the first place
(see launch_training.py / prepare_language_shards.py). Sorting by raw
bytes descending is already the same order as sorting by observed ratio
descending, since eng_Latn's byte count is a constant positive
denominator shared by every row -- so this is folded into the same table
rather than a second one. Rows whose observed ratio deviates from the
expected one by more than --deviation-tolerance-pct (default 20%) get an
explicit "<-- deviates" flag, since that's a sign the actual sampling
mix drifted from what data.sources was configured to produce (could be
normal short-run sampling noise, or worth a closer look if it's large
and persists over a longer run).

--legacy MODE: for logs produced by the OLD, pre-pid-fix version of
trace_source_bytes.py, which named every file "source_bytes.rank0.jsonl"
regardless of which actual worker process wrote it -- meaning multiple
independent, concurrently-running counters all appended to ONE file,
interleaved. This mode attempts a best-effort reconstruction: since a
single real worker's per-source byte counts can only ever increase, never
decrease, each line is greedily matched to whichever currently-tracked
worker state it's consistent with (every source's count in that line is
>= that worker's last known value for that source, checked across ALL
~20 sources simultaneously, not just one -- a much stronger signal than
any single source alone). If no tracked worker fits, a new one is
started. Ties are broken toward the worker requiring the smallest total
increase (most likely direct continuation).

This is a HEURISTIC, not a guarantee -- early in a run, before workers
have drawn enough to diverge, two different workers can legitimately
look consistent with the same prior state, and the tie-break is a best
guess rather than a certainty. Treat --legacy output as exploratory, and
lean on the metrics.jsonl cross-check below to gauge how much to trust
it. Prefer a clean pid-based rerun whenever that's an option.

Usage:
    python summarize_source_bytes.py <dump_dir>
    python summarize_source_bytes.py <dump_dir> --legacy
    python summarize_source_bytes.py <dump_dir> --deviation-tolerance-pct 15
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

DEFAULT_LANGS_CSV = "training_setup/langs/langs_chosen.csv"
DEFAULT_DEVIATION_TOLERANCE_PCT = 20.0


def read_last_json_line(path: str) -> dict | None:
    """Counts are cumulative within a process's log, so only the LAST
    valid line is needed -- earlier lines are just earlier snapshots of
    the same running totals."""
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


def read_all_json_lines(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def disentangle_interleaved_workers(rows: list[dict]) -> list[dict]:
    """Best-effort reconstruction of N independently-growing workers'
    final states from one file where their lines got interleaved (see
    module docstring's --legacy explanation). Returns a list of final
    {"bytes_by_source": ..., "draws_by_source": ...} states, one per
    detected worker."""
    workers: list[dict] = []  # each: {"bytes_by_source": {...}, "draws_by_source": {...}}

    def is_consistent(worker: dict, row: dict) -> bool:
        for source, n in row.get("bytes_by_source", {}).items():
            if n < worker["bytes_by_source"].get(source, 0):
                return False
        return True

    def total_delta(worker: dict, row: dict) -> int:
        delta = 0
        for source, n in row.get("bytes_by_source", {}).items():
            delta += n - worker["bytes_by_source"].get(source, 0)
        return delta

    for row in rows:
        candidates = [w for w in workers if is_consistent(w, row)]
        if candidates:
            best = min(candidates, key=lambda w: total_delta(w, row))
            best["bytes_by_source"] = dict(row.get("bytes_by_source", {}))
            best["draws_by_source"] = dict(row.get("draws_by_source", {}))
        else:
            workers.append({
                "bytes_by_source": dict(row.get("bytes_by_source", {})),
                "draws_by_source": dict(row.get("draws_by_source", {})),
            })

    return workers


def load_expected_ratios(langs_csv: str) -> dict[str, float]:
    """Returns {language_code: ratio_vs_english} from langs_csv -- the
    same column that determined data.sources' sampling weights (see
    prepare_language_shards.py). Returns {} if the file doesn't exist or
    has no usable rows, rather than raising -- the ratio comparison is a
    nice-to-have, not something that should crash the whole summary if
    the CSV path is wrong."""
    if not os.path.exists(langs_csv):
        return {}
    ratios = {}
    with open(langs_csv, newline="") as f:
        for row in csv.DictReader(f):
            code = row.get("language_code")
            raw = row.get("ratio_vs_english")
            if not code or raw in (None, "", "n/a"):
                continue
            try:
                ratios[code] = float(raw)
            except ValueError:
                continue
    return ratios


def summarize_and_print(total_bytes: dict, total_draws: dict, dump_dir: str, source_desc: str,
                         langs_csv: str, deviation_tolerance_pct: float) -> None:
    """Shared reporting logic: prints the per-language table (bytes,
    share, draws, and observed-vs-expected ratio-to-English), then runs
    the metrics.jsonl cross-check. Used by both the normal (clean
    pid-file) path and the --legacy reconstruction path."""
    if not total_bytes:
        print("No usable data found -- nothing to summarize.")
        return

    grand_total = sum(total_bytes.values())
    print(f"({source_desc})\n")

    eng_bytes = total_bytes.get("eng_Latn")
    expected_ratios = load_expected_ratios(langs_csv) if eng_bytes else {}
    show_ratio_cols = bool(eng_bytes)

    if eng_bytes and not expected_ratios:
        print(f"  NOTE: could not load expected ratios from {langs_csv!r} -- ratio "
              f"comparison columns will be blank.\n")
    if not eng_bytes:
        print("  NOTE: eng_Latn not found in traced data -- skipping ratio-vs-English "
              "comparison.\n")

    if show_ratio_cols:
        header = (f"{'language':<12} {'bytes':>16} {'share':>8} {'draws':>10} "
                  f"{'ratio_obs':>10} {'ratio_exp':>10} {'dev%':>8}")
    else:
        header = f"{'language':<12} {'bytes':>16} {'share':>8} {'draws':>10}"
    print(header)
    print("-" * len(header))

    for source, n in sorted(total_bytes.items(), key=lambda kv: kv[1], reverse=True):
        share = n / grand_total if grand_total else 0
        draws = total_draws.get(source, 0)
        row = f"{source:<12} {n:>16,} {share:>7.2%} {draws:>10,}"

        if show_ratio_cols:
            ratio_obs = n / eng_bytes
            ratio_exp = expected_ratios.get(source)
            if ratio_exp:
                dev_pct = (ratio_obs - ratio_exp) / ratio_exp * 100
                row += f" {ratio_obs:>10.3f} {ratio_exp:>10.3f} {dev_pct:>+7.1f}%"
                if abs(dev_pct) > deviation_tolerance_pct:
                    row += f"  <-- deviates > {deviation_tolerance_pct:.0f}%"
            else:
                row += f" {ratio_obs:>10.3f} {'n/a':>10} {'n/a':>8}"

        print(row)

    print("-" * len(header))
    print(f"{'TOTAL':<12} {grand_total:>16,}")

    # Cross-check against metrics.jsonl's own n_bytes logging -- this is
    # the direct empirical check for whether the trace (or, in --legacy
    # mode, the reconstruction) captured everything.
    metrics_path = os.path.join(dump_dir, "metrics.jsonl")
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
              f"{metrics_total:,}  (differs by {diff_pct:.2f}%)")
        if diff_pct > 5:
            print("  WARNING: >5% mismatch -- treat the per-language breakdown above "
                  "with caution.")
        else:
            print("  Close match -- the per-language breakdown above is very likely "
                  "capturing the full picture.")
    else:
        print(f"\n(no n_bytes rows found in {metrics_path} to cross-check against)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump_dir")
    parser.add_argument("--legacy", action="store_true",
                         help="Reconstruct totals from an old-style source_bytes.rank*.jsonl "
                              "log where multiple workers' output got interleaved into one "
                              "file, via best-effort monotonic matching. See module "
                              "docstring -- treat results as exploratory, not authoritative.")
    parser.add_argument("--langs-csv", default=DEFAULT_LANGS_CSV,
                         help=f"CSV with each language's expected ratio_vs_english, for "
                              f"the observed-vs-expected comparison columns (default "
                              f"{DEFAULT_LANGS_CSV}).")
    parser.add_argument("--deviation-tolerance-pct", type=float, default=DEFAULT_DEVIATION_TOLERANCE_PCT,
                         help=f"Flag a language's row if its observed ratio-to-English "
                              f"deviates from the expected ratio_vs_english by more than "
                              f"this percent (default {DEFAULT_DEVIATION_TOLERANCE_PCT:.0f}%%).")
    args = parser.parse_args()

    log_dir = os.path.join("dumps", args.dump_dir, "source_bytes")

    if args.legacy:
        rank_files = sorted(glob.glob(os.path.join(log_dir, "source_bytes.rank*.jsonl")))
        if not rank_files:
            print(f"No source_bytes.rank*.jsonl files found under {log_dir} -- "
                  f"nothing to reconstruct in --legacy mode.")
            return

        total_bytes = defaultdict(int)
        total_draws = defaultdict(int)
        n_workers_total = 0
        for path in rank_files:
            rows = read_all_json_lines(path)
            if not rows:
                print(f"  WARNING: {path} has no valid JSON lines, skipping.")
                continue
            workers = disentangle_interleaved_workers(rows)
            n_workers_total += len(workers)
            print(f"  {os.path.basename(path)}: reconstructed {len(workers)} distinct "
                  f"worker(s) from {len(rows)} interleaved line(s).")
            for worker in workers:
                for source, n in worker["bytes_by_source"].items():
                    total_bytes[source] += n
                for source, n in worker["draws_by_source"].items():
                    total_draws[source] += n

        print()
        summarize_and_print(
            dict(total_bytes), dict(total_draws), args.dump_dir,
            f"--legacy reconstruction: {n_workers_total} worker(s) recovered from "
            f"{len(rank_files)} file(s) -- BEST-EFFORT, see module docstring",
            args.langs_csv, args.deviation_tolerance_pct,
        )
        return

    pid_files = sorted(glob.glob(os.path.join(log_dir, "source_bytes.pid*.jsonl")))
    if not pid_files:
        print(f"No source_bytes.pid*.jsonl files found under {log_dir} -- "
              f"was this run launched with --trace-source-bytes? (Or, if this is an "
              f"older run with source_bytes.rank*.jsonl files instead, try --legacy.)")
        return

    total_bytes = defaultdict(int)
    total_draws = defaultdict(int)
    for path in pid_files:
        last = read_last_json_line(path)
        if last is None:
            print(f"  WARNING: {path} has no valid JSON lines, skipping.")
            continue
        for source, n in last.get("bytes_by_source", {}).items():
            total_bytes[source] += n
        for source, n in last.get("draws_by_source", {}).items():
            total_draws[source] += n

    summarize_and_print(
        dict(total_bytes), dict(total_draws), args.dump_dir,
        f"{len(pid_files)} process file(s) found: "
        f"{[os.path.basename(p) for p in pid_files]}",
        args.langs_csv, args.deviation_tolerance_pct,
    )


if __name__ == "__main__":
    main()