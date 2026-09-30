"""
run_cumulative_patching.py

"Constant information" patching over ALREADY-COMPUTED entropy scores --
no model forward pass, no recomputation. Reads the per-language JSON files
in --results-dir (bytes_entropies, and chars_entropies from
add_char_entropies.py), calibrates on English, patches every language,
and writes the results into the same eval_modes / char_eval_modes keys
run_patching.py uses, so results_to_CSV.py / results_to_JS.py /
results_to_txt_premiums.py can consume them (see INTEGRATION below).

BOUNDARY RULE (per unit i = one byte or one character, scores s_i = raw
entropy, per-char summed raw entropy in chars mode):

    cum   = sum of s over the units already in the current patch
    budget_reached(i)  :=  cum + s_i > T
    rising(i)          :=  s_i - s_{i-1} > m          (m = None -> always True)

    unit i STARTS a new patch  iff  current patch is non-empty
                                    AND budget_reached(i) AND rising(i)

  - m = None ("nomono"): pure constant-information patching. A patch holds
    at most T nats of (predicted) entropy; the unit that would overflow the
    budget opens the next patch. So, as in plain BLT, surprising units tend
    to START patches rather than end them.
  - m set: once the budget is reached, the cut is DEFERRED to the next unit
    where entropy rises by more than m. T acts as a minimum information
    content per patch, m decides WHERE the cut lands.
  - Limiting cases: at T=0 this is exactly raw_monotonicity with threshold
    m; at m=None it is pure cumulative. The family interpolates between the
    two. A larger m leaves fewer candidate cut points, so each m has a
    maximum achievable pps (reached at T=0); targets above it are reported
    as INFEASIBLE, not silently clamped.

MONOTONICITY VALUES (--mono): comma-separated, each one of
    none        no monotonicity condition
    <float>     absolute delta threshold, same at every bound, e.g. 0 or 0.25
    rel:<f>     f * m_star, where m_star is the delta threshold at which PURE
                monotonicity patching gives English a given pps target,
                calibrated here for THIS model and THIS score source.
Absolute values mean different things for bytes vs chars (summed char
scores have larger deltas) and across checkpoints, so rel: is the default.

--mono-ref per-bound (default): each bound uses the m_star of its OWN pps
target (m_star_low at 36 pps, m_star_anchor at 32.77, m_star_high at 23),
so f means the same thing at every operating point: f=0 cuts at any rise,
f=1 reproduces raw_monotonicity at that bound exactly (T collapses to 0),
and every f < 1 is feasible at every bound.
--mono-ref low: every bound uses m_star_low (the older behavior). Then f is
a relatively weaker constraint at coarser bounds, since m_star grows as
the pps target drops.

CALIBRATION mirrors calibrate_thresholds.py: for each case, binary-search T
so English hits --pps-low (t_low), --pps-high (t_high), --target-pps
(t_anchor), each with that bound's m. The midpoint is the midpoint in BOTH
parameters: t_mid = (t_low + t_high) / 2 and m_mid = f * (m_star_low +
m_star_high) / 2, mirroring the pipeline's threshold-midpoint convention;
its pps lands wherever it lands. Written to the summary CSV in the same
column format as calibrate_thresholds.py, plus m_low/m_mid/m_high/m_anchor.

CASE NAMES: cumulative_nomono, cumulative_mono_0, cumulative_mono_rel0p25,
cumulative_mono_0p5 ... (dots -> "p" so the _t_<value> suffix stays
parseable).

EXTRA DIAGNOSTICS (--diag-csv), per language x case x bound:
    pps, bpp, word_start_precision (share of cuts that land on a word
    start), word_start_recall (share of word starts that get a cut), and
    entropy_sum_premium (mean summed entropy per sentence / English's).
For pure cumulative patching, pps_premium should track entropy_sum_premium
closely -- that is the whole "constant information" prediction. Word-start
metrics are only meaningful for whitespace-delimited scripts.

INTEGRATION with the existing pipeline (three small edits, see the chat):
  1. run_patching.py: level-1 stale cleanup must not delete cases it
     doesn't own, and load_cases() must accept cumulative_* rows.
  2. results_to_txt_premiums.py: parse_premium_column must accept
     cumulative_* case names.
  3. results_to_CSV.py: pass --cases raw_entropy,raw_monotonicity,cumulative_nomono,...
     (or --cases all).

Usage (from repo root):
    python model_eval/run_cumulative_patching.py --results-dir <dir> --score-source chars
    python model_eval/run_cumulative_patching.py --results-dir <dir> --score-source both \\
        --mono none,0,rel:0.25,rel:0.5
    # every bound's m as a fraction of the LOW bound's m_star instead:
    python model_eval/run_cumulative_patching.py --results-dir <dir> --mono-ref low
    # custom 2-byte encoding checkpoint: double the pps targets as usual
    python model_eval/run_cumulative_patching.py --results-dir <dir> \\
        --target-pps 65.54 --pps-low 72 --pps-high 46
    # calibrate + diagnostics only, don't touch the JSON files or summary CSV:
    python model_eval/run_cumulative_patching.py --results-dir <dir> --no-write
"""

import argparse
import csv
import json
import os
from pathlib import Path

from tqdm import tqdm

ENGLISH = "eng_Latn"
DEFAULT_TARGET_PPS = 32.77
DEFAULT_PPS_LOW = 36.0
DEFAULT_PPS_HIGH = 23.0
EVAL_MODES_KEY = {"bytes": "eval_modes", "chars": "char_eval_modes"}
CASE_PREFIX = "cumulative_"
BOUND_NAMES = ["low", "mid", "high", "anchor"]
PPS_TOL = 0.05


# ── score extraction (same convention as run_patching.py) ─────────────────────

def get_scores_and_byte_counts(sentence, source):
    if source == "bytes":
        scores = [be[1] for be in sentence["bytes_entropies"]]
        byte_counts = [1] * len(scores)
    else:
        scores = [ce[1] for ce in sentence["chars_entropies"]]
        byte_counts = [ce[2] for ce in sentence["chars_entropies"]]
    return scores, byte_counts


def word_start_units(sentence, source):
    """Set of unit indices (bytes or chars) that begin a word, i.e. a
    non-whitespace character preceded by whitespace. Index 0 excluded (it
    always starts a patch). Returns None if byte widths can't be aligned."""
    text = sentence["text"]
    starts_chars = {i for i in range(1, len(text))
                    if not text[i].isspace() and text[i - 1].isspace()}
    if source == "chars":
        return starts_chars
    if "chars_entropies" in sentence:
        widths = [ce[2] for ce in sentence["chars_entropies"]]
    else:
        widths = [len(c.encode("utf-8")) for c in text]
    if len(widths) != len(text) or sum(widths) != len(sentence["bytes_entropies"]):
        return None
    offsets, pos = [], 0
    for w in widths:
        offsets.append(pos)
        pos += w
    return {offsets[i] for i in starts_chars}


# ── the patcher ───────────────────────────────────────────────────────────────

def cumulative_patch_lengths(scores, T, m):
    """Patch lengths in units. See module docstring for the rule."""
    lengths = []
    cur_len = 0
    cum = 0.0
    prev = None
    for s in scores:
        if cur_len > 0 and cum + s > T and (m is None or s - prev > m):
            lengths.append(cur_len)
            cur_len = 0
            cum = 0.0
        cur_len += 1
        cum += s
        prev = s
    if cur_len:
        lengths.append(cur_len)
    return lengths


def byte_lengths_for_patches(lengths_units, byte_counts):
    out, idx = [], 0
    for pl in lengths_units:
        out.append(sum(byte_counts[idx:idx + pl]))
        idx += pl
    return out


# ── calibration ───────────────────────────────────────────────────────────────

def mean_pps(score_lists, T, m):
    return sum(len(cumulative_patch_lengths(s, T, m)) for s in score_lists) / len(score_lists)


def search_T(score_lists, target, m, t_high_init):
    """Binary search T >= 0 for mean pps == target (higher T -> fewer
    patches). Returns (T, achieved_pps) or (None, pps_at_T0) if infeasible."""
    pps0 = mean_pps(score_lists, 0.0, m)
    if pps0 < target - PPS_TOL:
        return None, pps0
    lo, hi = 0.0, t_high_init
    mid, pps = hi, None
    for _ in range(60):
        mid = (lo + hi) / 2
        pps = mean_pps(score_lists, mid, m)
        if abs(pps - target) < PPS_TOL:
            break
        if pps > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-7:
            break
    return mid, pps


def search_m_star(score_lists, target):
    """Delta threshold at which PURE monotonicity (= cumulative at T=0)
    gives English `target` pps. Higher m -> fewer patches."""
    all_deltas = [b - a for s in score_lists for a, b in zip(s, s[1:])]
    lo, hi = min(all_deltas) - 1e-6, max(all_deltas) + 1e-6
    mid = hi
    for _ in range(60):
        mid = (lo + hi) / 2
        pps = mean_pps(score_lists, 0.0, mid)
        if abs(pps - target) < PPS_TOL:
            break
        if pps > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-7:
            break
    return mid


def parse_mono_specs(spec):
    out = []
    for part in spec.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part == "none":
            out.append(("none", None))
        elif part.startswith("rel:"):
            out.append(("rel", float(part[4:])))
        else:
            out.append(("abs", float(part)))
    return out


def case_name(kind, val):
    if kind == "none":
        return f"{CASE_PREFIX}nomono"
    tag = f"{val:g}".replace(".", "p").replace("-", "neg")
    return f"{CASE_PREFIX}mono_{'rel' if kind == 'rel' else ''}{tag}"


def run_stem(results_dir):
    """e.g. results/own_models/<run>/step_0000006000 -> <run>_step_0000006000"""
    return "_".join(Path(results_dir).resolve().parts[-2:])


def threshold_key(t):
    return f"t_{t:.4f}"


def calibrate(eng_sentences, source, mono_specs, args):
    score_lists = [get_scores_and_byte_counts(s, source)[0] for s in eng_sentences]
    t_high_init = max(sum(s) for s in score_lists) + 1.0
    targets = {"low": args.pps_low, "high": args.pps_high, "anchor": args.target_pps}

    # m_star per bound: the delta threshold at which PURE monotonicity gives
    # English that bound's pps target (= the feasibility ceiling for m there).
    m_star = {b: search_m_star(score_lists, t) for b, t in targets.items()}
    m_star["mid"] = (m_star["low"] + m_star["high"]) / 2
    print("  m_star (pure monotonicity): "
          + "  ".join(f"{b}={m_star[b]:.4f}" for b in BOUND_NAMES))

    def m_for(kind, val, bound):
        if kind == "none":
            return None
        if kind == "abs":
            return val
        ref = m_star[bound] if args.mono_ref == "per-bound" else m_star["low"]
        return val * ref

    cases = {}
    for kind, val in mono_specs:
        name = case_name(kind, val)
        row = {"case": name}
        ok = True
        for bound in ("low", "high", "anchor"):
            m = m_for(kind, val, bound)
            T, pps = search_T(score_lists, targets[bound], m, t_high_init)
            if T is None:
                print(f"  {name:32s} {bound}: INFEASIBLE with m={m:.4f} "
                      f"(pps at T=0 is {pps:.2f} < {targets[bound]})")
                ok = False
                break
            row[f"t_{bound}"], row[f"pps_{bound}"], row[f"m_{bound}"] = T, pps, m
        if not ok:
            continue
        # Midpoint: midpoint in BOTH parameters, mirroring the pipeline's
        # t_mid = (t_low + t_high) / 2 convention. Its pps lands wherever
        # it lands, as for the existing cases.
        row["m_mid"] = m_for(kind, val, "mid")
        row["t_mid"] = (row["t_low"] + row["t_high"]) / 2
        row["pps_mid"] = mean_pps(score_lists, row["t_mid"], row["m_mid"])

        keys = [threshold_key(row[f"t_{b}"]) for b in BOUND_NAMES]
        if len(set(keys)) < len(keys):
            print(f"  WARNING: {name} has two bounds with the same threshold key "
                  f"{keys} -- their results would overwrite each other in the JSON.")

        fmt_m = lambda m: "-" if m is None else f"{m:.4f}"
        print(f"  {name:28s} "
              + "  ".join(f"{b}: T={row[f't_{b}']:.4f} m={fmt_m(row[f'm_{b}'])} "
                          f"pps={row[f'pps_{b}']:.2f}" for b in BOUND_NAMES))
        cases[name] = row
    return cases, m_star


def write_summary_csv(path, cases, eng_sentences, source):
    """Same columns as calibrate_thresholds.py's summary CSV (plus m)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "fixed_t",
                    "t_low", "pps_low", "bpp_low", "t_mid", "pps_mid", "bpp_mid",
                    "t_high", "pps_high", "bpp_high", "t_anchor", "pps_anchor", "bpp_anchor",
                    "final_thresholds", "m_low", "m_mid", "m_high", "m_anchor"])
        for name, r in cases.items():
            cells = []
            for b in BOUND_NAMES:
                n_p = n_b = 0
                for s in eng_sentences:
                    sc, bc = get_scores_and_byte_counts(s, source)
                    lu = cumulative_patch_lengths(sc, r[f"t_{b}"], r[f"m_{b}"])
                    n_p += len(lu)
                    n_b += sum(bc)
                cells += [f"{r[f't_{b}']:.4f}", f"{r[f'pps_{b}']:.2f}", f"{n_b / n_p:.4f}"]
            finals = sorted(r[f"t_{b}"] for b in BOUND_NAMES)
            ms = ["" if r[f"m_{b}"] is None else f"{r[f'm_{b}']:.4f}" for b in BOUND_NAMES]
            w.writerow([name, ""] + cells + [str([f"{t:.4f}" for t in finals])] + ms)
    print(f"  summary -> {path}")


# ── patching all languages ────────────────────────────────────────────────────

def process_language(sentences, source, cases, write, diag):
    modes_key = EVAL_MODES_KEY[source]
    agg = {}  # (case, bound) -> counters
    for s in sentences:
        scores, byte_counts = get_scores_and_byte_counts(s, source)
        ws = word_start_units(s, source)
        if write:
            modes = s.setdefault(modes_key, {})
            for k in [k for k in modes if k.startswith(CASE_PREFIX) and k not in cases]:
                del modes[k]  # stale cumulative case from an earlier run
        for name, r in cases.items():
            if write:
                valid = {threshold_key(r[f"t_{b}"]) for b in BOUND_NAMES}
                entry = modes.setdefault(name, {})
                for k in [k for k in entry if k not in valid]:
                    del entry[k]
            for b in BOUND_NAMES:
                lu = cumulative_patch_lengths(scores, r[f"t_{b}"], r[f"m_{b}"])
                lb = byte_lengths_for_patches(lu, byte_counts)
                if write:
                    rec = {"n_patches": len(lu),
                           "avg_bytes_per_patch": round(sum(lb) / len(lu), 4)}
                    if source == "bytes":
                        rec["patch_lengths"] = lu
                    else:
                        rec["patch_lengths_chars"] = lu
                        rec["patch_lengths_bytes"] = lb
                    entry[threshold_key(r[f"t_{b}"])] = rec
                a = agg.setdefault((name, b), {"n_sent": 0, "n_patches": 0, "n_bytes": 0,
                                               "cuts": 0, "cuts_at_ws": 0, "ws": 0, "ws_cut": 0})
                a["n_sent"] += 1
                a["n_patches"] += len(lu)
                a["n_bytes"] += sum(lb)
                if ws is not None:
                    starts, pos = set(), 0
                    for pl in lu[:-1]:
                        pos += pl
                        starts.add(pos)
                    a["cuts"] += len(starts)
                    a["cuts_at_ws"] += len(starts & ws)
                    a["ws"] += len(ws)
                    a["ws_cut"] += len(ws & starts)
    ent_sum = sum(sum(get_scores_and_byte_counts(s, source)[0]) for s in sentences) / len(sentences)
    return agg, ent_sum


def run_source(source, args, mono_specs, paths):
    print(f"\n══ score source: {source} ══")
    eng_path = Path(args.results_dir) / f"{ENGLISH}.json"
    with open(eng_path, encoding="utf-8") as f:
        eng = json.load(f)
    if source == "chars" and "chars_entropies" not in eng[0]:
        raise ValueError(f"No chars_entropies in {eng_path} -- run add_char_entropies.py first.")

    cases, m_star = calibrate(eng, source, mono_specs, args)
    if not cases:
        print("  nothing feasible, skipping")
        return []

    summary_path = args.summary_csv or os.path.join(
        "calibrated_thresholds", "cumulative",
        *(["char_level"] if source == "chars" else []),
        f"{run_stem(args.results_dir)}_cumulative_thresholds_summary.csv")
    if args.no_write:
        # A dry run must not replace the calibration the JSONs were written
        # with -- results_to_txt_premiums.py looks thresholds up in this file.
        print(f"  --no-write: summary CSV not written ({summary_path.replace('{source}', source)})")
    else:
        write_summary_csv(summary_path.replace("{source}", source), cases, eng, source)

    diag_rows, eng_ent = [], None
    per_lang = {}
    for path in tqdm(paths, desc=f"patching ({source})"):
        with open(path, encoding="utf-8") as f:
            sentences = json.load(f)
        if source == "chars" and sentences and "chars_entropies" not in sentences[0]:
            print(f"  {path.stem}: no chars_entropies, skipped")
            continue
        agg, ent = process_language(sentences, source, cases, not args.no_write, True)
        per_lang[path.stem] = (agg, ent)
        if not args.no_write:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(sentences, f, ensure_ascii=False, indent=2)

    eng_agg, eng_ent = per_lang[ENGLISH]
    for lang, (agg, ent) in per_lang.items():
        for (name, b), a in agg.items():
            pps = a["n_patches"] / a["n_sent"]
            e = eng_agg[(name, b)]
            eng_pps = e["n_patches"] / e["n_sent"]
            diag_rows.append({
                "score_source": source, "lang": lang, "case": name, "bound": b,
                "m": "" if cases[name][f"m_{b}"] is None else round(cases[name][f"m_{b}"], 4),
                "T": round(cases[name][f"t_{b}"], 4),
                "pps": round(pps, 4),
                "bpp": round(a["n_bytes"] / a["n_patches"], 4),
                "pps_premium": round(pps / eng_pps, 4),
                "entropy_sum_premium": round(ent / eng_ent, 4),
                "word_start_precision": round(a["cuts_at_ws"] / a["cuts"], 4) if a["cuts"] else "",
                "word_start_recall": round(a["ws_cut"] / a["ws"], 4) if a["ws"] else "",
            })
    return diag_rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--score-source", choices=("bytes", "chars", "both"), default="chars")
    p.add_argument("--mono", default="none,0,rel:0.25,rel:0.5",
                   help="Comma-separated: none | <float> | rel:<f> (see docstring).")
    p.add_argument("--mono-ref", choices=("per-bound", "low"), default="per-bound",
                   help="What rel:<f> is a fraction of: m_star at EACH bound's own pps "
                        "target (default), or m_star at the low bound for all bounds.")
    p.add_argument("--target-pps", type=float, default=DEFAULT_TARGET_PPS)
    p.add_argument("--pps-low", type=float, default=DEFAULT_PPS_LOW)
    p.add_argument("--pps-high", type=float, default=DEFAULT_PPS_HIGH)
    p.add_argument("--summary-csv", default=None,
                   help="Override summary CSV path; may contain {source}.")
    p.add_argument("--diag-csv", default=None,
                   help="Diagnostics CSV (default: results/cumulative_diag/<dir name>.csv).")
    p.add_argument("--no-write", action="store_true",
                   help="Calibrate + diagnostics only; leave the JSON files AND the summary CSV untouched.")
    args = p.parse_args()

    mono_specs = parse_mono_specs(args.mono)
    sources = ["bytes", "chars"] if args.score_source == "both" else [args.score_source]
    paths = sorted(Path(args.results_dir).glob("*.json"))
    if not any(pth.stem == ENGLISH for pth in paths):
        raise FileNotFoundError(f"{ENGLISH}.json not found in {args.results_dir}")
    print(f"{len(paths)} language files in {args.results_dir}")
    print(f"pps targets: low={args.pps_low} anchor={args.target_pps} high={args.pps_high}")

    rows = []
    for source in sources:
        rows += run_source(source, args, mono_specs, paths)

    diag_path = args.diag_csv or os.path.join(
        "results", "cumulative_diag", f"{run_stem(args.results_dir)}.csv")
    os.makedirs(os.path.dirname(diag_path) or ".", exist_ok=True)
    if rows:
        with open(diag_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\ndiagnostics -> {diag_path}")


if __name__ == "__main__":
    main()