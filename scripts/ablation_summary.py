#!/usr/bin/env python3
"""
ablation_summary.py
-------------------
Aggregate per-run ablation CSV into mean / median / stddev tables.

Input  : all_runs.csv written by run_ablation_study.sh (one row per run).
Output : a markdown summary grouped by (design, variant), and a CSV with
         the same aggregated rows for downstream plotting.

Columns reported per (design, variant):
    runs, success_rate, tokens mean/median/std, assertions mean/median/std,
    tokens_per_assertion mean/median.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Tuple


def _f(v: str) -> float | None:
    try:
        return float(v) if v not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def _stats(xs: List[float]) -> Dict[str, float]:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return {"n": 0, "mean": 0.0, "median": 0.0, "std": 0.0,
                "min": 0.0, "max": 0.0}
    return {
        "n": len(xs),
        "mean": mean(xs),
        "median": median(xs),
        "std": stdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
    }


def aggregate(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict]:
    by_key: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_key[(r["design"], r["variant"])].append(r)

    agg: Dict[Tuple[str, str], Dict] = {}
    for (design, variant), group in by_key.items():
        total = len(group)
        ok = sum(1 for r in group if r["return_code"] == "0")
        tokens = [_f(r["total_tokens"]) for r in group]
        prompt = [_f(r["prompt_tokens"]) for r in group]
        compl = [_f(r["completion_tokens"]) for r in group]
        asserts = [_f(r["final_assertions"]) for r in group]
        tpa = [_f(r["tokens_per_assertion"]) for r in group]
        elapsed = [_f(r["elapsed_sec"]) for r in group]

        agg[(design, variant)] = {
            "runs": total,
            "succeeded": ok,
            "success_rate": ok / total if total else 0.0,
            "tokens":    _stats(tokens),
            "prompt":    _stats(prompt),
            "completion": _stats(compl),
            "assertions": _stats(asserts),
            "tokens_per_assertion": _stats(tpa),
            "elapsed_sec": _stats(elapsed),
        }
    return agg


def write_markdown(
    agg: Dict[Tuple[str, str], Dict],
    out_path: Path,
    all_rows: List[Dict[str, str]] | None = None,
) -> None:
    # Collect designs and a canonical variant order
    designs = sorted({d for (d, _) in agg.keys()})
    variant_order = ["full", "naive", "no-facts", "flat-facts",
                     "no-ast", "no-repair", "no-feedback"]

    lines: List[str] = []
    lines.append("# Ablation Study — Aggregated Results")
    lines.append("")
    total_runs = sum(a["runs"] for a in agg.values())
    total_ok   = sum(a["succeeded"] for a in agg.values())
    lines.append(
        f"**{total_runs} runs so far** ({total_ok} ok, {total_runs - total_ok} failed) "
        f"across {len(agg)} (design, variant) cells."
    )
    lines.append(f"Per-run raw data: `all_runs.csv`.")
    lines.append("")

    for design in designs:
        variants_present = [v for v in variant_order
                            if (design, v) in agg] + \
                           [v for (d, v) in agg if d == design
                            and v not in variant_order]
        lines.append(f"## {design}")
        lines.append("")
        lines.append(
            "| Variant | Runs | OK | Tokens (mean ± std) | Tokens (median) | "
            "Assertions (mean ± std) | Assertions (median) | "
            "Tokens/assert (median) | Elapsed (median s) |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for variant in variants_present:
            k = (design, variant)
            a = agg[k]
            t = a["tokens"]
            n = a["assertions"]
            tpa = a["tokens_per_assertion"]
            el = a["elapsed_sec"]
            lines.append(
                f"| {variant} "
                f"| {a['runs']} "
                f"| {a['succeeded']}/{a['runs']} "
                f"| {t['mean']:,.0f} ± {t['std']:,.0f} "
                f"| {t['median']:,.0f} "
                f"| {n['mean']:.1f} ± {n['std']:.1f} "
                f"| {n['median']:.0f} "
                f"| {tpa['median']:,.0f} "
                f"| {el['median']:.0f} |"
            )
        lines.append("")

    # Global cross-design headline: median tokens/assertion for 'full'
    # vs 'no-facts' across designs.
    headline_rows = []
    for d in designs:
        for v in ("full", "no-facts"):
            if (d, v) in agg:
                headline_rows.append((d, v,
                    agg[(d, v)]["tokens_per_assertion"]["median"],
                    agg[(d, v)]["assertions"]["median"]))
    if headline_rows:
        lines.append("## Headline — full vs no-facts (medians across runs)")
        lines.append("")
        lines.append("| Design | Variant | Tokens/assert (median) | Assertions (median) |")
        lines.append("|---|---|---:|---:|")
        for d, v, tpa, n in headline_rows:
            lines.append(f"| {d} | {v} | {tpa:,.0f} | {n:.0f} |")
        lines.append("")

    # Chronological run log: every run, in the order it completed.
    if all_rows:
        lines.append("## Run log (chronological)")
        lines.append("")
        lines.append(
            "| # | Design | Variant | Run | Tokens | Assertions | Tok/assert | Elapsed | Status |"
        )
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|---|")
        for i, r in enumerate(all_rows, start=1):
            rc = r.get("return_code", "")
            lint = r.get("lint_status", "") or ("OK" if rc == "0" else "FAIL")
            tok = r.get("total_tokens") or "0"
            fa = r.get("final_assertions") or "0"
            tpa = r.get("tokens_per_assertion") or ""
            el = r.get("elapsed_sec") or ""
            try:
                tok_fmt = f"{int(tok):,}"
            except ValueError:
                tok_fmt = tok
            try:
                tpa_fmt = f"{float(tpa):,.0f}" if tpa else ""
            except ValueError:
                tpa_fmt = tpa
            lines.append(
                f"| {i} | {r['design']} | {r['variant']} | {r['run']} "
                f"| {tok_fmt} | {fa} | {tpa_fmt} | {el} | {lint} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines))


def write_csv(agg: Dict[Tuple[str, str], Dict], out_path: Path) -> None:
    fields = [
        "design", "variant", "runs", "succeeded", "success_rate",
        "tokens_mean", "tokens_median", "tokens_std",
        "prompt_mean", "prompt_median",
        "completion_mean", "completion_median",
        "assertions_mean", "assertions_median", "assertions_std",
        "tokens_per_assertion_mean", "tokens_per_assertion_median",
        "elapsed_median",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for (design, variant), a in sorted(agg.items()):
            w.writerow([
                design, variant,
                a["runs"], a["succeeded"], f"{a['success_rate']:.3f}",
                f"{a['tokens']['mean']:.1f}",
                f"{a['tokens']['median']:.1f}",
                f"{a['tokens']['std']:.1f}",
                f"{a['prompt']['mean']:.1f}",
                f"{a['prompt']['median']:.1f}",
                f"{a['completion']['mean']:.1f}",
                f"{a['completion']['median']:.1f}",
                f"{a['assertions']['mean']:.1f}",
                f"{a['assertions']['median']:.1f}",
                f"{a['assertions']['std']:.1f}",
                f"{a['tokens_per_assertion']['mean']:.1f}",
                f"{a['tokens_per_assertion']['median']:.1f}",
                f"{a['elapsed_sec']['median']:.1f}",
            ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--csv", required=True,
                    help="Path to the per-run CSV (all_runs.csv).")
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()

    src = Path(args.csv)
    if not src.exists():
        raise SystemExit(f"no such file: {src}")

    with src.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"no rows in {src}")

    agg = aggregate(rows)
    write_markdown(agg, Path(args.output_md), all_rows=rows)
    write_csv(agg, Path(args.output_csv))
    print(f"Aggregated {len(rows)} runs across {len(agg)} (design, variant) pairs.")
    print(f"  MD  : {args.output_md}")
    print(f"  CSV : {args.output_csv}")


if __name__ == "__main__":
    main()
