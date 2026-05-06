"""
i2c_sanity_compare — print a one-screen comparison of an I2C-Master run
against AssertLLM's paper-reported Table 2 row.

Usage:
    python scripts/i2c_sanity_compare.py [--out-dir i2c_sanity_<TS>]

Reads:
    - assertllm/run_<NN>/sva.sv + token_summary.json
    - specguard_full/run_<NN>/sva.sv + token_summary.json (optional)
    - specguard_no_facts/run_<NN>/sva.sv + token_summary.json (optional)

Prints a table:
    - Paper's reported I2C numbers (Table 2)
    - Our AssertLLM reproduction's numbers
    - SpecGuard for cross-reference
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from statistics import mean


PAPER_I2C = {
    "from_nl_generated": 56,
    "from_nl_syntax_correct": 56,
    "from_nl_fpv_pass": 48,
    "from_waveform_generated": 9,
    "from_waveform_syntax_correct": 9,
    "from_waveform_fpv_pass": 6,
    "total_generated": 65,
    "total_syntax_correct": 65,
    "total_fpv_pass": 56,
    "fpv_pass_rate": 86,           # %
    "coi_coverage_pct": 93,        # %
}


def _classify_assertions(text: str) -> dict:
    """Classify assertions in a SVA file as width / connectivity /
    function. Approximate but consistent across files."""
    lines = [l.strip() for l in text.splitlines() if re.match(r"^\s*assert", l)]
    width = sum(1 for l in lines if "$bits(" in l)
    # The assertllm output has explicit category headers as comments
    # we honour them when present. Otherwise fall back to heuristics.
    sections = {"width": 0, "connectivity": 0, "function": 0}
    cur = None
    for ln in text.splitlines():
        m = re.search(r"\[(width|connectivity|function)\]", ln, re.IGNORECASE)
        if m:
            cur = m.group(1).lower(); continue
        if cur and re.match(r"^\s*assert", ln):
            sections[cur] += 1
    if any(sections.values()):
        return {"total": len(lines), **sections, "width_via_bits": width}
    # Heuristic fallback for SpecGuard output (no section headers)
    seq = sum(1 for l in lines if "|=>" in l or "|->" in l)
    return {
        "total": len(lines),
        "width": width,
        "connectivity": 0,
        "function": seq,
        "width_via_bits": width,
    }


def _read_run(run_dir: Path) -> dict:
    sva = run_dir / "sva.sv"
    summary = run_dir / "token_summary.json"
    out = {"run_dir": str(run_dir), "tokens": 0, "calls": 0,
           "asserts": 0, "tok_per_a": None, "classify": {}}
    if summary.exists():
        j = json.loads(summary.read_text())
        out["tokens"] = j.get("total_tokens", 0)
        out["calls"]  = j.get("llm_calls", 0)
        out["asserts"] = j.get("final_assertions", 0)
        out["tok_per_a"] = j.get("tokens_per_assertion")
    if sva.exists():
        out["classify"] = _classify_assertions(sva.read_text())
    return out


def _aggregate(runs: list[dict]) -> dict:
    if not runs:
        return {"n": 0}
    n = len(runs)
    out = {"n": n}
    for k in ("tokens", "calls", "asserts"):
        vals = [r[k] for r in runs if r.get(k) is not None]
        out[f"{k}_mean"] = mean(vals) if vals else 0
    cl_keys = ("total", "width", "connectivity", "function")
    for k in cl_keys:
        vals = [r["classify"].get(k, 0) for r in runs]
        out[f"cls_{k}_mean"] = mean(vals) if vals else 0
    tpas = [r["tok_per_a"] for r in runs if r.get("tok_per_a") not in (None, "", "None")]
    out["tpa_mean"] = mean(tpas) if tpas else None
    return out


def _print_run_table(label: str, agg: dict) -> None:
    print(f"  {label:32s}  N={agg.get('n',0)}")
    if agg.get("n", 0) == 0:
        print(f"  {'(no runs found)':<32s}")
        return
    print(f"  {'  total assertions (mean)':32s}  {agg.get('cls_total_mean', 0):.1f}")
    print(f"  {'  width  (mean)':32s}  {agg.get('cls_width_mean', 0):.1f}")
    print(f"  {'  connectivity (mean)':32s}  {agg.get('cls_connectivity_mean', 0):.1f}")
    print(f"  {'  function (mean)':32s}  {agg.get('cls_function_mean', 0):.1f}")
    print(f"  {'  tokens (mean)':32s}  {agg.get('tokens_mean', 0):,.0f}")
    print(f"  {'  LLM calls (mean)':32s}  {agg.get('calls_mean', 0):.0f}")
    if agg.get("tpa_mean"):
        print(f"  {'  tokens per assertion (mean)':32s}  {agg['tpa_mean']:,.0f}")
    print()


def _print_paper_row() -> None:
    p = PAPER_I2C
    print(f"  {'AssertLLM paper (Table 2 row)':32s}  reported on I²C-Master Core")
    print(f"  {'  total assertions':32s}  {p['total_generated']} "
          f"(NL: {p['from_nl_generated']} + waveform: {p['from_waveform_generated']})")
    print(f"  {'  syntax-correct':32s}  {p['total_syntax_correct']}/{p['total_generated']} (100 %)")
    print(f"  {'  FPV-pass':32s}  {p['total_fpv_pass']}/{p['total_generated']} ({p['fpv_pass_rate']} %)")
    print(f"  {'  COI coverage':32s}  {p['coi_coverage_pct']} %")
    print()


def _verdict(assertllm_agg: dict) -> None:
    if assertllm_agg.get("n", 0) == 0:
        print("[verdict] No AssertLLM I2C runs found yet — skip verdict.")
        return
    n_ours = assertllm_agg.get("cls_total_mean", 0)
    paper_nl = PAPER_I2C["from_nl_generated"]   # what we should target (no waveform)
    paper_total = PAPER_I2C["total_generated"]  # what they got with everything
    pct_nl    = (n_ours / paper_nl)    * 100 if paper_nl    else 0
    pct_total = (n_ours / paper_total) * 100 if paper_total else 0

    print("[verdict]")
    print(f"  Our reproduction emitted {n_ours:.1f} assertions on I²C-Master.")
    print(f"  Paper's NL-only target (Table 2): {paper_nl} → reproduction is at {pct_nl:.0f} %.")
    print(f"  Paper's total (NL + waveform):   {paper_total} → reproduction is at {pct_total:.0f} %.")
    if pct_nl >= 70:
        print("  → Reproduction is faithful to the NL-only path (≥ 70 % of paper).")
    elif pct_nl >= 40:
        print("  → Reproduction is partial (40-70 %). Spec quality may be the bottleneck;")
        print("    check the per-signal interconnection field of signal_specs.json.")
    else:
        print("  → Reproduction is far below paper (< 40 %). Likely either the LLM #1 input")
        print("    quality is much weaker than the paper's PDF, or the LLM is producing")
        print("    mostly empty per-signal records on this benchmark. Inspect signal_specs.json.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=None,
                    help="Path to a sanity-check results dir. "
                         "If omitted, finds the latest i2c_sanity_*.")
    args = ap.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        candidates = sorted(glob.glob("i2c_sanity_*"))
        if not candidates:
            print("No i2c_sanity_* directory found. Run ./run_i2c_sanity_check.sh first.",
                  file=sys.stderr)
            sys.exit(1)
        out_dir = Path(candidates[-1])

    print(f"Reading: {out_dir}")
    print()
    print("=" * 78)
    print("I²C-Master sanity check — reproduction vs paper")
    print("=" * 78)
    print()

    _print_paper_row()

    al_runs = []
    for r in sorted(glob.glob(str(out_dir / "assertllm" / "run_*"))):
        al_runs.append(_read_run(Path(r)))
    al_agg = _aggregate(al_runs)
    _print_run_table("AssertLLM reproduction (ours)", al_agg)

    for label, sub in [("SpecGuard full (with facts)", "specguard_full"),
                       ("SpecGuard no-facts default", "specguard_no_facts")]:
        runs = []
        for r in sorted(glob.glob(str(out_dir / sub / "run_*"))):
            runs.append(_read_run(Path(r)))
        if runs:
            _print_run_table(label, _aggregate(runs))

    _verdict(al_agg)


if __name__ == "__main__":
    main()
