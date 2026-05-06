#!/usr/bin/env python3
"""
quality_weighted_cost.py
------------------------
Compare token efficiency across SVA-generation methods on a metric that
penalises trivial / duplicate / narrow-domain output.

For each `sva.sv` file under one or more results directories, scores
every assertion on three axes:

  - usable    : present in the post-lint .sva file (the orchestrators
                only emit lint-clean SVAs, so this defaults True; we
                additionally drop obvious malformed orphans like a bare
                `else $error(...)` without a preceding `assert`)
  - complex   : NOT a pure $bits() width check, NOT a duplicate of
                another in the same file, AND has at least one of:
                $past, $stable, $rose/$fell, $onehot, inside, |=>,
                or a multi-signal antecedent (>=2 signals joined by
                && / ||)
  - diverse   : the file as a whole covers N distinct functional
                domains (mux, passthrough, reset-imp, seq-update,
                handshake, stable-backpressure, onehot, inside-set,
                edge-detect, range, other)

Composite quality unit per file:
    Q = n_unique_complex × n_distinct_domains

Headline metric per file:
    tokens / Q   (tokens per quality-weighted assertion-unit)

Lower = better. The metric penalises three known inflation patterns:
  - AssertLLM-style trivial $bits() walks (low complexity)
  - duplicate copies of the same assertion (low uniqueness)
  - single-category output (low diversity)

USAGE
-----
    python scripts/quality_weighted_cost.py \\
        --results-dirs baselines/results_all_*/ \\
                       naive_then_large_ablation_*/ \\
        --output-md docs/quality_weighted_cost.md \\
        --output-csv docs/quality_weighted_cost.csv

Methods are auto-detected from the path layout:

    baselines/results_all_*/<method>/<design>/run_NN/sva.sv     → method
    naive_then_large_ablation_*/<design>/<variant>/run_NN/sva.sv → variant
    ablation_study_*/<design>/<variant>/run_NN/sva.sv            → variant

Use `--methods` to filter (e.g. `--methods assertllm chiraag full`).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Per-assertion classification
# ---------------------------------------------------------------------------

DOMAIN_LABELS = (
    "stable-backpressure", "onehot", "inside-set", "seq-update",
    "past-other", "edge-detect", "handshake", "reset-imp",
    "implication-other", "other",
)


def categorise(line: str) -> str:
    """Single-domain label per assertion (mirrors facts_card_qualitative_review)."""
    L = line.strip()
    if not L.startswith("assert"):
        return "non-assert"
    if "$stable" in L: return "stable-backpressure"
    if "$onehot" in L: return "onehot"
    if "inside" in L: return "inside-set"
    has_past = "$past" in L
    if has_past and "|=>" in L: return "seq-update"
    if has_past: return "past-other"
    if "$rose" in L or "$fell" in L: return "edge-detect"
    has_handshake = (
        ("pvld" in L and "prdy" in L)
        or (re.search(r"\bvalid\b", L) and re.search(r"\bready\b", L))
    )
    if has_handshake: return "handshake"
    if re.search(r"!\s*[a-zA-Z_]\w*\s*\|->", L): return "reset-imp"
    if "|->" in L: return "implication-other"
    return "other"


_BITS_PURE = re.compile(r"\$bits\s*\(\s*\w+(?:\.\w+|\[[^\]]+\])*\s*\)\s*==\s*\d")
_RESET_BARE = re.compile(
    r"^\s*assert\b[^;]*?!\s*\w+\s*\|->\s*\w+\s*==\s*"
    r"(?:\d+|'\w+|\d+'[bdho]\w+|\{[^}]+\})\s*\)?[^;]*$"
)


def is_trivial(line: str) -> bool:
    """A bare $bits() width check or a single-signal reset-to-constant."""
    L = line.strip()
    if _BITS_PURE.search(L):
        # Width check is trivial unless paired with another property in
        # the same expression.
        if "&&" not in L and "||" not in L and "|->" not in L:
            return True
    return False


_DOLLAR_BUILTIN = re.compile(r"\$(past|stable|rose|fell|onehot|onehot0|isunknown)\b")
_RTL_KW = {
    "assert","property","disable","iff","posedge","negedge","else","error",
    "and","or","not","if","logic","wire","reg","input","output","always_ff",
    "always_comb","module","endmodule","begin","end","b","h","d","x","z",
    "inside","throughout","until","first_match",
}


def antecedent_signals(line: str) -> int:
    """Best-effort count of distinct signals on the LHS of |-> or |=>."""
    m = re.search(r"\)\s*([^|]+?)\s*\|[->=]+\s*", line)
    if not m:
        return 0
    body = re.sub(r'"[^"]*"', "", m.group(1))
    body = re.sub(r"\$\w+", "", body)
    body = re.sub(r"\d+'[bdho][\w?]+", "", body)
    toks = re.findall(r"\b[a-zA-Z_]\w*\b", body)
    sigs = {t for t in toks if t not in _RTL_KW and not t.isdigit()}
    return len(sigs)


def is_complex(line: str) -> bool:
    """Has at least one of: $past/$stable/$rose/$fell/$onehot, inside,
    |=> (next-cycle), multi-signal antecedent (>=2 signals)."""
    L = line.strip()
    if _DOLLAR_BUILTIN.search(L): return True
    if "inside" in L: return True
    if "|=>" in L: return True
    if antecedent_signals(L) >= 2: return True
    return False


_NORM_WS = re.compile(r"\s+")


def normalise(line: str) -> str:
    """Collapse whitespace + strip comments + lowercase for dedup."""
    s = re.sub(r"//.*", "", line)
    s = re.sub(r"\$error\([^)]*\)", "$error(...)", s)  # error msg is just commentary
    return _NORM_WS.sub(" ", s).strip().lower()


# ---------------------------------------------------------------------------
# Per-file metrics
# ---------------------------------------------------------------------------

@dataclass
class FileMetrics:
    n_total: int = 0
    n_usable: int = 0
    n_unique: int = 0
    n_complex: int = 0
    n_complex_unique: int = 0
    n_trivial: int = 0
    n_malformed: int = 0
    n_domains: int = 0
    domains: Counter = field(default_factory=Counter)

    @property
    def quality_unit(self) -> int:
        """Q = n_complex_unique × n_distinct_domains."""
        return self.n_complex_unique * self.n_domains


def detect_malformed(text: str) -> int:
    """Count orphan `else $error(...)` lines without a preceding assert."""
    lines = text.splitlines()
    n = 0
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("else") and "$error" in s:
            preceding = " ".join(lines[max(0, i - 3):i])
            if "assert" not in preceding:
                n += 1
    return n


# `property NAME; ... endproperty` declarations — captured so that an
# `assert property (NAME)` line can be analysed against the property
# body (which carries the $past / $stable / |=> / multi-signal cues),
# not against the bare `assert property (NAME)` line itself.
_PROPERTY_BLOCK_RE = re.compile(
    r"\bproperty\s+([A-Za-z_]\w*)\s*[;]\s*(.*?)\s*\bendproperty\b",
    re.DOTALL | re.IGNORECASE,
)
_ASSERT_PROP_REF_RE = re.compile(
    r"\bassert\s+property\s*\(\s*([A-Za-z_]\w*)\s*\)",
    re.IGNORECASE,
)


def extract_property_bodies(text: str) -> Dict[str, str]:
    """Return {property_name -> body_text} for every `property ...
    endproperty` block in the file.  The body excludes the `property
    NAME;` header and the `endproperty` keyword."""
    out: Dict[str, str] = {}
    for m in _PROPERTY_BLOCK_RE.finditer(text):
        name, body = m.group(1), m.group(2).strip()
        out[name] = body
    return out


def _expand_with_property_body(
    line: str,
    prop_bodies: Dict[str, str],
    nearest_body: Optional[str] = None,
) -> str:
    """If `line` is a bare `assert property (NAME);` reference, splice
    the body of the named property into the line so that the line-based
    categorise / triviality / complexity helpers can score on the
    actual property text rather than the bare reference.

    Resolution order:
      1. Exact name match in `prop_bodies`.
      2. Fallback to `nearest_body` (the immediately preceding
         `property ... endproperty` block in the same file) — this
         covers files where the assert's reference name does not match
         the declared property name, e.g. Assertain's older refiner
         that derived the assert-side name from a different string than
         the LLM-emitted property header.
      3. No match → return `line` unchanged.
    """
    m = _ASSERT_PROP_REF_RE.search(line)
    if not m:
        return line
    body = prop_bodies.get(m.group(1)) or nearest_body
    if not body:
        return line
    flat_body = re.sub(r"\s+", " ", body).strip()
    return line.rstrip(";").rstrip() + "  " + flat_body + ";"


def analyse_file(path: Path) -> FileMetrics:
    fm = FileMetrics()
    if not path.exists():
        return fm
    text = path.read_text(errors="ignore")
    fm.n_malformed = detect_malformed(text)
    prop_bodies = extract_property_bodies(text)

    seen: Set[str] = set()
    nearest_body: Optional[str] = None    # most-recent property body seen
    in_prop = False
    prop_buf: List[str] = []
    for ln in text.splitlines():
        # Track the most-recent `property ... endproperty` block so the
        # next `assert property (NAME)` line can fall back to it when
        # the named lookup fails (e.g. files where the assert's
        # reference name doesn't match the declared property name).
        if not in_prop and re.search(r"\bproperty\s+[A-Za-z_]\w*", ln):
            in_prop = True
            prop_buf = [ln]
        elif in_prop:
            prop_buf.append(ln)
            if "endproperty" in ln:
                joined = "\n".join(prop_buf)
                bm = re.search(
                    r"\bproperty\s+[A-Za-z_]\w*\s*[;]\s*(.*?)\bendproperty\b",
                    joined, re.DOTALL,
                )
                if bm:
                    nearest_body = bm.group(1).strip()
                in_prop = False
                prop_buf = []
        s = ln.strip()
        if not s.startswith("assert"):
            continue
        fm.n_total += 1
        fm.n_usable += 1
        # Inline the named-property body before scoring so that
        # `assert property (p_xyz);` is judged on what `p_xyz` actually
        # asserts, not on the bare reference line.
        scoring_line = _expand_with_property_body(ln, prop_bodies, nearest_body)
        cat = categorise(scoring_line)
        fm.domains[cat] += 1
        norm = normalise(scoring_line)
        if norm in seen:
            continue
        seen.add(norm)
        fm.n_unique += 1
        triv = is_trivial(scoring_line)
        if triv:
            fm.n_trivial += 1
        if is_complex(scoring_line) and not triv:
            fm.n_complex += 1
            fm.n_complex_unique += 1

    # Domains that actually contributed any assertion (post-categorisation).
    fm.n_domains = sum(1 for k, v in fm.domains.items() if v > 0 and k != "non-assert")
    return fm


# ---------------------------------------------------------------------------
# Per-run aggregation
# ---------------------------------------------------------------------------

@dataclass
class RunRow:
    method: str
    design: str
    run: str
    sva_path: Path
    summary_path: Optional[Path]
    metrics: FileMetrics
    tokens: int = 0
    elapsed: float = 0.0


def load_token_summary(path: Path) -> Tuple[int, float]:
    if not path or not path.exists():
        return 0, 0.0
    try:
        d = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return 0, 0.0
    return int(d.get("total_tokens", 0) or 0), float(d.get("elapsed_sec", 0) or 0)


# Discover (method, design, run) from the directory shape.
_KNOWN_METHODS = {
    "assertllm", "assertain", "chiraag",
    "full", "no-facts", "flat-facts", "no-ast",
    "no-repair", "no-feedback", "naive",
}


def _infer_method_design(root: Path, parts: Tuple[str, ...]) -> Optional[Tuple[str, str]]:
    """Infer (method, design) from the path components of a found sva.sv.

    Supported layouts (relative to `root`):
      <method>/<design>/run_NN/sva.sv          — run_all_baselines.sh
      <design>/<variant>/run_NN/sva.sv         — naive_then_large_ablation, ablation_study
      <design>/run_NN/sva.sv                   — single-method baseline (e.g. results_20260429_1009)
                                                  → method inferred from root name (e.g. "assertllm")
    """
    if len(parts) >= 3 and parts[-2].startswith("run_"):
        a, b, run_dir = parts[-3], parts[-2], parts[-1]
        # 3-level layout: <design>/run_NN/sva.sv  → method = root-stem
        # We're at parts = (design, run_NN, sva.sv).
        if len(parts) == 3:
            method = _method_from_root_name(root)
            return method, a
    if len(parts) >= 4 and parts[-2].startswith("run_"):
        a, b, run_dir = parts[-4], parts[-3], parts[-2]
        # 4-level layout: <X>/<Y>/run_NN/sva.sv
        if a in _KNOWN_METHODS:
            return a, b
        if b in _KNOWN_METHODS:
            return b, a
        return a, b  # best-effort: assume <method>/<design>
    return None


def _method_from_root_name(root: Path) -> str:
    """Heuristic: pull a method name out of a top-level directory name.

    `baselines/results_<TS>/` (the assertllm-only batch) → "assertllm".
    `baselines/results_all_<TS>/` (the multi-baseline batch) → "" (each
        sub-method already has its own dir).
    `naive_then_large_ablation_<TS>/` → "" (each sub-variant has its own dir).
    Anything else → the directory name itself.
    """
    name = root.name.lower()
    if name.startswith("results_") and "all" not in name:
        return "assertllm"
    if name.startswith("results_all_"):
        return ""
    if "ablation" in name:
        return ""
    return root.name


def iter_runs(roots: Iterable[Path], methods_filter: Optional[Set[str]]) -> Iterable[RunRow]:
    for root in roots:
        if not root.exists():
            continue
        for sva in root.rglob("sva.sv"):
            try:
                rel = sva.relative_to(root)
            except ValueError:
                continue
            parts = rel.parts
            if len(parts) < 3:
                continue
            inferred = _infer_method_design(root, parts)
            if inferred is None:
                continue
            method, design = inferred
            if not method:
                continue
            if methods_filter and method not in methods_filter:
                continue

            run_id = parts[-2][4:] if parts[-2].startswith("run_") else parts[-2]
            metrics = analyse_file(sva)
            summary = sva.parent / "token_summary.json"
            tokens, elapsed = load_token_summary(summary)
            yield RunRow(
                method=method, design=design, run=run_id,
                sva_path=sva,
                summary_path=summary if summary.exists() else None,
                metrics=metrics,
                tokens=tokens,
                elapsed=elapsed,
            )


# ---------------------------------------------------------------------------
# Cell + grand aggregation
# ---------------------------------------------------------------------------

@dataclass
class CellSummary:
    method: str
    design: str
    n_runs: int
    asserts_mean: float
    unique_mean: float
    complex_unique_mean: float
    domains_mean: float
    quality_unit_mean: float
    tokens_mean: float
    tokens_per_assert: Optional[float]
    tokens_per_unique: Optional[float]
    tokens_per_complex: Optional[float]
    tokens_per_quality: Optional[float]
    trivial_share: float
    malformed_total: int


def _safe_div(num: float, den: float) -> Optional[float]:
    return num / den if den > 0 else None


def aggregate(rows: List[RunRow]) -> List[CellSummary]:
    by_cell: Dict[Tuple[str, str], List[RunRow]] = defaultdict(list)
    for r in rows:
        by_cell[(r.method, r.design)].append(r)
    out: List[CellSummary] = []
    for (m, d), cell in sorted(by_cell.items()):
        n = len(cell)
        asserts = [r.metrics.n_total for r in cell]
        uniques = [r.metrics.n_unique for r in cell]
        complexes = [r.metrics.n_complex_unique for r in cell]
        domains = [r.metrics.n_domains for r in cell]
        quality = [r.metrics.quality_unit for r in cell]
        tokens = [r.tokens for r in cell if r.tokens > 0]
        trivial = sum(r.metrics.n_trivial for r in cell)
        total = sum(asserts) or 1
        a_mean = statistics.mean(asserts)
        u_mean = statistics.mean(uniques)
        c_mean = statistics.mean(complexes)
        d_mean = statistics.mean(domains)
        q_mean = statistics.mean(quality)
        t_mean = statistics.mean(tokens) if tokens else 0.0
        out.append(CellSummary(
            method=m, design=d, n_runs=n,
            asserts_mean=a_mean,
            unique_mean=u_mean,
            complex_unique_mean=c_mean,
            domains_mean=d_mean,
            quality_unit_mean=q_mean,
            tokens_mean=t_mean,
            tokens_per_assert=_safe_div(t_mean, a_mean),
            tokens_per_unique=_safe_div(t_mean, u_mean),
            tokens_per_complex=_safe_div(t_mean, c_mean),
            tokens_per_quality=_safe_div(t_mean, q_mean),
            trivial_share=trivial / total,
            malformed_total=sum(r.metrics.n_malformed for r in cell),
        ))
    return out


def grand_aggregate(cells: List[CellSummary]) -> List[Dict]:
    by_method: Dict[str, List[CellSummary]] = defaultdict(list)
    for c in cells:
        by_method[c.method].append(c)
    out = []
    for m, cells_m in sorted(by_method.items()):
        n_designs = len(cells_m)
        # Average per-design means (so big designs don't dominate).
        avg = lambda key: statistics.mean(getattr(c, key) for c in cells_m if getattr(c, key) is not None)
        avg_opt = lambda key: (
            statistics.mean(getattr(c, key) for c in cells_m if getattr(c, key) is not None)
            if any(getattr(c, key) is not None for c in cells_m) else None
        )
        out.append({
            "method": m,
            "n_designs": n_designs,
            "asserts_mean": avg("asserts_mean"),
            "complex_unique_mean": avg("complex_unique_mean"),
            "domains_mean": avg("domains_mean"),
            "tokens_mean": avg("tokens_mean"),
            "tokens_per_assert": avg_opt("tokens_per_assert"),
            "tokens_per_complex": avg_opt("tokens_per_complex"),
            "tokens_per_quality": avg_opt("tokens_per_quality"),
            "trivial_share": avg("trivial_share"),
            "malformed_per_design": statistics.mean(c.malformed_total for c in cells_m),
        })
    return out


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------

def render_markdown(cells: List[CellSummary], grand: List[Dict]) -> str:
    out: List[str] = []
    out.append("# Quality-Weighted Cost Comparison\n")
    out.append("Metric definitions:")
    out.append("- **A** = total assertions in the file")
    out.append("- **U** = unique (post-dedup)")
    out.append("- **CU** = complex AND unique (uses `$past` / `$stable` / `$rose` / `$fell` / `$onehot` / `inside` / `|=>` / multi-signal antecedent; not a bare `$bits()` width check)")
    out.append("- **D** = distinct functional domains in the file")
    out.append("- **Q** = CU × D — composite quality unit (penalises trivial inflation, duplicate copies, single-domain output)")
    out.append("- **tok/Q** = total_tokens / Q — *tokens per usable, complex, diverse assertion-unit*. Lower is better.\n")

    out.append("## Headline (per method, mean across designs)\n")
    out.append("| Method | Designs | Tokens μ | A μ | CU μ | D μ | tok/A | tok/CU | **tok/Q** | Trivial share | Malformed/design |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for g in grand:
        out.append("| {method} | {n_designs} | {t:,.0f} | {a:.1f} | {cu:.1f} | {d:.1f} | {ta} | {tcu} | **{tq}** | {ts:.0%} | {m:.1f} |".format(
            method=g["method"],
            n_designs=g["n_designs"],
            t=g["tokens_mean"],
            a=g["asserts_mean"],
            cu=g["complex_unique_mean"],
            d=g["domains_mean"],
            ta="—" if g["tokens_per_assert"] is None else f"{g['tokens_per_assert']:,.0f}",
            tcu="—" if g["tokens_per_complex"] is None else f"{g['tokens_per_complex']:,.0f}",
            tq="—" if g["tokens_per_quality"] is None else f"{g['tokens_per_quality']:,.0f}",
            ts=g["trivial_share"],
            m=g["malformed_per_design"],
        ))

    out.append("\n## Per (method, design) cell\n")
    out.append("| Method | Design | Runs | Tokens μ | A μ | U μ | CU μ | D μ | tok/A | tok/CU | **tok/Q** | Trivial | Malf |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in cells:
        out.append("| {m} | {d} | {n} | {t:,.0f} | {a:.1f} | {u:.1f} | {cu:.1f} | {dm:.1f} | {ta} | {tcu} | **{tq}** | {ts:.0%} | {mf} |".format(
            m=c.method, d=c.design, n=c.n_runs,
            t=c.tokens_mean, a=c.asserts_mean, u=c.unique_mean,
            cu=c.complex_unique_mean, dm=c.domains_mean,
            ta="—" if c.tokens_per_assert is None else f"{c.tokens_per_assert:,.0f}",
            tcu="—" if c.tokens_per_complex is None else f"{c.tokens_per_complex:,.0f}",
            tq="—" if c.tokens_per_quality is None else f"{c.tokens_per_quality:,.0f}",
            ts=c.trivial_share, mf=c.malformed_total,
        ))
    return "\n".join(out) + "\n"


def render_csv(cells: List[CellSummary]) -> str:
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "method", "design", "n_runs",
        "tokens_mean", "asserts_mean", "unique_mean",
        "complex_unique_mean", "domains_mean", "quality_unit_mean",
        "tokens_per_assert", "tokens_per_unique",
        "tokens_per_complex", "tokens_per_quality",
        "trivial_share", "malformed_total",
    ])
    for c in cells:
        w.writerow([
            c.method, c.design, c.n_runs,
            f"{c.tokens_mean:.1f}",
            f"{c.asserts_mean:.2f}", f"{c.unique_mean:.2f}",
            f"{c.complex_unique_mean:.2f}", f"{c.domains_mean:.2f}",
            f"{c.quality_unit_mean:.2f}",
            "" if c.tokens_per_assert is None else f"{c.tokens_per_assert:.1f}",
            "" if c.tokens_per_unique is None else f"{c.tokens_per_unique:.1f}",
            "" if c.tokens_per_complex is None else f"{c.tokens_per_complex:.1f}",
            "" if c.tokens_per_quality is None else f"{c.tokens_per_quality:.1f}",
            f"{c.trivial_share:.4f}",
            c.malformed_total,
        ])
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dirs", nargs="+", required=True,
                    type=Path,
                    help="One or more results directories to scan.")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="Filter to these method/variant names.")
    ap.add_argument("--output-md", type=Path, default=None,
                    help="Write the markdown report to this path.")
    ap.add_argument("--output-csv", type=Path, default=None,
                    help="Write the per-cell CSV to this path.")
    ap.add_argument("--print-stdout", action="store_true",
                    help="Also print the markdown to stdout.")
    args = ap.parse_args()

    methods_filter = set(args.methods) if args.methods else None
    rows = list(iter_runs(args.results_dirs, methods_filter))
    if not rows:
        print("No SVA files found under the supplied results directories.", file=sys.stderr)
        return 1

    cells = aggregate(rows)
    grand = grand_aggregate(cells)

    md = render_markdown(cells, grand)
    csv_text = render_csv(cells)

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md)
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        args.output_csv.write_text(csv_text)
    if args.print_stdout or (not args.output_md and not args.output_csv):
        sys.stdout.write(md)

    print(f"\n[summary] {len(rows)} runs, {len(cells)} (method, design) cells, "
          f"{len(grand)} methods.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
