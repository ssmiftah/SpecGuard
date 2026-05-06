#!/usr/bin/env python3
"""
specguard_quality_metric.py
---------------------------

Quality + cost framework for comparing SpecGuard against baselines.

Per assertion we compute four boolean axes (Tier 1, "structural"):

    lint_pass   pyslang accepts the syntax (UndeclaredIdentifier ignored).
    resolved    every identifier in the assertion is in the design's
                signal_map (no hallucinations).
    non_trivial not a bare $bits()==const, not a `1'b1` body.
    unique      not a string-normalised duplicate of an earlier assertion.

T1 = lint_pass AND resolved AND non_trivial AND unique.

We also compute Tier 2 (coverage of documented intent):

    coverage_T2 = (# documented properties matched by some T1 assertion)
                  / (# documented properties in the design's docs).

Documented properties are extracted from `docs_dir` via
`sva_pipeline.rtl_facts._extract_documented_properties` — the same code
the pipeline uses to inject doc-property hints into prompts. A docprop
is "covered" when its signal set is a subset of some T1 assertion's
identifier set.

Headline metric:

    useful_score = n_T1 + alpha * n_docs_covered      (alpha=1)
    cost_useful  = total_tokens / useful_score        (lower is better)

`cost_per_T1 = total_tokens / n_T1` is reported alongside as a
diagnostic. Raw tokens, n_T1, and n_docs_covered are always reported
so the headline can be sanity-checked.

USAGE
-----
    python scripts/specguard_quality_metric.py \\
        --results-dir specguard_full_5rep_20260505_1943:specguard \\
        --results-dir baselines/results_all_20260505_1239:auto \\
        --output-md docs/specguard_quality.md \\
        --output-csv docs/specguard_quality.csv

Each `--results-dir` takes one of:
    PATH                — auto-detect method/design layout
    PATH:LABEL          — force the method label (e.g. ``:specguard``)
    PATH:auto           — same as no label

Layouts recognised (rooted at PATH):
    <method>/<design>/run_NN/sva.sv     — baselines/results_all_*
    <design>/run_NN/sva.sv              — single-method dir (label required
                                          or inferred from PATH name)

Each `sva.sv` must have a sibling `token_summary.json`. The metric reads
the `total_tokens`, `prompt_tokens`, `completion_tokens` fields from it.
The `config` column from a sibling `all_runs.csv` is used to find the
design's RTL/docs paths so we can extract the signal map and
documented-property set.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Suppress pipeline INFO logs while we batch-extract facts; they pollute stdout.
logging.basicConfig(level=logging.WARNING)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import yaml  # noqa: E402

from sva_pipeline.lint_loop import (  # noqa: E402
    split_assertions, lint_single_assertion,
)
from sva_pipeline.rtl_facts import (  # noqa: E402
    extract_rtl_facts, _extract_documented_properties,
)
from sva_pipeline.ast_assertions import extract_patterns  # noqa: E402


# =============================================================================
# Config + design fact extraction
# =============================================================================

def _resolve_path(p: str) -> str:
    """Interpret YAML-relative paths against the project root."""
    if not p:
        return p
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str(_PROJECT_ROOT / pp)


def load_design_paths(yaml_path: Path) -> Optional[Tuple[str, str, str]]:
    """Return (rtl_dir, top_module, docs_dir) extracted from a pipeline
    YAML. None on read failure or missing required keys."""
    if not yaml_path.exists():
        return None
    try:
        cfg = yaml.safe_load(yaml_path.read_text()) or {}
    except yaml.YAMLError:
        return None
    d = cfg.get("design", {}) or {}
    rtl = _resolve_path(d.get("rtl_dir", ""))
    top = d.get("top_module", "") or ""
    docs = _resolve_path(d.get("docs_dir", "") or "")
    if not rtl:
        return None
    return rtl, top, docs


@dataclass
class CoverageEvent:
    """A single point of design behaviour the assertion suite is
    expected to cover.  Each event carries a 'kind' (reset / doc /
    case_branch / fsm / handshake / mux), a stable key (for dedup
    across methods), and the set of identifiers an assertion must
    reference to count as covering it.
    """
    kind: str
    key: str
    required_signals: frozenset


@dataclass
class DesignFacts:
    rtl_dir: str
    top_module: str
    docs_dir: str
    signal_set: Set[str] = field(default_factory=set)
    documented_properties: List[Dict] = field(default_factory=list)
    coverage_events: List[CoverageEvent] = field(default_factory=list)


_FACTS_CACHE: Dict[Tuple[str, str], Optional[DesignFacts]] = {}
_LINT_CACHE: Dict[str, bool] = {}


def get_design_facts(yaml_path: Path) -> Optional[DesignFacts]:
    """Cached extraction of (signal_set, documented_properties) for a
    design — keyed on (rtl_dir, top_module). Multiple runs of the same
    design + method (and across methods) share a single cache entry."""
    paths = load_design_paths(yaml_path)
    if not paths:
        return None
    rtl, top, docs = paths
    key = (rtl, top)
    if key in _FACTS_CACHE:
        return _FACTS_CACHE[key]
    facts = None
    try:
        facts = extract_rtl_facts(rtl, top_module=top, docs_dir=docs)
    except Exception as exc:
        sys.stderr.write(
            f"[warn] facts extraction failed for {rtl} (top={top}): {exc}\n"
        )
        _FACTS_CACHE[key] = None
        return None
    doc_props = _extract_documented_properties(docs, facts) if docs else []
    # Merge clock + reset names into the resolution set; pyslang tracks
    # them in separate fields, but they're real signals from the
    # assertion's perspective.  Without this, every assertion that
    # references a clock or reset is flagged as hallucination-positive.
    sig = set(facts.all_signals) \
        | set(getattr(facts, "clock_signals", set()) or set()) \
        | set(getattr(facts, "reset_signals", set()) or set()) \
        | set((getattr(facts, "signal_widths", {}) or {}).keys()) \
        | set((getattr(facts, "signal_frequencies", {}) or {}).keys())
    # Note: we do NOT whitelist loop / generate index variables (`i`,
    # `j`, etc.). Outside a generate block they're genuinely
    # unresolved — the metric reflects what the post-processor's
    # validate_loop_var_scope detector tries to catch.

    # Re-run AST pattern extraction to enumerate case-branch coverage
    # events.  Pyslang doesn't surface case branches as a structured
    # field on RTLFacts, but the regex AST extractor does.
    patterns: List = []
    if rtl and Path(rtl).exists():
        rtl_text_parts: List[str] = []
        for root, _, files in __import__("os").walk(rtl):
            for fn in sorted(files):
                if not fn.endswith((".v", ".sv")):
                    continue
                fpath = Path(root) / fn
                try:
                    rtl_text_parts.append(
                        fpath.read_text(errors="ignore")
                    )
                except OSError:
                    pass
        if rtl_text_parts:
            try:
                patterns = extract_patterns(
                    "\n\n".join(rtl_text_parts),
                    clock=next(iter(getattr(facts, "clock_signals", []) or []),
                               None),
                    reset=next(iter(getattr(facts, "reset_signals", []) or []),
                               None),
                )
            except Exception as exc:
                sys.stderr.write(
                    f"[warn] AST extraction failed for {rtl}: {exc}\n"
                )

    coverage = _extract_coverage_events(facts, doc_props, patterns)

    df = DesignFacts(
        rtl_dir=rtl, top_module=top, docs_dir=docs,
        signal_set=sig,
        documented_properties=doc_props,
        coverage_events=coverage,
    )
    _FACTS_CACHE[key] = df
    return df


def _extract_coverage_events(
    facts,
    doc_props: List[Dict],
    patterns: List,
) -> List[CoverageEvent]:
    """Enumerate the design's coverage points across three axes:

    * **reset** — one event per resettable register.  An assertion
      covers it by referencing the register name (regardless of
      whether the rhs matches; the structural metric already filters
      lint-clean assertions).
    * **doc-property** — one event per documented sentence with at
      least one resolvable signal.  An assertion covers it when its
      identifier set overlaps the docprop's signal set above the T2
      coverage threshold.
    * **case-branch** — one event per (case_selector, case_value, lhs)
      triple from the AST extractor.  An assertion covers it by
      referencing the lhs AND mentioning the value (literally — the
      branch literal must appear in the assertion text).
    """
    events: List[CoverageEvent] = []
    seen_keys: Set[str] = set()

    # Axis 1 — reset events.
    # required_signals is just {reg} on purpose: the reset signal is
    # SHARED across every reset event in a design (e.g. all 843 cdp
    # registers reset by the same `nvdla_core_rstn`).  If we kept
    # rst_sig in the required set, the >=50% overlap rule would let a
    # single assertion that merely *mentions* the reset signal claim
    # coverage of every reset event in the design — ChIRAAG was hitting
    # "843 / 843 reset events on cdp" with a single T1 assertion that
    # said nothing about any particular register. By requiring the
    # register name itself, the event is hit only when the assertion
    # actually constrains that register.
    for reg, (rst_sig, rst_val) in (
            getattr(facts, "reset_values", {}) or {}).items():
        key = f"reset::{reg}::{rst_val}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        events.append(CoverageEvent(
            kind="reset", key=key, required_signals=frozenset({reg}),
        ))

    # Axis 2 — documented properties (mirrors the T2 metric).
    for i, p in enumerate(doc_props):
        signals = tuple(sorted(p.get("signals", []) or []))
        if not signals:
            continue
        key = f"doc::{p.get('source_file','?')}::{p.get('source_line','?')}::{i}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        events.append(CoverageEvent(
            kind="doc", key=key, required_signals=frozenset(signals),
        ))

    # Axis 3 — case-branch events from the AST extractor.
    for p in patterns:
        if getattr(p, "pattern_type", "") != "case_branch":
            continue
        sel = getattr(p, "selector", None) or ""
        cond = getattr(p, "condition", None) or ""
        lhs = getattr(p, "lhs", "") or ""
        if not (sel and lhs and cond):
            continue
        key = f"case::{lhs}::{sel}::{cond}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        # Required: the LHS appears + the case value literal appears.
        # Extract the bare identifier from the selector if it's a
        # simple name (otherwise leave the full text and rely on
        # substring matching downstream).
        sel_ids = set(re.findall(r"\b[A-Za-z_]\w*\b", sel))
        required = {lhs} | (sel_ids & {x for x in sel_ids
                                       if not x.lower() in _RTL_KW})
        events.append(CoverageEvent(
            kind="case", key=key, required_signals=frozenset(required),
        ))

    return events


def assertion_covers_event(
    event: CoverageEvent,
    assertion_ids: Set[str],
    assertion_text_lower: str,
    threshold: float = 0.5,
) -> bool:
    """Decide whether one assertion covers one coverage event.

    Generic rule: the assertion's identifier set overlaps the event's
    required signals at >= ``threshold`` fraction (with at least one
    shared identifier).  For ``case`` events we additionally require
    the case-value literal to appear textually in the assertion (so
    a generic "lhs is checked somewhere" doesn't trivially count).
    """
    target = event.required_signals
    if not target:
        return False
    overlap = len(target & assertion_ids)
    need = max(1, int(len(target) * threshold + 0.999))
    if overlap < need:
        return False
    if event.kind == "case":
        # Check the case-value literal is present textually.
        # Key format: "case::{lhs}::{selector}::{value}"
        try:
            value = event.key.split("::", 3)[3]
        except IndexError:
            return True
        # Loose match — value strings often contain typed literals
        # ("4'b0001"); strip whitespace and lowercase before contains.
        v_norm = re.sub(r"\s+", "", value).lower()
        if v_norm and v_norm not in re.sub(r"\s+", "", assertion_text_lower):
            return False
    return True


# =============================================================================
# Per-assertion classifiers
# =============================================================================

_RTL_KW = {
    "assert", "property", "endproperty", "disable", "iff", "posedge",
    "negedge", "else", "and", "or", "not", "if", "logic", "wire", "reg",
    "input", "output", "always_ff", "always_comb", "module", "endmodule",
    "begin", "end", "b", "h", "d", "x", "z", "inside", "throughout",
    "until", "first_match", "signed", "unsigned", "null", "void",
    "true", "false",
}


def assertion_identifiers(line: str) -> Set[str]:
    """Tokenise an assertion and return non-keyword identifiers."""
    s = re.sub(r'"[^"]*"', "", line)            # drop string literals
    s = re.sub(r"//.*", "", s)                   # drop line comments
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    s = re.sub(r"\$\w+", "", s)                  # drop $-builtins
    s = re.sub(r"\d+'[bdhoBDHO][\w?xzXZ]+", "", s)  # drop typed literals
    s = re.sub(r"\b\d+\b", "", s)                # drop bare numbers
    toks = re.findall(r"\b[A-Za-z_]\w*\b", s)
    return {t for t in toks if t.lower() not in _RTL_KW}


def has_unresolved_signals(line: str, signal_set: Set[str]) -> bool:
    """True when at least one identifier in `line` isn't in `signal_set`.

    When the design's signal_set is empty (extraction failed), we cannot
    judge resolution, so we return False (treat as resolved).
    """
    if not signal_set:
        return False
    ids = assertion_identifiers(line)
    return any(t not in signal_set for t in ids)


_TRIVIAL_BITS = re.compile(
    r"\$bits\s*\(\s*\w+(?:\.\w+|\[[^\]]+\])*\s*\)\s*==\s*\d"
)
_VACUOUS_TRUE = re.compile(
    r"^\s*assert\s*(?:property\s*)?\(?\s*1'b1\s*\)?\s*\)?\s*;",
    re.IGNORECASE,
)


def is_trivial(line: str) -> bool:
    """Detect bare width-equality and `assert (1'b1)`-style vacuity."""
    L = line.strip()
    if _TRIVIAL_BITS.search(L) and "&&" not in L and "||" not in L \
            and "|->" not in L and "|=>" not in L:
        return True
    if _VACUOUS_TRUE.search(L):
        return True
    return False


_NORM_WS = re.compile(r"\s+")


def normalise(line: str) -> str:
    """Whitespace-collapsed, lowercased, comment-stripped form for
    string-dedup. The error message is canonicalised so two assertions
    that differ only in the $error string are still detected as dups."""
    s = re.sub(r"//.*", "", line)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    s = re.sub(r"\$error\s*\([^)]*\)", "$error(...)", s)
    s = re.sub(r"\$display\s*\([^)]*\)", "$display(...)", s)
    return _NORM_WS.sub(" ", s).strip().lower()


# =============================================================================
# Complexity classifier (T1_simple vs T1_complex)
# =============================================================================

# Anything in this set marks an assertion as "complex" — uses temporal
# operators, set membership, or other constructs beyond bare equality.
_COMPLEX_DOLLAR = re.compile(
    r"\$(past|stable|rose|fell|onehot|onehot0|countones|isunknown|changed)\b"
)
_COMPLEX_KEYWORDS = re.compile(
    r"\b(inside|throughout|until|first_match)\b"
)


def _antecedent_signal_count(line: str) -> int:
    """Best-effort count of distinct signals in the antecedent of an
    implication (the LHS of |-> or |=>). Returns 0 when there's no
    implication operator. Used to flag multi-signal-condition
    assertions as complex even without a $-builtin or temporal op.
    """
    m = re.search(r"(.+?)\s*\|[->=]+\s*", line)
    if not m:
        return 0
    body = m.group(1)
    body = re.sub(r'"[^"]*"', "", body)
    body = re.sub(r"\$\w+", "", body)
    body = re.sub(r"\d+'[bdho][\w?xz]+", "", body, flags=re.IGNORECASE)
    body = re.sub(r"@\s*\([^)]*\)", "", body)
    body = re.sub(r"\bdisable\s+iff\s*\([^)]*\)", "", body)
    toks = re.findall(r"\b[A-Za-z_]\w*\b", body)
    sigs = {t for t in toks if t.lower() not in _RTL_KW and not t.isdigit()}
    return len(sigs)


def is_complex(line: str) -> bool:
    """An assertion is complex when it uses a temporal/set/onehot
    operator, an implication (|-> or |=>), or a multi-signal antecedent
    (≥2 distinct signals before the implication). Bare equalities
    against constants are simple."""
    L = line.strip()
    if _COMPLEX_DOLLAR.search(L):
        return True
    if _COMPLEX_KEYWORDS.search(L):
        return True
    if "|=>" in L:
        return True
    if "|->" in L and _antecedent_signal_count(L) >= 2:
        return True
    return False


# =============================================================================
# T2 coverage scoring
# =============================================================================

# Default fraction of a docprop's signals that must appear in some
# T1 assertion's identifier set for the docprop to count as covered.
# 1.0 = strict subset (the original rule); 0.5 = "majority of the
# signals are referenced". Empirically, strict subset under-reports
# coverage on designs whose docs reference compound signal sets that
# the LLM splits across several assertions; 0.5 catches those without
# being so loose that a single shared signal triggers a match.
COVERAGE_OVERLAP_THRESHOLD = 0.5


def coverage_count(t1_assertions: List[str],
                   doc_props: List[Dict],
                   threshold: float = COVERAGE_OVERLAP_THRESHOLD) -> int:
    """Count documented properties matched by ≥1 T1 assertion.

    A docprop is matched when some T1 assertion's identifier set
    contains at least ``threshold`` fraction of the docprop's target
    signals (and at least one signal — empty-signal docprops are
    skipped). At threshold=1.0 this reduces to the strict-subset rule.
    """
    if not doc_props:
        return 0
    assertion_ids = [assertion_identifiers(a) for a in t1_assertions]
    covered = 0
    for p in doc_props:
        target = set(p.get("signals", []) or [])
        if not target:
            continue
        # Floor: at least one shared signal AND >= threshold fraction.
        need = max(1, int(len(target) * threshold + 0.999))  # ceil
        for ids in assertion_ids:
            overlap = len(target & ids)
            if overlap >= need:
                covered += 1
                break
    return covered


# =============================================================================
# Property body inlining (for `assert property (NAME)` references)
# =============================================================================

# `property NAME; ... endproperty` → captured so that bare
# `assert property (NAME)` references can be evaluated against the
# real assertion body for resolution + complexity scoring.
_PROPERTY_BLOCK_RE = re.compile(
    r"\bproperty\s+([A-Za-z_]\w*)\s*[;]\s*(.*?)\s*\bendproperty\b",
    re.DOTALL | re.IGNORECASE,
)
_ASSERT_PROP_REF_RE = re.compile(
    r"\bassert\s+property\s*\(\s*([A-Za-z_]\w*)\s*\)\s*;?",
    re.IGNORECASE,
)


def _extract_property_bodies(text: str) -> Dict[str, str]:
    """Return {property_name -> body_text} for every property block."""
    out: Dict[str, str] = {}
    for m in _PROPERTY_BLOCK_RE.finditer(text):
        name, body = m.group(1), m.group(2).strip()
        out[name] = body
    return out


def _expand_assertion(line: str, prop_bodies: Dict[str, str]) -> str:
    """If `line` is a bare `assert property (NAME);`, splice the
    body of the named property into the line so downstream lint /
    resolution / complexity scorers see the real assertion text.
    No-op when the assertion already contains its body inline."""
    m = _ASSERT_PROP_REF_RE.search(line)
    if not m:
        return line
    body = prop_bodies.get(m.group(1))
    if not body:
        return line
    flat_body = re.sub(r"\s+", " ", body).strip().rstrip(";")
    return f"assert property ({flat_body});"


# =============================================================================
# Per-file analyser
# =============================================================================

@dataclass
class FileMetrics:
    n_total: int = 0
    n_lint_pass: int = 0
    n_resolved: int = 0
    n_non_trivial: int = 0
    n_unique: int = 0
    n_t1: int = 0
    n_t1_simple: int = 0   # T1 assertions with no temporal / set / multi-sig features
    n_t1_complex: int = 0  # T1 assertions that use $past/$stable/inside/|->/|=>/multi-sig antecedent
    n_docs_total: int = 0
    n_docs_covered: int = 0
    # Functional coverage — counted per axis, plus a union total.
    n_events_total: int = 0
    n_events_hit: int = 0
    n_reset_total: int = 0
    n_reset_hit: int = 0
    n_doc_total: int = 0
    n_doc_hit: int = 0
    n_case_total: int = 0
    n_case_hit: int = 0


def analyse_file(sva_path: Path,
                 df: Optional[DesignFacts]) -> FileMetrics:
    fm = FileMetrics()
    # events_total is design-fixed — set it up front so empty / missing
    # SVA files still report the correct denominator (otherwise per-cell
    # mean(ev_tot) under-counts, since methods that emit zero-assertion
    # files would carry ev_tot=0 and pull the cell average below the
    # design's actual event count).
    if df is not None:
        events = df.coverage_events or []
        fm.n_events_total = len(events)
        fm.n_reset_total = sum(1 for e in events if e.kind == "reset")
        fm.n_doc_total   = sum(1 for e in events if e.kind == "doc")
        fm.n_case_total  = sum(1 for e in events if e.kind == "case")
        fm.n_docs_total  = len(df.documented_properties)
    if not sva_path.exists():
        return fm
    text = sva_path.read_text(errors="ignore")
    # Pre-scan for `property NAME; ... endproperty` blocks so we can
    # inline them into bare `assert property (NAME)` references.
    # Without this, ChiRAAG / AssertLLM-style outputs (which separate
    # the property declaration from the assert) would all fail
    # resolution because the assert line only mentions the property
    # name, not the underlying signals.
    prop_bodies = _extract_property_bodies(text)
    entries = split_assertions(text)
    if not entries:
        return fm
    fm.n_total = len(entries)

    seen_norm: Set[str] = set()
    t1_lines: List[str] = []
    signal_set = df.signal_set if df else set()

    for entry in entries:
        # Inline the property body before scoring (no-op when the
        # assertion already contains its body inline).
        a = _expand_assertion(entry["assertion"], prop_bodies)
        # Lint (pyslang already filters UndeclaredIdentifier).
        # Memoised to avoid recompiling identical or near-identical
        # assertions (LUT-style outputs can have hundreds of clones
        # that lint to the same result).
        lint_key = a.strip()
        if lint_key in _LINT_CACHE:
            lint_pass = _LINT_CACHE[lint_key]
        else:
            try:
                lint_pass = lint_single_assertion(
                    a, reject_assert_property=False
                )["status"] == "PASS"
            except Exception:
                lint_pass = False
            _LINT_CACHE[lint_key] = lint_pass

        resolved = not has_unresolved_signals(a, signal_set)
        non_trivial = not is_trivial(a)

        norm = normalise(a)
        unique = norm not in seen_norm
        seen_norm.add(norm)

        if lint_pass:
            fm.n_lint_pass += 1
        if resolved:
            fm.n_resolved += 1
        if non_trivial:
            fm.n_non_trivial += 1
        if unique:
            fm.n_unique += 1
        if lint_pass and resolved and non_trivial and unique:
            fm.n_t1 += 1
            t1_lines.append(a)
            if is_complex(a):
                fm.n_t1_complex += 1
            else:
                fm.n_t1_simple += 1

    if df:
        fm.n_docs_total = len(df.documented_properties)
        fm.n_docs_covered = coverage_count(t1_lines, df.documented_properties)

        # --- Functional coverage (3-axis: reset / doc / case-branch) ---
        # For each event, check whether ANY T1 assertion's identifier
        # set covers it.  Per-axis counters fall out of the union for
        # downstream reporting.
        events = df.coverage_events or []
        fm.n_events_total = len(events)
        fm.n_reset_total = sum(1 for e in events if e.kind == "reset")
        fm.n_doc_total = sum(1 for e in events if e.kind == "doc")
        fm.n_case_total = sum(1 for e in events if e.kind == "case")

        if events and t1_lines:
            t1_id_sets = [assertion_identifiers(a) for a in t1_lines]
            t1_lower = [a.lower() for a in t1_lines]
            for event in events:
                hit = False
                for ids, txt_lower in zip(t1_id_sets, t1_lower):
                    if assertion_covers_event(event, ids, txt_lower):
                        hit = True
                        break
                if hit:
                    fm.n_events_hit += 1
                    if event.kind == "reset":
                        fm.n_reset_hit += 1
                    elif event.kind == "doc":
                        fm.n_doc_hit += 1
                    elif event.kind == "case":
                        fm.n_case_hit += 1

    return fm


def _normalise_design_name(d: str) -> str:
    """Map baseline-prefixed names ("nvdla_cmac", "coral_fifox") to the
    bare form used by SpecGuard's matrix ("cmac", "fifox"). Unknown
    prefixes are returned as-is."""
    for pfx in ("nvdla_", "coral_"):
        if d.startswith(pfx):
            return d[len(pfx):]
    # Special case: chisel/coral RVV is named "rvv_backend" in some
    # configs and "rvv" in others. Collapse to "rvv".
    if d in ("rvv_backend",):
        return "rvv"
    return d


# =============================================================================
# Run discovery
# =============================================================================

@dataclass
class RunRow:
    method: str
    design: str
    run: str
    sva_path: Path
    metrics: FileMetrics
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    elapsed_sec: float = 0.0


def _load_token_summary(p: Path) -> Tuple[int, int, int, int, float]:
    """Return (total, prompt, completion, llm_calls, elapsed_sec)."""
    if not p.exists():
        return 0, 0, 0, 0, 0.0
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return 0, 0, 0, 0, 0.0
    return (
        int(d.get("total_tokens", 0) or 0),
        int(d.get("prompt_tokens", 0) or 0),
        int(d.get("completion_tokens", 0) or 0),
        int(d.get("llm_calls", 0) or 0),
        float(d.get("elapsed_sec", 0) or 0),
    )


def _read_all_runs_csv(root: Path) -> Dict[Tuple[str, str, str], str]:
    """Return {(method_or_variant, design, run) -> config_path} from
    `all_runs.csv` if present. Used to look up the YAML for each run.
    """
    out: Dict[Tuple[str, str, str], str] = {}
    csv_path = root / "all_runs.csv"
    if not csv_path.exists():
        return out
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            design = row.get("design", "")
            raw_run = (row.get("run", "") or "")
            cfg = row.get("config", "")
            method = row.get("variant", "") or row.get("method", "")
            if not (design and cfg):
                continue
            # Index under every plausible run-id form because callers
            # may query with the raw form ("01") OR the lstripped form
            # ("1") depending on whether they got it from the filesystem
            # (zero-padded) or from the CSV (sometimes not).
            run_forms = {raw_run, raw_run.lstrip("0") or "0"}
            for rf in run_forms:
                out[(method, design, rf)] = cfg
                out[("", design, rf)] = cfg  # method-agnostic fallback
    return out


# 4-level: <root>/<method>/<design>/run_NN/sva.sv
# 3-level: <root>/<design>/run_NN/sva.sv
def _infer_layout(rel_parts: Tuple[str, ...]) -> Optional[Tuple[str, str, str]]:
    """Return (method_or_None, design, run) from path parts under root.
    method can be None for 3-level layouts (caller must supply label).
    """
    if len(rel_parts) >= 3 and rel_parts[-2].startswith("run_") \
            and rel_parts[-1] == "sva.sv":
        run = rel_parts[-2][4:]
        if len(rel_parts) >= 4:
            method = rel_parts[-4]
            design = rel_parts[-3]
            return method, design, run
        return None, rel_parts[-3], run
    return None


def discover_runs(root: Path,
                  forced_label: Optional[str]) -> List[RunRow]:
    """Walk `root` for `sva.sv` files and produce one RunRow each."""
    rows: List[RunRow] = []
    if not root.exists():
        return rows
    csv_lookup = _read_all_runs_csv(root)
    sva_files = sorted(root.rglob("sva.sv"))
    sys.stderr.write(f"[scan]   {len(sva_files)} sva.sv files under {root}\n")
    sys.stderr.flush()
    import time
    t_start = time.monotonic()
    for i, sva in enumerate(sva_files, 1):
        t_file_start = time.monotonic()
        try:
            rel = sva.relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        layout = _infer_layout(parts)
        if layout is None:
            continue
        inferred_method, design, run = layout
        method = forced_label or inferred_method
        if not method:
            method = root.name
        # Look up YAML config from all_runs.csv
        cfg_str = (
            csv_lookup.get((inferred_method or "", design, run))
            or csv_lookup.get(("", design, run))
            or ""
        )
        cfg_path = (_PROJECT_ROOT / cfg_str) if cfg_str else None
        df = get_design_facts(cfg_path) if cfg_path else None
        # Per-file analysis
        metrics = analyse_file(sva, df)
        # Token summary
        tot, pt, ct, calls, elapsed = _load_token_summary(
            sva.parent / "token_summary.json"
        )
        rows.append(RunRow(
            method=method, design=_normalise_design_name(design), run=run,
            sva_path=sva, metrics=metrics,
            total_tokens=tot, prompt_tokens=pt,
            completion_tokens=ct, llm_calls=calls,
            elapsed_sec=elapsed,
        ))
        sys.stderr.write(
            f"[scan]   ({i}/{len(sva_files)}) {method}/{design}/run_{run} "
            f"asserts={metrics.n_total} t1={metrics.n_t1} "
            f"docs={metrics.n_docs_covered}/{metrics.n_docs_total} "
            f"({time.monotonic()-t_file_start:.1f}s)\n"
        )
        sys.stderr.flush()
    return rows


# =============================================================================
# Aggregation
# =============================================================================

ALPHA_DOC_WEIGHT = 1.0       # weight on n_docs_covered in the headline
WEIGHT_T1_COMPLEX = 2.0      # complex T1 assertions count this many simple ones
WEIGHT_T1_SIMPLE = 1.0


def _useful_score(m: FileMetrics) -> float:
    """useful = simple + 2*complex + 1*docs_covered.

    Complex assertions encode temporal / set / multi-signal intent
    that bare equalities cannot, so they count for more. Documented
    properties covered are added on top to reward suites that target
    documented intent rather than just structural shape.
    """
    return (
        WEIGHT_T1_SIMPLE * float(m.n_t1_simple)
        + WEIGHT_T1_COMPLEX * float(m.n_t1_complex)
        + ALPHA_DOC_WEIGHT * float(m.n_docs_covered)
    )


def _safe_div(num: float, den: float) -> Optional[float]:
    return num / den if den > 0 else None


@dataclass
class CellSummary:
    method: str
    design: str
    n_runs: int
    asserts_mean: float
    t1_mean: float
    t1_simple_mean: float
    t1_complex_mean: float
    complex_share: float    # n_t1_complex / n_t1, NaN if no T1
    docs_mean: float
    docs_cov_mean: float
    coverage_pct: float
    tokens_mean: float
    cost_per_T1: Optional[float]
    cost_useful: Optional[float]
    lint_pass_rate: float
    resolve_rate: float
    # Functional coverage — the new headline.
    events_total_mean: float
    events_hit_mean: float
    fc_pct: float           # events_hit / events_total
    cost_per_cov: Optional[float]   # tokens / events_hit
    reset_pct: float
    doc_pct: float
    case_pct: float


def aggregate(rows: List[RunRow]) -> List[CellSummary]:
    by_cell: Dict[Tuple[str, str], List[RunRow]] = defaultdict(list)
    for r in rows:
        by_cell[(r.method, r.design)].append(r)
    out: List[CellSummary] = []
    for (m, d), cell in sorted(by_cell.items()):
        n = len(cell)
        asserts = [r.metrics.n_total for r in cell]
        t1s = [r.metrics.n_t1 for r in cell]
        t1_simple = [r.metrics.n_t1_simple for r in cell]
        t1_complex = [r.metrics.n_t1_complex for r in cell]
        docs = [r.metrics.n_docs_total for r in cell]
        docs_cov = [r.metrics.n_docs_covered for r in cell]
        useful = [_useful_score(r.metrics) for r in cell]
        tokens = [r.total_tokens for r in cell]
        lint_pass = sum(r.metrics.n_lint_pass for r in cell)
        resolved = sum(r.metrics.n_resolved for r in cell)
        total = sum(asserts) or 1
        a_mean = statistics.mean(asserts) if asserts else 0.0
        t_mean = statistics.mean(tokens) if tokens else 0.0
        u_mean = statistics.mean(useful) if useful else 0.0
        t1_mean = statistics.mean(t1s) if t1s else 0.0
        ts_mean = statistics.mean(t1_simple) if t1_simple else 0.0
        tc_mean = statistics.mean(t1_complex) if t1_complex else 0.0
        d_mean = statistics.mean(docs) if docs else 0.0
        c_mean = statistics.mean(docs_cov) if docs_cov else 0.0
        cs_total = sum(t1_complex)
        t1_total = sum(t1s)
        # Functional-coverage aggregates.
        ev_tot = [r.metrics.n_events_total for r in cell]
        ev_hit = [r.metrics.n_events_hit for r in cell]
        rst_tot = [r.metrics.n_reset_total for r in cell]
        rst_hit = [r.metrics.n_reset_hit for r in cell]
        d_cov_tot = [r.metrics.n_doc_total for r in cell]
        d_cov_hit = [r.metrics.n_doc_hit for r in cell]
        cs_tot = [r.metrics.n_case_total for r in cell]
        cs_hit = [r.metrics.n_case_hit for r in cell]
        et_mean = statistics.mean(ev_tot) if ev_tot else 0.0
        eh_mean = statistics.mean(ev_hit) if ev_hit else 0.0
        out.append(CellSummary(
            method=m, design=d, n_runs=n,
            asserts_mean=a_mean,
            t1_mean=t1_mean,
            t1_simple_mean=ts_mean,
            t1_complex_mean=tc_mean,
            complex_share=(cs_total / t1_total) if t1_total > 0 else 0.0,
            docs_mean=d_mean,
            docs_cov_mean=c_mean,
            coverage_pct=(c_mean / d_mean * 100.0) if d_mean > 0 else 0.0,
            tokens_mean=t_mean,
            cost_per_T1=_safe_div(t_mean, t1_mean),
            cost_useful=_safe_div(t_mean, u_mean),
            lint_pass_rate=lint_pass / total,
            resolve_rate=resolved / total,
            events_total_mean=et_mean,
            events_hit_mean=eh_mean,
            fc_pct=(eh_mean / et_mean * 100.0) if et_mean > 0 else 0.0,
            cost_per_cov=_safe_div(t_mean, eh_mean),
            reset_pct=(statistics.mean(rst_hit) / statistics.mean(rst_tot) * 100.0)
                      if rst_tot and statistics.mean(rst_tot) > 0 else 0.0,
            doc_pct=(statistics.mean(d_cov_hit) / statistics.mean(d_cov_tot) * 100.0)
                    if d_cov_tot and statistics.mean(d_cov_tot) > 0 else 0.0,
            case_pct=(statistics.mean(cs_hit) / statistics.mean(cs_tot) * 100.0)
                     if cs_tot and statistics.mean(cs_tot) > 0 else 0.0,
        ))
    return out


def grand_aggregate(cells: List[CellSummary]) -> List[Dict]:
    """Macro-average per method across designs (each design weighted
    equally regardless of how many assertions it produced)."""
    by_method: Dict[str, List[CellSummary]] = defaultdict(list)
    for c in cells:
        by_method[c.method].append(c)
    out = []

    def avg(cells_m: List[CellSummary], key: str) -> float:
        vals = [getattr(c, key) for c in cells_m]
        return statistics.mean(vals) if vals else 0.0

    def avg_opt(cells_m: List[CellSummary], key: str) -> Optional[float]:
        vals = [getattr(c, key) for c in cells_m if getattr(c, key) is not None]
        return statistics.mean(vals) if vals else None

    for m, cells_m in sorted(by_method.items()):
        out.append({
            "method": m,
            "n_designs": len(cells_m),
            "asserts_mean": avg(cells_m, "asserts_mean"),
            "t1_mean": avg(cells_m, "t1_mean"),
            "t1_simple_mean": avg(cells_m, "t1_simple_mean"),
            "t1_complex_mean": avg(cells_m, "t1_complex_mean"),
            "complex_share": avg(cells_m, "complex_share"),
            "docs_cov_mean": avg(cells_m, "docs_cov_mean"),
            "coverage_pct": avg(cells_m, "coverage_pct"),
            "tokens_mean": avg(cells_m, "tokens_mean"),
            "cost_per_T1": avg_opt(cells_m, "cost_per_T1"),
            "cost_useful": avg_opt(cells_m, "cost_useful"),
            "lint_pass_rate": avg(cells_m, "lint_pass_rate"),
            "resolve_rate": avg(cells_m, "resolve_rate"),
            # Functional coverage rollups.
            "events_total_mean": avg(cells_m, "events_total_mean"),
            "events_hit_mean": avg(cells_m, "events_hit_mean"),
            "fc_pct": avg(cells_m, "fc_pct"),
            "cost_per_cov": avg_opt(cells_m, "cost_per_cov"),
            "reset_pct": avg(cells_m, "reset_pct"),
            "doc_pct": avg(cells_m, "doc_pct"),
            "case_pct": avg(cells_m, "case_pct"),
        })
    return out


# =============================================================================
# Output rendering
# =============================================================================

def _fmt_opt(v: Optional[float], fmt: str = "{:,.0f}") -> str:
    if v is None:
        return "—"
    return fmt.format(v)


def render_markdown(cells: List[CellSummary],
                    grand: List[Dict]) -> str:
    out: List[str] = []
    out.append("# SpecGuard Quality + Cost Comparison\n")
    out.append("Per assertion four boolean axes (Tier 1, structural):\n")
    out.append("- **lint** — pyslang accepts the syntax")
    out.append("- **resolved** — every identifier appears in the design's signal map (no hallucination)")
    out.append("- **non-trivial** — not a bare `$bits()==const`, not a `1'b1` body")
    out.append("- **unique** — not a string-normalised duplicate of an earlier assertion\n")
    out.append("**N_T1** = lint ∧ resolved ∧ non-trivial ∧ unique.\n")
    out.append("**T1 stratification** (within N_T1):")
    out.append("- **simple** — bare equalities, single-signal expressions, no implication")
    out.append("- **complex** — uses `$past`/`$stable`/`$rose`/`$fell`/`$onehot`/`$countones`/`inside`/`|->` (with multi-signal antecedent)/`|=>`/`throughout`/`until`\n")
    out.append("Tier 2 (coverage of documented intent):\n")
    out.append(f"- **docs covered / total** — documented sentences (in `docs/`) where ≥{COVERAGE_OVERLAP_THRESHOLD*100:.0f}% of the docprop's signals appear in some N_T1 assertion's identifier set (and at least one signal overlaps)\n")
    out.append("Headline (lower `tok/useful` = better):\n")
    out.append(
        f"- **useful_score** = {WEIGHT_T1_SIMPLE:g}×T1_simple + "
        f"{WEIGHT_T1_COMPLEX:g}×T1_complex + {ALPHA_DOC_WEIGHT:g}×docs_covered"
    )
    out.append("- **cost_useful** = total_tokens / useful_score")
    out.append("- **cost_per_T1** = total_tokens / N_T1   (diagnostic, no complexity weighting)\n")

    out.append("\n## Functional coverage — the headline\n")
    out.append("Each design is decomposed into discrete coverage events across three axes:")
    out.append("- **reset** — one event per resettable register (from `facts.reset_values`)")
    out.append("- **doc** — one event per documented sentence with at least one design signal")
    out.append("- **case** — one event per (case_selector, case_value, lhs) triple from the AST extractor\n")
    out.append("An event is **hit** when at least one N_T1 assertion's identifier set "
               f"overlaps the event's required signals at ≥{COVERAGE_OVERLAP_THRESHOLD*100:.0f}%. "
               "Case events additionally require the case value literal to appear in the assertion text.\n")
    out.append("**cost_per_cov** = total_tokens / events_hit  (lower is better).")
    out.append("**FC%** = events_hit / events_total — directly comparable across methods on the same design.\n")
    out.append("| Method | Designs | Tokens μ | Events μ | Hit μ | **FC%** | Reset% | Doc% | Case% | **tok/cov** |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for g in grand:
        out.append(
            "| {m} | {n} | {t:,.0f} | {et:.1f} | {eh:.1f} | "
            "**{fc:.0f}%** | {rp:.0f}% | {dp:.0f}% | {cp:.0f}% | "
            "**{cc}** |".format(
                m=g["method"], n=g["n_designs"],
                t=g["tokens_mean"],
                et=g["events_total_mean"], eh=g["events_hit_mean"],
                fc=g["fc_pct"],
                rp=g["reset_pct"], dp=g["doc_pct"], cp=g["case_pct"],
                cc=_fmt_opt(g["cost_per_cov"]),
            )
        )

    out.append("\n## Structural quality (T1) — diagnostic\n")
    out.append("| Method | Designs | Tokens μ | Asserts μ | N_T1 μ | T1_S | T1_C | Cmplx% | DocsCov μ | Cov% | Lint% | Resolve% | tok/T1 | tok/useful |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for g in grand:
        out.append(
            "| {m} | {n} | {t:,.0f} | {a:.1f} | {t1:.1f} | {ts:.1f} | "
            "{tc:.1f} | {cmpct:.0%} | {dc:.1f} | {cp:.0f}% | {lp:.0%} | "
            "{rs:.0%} | {ct1} | {cu} |".format(
                m=g["method"], n=g["n_designs"],
                t=g["tokens_mean"], a=g["asserts_mean"],
                t1=g["t1_mean"],
                ts=g["t1_simple_mean"], tc=g["t1_complex_mean"],
                cmpct=g["complex_share"],
                dc=g["docs_cov_mean"],
                cp=g["coverage_pct"], lp=g["lint_pass_rate"],
                rs=g["resolve_rate"],
                ct1=_fmt_opt(g["cost_per_T1"]),
                cu=_fmt_opt(g["cost_useful"]),
            )
        )

    out.append("\n## Per (method, design) cell — functional coverage\n")
    out.append("| Method | Design | Runs | Tokens μ | Events | Hit | **FC%** | Reset% | Doc% | Case% | **tok/cov** |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in cells:
        out.append(
            "| {m} | {d} | {n} | {t:,.0f} | {et:.0f} | {eh:.0f} | "
            "**{fc:.0f}%** | {rp:.0f}% | {dp:.0f}% | {cp:.0f}% | "
            "**{cc}** |".format(
                m=c.method, d=c.design, n=c.n_runs,
                t=c.tokens_mean,
                et=c.events_total_mean, eh=c.events_hit_mean,
                fc=c.fc_pct,
                rp=c.reset_pct, dp=c.doc_pct, cp=c.case_pct,
                cc=_fmt_opt(c.cost_per_cov),
            )
        )

    out.append("\n## Per (method, design) cell — structural T1 (diagnostic)\n")
    out.append("| Method | Design | Runs | Tokens μ | A μ | N_T1 | T1_S | T1_C | Cmplx% | Docs | Cov | Cov% | Lint% | Resolve% | tok/T1 | tok/useful |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in cells:
        out.append(
            "| {m} | {d} | {n} | {t:,.0f} | {a:.1f} | {t1:.1f} | "
            "{ts:.1f} | {tc:.1f} | {cmpct:.0%} | "
            "{dt:.1f} | {dc:.1f} | {cp:.0f}% | {lp:.0%} | {rs:.0%} | "
            "{ct1} | {cu} |".format(
                m=c.method, d=c.design, n=c.n_runs,
                t=c.tokens_mean, a=c.asserts_mean, t1=c.t1_mean,
                ts=c.t1_simple_mean, tc=c.t1_complex_mean,
                cmpct=c.complex_share,
                dt=c.docs_mean, dc=c.docs_cov_mean,
                cp=c.coverage_pct, lp=c.lint_pass_rate,
                rs=c.resolve_rate,
                ct1=_fmt_opt(c.cost_per_T1),
                cu=_fmt_opt(c.cost_useful),
            )
        )
    return "\n".join(out) + "\n"


def render_csv(cells: List[CellSummary]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "method", "design", "n_runs",
        "tokens_mean", "asserts_mean",
        "t1_mean", "t1_simple_mean", "t1_complex_mean", "complex_share",
        "docs_total_mean", "docs_covered_mean", "coverage_pct",
        "lint_pass_rate", "resolve_rate",
        "cost_per_T1", "cost_useful",
        "events_total_mean", "events_hit_mean", "fc_pct",
        "reset_pct", "doc_pct", "case_pct", "cost_per_cov",
    ])
    for c in cells:
        w.writerow([
            c.method, c.design, c.n_runs,
            f"{c.tokens_mean:.1f}",
            f"{c.asserts_mean:.2f}",
            f"{c.t1_mean:.2f}",
            f"{c.t1_simple_mean:.2f}",
            f"{c.t1_complex_mean:.2f}",
            f"{c.complex_share:.4f}",
            f"{c.docs_mean:.2f}",
            f"{c.docs_cov_mean:.2f}",
            f"{c.coverage_pct:.2f}",
            f"{c.lint_pass_rate:.4f}",
            f"{c.resolve_rate:.4f}",
            "" if c.cost_per_T1 is None else f"{c.cost_per_T1:.2f}",
            "" if c.cost_useful is None else f"{c.cost_useful:.2f}",
            f"{c.events_total_mean:.2f}",
            f"{c.events_hit_mean:.2f}",
            f"{c.fc_pct:.2f}",
            f"{c.reset_pct:.2f}",
            f"{c.doc_pct:.2f}",
            f"{c.case_pct:.2f}",
            "" if c.cost_per_cov is None else f"{c.cost_per_cov:.2f}",
        ])
    return buf.getvalue()


# =============================================================================
# CLI
# =============================================================================

def _parse_results_arg(arg: str) -> Tuple[Path, Optional[str]]:
    """`PATH` or `PATH:LABEL` — `:auto` is treated as no label."""
    if ":" in arg:
        # Avoid splitting Windows-style drive letters.
        head, _, tail = arg.rpartition(":")
        if head and Path(head).exists() and not Path(arg).exists():
            label = tail or None
            if label and label.lower() == "auto":
                label = None
            return Path(head), label
    return Path(arg), None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--results-dir", action="append", required=True, dest="results_dirs",
        help="Repeatable. PATH or PATH:LABEL. LABEL forces the method "
             "name for runs found under PATH. Use `:auto` (or omit) to "
             "infer from the directory layout.",
    )
    ap.add_argument(
        "--output-md", type=Path, default=None,
        help="Write the markdown report to this path.",
    )
    ap.add_argument(
        "--output-csv", type=Path, default=None,
        help="Write the per-cell CSV to this path.",
    )
    ap.add_argument(
        "--print-stdout", action="store_true",
        help="Also write the markdown to stdout.",
    )
    args = ap.parse_args(argv)

    rows: List[RunRow] = []
    for spec in args.results_dirs:
        path, label = _parse_results_arg(spec)
        if not path.exists():
            sys.stderr.write(f"[warn] results dir not found: {path}\n")
            continue
        sys.stderr.write(f"[scan] {path}  label={label or '(auto)'}\n")
        rows.extend(discover_runs(path, label))

    if not rows:
        sys.stderr.write("No SVA files discovered.\n")
        return 1

    sys.stderr.write(
        f"[scan] {len(rows)} run(s) across "
        f"{len({(r.method, r.design) for r in rows})} (method, design) cell(s)\n"
    )

    cells = aggregate(rows)
    grand = grand_aggregate(cells)

    md = render_markdown(cells, grand)
    csv_text = render_csv(cells)

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md)
        sys.stderr.write(f"[write] {args.output_md}\n")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        args.output_csv.write_text(csv_text)
        sys.stderr.write(f"[write] {args.output_csv}\n")
    if args.print_stdout or (not args.output_md and not args.output_csv):
        sys.stdout.write(md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
