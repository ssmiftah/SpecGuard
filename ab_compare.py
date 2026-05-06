"""
ab_compare.py
-------------
Compare two pipeline runs (baseline vs augmented) to evaluate Stage 2
RTL facts prompt augmentation.

Each "run" is identified by a base path; the script reads the matching
``*_sva.sv``, ``*_log.txt``, and ``*_lint.json`` files.

Usage::

    python ab_compare.py <baseline_base_path> <augmented_base_path>

Example::

    python ab_compare.py \\
        nvdla_cmac_test/cmac_baseline \\
        nvdla_cmac_test/cmac_ollama_facts

The script prints a side-by-side table of:
- Final assertion count
- Per-post-processor removal counts (parsed from the log)
- Lint failure count
- Hallucination denylist size (if facts.json present)

It does NOT make value judgments — it just reports the numbers so you
can decide whether the augmentation is helping.
"""
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Lines we look for in the pipeline log. Each entry is
# (label, compiled regex, group index of the count).
#
# IMPORTANT: most post-processors only emit a summary line when they
# actually did something — a count of zero means "0 actions" but ALSO
# means there is no log line, so we report 0 either way.
LOG_PATTERNS: List[Tuple[str, "re.Pattern", int]] = [
    ("validate_signals (hallucinated)",
     re.compile(r"Signal validation: removed (\d+) assertion"), 1),
    ("verify_constant_signal_pairs",
     re.compile(r"Constant-signal verification: removed (\d+) mismatched"), 1),
    ("validate_signal_widths",
     re.compile(r"Width validation: removed (\d+) width-mismatched"), 1),
    ("validate_reset_values",
     re.compile(r"Reset value validation: removed (\d+) wrong-reset-value"), 1),
    ("check_case_selector_mismatch",
     re.compile(r"Data-flow check: removed (\d+) case-selector"), 1),
    ("fix_next_cycle_on_combinational",
     re.compile(r"Fixed (\d+) assertion\(s\): \|=> to \|->"), 1),
    ("remove_trivial_assertions",
     re.compile(r"Removed (\d+) trivially true assertion"), 1),
    ("deduplicate_assertions",
     re.compile(r"Deduplicated (\d+) assertion"), 1),
    ("semantic_deduplicate",
     re.compile(r"Semantic deduplication: removed (\d+) assertion"), 1),
    ("remove_subsumed_and_contradicting",
     re.compile(r"Removed (\d+) subsumed/contradicting"), 1),
    ("remove_wrong_style_assertions",
     re.compile(r"Removed (\d+) wrong-style assertion"), 1),
    ("fix_bare_property_fragments",
     re.compile(r"Wrapped (\d+) bare property fragment"), 1),
    ("fix_immediate_and_form",
     re.compile(r"Fixed (\d+) immediate AND-form"), 1),
]


# Final assertion count is logged at the end of run_lint_loop.
_FINAL_COUNT_RE = re.compile(
    r"Lint loop complete: (\d+) assertion\(s\) in final output"
)
# Lint summary line: "Lint summary: P passed, F failed (out of T)"
_LINT_SUMMARY_RE = re.compile(
    r"Lint summary: (\d+) passed, (\d+) failed \(out of (\d+)\)"
)


def count_assertions(sva_path: Path) -> int:
    """Count assert statements in an SVA file."""
    if not sva_path.exists():
        return -1
    text = sva_path.read_text()
    # Match `assert (...)` and `assert property (...)` at line start
    # (allow leading whitespace).
    return len(re.findall(r"^\s*assert\b", text, re.MULTILINE))


def parse_log_counts(log_path: Path) -> Dict[str, int]:
    """
    Walk a pipeline log and sum every removal/conversion count per
    post-processor. Multiple log lines for the same processor are
    summed (the lint loop runs through phases multiple times across
    refinement iterations).
    """
    counts: Dict[str, int] = {label: 0 for label, _, _ in LOG_PATTERNS}
    if not log_path.exists():
        return counts

    with open(log_path, "r", errors="replace") as f:
        for line in f:
            for label, pattern, group in LOG_PATTERNS:
                m = pattern.search(line)
                if m:
                    try:
                        counts[label] += int(m.group(group))
                    except (ValueError, IndexError):
                        pass
    return counts


def parse_final_count(log_path: Path) -> int:
    """
    Find the last "Lint loop complete: N assertion(s) in final output"
    line in the runtime log and return N.
    """
    if not log_path.exists():
        return -1
    last = -1
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = _FINAL_COUNT_RE.search(line)
            if m:
                last = int(m.group(1))
    return last


def parse_lint_summary(log_path: Path) -> Tuple[int, int, int]:
    """
    Return (passed, failed, total) from the LAST "Lint summary" line.
    Returns (-1, -1, -1) if not found.
    """
    if not log_path.exists():
        return (-1, -1, -1)
    last = (-1, -1, -1)
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = _LINT_SUMMARY_RE.search(line)
            if m:
                last = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return last


def count_lint_failures(lint_path: Path) -> int:
    """Count entries in a lint failures JSON file."""
    if not lint_path.exists():
        return -1
    try:
        with open(lint_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            # Some pipelines store {"failures": [...]} or {"assertions": [...]}.
            for key in ("failures", "assertions", "lint_failures"):
                if key in data and isinstance(data[key], list):
                    return len(data[key])
            return len(data)
    except (json.JSONDecodeError, OSError):
        return -1
    return -1


def find_denylist(rtl_dir_hint: Optional[str], deny_dir: str) -> Optional[Path]:
    """
    Look for a hallucination denylist matching the design (best effort).
    We accept any file in the deny_dir whose name starts with the rtl
    directory's basename.
    """
    p = Path(deny_dir)
    if not p.exists():
        return None
    if rtl_dir_hint:
        prefix = Path(rtl_dir_hint).name
        for f in p.glob(f"{prefix}*.json"):
            return f
    files = list(p.glob("*.json"))
    return files[0] if files else None


def count_denylist(deny_path: Optional[Path]) -> Tuple[int, int]:
    """Return (unique_names, total_count) from a denylist file."""
    if deny_path is None or not deny_path.exists():
        return -1, -1
    try:
        with open(deny_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return -1, -1
        return len(data), sum(int(v) for v in data.values() if isinstance(v, (int, float)))
    except (json.JSONDecodeError, OSError, ValueError):
        return -1, -1


def collect(base: Path, runtime_log: Optional[Path] = None) -> dict:
    """
    Collect every metric from one run, given the base path.

    Parameters
    ----------
    base : Path
        Base path used to derive ``{base}_sva.sv``, ``{base}_log.txt``,
        and ``{base}_lint.json``.
    runtime_log : Path, optional
        Path to the actual stdout/stderr log captured by ``tee``. If
        omitted, the post-processor counts come from the ``*_log.txt``
        file (which usually only contains the SVA output, not the run
        log) and will be zero. Pass the real ``*_run.log`` path here
        for accurate counts.
    """
    sva = base.with_name(base.name + "_sva.sv")
    log = base.with_name(base.name + "_log.txt")
    lint = base.with_name(base.name + "_lint.json")

    # Prefer the explicit runtime log; fall back to *_log.txt for the
    # rare case where the user redirected stdout there directly.
    log_for_counts = runtime_log if runtime_log else log

    return {
        "sva_path": sva,
        "log_path": log,
        "lint_path": lint,
        "runtime_log": log_for_counts,
        "n_assertions": count_assertions(sva),
        "log_counts": parse_log_counts(log_for_counts),
        "n_lint_failures": count_lint_failures(lint),
        "final_count": parse_final_count(log_for_counts),
        "lint_summary": parse_lint_summary(log_for_counts),
    }


def render(label: str, baseline: dict, augmented: dict) -> None:
    """Print a single comparison row."""
    b = baseline.get(label, 0) if isinstance(baseline, dict) else baseline
    a = augmented.get(label, 0) if isinstance(augmented, dict) else augmented
    delta = a - b
    arrow = ""
    if delta < 0:
        arrow = " (better)"
    elif delta > 0:
        arrow = " (worse)"
    print(f"  {label:<40} {b:>6} -> {a:>6}  ({delta:+d}){arrow}")


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print(__doc__)
        print(
            "Usage: python ab_compare.py BASELINE_BASE AUGMENTED_BASE "
            "[BASELINE_RUN_LOG] [AUGMENTED_RUN_LOG]"
        )
        sys.exit(1)

    baseline_base = Path(sys.argv[1])
    augmented_base = Path(sys.argv[2])
    baseline_runlog = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    augmented_runlog = Path(sys.argv[4]) if len(sys.argv) > 4 else None

    baseline = collect(baseline_base, baseline_runlog)
    augmented = collect(augmented_base, augmented_runlog)

    print("=" * 72)
    print("A/B comparison")
    print("=" * 72)
    print(f"Baseline:  {baseline_base}")
    print(f"Augmented: {augmented_base}")
    if baseline_runlog or augmented_runlog:
        print(f"Baseline runtime log:  {baseline_runlog or '(none — counts will be 0)'}")
        print(f"Augmented runtime log: {augmented_runlog or '(none — counts will be 0)'}")
    print()

    print("Files found:")
    for label, run in (("baseline", baseline), ("augmented", augmented)):
        for kind in ("sva_path", "lint_path", "runtime_log"):
            p = run[kind]
            if p is None:
                continue
            mark = "OK" if p.exists() else "MISSING"
            print(f"  {label:<10} {kind:<12} {mark:<8} {p}")
    print()

    print("Final assertion counts:")
    render("total_assert_statements (from *_sva.sv)",
           baseline["n_assertions"], augmented["n_assertions"])
    render("Lint loop final count (from runtime log)",
           baseline["final_count"], augmented["final_count"])
    print()

    print("Post-processor actions (parsed from runtime log):")
    all_labels = sorted({l for l, _, _ in LOG_PATTERNS})
    for label in all_labels:
        render(label,
               baseline["log_counts"].get(label, 0),
               augmented["log_counts"].get(label, 0))
    total_b = sum(baseline["log_counts"].values())
    total_a = sum(augmented["log_counts"].values())
    print("  " + "-" * 64)
    render("TOTAL post-processor actions", total_b, total_a)
    print()

    print("Lint summary (last iteration, from runtime log):")
    bp, bf, bt = baseline["lint_summary"]
    ap, af, at = augmented["lint_summary"]
    render("lint passed", bp, ap)
    render("lint failed", bf, af)
    render("lint total", bt, at)
    print()

    print("Lint failure file (from *_lint.json):")
    render("lint_failures", baseline["n_lint_failures"], augmented["n_lint_failures"])
    print()

    # Try to read the denylist (single, design-wide; not per-run).
    deny = find_denylist(None, "indices/hallucinations")
    if deny is not None:
        unique, total = count_denylist(deny)
        print(f"Hallucination denylist (cumulative across runs): {deny}")
        print(f"  unique names: {unique}")
        print(f"  total observations: {total}")
        print()

    print("Notes:")
    print("- 'better' for post-processor counts means FEWER fixes were needed,")
    print("  i.e. the LLM produced cleaner output upstream.")
    print("- 'better' for total assertions can go either way; what matters is")
    print("  whether the FINAL count is similar or higher with FEWER removals.")
    print("- The denylist is cumulative across runs and grows over time;")
    print("  comparing single runs is not meaningful unless you reset between.")
    print("- Provide the runtime *_run.log path as the 3rd/4th argument so the")
    print("  script can parse post-processor counts. *_log.txt usually contains")
    print("  the SVA output, not the runtime log.")
    print()


if __name__ == "__main__":
    main()
