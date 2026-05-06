"""
Ablation study for the SVA generation pipeline.

Measures the contribution of each post-processing component by running
the full pipeline with components selectively enabled/disabled.

Uses cached raw SVA from a recent pipeline run to avoid re-running the
LLM (which is non-deterministic and slow).
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
from sva_pipeline.lint_loop import (
    fix_immediate_implication, fix_double_negation,
    fix_condition_only_assertions, fix_immediate_and_form,
    fix_next_cycle_on_combinational, fix_bare_property_fragments,
    deduplicate_assertions, semantic_deduplicate,
    remove_trivial_assertions, remove_wrong_style_assertions,
    remove_subsumed_and_contradicting, validate_signals,
    split_assertions, lint_all_assertions,
)
from sva_pipeline.ast_assertions import (
    generate_ast_assertions, format_skeletons_as_sva,
)
from sva_pipeline.slang_frontend import build_design_info


def count_assertions(sva: str) -> int:
    """Count valid assertion statements (lines starting with 'assert')."""
    return len(re.findall(r'^\s*assert\s', sva, re.MULTILINE))


def lint_count(sva: str) -> int:
    """Count assertions that pass pyslang lint."""
    if not sva.strip():
        return 0
    entries = split_assertions(sva)
    passed, _ = lint_all_assertions(entries, reject_assert_property=False)
    return len(passed)


# ---------------------------------------------------------------------------
# Setup: load design and AST baseline
# ---------------------------------------------------------------------------

print("=" * 70)
print("SVA Pipeline Ablation Study")
print("Design: NVDLA CMAC (NV_NVDLA_CMAC_CORE_MAC_mul)")
print("=" * 70)

rtl_dir = "nvdla_cmac_test/rtl"
top_file = "NV_NVDLA_CMAC_CORE_MAC_mul.v"
top_module = "NV_NVDLA_CMAC_CORE_MAC_mul"

print("\n[1/3] Building design info...")
info = build_design_info(rtl_dir, top_module, top_file)

print(f"  Modules: {len(info.modules)}")
print(f"  Signal map: {len(info.signal_map)} ports")
print(f"  Clock: {info.clock_signal}, Reset: {info.reset_signal}")

# AST extraction from full RTL.
print("\n[2/3] Extracting AST skeletons...")
src = ""
for f in sorted(Path(rtl_dir).glob("*.v")):
    src += f.read_text(errors="replace") + "\n"

skeletons = generate_ast_assertions(src, info.clock_signal, info.reset_signal)
ast_sva = format_skeletons_as_sva(skeletons)
ast_count = count_assertions(ast_sva)
print(f"  AST skeletons: {len(skeletons)}")
print(f"  AST formatted: {ast_count} assertions")

# Load raw LLM output from the most recent run's saved file.
# We use the current sva_output as proxy for "AST + LLM raw".
print("\n[3/3] Loading recent pipeline output as 'AST+LLM raw'...")
recent_output = Path("nvdla_cmac_test/cmac_ast_llm_sva.sv").read_text()
combined_raw = ast_sva + "\n\n" + recent_output  # deliberately combine

print(f"  Combined raw: {count_assertions(combined_raw)} assertions")


# ---------------------------------------------------------------------------
# Mock config object
# ---------------------------------------------------------------------------
class MockConfig:
    rtl_dir = "nvdla_cmac_test/rtl"
    reject_assert_property = False
    use_self_review = False


config = MockConfig()


# ---------------------------------------------------------------------------
# Ablation runs
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Mode Ablation (different generation strategies)")
print("=" * 70)

# Mode 1: AST-only
ast_only_lint = lint_count(ast_sva)
print(f"\n  AST-only:        {ast_count:3d} raw → {ast_only_lint:3d} pass lint")

# Mode 2: AST+LLM raw (no post-processing)
raw_count = count_assertions(combined_raw)
raw_lint = lint_count(combined_raw)
print(f"  AST+LLM raw:     {raw_count:3d} raw → {raw_lint:3d} pass lint")


print("\n" + "=" * 70)
print("Post-Processing Ablation (incremental)")
print("=" * 70)
print(f"\n  Starting from AST+LLM raw: {raw_count} assertions")

steps = [
    ("None (raw)", lambda s: s),
    # Phase 1: TRANSFORM
    ("[P1] wrap bare fragments", lambda s: fix_bare_property_fragments(s, info.clock_signal, info.reset_signal)),
    ("[P1] fix |-> in immediate", fix_immediate_implication),
    ("[P1] fix double negation", fix_double_negation),
    ("[P1] fix AND-form", fix_immediate_and_form),
    ("[P1] fix condition-only", fix_condition_only_assertions),
    ("[P1] fix |=> on comb sigs", lambda s: fix_next_cycle_on_combinational(s, rtl_dir=config.rtl_dir)),
    # Phase 2: REMOVE WRONG
    ("[P2] remove trivial", remove_trivial_assertions),
    ("[P2] remove wrong style", lambda s: remove_wrong_style_assertions(s, config)),
    # Phase 3: VALIDATE & DEDUPLICATE
    ("[P3] validate signals", lambda s: validate_signals(s, info.signal_map, rtl_dir=config.rtl_dir)),
    ("[P3] string dedup", deduplicate_assertions),
    ("[P3] semantic dedup", semantic_deduplicate),
    ("[P3] subsumption/contradiction", remove_subsumed_and_contradicting),
]

current = combined_raw
print(f"\n  {'Step':<35} {'Count':>6} {'Lint OK':>8} {'Δ':>5}")
print(f"  {'-'*35} {'-'*6} {'-'*8} {'-'*5}")

prev_count = count_assertions(current)
prev_lint = lint_count(current)
print(f"  {'Initial raw':<35} {prev_count:>6} {prev_lint:>8} {'':>5}")

for label, fn in steps[1:]:  # skip "None"
    current = fn(current)
    new_count = count_assertions(current)
    new_lint = lint_count(current)
    delta = new_count - prev_count
    sign = "+" if delta > 0 else ""
    print(f"  {label:<35} {new_count:>6} {new_lint:>8} {sign}{delta:>4}")
    prev_count = new_count


print("\n" + "=" * 70)
print("Component Removal Ablation (full minus one)")
print("=" * 70)


def run_full_pipeline(sva, skip=None):
    """Run full pipeline, optionally skipping one component.

    Mirrors the 3-phase structure of run_lint_loop.
    """
    skip = skip or set()

    # Phase 1: TRANSFORM
    if "bare_fragments" not in skip:
        sva = fix_bare_property_fragments(sva, info.clock_signal, info.reset_signal)
    if "immediate_implication" not in skip:
        sva = fix_immediate_implication(sva)
    if "double_negation" not in skip:
        sva = fix_double_negation(sva)
    if "and_form" not in skip:
        sva = fix_immediate_and_form(sva)
    if "condition_only" not in skip:
        sva = fix_condition_only_assertions(sva)
    if "combinational_fix" not in skip:
        sva = fix_next_cycle_on_combinational(sva, rtl_dir=config.rtl_dir)

    # Phase 2: REMOVE WRONG
    if "trivial" not in skip:
        sva = remove_trivial_assertions(sva)
    if "wrong_style" not in skip:
        sva = remove_wrong_style_assertions(sva, config)

    # Phase 3: VALIDATE & DEDUPLICATE
    if "signal_validation" not in skip:
        sva = validate_signals(sva, info.signal_map, rtl_dir=config.rtl_dir)
    if "string_dedup" not in skip:
        sva = deduplicate_assertions(sva)
    if "semantic_dedup" not in skip:
        sva = semantic_deduplicate(sva)
    if "subsumption" not in skip:
        sva = remove_subsumed_and_contradicting(sva)

    return sva


full_sva = run_full_pipeline(combined_raw)
full_count = count_assertions(full_sva)
full_lint = lint_count(full_sva)
print(f"\n  Full pipeline: {full_count} assertions, {full_lint} pass lint")
print(f"\n  {'Removed component':<30} {'Count':>6} {'Lint':>6} {'Δlint':>7}")
print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*7}")

components = [
    "bare_fragments", "and_form", "condition_only",
    "combinational_fix", "trivial", "string_dedup",
    "signal_validation", "semantic_dedup", "subsumption",
]

for comp in components:
    sva = run_full_pipeline(combined_raw, skip={comp})
    cnt = count_assertions(sva)
    lc = lint_count(sva)
    dlint = lc - full_lint
    sign = "+" if dlint > 0 else ""
    print(f"  {comp:<30} {cnt:>6} {lc:>6} {sign}{dlint:>6}")


print("\n" + "=" * 70)
print("Done.")
print("=" * 70)
