"""
Regression check for the unified-facts pipeline.

After Stage 1 of the refactor, the regex-based extractors in lint_loop.py
were deleted. This script verifies that the new RTLFacts-based extractors
+ post-processors still work correctly on synthetic test inputs across
the three reference designs (CMAC, AES, nvdla_mul).

It is no longer a strict parity check (there's nothing to compare against
now that the old extractors are gone) — it's a functional regression test.

Run from the repo root:

    python parity_check.py
"""
import sys

sys.path.insert(0, ".")

from sva_pipeline.rtl_facts import extract_rtl_facts
from sva_pipeline.lint_loop import (
    fix_next_cycle_on_combinational,
    verify_constant_signal_pairs,
    validate_signal_widths,
    validate_reset_values,
    validate_signals,
    check_case_selector_mismatch,
)


# ---------------------------------------------------------------------------
# Test SVA fixtures (one per post-processor)
# ---------------------------------------------------------------------------

CMAC_TEST_FIX_NEXT_CYCLE = '''\
// Should be converted: out_data is combinational
assert property (@(posedge clk) (cond) |=> out_data == 17'h10000) else $error("test1");
// Should be untouched: cfg_is_int8_d1 is sequential
assert property (@(posedge clk) (cfg_reg_en) |=> cfg_is_int8_d1 == $past(cfg_is_int8)) else $error("test2");
'''

CMAC_TEST_VERIFY_CONST = '''\
// Wrong: 32'h55005500 belongs to res_a_gate, not res_a
assert (res_a == 32'h55005500) else $error("test wrong");
// Correct
assert (res_a_gate == 32'h55005500) else $error("test correct");
// Constant not in RTL — should not trigger
assert (out_data == 17'hdead) else $error("test unrelated");
'''

CMAC_TEST_VALIDATE_SIGNALS = '''\
// Valid signals
assert (in_code == 4'b0011) else $error("test1");
assert (out_data == 17'h10000) else $error("test2");
// Hallucinated signals
assert (fake_signal == 4'b0011 && nonexistent == 1'b0) else $error("test3");
'''

CMAC_TEST_CASE_MISMATCH = '''\
// AST-style (uses {is_8bit, in_code} like RTL): keep
assert (!({is_8bit, in_code} == 4'b0011) || (out_inv == 1'b0)) else $error("ast");
// LLM-style (uses raw 'code'): drop because in_code is derived from code
assert (!((code == 3'b011) || (code == 3'b100)) || (out_inv == 1'b1)) else $error("llm");
'''

CMAC_TEST_WIDTH = '''\
// Valid: in_code is 3 bits, comparing to 3'b011 is fine
assert (in_code == 3'b011) else $error("ok1");
// Valid: out_data[16:9] is in range
assert (out_data[16:9] == 8'b0) else $error("ok2");
// Invalid: in_code is 3 bits, comparing to 4'b0011 is too wide
assert (in_code == 4'b0011) else $error("wrong1");
// Invalid: out_data[17] is out of range (max index is 16)
assert (out_data[17] == 1'b0) else $error("wrong2");
'''

CMAC_TEST_RESET = '''\
// Valid: cfg_is_int8 resets to 1'b0
assert property (@(posedge clk) !nvdla_core_rstn |-> cfg_is_int8 == 1'b0) else $error("ok1");
// Valid: cfg_is_int16 resets to 1'b1 (the unusual one)
assert property (@(posedge clk) !nvdla_core_rstn |-> cfg_is_int16 == 1'b1) else $error("ok2");
// Wrong: cfg_is_int16 actually resets to 1'b1, not 1'b0
assert property (@(posedge clk) !nvdla_core_rstn |-> cfg_is_int16 == 1'b0) else $error("wrong1");
// Wrong: proc_precision actually resets to 2'b01
assert property (@(posedge clk) !nvdla_core_rstn |-> proc_precision == 2'b00) else $error("wrong2");
'''


# ---------------------------------------------------------------------------
# Expected outcomes per (post_processor, design)
# ---------------------------------------------------------------------------

EXPECTED_CHECKS = {
    "fix_next_cycle_on_combinational": {
        "CMAC": (
            CMAC_TEST_FIX_NEXT_CYCLE,
            # First assertion's |=> should become |->, second should stay |=>
            lambda out: "out_data == 17'h10000" in out
                        and out.count("|->") >= 1
                        and "cfg_is_int8_d1 == $past(cfg_is_int8)" in out
                        and "|=>" in out,
        ),
    },
    "verify_constant_signal_pairs": {
        "CMAC": (
            CMAC_TEST_VERIFY_CONST,
            # Wrong assertion (res_a) should be removed; correct (res_a_gate) kept
            lambda out: "res_a == 32'h55005500" not in out
                        and "res_a_gate == 32'h55005500" in out
                        and "17'hdead" in out,
        ),
    },
    "validate_signals": {
        "CMAC": (
            CMAC_TEST_VALIDATE_SIGNALS,
            # Hallucinated assertion removed, real ones kept
            lambda out: "fake_signal" not in out
                        and "in_code == 4'b0011" in out
                        and "out_data == 17'h10000" in out,
        ),
    },
    "check_case_selector_mismatch": {
        "CMAC": (
            CMAC_TEST_CASE_MISMATCH,
            # AST kept, LLM dropped because code → in_code transformation
            lambda out: "{is_8bit, in_code}" in out
                        and "(code == 3'b011)" not in out,
        ),
    },
    "validate_signal_widths": {
        "CMAC": (
            CMAC_TEST_WIDTH,
            # Valid kept, invalid removed
            lambda out: "in_code == 3'b011" in out
                        and "out_data[16:9]" in out
                        and "in_code == 4'b0011" not in out
                        and "out_data[17]" not in out,
        ),
    },
    "validate_reset_values": {
        "CMAC": (
            CMAC_TEST_RESET,
            # Valid (matching) kept, wrong removed
            lambda out: "cfg_is_int8 == 1'b0" in out
                        and "cfg_is_int16 == 1'b1" in out
                        and "cfg_is_int16 == 1'b0" not in out
                        and "proc_precision == 2'b00" not in out,
        ),
    },
}


def main():
    print("=" * 72)
    print("Unified-Facts Regression Check")
    print("=" * 72)

    designs = [
        ("CMAC", "nvdla_cmac_test/rtl"),
        ("AES", "RTL Cases/AES-Verilog/SourceCode"),
        ("nvdla_mul", "nvdla_mul_test/rtl"),
    ]

    # Build facts for every design once and report counts.
    print("\n--- RTL fact extraction ---")
    facts_by_design = {}
    for label, rtl_dir in designs:
        facts = extract_rtl_facts(rtl_dir)
        facts_by_design[label] = facts
        status = "OK" if facts.is_complete else "PARTIAL"
        print(f"  {label:<10} {status}: "
              f"{len(facts.signal_definitions)} sig_defs, "
              f"{len(facts.case_selectors)} case-driven, "
              f"{len(facts.combinational_signals)} comb, "
              f"{len(facts.constant_signal_pairs)} const, "
              f"{len(facts.all_signals)} total")
        if facts.parse_warnings:
            for w in facts.parse_warnings[:3]:
                print(f"      warning: {w}")

    # Run functional checks (CMAC only — fixtures are CMAC-specific).
    print("\n--- CMAC functional checks ---")
    cmac_facts = facts_by_design["CMAC"]

    pp_map = {
        "fix_next_cycle_on_combinational": fix_next_cycle_on_combinational,
        "verify_constant_signal_pairs": verify_constant_signal_pairs,
        "validate_signals": validate_signals,
        "check_case_selector_mismatch": check_case_selector_mismatch,
        "validate_signal_widths": validate_signal_widths,
        "validate_reset_values": validate_reset_values,
    }

    passed = 0
    total = 0
    for pp_name, design_tests in EXPECTED_CHECKS.items():
        pp = pp_map[pp_name]
        for design_label, (test_sva, predicate) in design_tests.items():
            total += 1
            try:
                out = pp(test_sva, cmac_facts)
                ok = predicate(out)
            except Exception as exc:
                print(f"  {pp_name:<35} {design_label:<10} ERROR: {exc}")
                continue
            status = "PASS" if ok else "FAIL"
            print(f"  {pp_name:<35} {design_label:<10} {status}")
            if ok:
                passed += 1
            else:
                # Show what came out for debugging
                preview = out.replace("\n", "\\n")[:200]
                print(f"      output: {preview}")

    print()
    print("=" * 72)
    print(f"Result: {passed}/{total} checks passed")
    print("=" * 72)


if __name__ == "__main__":
    main()
