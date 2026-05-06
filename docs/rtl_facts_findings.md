# RTL Facts Pipeline — Experimental Findings

This document records A/B test results from Stages 2-3 of the RTL facts
prompt augmentation pipeline. All tests used Qwen3-14B (Q4_K_M via Ollama
or int4 via bitsandbytes) on the AST+LLM assertion generation path.

## Stage 2: RTL Facts Prompt Augmentation

### Hypothesis
Injecting structured RTL facts (signal widths, drive kinds, reset values,
case selectors, clock/reset pairs) into the LLM system prompt reduces
post-processor corrections by giving the LLM ground-truth signal info.

### Design: CMAC (5 modules, 124 signals, 1 clock domain)

**Baseline (no facts) vs Stage 2 (facts ON), HF local backend:**

| Metric                          | Baseline | Facts ON | Change |
|---------------------------------|----------|----------|--------|
| Final assertions                | 110      | 122      | +12    |
| fix_next_cycle_on_combinational | 33       | 0        | -33    |
| deduplicate_assertions          | 23       | 4        | -19    |
| remove_trivial_assertions       | 6        | 0        | -6     |
| validate_signals (hallucinated) | 1        | 0        | -1     |
| fix_bare_property_fragments     | 1        | 0        | -1     |
| **TOTAL post-proc actions**     | **57**   | **4**    | **-93%** |

**Key finding:** The drive-kind annotations `(c)`/`(s)` in the signal
widths table eliminated `|=>` misuse entirely (33 → 0). The LLM used
`|->` correctly for all combinational signals without any post-processing.

### Stage 2 Improvements

**Instruction sandwich reminders (Ollama backend):**

| Metric                      | Facts only | + Reminders | Change |
|-----------------------------|------------|-------------|--------|
| Final assertions            | 94         | 104         | +10    |
| fix_bare_property_fragments | 8          | 0           | -8     |
| deduplicate_assertions      | 7          | 1           | -6     |
| TOTAL post-proc             | 15         | 4           | -11    |

The instruction sandwich (compact reminders at the end of the system
prompt) fixed the bare-fragment regression on the Ollama GGUF model.

**Backend-aware aggressive hint — COUNTERPRODUCTIVE:**

Adding `"IMPORTANT: You MUST emit each assertion as a COMPLETE, SINGLE-LINE
statement..."` to the prompt increased bare-fragment errors from 8 to 18
on the GGUF model. Smaller/quantized models fixate on the negative pattern
when heavily emphasized, paradoxically increasing the error rate. The
gentler bad-pattern example in the facts block was sufficient. The
aggressive hint was removed.

**Per-batch signal facts (Stage 2b) — NEUTRAL on CMAC:**

Injecting per-batch signal-specific facts into the user message (widths,
drive kinds for signals in the current batch's skeletons) was tested and
found approximately neutral on CMAC (2 batches). The "already-covered"
list (telling batch 2 what batch 1 produced) was counterproductive —
it suppressed LLM output (104 → 70 assertions) and increased bare-fragment
errors. Per-batch facts are gated behind `use_per_batch_facts: false`.

## Stage 3: Module-Scoped Facts

### Hypothesis
For multi-module designs, scoping the facts block to the top module's
hierarchy (instead of dumping all signals flat) gives the LLM a more
focused context, producing more and better assertions.

### Design: nvdla_mul (19 modules, 311 signals, 61 name collisions, 2 clock domains)

**6-way depth comparison (Ollama backend):**

| Depth | Signals | Final assertions | TOTAL post-proc | fix_next_cycle | width_mismatch |
|-------|---------|-----------------|-----------------|----------------|----------------|
| flat  | 311     | 161             | 10              | 9              | 1              |
| 0     | 36      | 199             | 16              | 8              | 0              |
| 1     | 50      | 218             | 19              | 17             | 0              |
| **2** | **238** | **224**         | **13**          | **12**         | **0**          |
| 3     | 260     | 188             | 20              | 11             | 4              |
| 4     | 280     | 211             | 23              | 12             | 8              |

**Key findings:**

1. **Module scoping dramatically increases assertion count.** The flat
   view (311 signals) produced only 161 assertions. Depth=2 (238 signals)
   produced 224 — a 39% increase. The smaller, focused facts block gives
   the LLM clearer context about the design's structure.

2. **The relationship is non-monotonic.** Too narrow (depth=0, 36 signals)
   loses submodule context. Too broad (depth=3-4, 260-280 signals) adds
   noise — `width_mismatch` errors increase from 0 to 4 to 8 as more
   deep submodule signals are included.

3. **Depth=2 is the sweet spot.** Highest assertion count (224), second-
   lowest post-proc actions (13), zero width mismatches, zero dedups.

4. **Fix 6 (submodule drive-kind list) works.** When module scoping
   removes signals from the detailed widths table, a compact one-line
   list of combinational/sequential submodule signal names is appended.
   This preserves the `|->` vs `|=>` guidance without bloating the prompt.

### Design: CMAC (5 modules, 124 signals) — NEUTRAL

Module scoping on CMAC was neutral-to-slightly-negative (104 → 84
assertions, 4 → 9 post-proc actions). CMAC's flat view is already clean
with no name collisions, so scoping provides no benefit. Module scoping
should be enabled only for designs with 10+ modules.

## Recommended Configuration

### Small designs (< 10 modules, < 200 signals)
```yaml
agent:
  use_rtl_facts: true           # Stage 2 — always beneficial
  module_facts_mode: "off"      # scoping not needed
```

### Large designs (10+ modules, 200+ signals)
```yaml
agent:
  use_rtl_facts: true
  module_facts_mode: "lazy"     # or "full" for collision-proof
  module_scope_depth: 2         # empirically optimal on nvdla_mul
```

### Experimental / ablation
```yaml
agent:
  use_per_batch_facts: true     # per-batch signal injection (neutral on CMAC)
  use_grammar_constraints: true # signal-name allowlist grammar (needs vLLM/outlines)
  use_hallucination_denylist: true  # inject accumulated denylist into prompt
```

## Methodology Notes

- All comparisons use the same model (Qwen3-14B), backend (Ollama with
  Q4_K_M GGUF or HF with bitsandbytes int4), and generation parameters
  (temperature=0.1, max_new_tokens=4096).
- LLM output is non-deterministic. Single-run comparisons show directional
  trends but have noise (e.g., `fix_next_cycle` varies 8-17 across runs
  on the same config). The strongest signals are large deltas (>50%)
  consistent across multiple runs.
- Post-processor action counts are parsed from runtime logs using
  `ab_compare.py`. Final assertion counts come from the output `.sv` files.
- The `ab_compare.py` tool and multi-config `abtest.sh` script are
  included in the repository for reproducing these results.
