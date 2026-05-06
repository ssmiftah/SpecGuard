# RQ3 Token-Efficiency Ablation — Findings

_Last updated: 2026-04-20. Designs: NVDLA CMAC (84-LOC multiplier) and
NVDLA CACC (~600 KB calculator). Model: qwen3:14b-32k via Ollama._

This document records the results of the token-efficiency ablation
study and the single design change it motivates: **`use_rtl_facts` now
defaults to `False`**. It also documents how the pipeline measures
token usage end-to-end, so the numbers below can be reproduced and
audited.

---

## 1. What we measured

For each pipeline component we produced an ablation config that
disables just that component, then ran it end-to-end on the same
design. Every other setting — model, temperature, retrieval, AST
frontend, post-processors — is held constant.

| Row | Config suffix | Knob toggled |
|---|---|---|
| Full pipeline | `_full` | (baseline, all features on) |
| Remove RTL facts card | `_ablation_no_facts` | `use_rtl_facts: false` |
| Remove module scoping | `_ablation_flat_facts` | `module_facts_mode: "off"` (facts still on) |
| Remove AST skeletons | `_ablation_no_ast` | `use_ast_assertions: false` |
| Remove deterministic repair | `_ablation_no_repair` | `enable_deterministic_repair: false` (feedback loop still active) |
| Remove selective feedback gating | `_ablation_no_feedback` | repair off **and** `max_refinement_iterations: 0` |

Runners: [run_cmac_ablations.sh](../run_cmac_ablations.sh) and
[run_cacc_ablations.sh](../run_cacc_ablations.sh).

---

## 2. How token usage is calculated

The pipeline instruments every LLM call and reports three counts —
prompt, completion, total — that are summed across the whole run.

### 2.1 Per-call capture (backend layer)

Each backend's `generate()` returns a third tuple element carrying the
token counts for that single request:

```python
def generate(messages, tools) -> Tuple[str, List[Dict], Dict[str, int]]:
    ...
    return text, tool_calls, {
        "prompt_tokens": N,
        "completion_tokens": M,
        "total_tokens": N + M,
    }
```

| Backend | Source of the numbers |
|---|---|
| **OpenAI / Ollama** ([openai_backend.py:123](../sva_pipeline/backends/openai_backend.py#L123)) | `response.usage.prompt_tokens` and `response.usage.completion_tokens`. Ollama's OpenAI-compatible endpoint populates both fields using the model's own tokenizer, so counts match what the model actually saw. |
| **Anthropic** ([anthropic_backend.py](../sva_pipeline/backends/anthropic_backend.py)) | `response.usage.input_tokens` / `response.usage.output_tokens`. |
| **Local HF** ([local.py](../sva_pipeline/backends/local.py)) | `prompt_len = inputs.input_ids.shape[1]`; `completion_len = outputs.shape[1] - prompt_len`. Exact, because we tokenize. |

No estimation or heuristic is involved — the numbers come from the
model's own tokenizer via the backend's reported usage.

### 2.2 Per-step accumulation (trace layer)

Every call site inside the agent (planning, execution, refinement,
direct, lint-feedback) forwards the returned `usage` dict to
[TraceLogger.log_step()](../sva_pipeline/trace_logger.py#L42). The
per-step record in `*_trace.csv` gets three columns — Prompt Tokens,
Completion Tokens, Total Tokens — and the JSON trace embeds the same
fields.

Non-LLM operations (AST extraction, Phase 1 deterministic repair, lint
iterations) are logged with zero tokens. That means the aggregate
total counts **only tokens the LLM actually consumed**, not tokens
equivalent to deterministic-repair work.

### 2.3 Run-level aggregation

At end of run, `TraceLogger.totals()` sums all per-step counts:

```python
totals = agent.trace.totals()
# -> {"prompt_tokens": …, "completion_tokens": …,
#     "total_tokens": …, "llm_calls": N}
```

`llm_calls` counts steps where `prompt_tokens > 0 or completion_tokens > 0` —
i.e., only genuine model calls. `main.py` then writes
`<sva_file>_token_summary.json` with the totals plus the count of
surviving assertions, derived by regexing `^\s*assert\b` on the final
output SVA file:

```json
{
  "design": "NV_NVDLA_cacc",
  "config": "examples/nvdla_cacc_full.yaml",
  "llm_calls": 59,
  "prompt_tokens": 539737,
  "completion_tokens": 119203,
  "total_tokens": 658940,
  "final_assertions": 244,
  "tokens_per_assertion": 2700.57
}
```

The summary is written **before** `sys.exit(1)` on a failed run, so
even zero-assertion runs emit a usable token record.

### 2.4 Matrix rollup

[scripts/token_cost_matrix.py](../scripts/token_cost_matrix.py) walks
a results directory, reads every `*_token_summary.json`, and emits
`summary.csv` + `summary.md` with one row per variant. The per-variant
tokens/assertion number is total_tokens ÷ final_assertions; it is
left blank when final_assertions == 0.

### 2.5 What "tokens per assertion" means, precisely

- **Numerator**: every prompt + completion token consumed by the LLM
  from pipeline start to end, across all phases (initial generation,
  lint-feedback refinement rounds, any re-prompting).
- **Denominator**: assertions that survived the full pipeline, i.e.
  written into the output SVA file after dedup, signal validation,
  Phase 1 repair, and N iterations of the lint loop. These are
  assertions that pyslang compiles successfully.

It is **not** a per-call or per-round metric; it is the amortized cost
of getting one usable assertion out the far end of the pipeline.

---

## 3. Results

### 3.1 CMAC — small scale (84 LOC)

| Variant | LLM calls | Prompt | Completion | Total | Assertions | Tok / assert |
|---|---:|---:|---:|---:|---:|---:|
| full | 3 | 81 979 | 4 166 | 86 145 | 55 | 1 566 |
| no-facts | 2 | 51 179 | 3 667 | **54 846** | 52 | **1 055** |
| flat-facts | 2 | 56 327 | 2 812 | 59 139 | 53 | 1 116 |
| **no-ast** | 4 | 112 812 | 8 192 | 121 004 | 11 | 11 000 |
| no-repair | 3 | 82 078 | 4 457 | 86 535 | 50 | 1 731 |
| **no-feedback** | 2 | 55 321 | 3 058 | 58 379 | **0** | — |

Results dir: `cmac_ablations_20260420_0929/`.

### 3.2 CACC — large scale (~600 KB, multi-file)

| Variant | LLM calls | Prompt | Completion | Total | Assertions | Tok / assert |
|---|---:|---:|---:|---:|---:|---:|
| full | 59 | 539 737 | 119 203 | 658 940 | 244 | 2 701 |
| no-facts | 59 | 442 217 | 115 571 | **557 788** | **326** | **1 711** |
| flat-facts | 59 | 543 543 | 119 854 | 663 397 | 270 | 2 457 |
| **no-ast** | 4 | 54 920 | 8 192 | 63 112 | 6 | 10 519 |
| no-repair | 59 | 529 762 | 120 312 | 650 074 | 315 | 2 064 |
| **no-feedback** | 57 | 514 070 | 115 841 | 629 911 | **0** | — |

Results dir: `cacc_ablations_20260420_0944/`.

---

## 4. What the two designs agree on

### Load-bearing components

- **AST skeletons.** Removing them cuts assertion yield 78 % on CMAC
  and 98 % on CACC; tokens-per-assertion explode to ~10 000 on both.
  Without AST the pipeline has nothing concrete to iterate on, so the
  lint loop fires only a handful of times and the LLM produces vague
  protocol stabs.
- **Lint feedback loop.** Zero assertions survive on both designs
  when refinement is disabled (~58 K and ~630 K tokens burned for
  zero output). The LLM's first-pass output needs the refine-on-error
  loop to become lint-clean.

### Not load-bearing

- **RTL facts card.** On both designs `no-facts` is cheaper *and*
  produces at least as many assertions as `full`. On CACC the gap is
  stark: −15 % tokens and +34 % assertions → **37 % lower tokens per
  correct assertion**.
- **Module scoping.** `flat-facts` matches `full` in cost (±1 %) and
  adds assertions (+10 % on CACC). Scoping inside the facts block is
  cost-neutral but isn't paying for itself either.
- **Deterministic repair.** `no-repair` *increases* assertion count
  and marginally improves tokens/assertion because Phase 1 drops some
  assertions the LLM would happily refine. The repair transforms are
  fast locally but not critical once the feedback loop is available.

---

## 5. Why removing facts *helps* coverage

Classifying the generated assertions by what they check reveals that
facts are not just cost-neutral — they degrade semantic quality.

### CACC — semantic coverage per variant

| Variant | Protocol I/O refs | Handshake | Reset-implications | Arith / sat |
|---|---:|---:|---:|---:|
| full | 8 | 2 | 45 | 35 |
| **no-facts** | **56** | **33** | **87** | **76** |
| flat-facts | 28 | 6 | 49 | 43 |
| no-repair | 37 | 7 | 40 | 90 |

Without the facts card the LLM produces **7× more port/protocol
references and 16× more handshake checks** on CACC. The AST-seeded
structural core (mux selects, passthroughs) is essentially identical
across variants, so the extra content in `no-facts` is net-new
semantic coverage drawn from the spec docs rather than template
expansion.

**Interpretation.** The facts card is a ~100 K-prompt-token budget
item across 59 calls. Evidence suggests it crowds out attention on the
spec-document context, so the LLM spends fewer cycles reasoning about
protocol behavior described in the docs and more cycles re-stating
what the facts block already said. Removing facts frees that attention
budget and the model uses it to generate protocol assertions.

### Quality caveat: no-facts has a small malformed-assertion rate

Six assertions (1.8 %) in the no-facts CACC output are bare fragments
like `(cond) |=> x == $past(y))` — missing the `assert property
(@(posedge clk) disable iff (...))` wrapper. These would be rejected
by a real simulator. Phase 1's `fix_bare_property_fragments` normally
catches these; a small regex edge case lets six slip through. Even
counting these as defects, no-facts has 320 good assertions vs full's
244 and still wins on tokens/assertion (1 745 vs 2 701). Tracked
separately as a Phase 1 bug, not a facts-card issue.

---

## 6. Decision: `use_rtl_facts` now defaults to `False`

Given the CMAC + CACC evidence, carrying the facts card by default is
a net negative on both cost and coverage. Changed in
[sva_pipeline/config.py:287](../sva_pipeline/config.py#L287):

```python
use_rtl_facts: bool = False  # was True
```

Configs that want to re-enable facts for experimentation (e.g. the
ablation runs themselves) must set `agent.use_rtl_facts: true`
explicitly. All ablation YAMLs in `examples/` already do this where
needed, so no existing runs break.

### What this does **not** change

- Module scoping (`module_facts_mode`, `module_scope_depth`) is
  unaffected. Scoping is only meaningful when facts are on.
- Per-batch fact injection (`use_per_batch_facts`) was already OFF by
  default and stays OFF.
- The denylist, grammar constraints, and hallucination controls are
  unchanged.

---

## 7. Caveats and follow-ups

- **Two designs is two data points.** CMAC is tiny and CACC is large,
  which is encouraging, but RVV (mid-size, hierarchical) and FifoX
  (Chisel-emitted) haven't been re-run with the new default. The
  expectation is that facts hurt or at best break even on both, but
  this should be measured, not assumed.
- **The no-facts malformed-fragment rate should be fixed at the
  Phase 1 level.** Those 6 lines expose a regex edge case in
  `fix_bare_property_fragments` — tracking separately.
- **Feedback-loop cost is the dominant variable now.** With facts
  off, most of the token budget is in refinement rounds. Future
  efficiency work should look at making fewer, smarter refinement
  passes rather than shrinking the initial prompt further.
- **Naive-vs-ours headline is untouched by this change.** The naive
  baseline runs with no facts, no AST, no scoping, no repair — it
  was already at the `no-ast + no-feedback` extreme. The headline
  reduction comes from AST + feedback, which this change doesn't
  affect.
