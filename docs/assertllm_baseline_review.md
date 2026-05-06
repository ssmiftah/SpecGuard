# AssertLLM Baseline — Quality Review

_Source: `baselines/results_20260429_1009/` — 11 designs × 1 rep_

This document audits the output of our standalone AssertLLM
reproduction against two questions:

1. **Is the output correct?** — does it lint, are widths right, are
   the assertions well-formed SystemVerilog?
2. **Is it useful?** — what kind of properties does it actually
   verify, and how does it compare to SpecGuard's output on the same
   designs?

Short answer: **mostly correct, mostly trivial**. The AssertLLM
reproduction emits lint-clean width checks at 5–60× our pipeline's
per-assertion token cost. Two minor bugs and a class of
verification-trivial outputs are documented below.

---

## 1. Per-design output inventory

Token counts are from the per-run `token_summary.json`. "$bits-only"
counts assertions whose body is a pure `$bits(sig) == N` width check
— either as immediate or wrapped in concurrent property syntax.

| Design | Total | $bits-only | Concurrent | Truncated | Sections (W / C / F) | Tokens | tok/A |
|---|---:|---:|---:|---:|---:|---:|---:|
| coral_fifox | 1 | 1 | 0 | 0 | 1 / 0 / 0 | 39 895 | 39 895 |
| coral_rvv | 17 | 17 | 0 | 0 | 16 / 0 / 1 | 32 511 | 1 912 |
| nvdla_cmac | 6 | 6 | 0 | 0 | 6 / 0 / 0 | 28 617 | 4 770 |
| nvdla_rubik | 7 | 7 | 0 | 0 | 7 / 0 / 0 | 81 448 | 11 635 |
| nvdla_cacc | 12 | 11 | 1 | **1** | 10 / 0 / 2 | 88 214 | 7 351 |
| nvdla_pdp | 9 | 8 | 1 | 0 | 8 / 0 / 1 | 127 166 | 14 130 |
| nvdla_cdp | 4 | 4 | 0 | 0 | 3 / 0 / 1 | 101 214 | 25 304 |
| nvdla_cmacfull | 82 | 79 | 1 | 0 | 80 / 1 / 1 | 792 272 | 9 662 |
| nvdla_cdma | 7 | 5 | 0 | 0 | 7 / 0 / 0 | 160 297 | 22 900 |
| nvdla_csc | 97 | 94 | 17 | **2** | 94 / 0 / 3 | 1 556 956 | 16 051 |
| nvdla_sdp | 17 | 15 | 4 | 0 | 15 / 0 / 2 | 235 332 | 13 843 |

### Key observations from the inventory

- **Width section dominates everywhere** — 91 % to 100 % of every
  design's output is `$bits()` width checks.
- **Connectivity section is essentially unused** — exactly **1**
  connectivity assertion across all 11 designs (in cmacfull). The
  prompt explicitly asks for them; the LLM almost never produces any.
- **Function section is sparse** — 1–4 function assertions per
  design, totalling 11 across the corpus.
- **CSC produced 17 concurrent assertions** but every single one is
  a `$bits()` check wrapped in `assert property (@(posedge clk)
  disable iff (!rstn) $bits(sig) == 8)`. Verification-trivial — see
  §3 below.

---

## 2. Are the widths actually correct?

Spot-checked AssertLLM's claimed widths against the actual RTL port
declarations on `nvdla_cacc`:

| Signal | AssertLLM says | RTL declares | Verdict |
|---|---|---|---|
| `mac_a2accu_pd` | 9 bits | `input [8:0]` (= 9) | ✓ correct |
| `mac_b2accu_data0` | 176 bits | `input [175:0]` (= 176) | ✓ correct |
| `mac_b2accu_mask` | 8 bits | `input [7:0]` (= 8) | ✓ correct |
| `mac_b2accu_mode` | 8 bits | `input [7:0]` (= 8) | ✓ correct |
| `nvdla_core_clk` | 1 bit | scalar | ✓ correct |
| `pwrbus_ram_pd` | 32 bits | `input [31:0]` | ✓ correct |

**Width values are reliably correct.** The Spec Analyzer phase pulls
the right width out of the docs (or the elaboration metadata it gets
fed). This is the assertion category the LLM is best at because
there's a deterministic right answer.

---

## 3. Bugs and quality smells

### 3.1 Two truncated assertions

`nvdla_cacc/run_01/sva.sv` line 21:

```sv
assert ( $bits(cacc2csb_resp_pd) == 34
```

No closing paren, no `else $error`. The LLM hit `max_tokens` mid-line
on the [width] section batch. This won't compile.

`nvdla_csc/run_01/sva.sv` has 2 similar truncations (out of 97).

**Impact:** all three would be rejected by a real simulator. They're
counted in our `final_assertions` because the regex `^\s*assert\b`
matches the start; pyslang lint missed them because we don't elaborate
the AssertLLM output.

### 3.2 `$bits()` wrapped in concurrent SVA — pointless

CSC has 17 assertions of this shape:

```sv
assert property (@(posedge nvdla_core_clk) disable iff (!nvdla_core_rstn)
                 $bits(sc2mac_dat_b_data96) == 8) else $error("...");
```

`$bits()` is a compile-time constant. Wrapping it in concurrent
property syntax means the simulator re-checks the same constant on
every clock edge — semantically identical to a one-time elaboration
check, but **80× more expensive** at runtime. A verification engineer
would write this as the immediate form `assert ($bits(...) == 8);` or
better, leave it to the linter.

### 3.3 Numbered-signal width-check inflation

`nvdla_cmacfull` and `nvdla_csc` produce 80 and 97 assertions
respectively — far more than any other design — but the explanation
is unflattering: both designs have many indexed copies of the same
signal (`sc2mac_dat_data10`, `sc2mac_dat_data100`, …,
`sc2mac_dat_data127`). AssertLLM emits one width-check per indexed
copy because it processes each as a "different" signal. They're all
8 bits, all redundant. The 80-line CMAC-full output reduces to ~3
distinct assertions if you dedup by `8'-bit-width-of-some-data-byte`.

### 3.4 Connectivity section is unused

The prompt explicitly asks for `[connectivity]` assertions —
"check that the signal connects correctly to the listed
interconnection signals." Across 11 designs and ~260 generated
assertions, **exactly 1 connectivity assertion was emitted**.

The cause: the Spec Analyzer's `interconnection` field is often
empty because our markdown spec docs don't enumerate per-signal
neighbours the way the AssertLLM paper's PDF specs do. Without that
input, the SVA Generator has nothing to ground a connectivity
assertion in, so it emits none. Honest output, but it means
~33 % of the prompt's three-way categorisation is dead weight on our
benchmark.

### 3.5 Function assertions are rare but include the only real-quality content

The 11 function assertions across all 11 designs are the only place
the AssertLLM reproduction produces verification content that matters.
The two best examples (both from `nvdla_sdp`):

```sv
assert property (@(posedge nvdla_core_clk) disable iff (!nvdla_core_rstn)
    (cacc2sdp_valid && !cacc2sdp_ready) |-> $stable(cacc2sdp_pd))
    else $error("cacc2sdp_pd must be stable during backpressure (valid=1 and ready=0)");

assert property (@(posedge nvdla_core_clk) disable iff (!nvdla_core_rstn)
    (cacc2sdp_valid && !cacc2sdp_ready) |-> (cacc2sdp_valid == $past(cacc2sdp_valid)))
    else $error("cacc2sdp_valid must remain stable during backpressure");
```

These are **textbook handshake-stability properties**, exactly what a
verification engineer would write by hand. They're also the *only*
properties in the entire corpus that exercise SVA temporal operators
(`$stable`, `$past`, `|->`). Note that across the 11 large-design
SpecGuard runs in the same week, **zero** assertions used `$stable`
or `$rose` either — so AssertLLM is the only method that produced
these 2 backpressure properties.

---

## 4. Side-by-side per-token efficiency vs SpecGuard

Comparing AssertLLM (n=1) against SpecGuard's no-facts default
(n=1–15 per design from prior batches):

| Design | AssertLLM tok/A | SpecGuard tok/A | AssertLLM is … |
|---|---:|---:|---|
| coral_fifox | 39 895 | 1 731 | **23× worse** |
| coral_rvv | 1 912 | 122 | **16× worse** |
| nvdla_cmac | 4 770 | 1 037 | 4.6× worse |
| nvdla_rubik | 11 635 | 4 091 | 2.8× worse |
| nvdla_cacc | 7 351 | 1 129 | 6.5× worse |
| nvdla_pdp | 14 130 | 2 409 | 5.9× worse |
| nvdla_cdp | 25 304 | 418 | **60× worse** |
| nvdla_cdma | 22 900 | 4 810 | 4.8× worse |
| nvdla_cmacfull | 9 662 | (no-facts INCOMPLETE) | n/a |
| nvdla_csc | 16 051 | (no-facts INCOMPLETE) | n/a |
| nvdla_sdp | 13 843 | (data-collection issue) | n/a |

**On every shared cell, AssertLLM is 2.8× to 60× more expensive per
emitted assertion.** The cause is straightforward: per-signal × 3-
category prompting fires 30–1 134 LLM calls per design. Each call is
small (avg ~1 K prompt tokens) but the call count is huge, and the
output per call is one tiny width check.

CDP is the most extreme case — SpecGuard hits 19 assertions in 8 K
tokens (tok/A 418); AssertLLM produces 4 width checks in 101 K tokens
(tok/A 25 304).

---

## 5. Verdict

### Is the output correct?

**Mostly yes, with minor caveats:**
- ✅ Widths claimed match the RTL.
- ✅ 99 % of generated assertions parse as valid SVA.
- ⚠️ 3 of ~260 assertions are truncated mid-line (max_tokens cutoff).
- ⚠️ The 17 CSC concurrent-`$bits()` assertions are a SVA misuse —
  technically valid but semantically pointless.

### Is the output useful?

**As a verification artefact: weak.** Almost all of it is bit-width
declarations re-stated from the RTL. The 11 function assertions
(across all 11 designs combined) are the only content that resembles
a property a human verifier would write by hand. The CSC/CMAC-full
"high yields" of 97 and 82 assertions are width-check inflation
across numbered signal copies, not real coverage.

**As a comparison baseline: ideal.** AssertLLM's pattern of "many
small calls, almost all width checks" gives us a clear contrast point
in the paper: SpecGuard's 30–400+ assertions per design are
structurally diverse (mux, reset, sequential `$past`, handshake) at
2.8–60× lower per-assertion token cost.

### What changes the picture

If we ran AssertLLM with **richer spec docs** that included
per-signal interconnection lists (as the paper's I2C/ECG/Pairing
specs do), the connectivity section would populate and the corpus
would diversify. Our current markdown specs are intentionally lean
(rationale: AssertLLM is being measured under matched conditions
with the same docs SpecGuard sees), so the connectivity gap is part
of the comparison, not a bug to fix.

If we wanted to make AssertLLM more competitive, the two interventions
that would help most are:
1. Batch signals — instead of one prompt per signal, ask for "all
   signals' width checks in one call". Cuts tok/A by ~10×.
2. Add a per-design pre-prompt with the spec's protocol section, so
   the function-category prompt sees handshake / credit / mode
   information instead of an empty interconnection list.

Both would dilute AssertLLM relative to the paper's described method,
so we leave them out of the reproduction and report the honest result.

---

## 6. Files referenced

- `baselines/results_20260429_1009/<design>/run_01/sva.sv`
- `baselines/results_20260429_1009/<design>/run_01/token_summary.json`
- `baselines/assertllm/spec_analyzer.py` (LLM #1 prompts)
- `baselines/assertllm/sva_generator.py` (LLM #3 prompts)
- AssertLLM paper: `Baselines/assertllm.pdf`
- Companion SpecGuard analyses: [rq_findings.md](rq_findings.md),
  [full_pipeline_quality_review.md](full_pipeline_quality_review.md)
