# Assertion Quality Review — Full-Pipeline Runs

_Sources: full-pipeline SVA files for Booth, CMAC, CACC, FifoX, RVV
under `nvdla_cmac_test/`, `nvdla_cacc_test/`, `coral_test/`. Model:
qwen3:14b-32k via Ollama. All counts computed directly from the
emitted SVA files and the paired `lint.json`._

This document characterises the **assertions the pipeline produces**
across the five test designs, then runs a **quality ablation** on
CMAC and CACC to show how each pipeline component contributes to
coverage and correctness. It complements — but doesn't repeat — the
token-efficiency finding in [rq_findings.md](rq_findings.md).

---

## 1. What we mean by "quality"

An assertion is high-quality if it satisfies all of:

1. **Syntactically valid SystemVerilog** — parses with pyslang and
   would be accepted by a commercial simulator.
2. **Well-formed as a property** — immediate assertions terminate
   with `;`, concurrent assertions have a clocking event and
   `disable iff` guard against reset.
3. **References only real signals** — no hallucinated identifiers
   (validated by the signal-map extractor).
4. **Expresses a non-trivial property** — not a tautology, not a
   duplicate of another assertion, not a rewrite of the RHS under
   the same condition.
5. **Covers a functional domain** — protocol behaviour, reset
   semantics, datapath correctness, handshake discipline, counter
   invariants, etc.

Points 1–3 are machine-checkable and enforced by the lint loop +
signal validator. Points 4–5 are what this review inspects.

### Domain taxonomy used throughout

| Tag | Pattern the assertion checks |
|---|---|
| **Mux / ternary** | `x == (sel ? a : b)` — datapath selector correctness |
| **Passthrough** | `port_out == internal_signal` — I/O forwarding |
| **Reset-implication** | `!rstn \|-> x == reset_value` — reset semantics |
| **Sequential update** | `en \|=> reg == $past(src)` — register causality |
| **Handshake** | `valid \|-> ready`, `$stable(data)` — protocol discipline |
| **Counter / FIFO** | `cnt < CAPACITY`, pointer arithmetic |
| **Credit / flow** | `credit_vld \|-> size != 0` |
| **Arithmetic** | saturation, sign, nan handling |
| **Range / bound** | `ptr < N` bound guards |

---

## 2. Per-design quality review

### 2.1 Booth — 32 assertions (AST-only)

| Metric | Value |
|---|---:|
| Assertions | 32 |
| LLM calls | 0 |
| Tokens consumed | 0 |
| ACR (lint pass rate) | 100 % |
| Domain coverage | 1 (case-branch datapath) |

Booth is a pure combinational 14-bit Booth recoder. The AST extractor
finds the `{is_8bit, in_code}` case statement and emits one assertion
per branch:

```systemverilog
assert (!({is_8bit, in_code} == 4'b0000) || (out_data == 17'h10000))
    else $error("out_data mismatch when {is_8bit, in_code}==4'b0000");
assert (!({is_8bit, in_code} == 4'b0110) || (out_data == {src_data[15], ~src_data}))
    else $error("out_data mismatch when {is_8bit, in_code}==4'b0110");
```

**Quality:** high-precision, one assertion per 4-bit Booth encoding.
Every branch of the case is covered. No LLM was ever invoked — AST
alone exhausted the design's testable surface. The pipeline correctly
detected that no LLM enrichment was needed.

**Gaps:** no protocol or reset assertions (Booth has no clock or
reset), no cross-input relational properties.

### 2.2 CMAC — 55 assertions

| Metric | Value |
|---|---:|
| Assertions | 55 |
| LLM calls | 3 (3 ⇒ spec-validation + 1 refinement) |
| Total tokens | 86 145 |
| ACR | 100 % |
| Domain coverage | 3 (passthrough, reset, seq-update) |

CMAC is the MAC multiplier core (1 574 LOC). The pipeline mixes
AST-seeded case-branch assertions with LLM-enriched register-update
properties:

```systemverilog
// AST-seeded case branch (immediate)
assert (!({is_8bit, in_code} == 4'b1111) || (out_data == 17'h100))
    else $error("out_data mismatch when {is_8bit, in_code}==4'b1111");

// LLM-enriched sequential register update
assert property (@(posedge nvdla_core_clk) disable iff (!nvdla_core_rstn)
    ((cfg_reg_en) == 1'b1) |=> cfg_is_fp16_d1 == $past({4{cfg_is_fp16}}))
    else $error("cfg_is_fp16_d1 functional update mismatch");

// LLM-enriched predicate (credit-like)
assert ((nvdla_cmac_a_s_pointer_0_wren) && (producer == reg_wr_data[0]))
    else $error("Producer must be updated to reg_wr_data[0] when wren is asserted");
```

**Quality:** 37 immediate case-branch checks plus 20 concurrent
properties; 5 of the concurrent ones are register-update causality
checks with correct `$past` timing (the `|=>` off-by-one fix
landed before these runs). No bare fragments, no duplicates, no
hallucinated signals.

**Gaps:** no handshake or arithmetic-overflow assertions; the LLM
focused on register updates and reset values.

### 2.3 CACC — 244 assertions

| Metric | Value |
|---|---:|
| Assertions | 244 |
| LLM calls | 59 (spec-validation batches + 1 refinement) |
| Total tokens | 658 940 |
| ACR | 100 % |
| Domain coverage | 6 (mux, passthrough, reset, handshake, credit, seq-update) |

CACC is the ~31 KLOC floating-point accumulator. The largest test
design. Output is a mix of every domain in the taxonomy:

```systemverilog
// Passthrough — port wired through from internal signal
assert (cacc2sdp_valid == dbuf_rd_valid_d3)
    else $error("cacc2sdp_valid passthrough mismatch");

// Sequential register update (LLM-enriched, index-walked)
assert property (@(posedge nvdla_core_clk) disable iff (!nvdla_core_rstn)
    ((calc_op_en_int[75]) == 1'b1) |=> calc_op0_int_75_d1 == $past(calc_elem_75_w))
    else $error("calc_op0_int_75_d1 functional update mismatch");

// Reset-value check
assert property (@(posedge nvdla_core_clk) !nvdla_core_rstn |->
    calc_wr_en_d2 == {8{1'b0}})
    else $error("calc_wr_en_d2 reset value mismatch");

// Conditional register update (immediate form post-repair)
assert (!(dlv_end_tag1_en) || (dlv_end_tag1_addr == $past(dlv_end_tag1_addr_w)))
    else $error("dlv_end_tag1_addr functional update mismatch");
```

**Quality distribution:**

| Domain | Count | % |
|---|---:|---:|
| Reset-implication | 182 | 75 % |
| Sequential update (`\|=>` + `$past`) | 132 | 54 % |
| Mux / ternary | 54 | 22 % |
| Protocol port refs | 32 | 13 % |
| FIFO / counter | 5 | 2 % |
| Passthrough | 2 | < 1 % |
| Credit | 1 | < 1 % |

Categories overlap (a register-update assertion typically includes
reset guarding), so percentages sum > 100.

**Gaps:** protocol/handshake coverage is thin (1 handshake assertion
out of 244). Credit-pulse discipline gets 1 assertion. The output
heavily favours mechanical register-update enumeration over
spec-level protocol properties — a direct consequence of the facts
block crowding out spec-document attention (see
[ablation_findings_cacc.md](ablation_findings_cacc.md), §5).

### 2.4 FifoX — 25 assertions

| Metric | Value |
|---|---:|
| Assertions | 25 |
| LLM calls | 2 |
| Total tokens | 57 529 |
| ACR | 100 % |
| Domain coverage | 4 (passthrough, reset, seq-update, range) |

FifoX is Chisel-emitted — SystemVerilog that uses generic types and
`_T_`/`_GEN_` intermediate wires. The pipeline handled it without
needing special-case support:

```systemverilog
// Chisel port passthrough
assert (io_in_ready == io_in_ready_0)
    else $error("io_in_ready passthrough mismatch");

// Reset of memory array
assert property (@(posedge clock) disable iff (reset) (reset) |=> mem_4 == 8'h0)
    else $error("mem_4 must reset to 8'h0");

// Bounds check on internal pointer
assert property (@(posedge clock) inxpos_2 < 4'd10)
    else $error("inxpos_2 must stay within 0-9");
```

**Quality:** 11 concurrent sequential properties, 6 range-bound
checks (pointer stays within capacity), 3 passthrough, 2 reset.
All compile; none reference hallucinated signals. The LLM
correctly inferred bound invariants from the FIFO depth parameter
even though the Chisel output names the pointer register cryptically.

**Gaps:** no handshake assertions for the Decoupled `valid/ready`
protocol, which is the canonical property for a Chisel-emitted FIFO.
This would be the headline assertion a human would write. The pipeline
misses it.

### 2.5 RVV — 92 assertions

| Metric | Value |
|---|---:|
| Assertions | 92 |
| LLM calls | 2 |
| Total tokens | 14 363 |
| ACR | 98 % |
| Domain coverage | 5 (mux, passthrough, reset, handshake, counters/FIFO) |

RVV is the Coral NPU RISC-V Vector backend — 12 files, 1 591 LOC,
hierarchical (ROB, arbitration, retire). Output shows the broadest
protocol coverage of any design:

```systemverilog
// Arithmetic — round-robin grant generation
assert (grant_tmp == ({req,req} & ~({req,req} - (2*REQ_NUM)'(prio))))
    else $error("grant_tmp assignment mismatch");

// Pipeline register forwarding
assert (datain_seq == datain)
    else $error("datain_seq passthrough mismatch");

// Conditional write enable composition
assert (w_frf[j] == ((rob2rt_write_data[j].w_type==FRF) && rob2rt_write_data[j].w_valid))
    else $error("w_frf[j] assignment mismatch");

// Handshake/ready composition
assert (vxsat2rt_ready == (~(w_vrf_valid&w_vxsat) | {3{vxsat2rt_write_ready}}))
    else $error("vxsat2rt_ready assignment mismatch");
```

**Quality:** 25 protocol-port references, 17 counter / ROB-pointer
invariants, 7 handshake-composition checks, 23 passthroughs, 10
mux-select correctness. This is the best-balanced distribution in
the set — likely because the design has more protocol surface per
LOC than NVDLA's datapath-heavy modules.

**Gaps:** only 3 reset-implication assertions despite the design
having multiple reset domains. A few sequential properties the
design obviously has (ROB-pointer monotonicity, one-hot grant)
aren't explicitly emitted.

---

## 3. Cross-design coverage summary

| Design | Assertions | ACR | Mux | Pass | Reset | Seq | Handshake | FIFO/cnt | Arith | Domains covered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Booth | 32 | 100 % | 32 | — | — | — | — | — | — | 1 |
| CMAC | 55 | 100 % | — | 1 | 20 | 5 | — | — | — | 3 |
| CACC | 244 | 100 % | 54 | 2 | 182 | 132 | 1 | 5 | — | 6 |
| FifoX | 25 | 100 % | 1 | 3 | 2 | 11 | — | 1 | — | 4 |
| RVV | 92 | 98 % | 10 | 23 | 3 | — | 7 | 17 | — | 5 |

### Consistent across designs

- **Zero hallucinated signals** in any full run. The signal-map
  validator catches everything.
- **Zero duplicates.** Dedup works.
- **100 % lint-clean** after feedback-loop convergence on every
  design except RVV (98 %, 2 assertions dropped at iteration limit).

### Inconsistent

- **Handshake coverage is weak-to-absent** on CACC, CMAC, FifoX.
  RVV is the only design where the LLM systematically writes
  handshake properties. This correlates with RVV's protocol-heavy
  surface, not pipeline quality.
- **Arithmetic / overflow** properties are missing across all
  designs. The LLM doesn't emit saturation or sign-extension
  assertions even on designs (CMAC, CACC, RVV) where these are
  clearly present in the RTL.
- **Register-update assertions dominate CACC output** (132/244 =
  54 %) because the AST skeletons seed heavily on always_ff blocks.
  This is mechanical coverage that reads as repetitive.

---

## 4. Quality ablation — which components drive which quality dimensions

Same design, same model, one component disabled per row. Data comes
from the CMAC (`cmac_ablations_20260420_0929/`) and CACC
(`cacc_ablations_20260420_0944/`) ablation batches.

### 4.1 CMAC quality by ablation

| Variant | Assertions | ACR | Domains | Reset-impl | Seq-update | Bare fragments | Notable |
|---|---:|---:|---:|---:|---:|---:|---|
| naive baseline | 0 | 0 % | 0 | 0 | 0 | — | Wholesale lint rejection (emits `property…endproperty` without `assert`) |
| no-ast | 11 | 100 % | 1 | 11 | 0 | 0 | Reset-only coverage; no datapath or sequential checks |
| no-facts | 52 | 100 % | 2 | 14 | 0 | 1 | Drops to 2 domains but 52 good assertions |
| flat-facts | 53 | 100 % | 3 | 15 | 5 | 0 | Restores seq-update coverage; equal to full minus 2 |
| no-repair | 50 | 100 % | 3 | 12 | 3 | 0 | LLM covers for missing Phase 1 via feedback loop |
| **full** | 55 | 100 % | 3 | 15 | 5 | 0 | Baseline — 3 domains, balanced |

**Reading the rows:**

- **no-ast vs full**: collapsing from 55 to 11 assertions is not
  just quantity — the 11 surviving are 100 % reset-implications.
  The mux and sequential-update domains vanish entirely. AST
  provides the structural scaffold the LLM builds on.
- **naive vs full**: 55 to 0 is the total-collapse case. No ACR
  either, because the output isn't even lint-legal.
- **no-facts vs full**: minor quality dip (55→52, 3→2 domains).
  Sequential-update assertions disappear — the LLM stops writing
  them without the facts block reminding it of flop semantics.
  One bare-fragment slip gets through.
- **flat-facts vs full**: nearly identical (53 vs 55). Dropping
  module scoping costs 2 assertions and nothing else.
- **no-repair vs full**: 50 vs 55. The 5 missing are ones Phase 1
  would have cleaned up locally; the feedback loop recovers most
  but not all.

### 4.2 CACC quality by ablation

| Variant | Assertions | ACR | Domains | Reset | Seq-update | Bare fragments | Notable |
|---|---:|---:|---:|---:|---:|---:|---|
| naive baseline | 1 | 33 % | 1 | 0 | 0 | 0 | Single handshake assertion survives |
| no-ast | 6 | 100 % | 4 | 6 | 0 | 0 | Vague protocol stabs; no mux, no register updates |
| no-facts | 326 | 98 % | **7** | 87 | 159 | 9 | Widest coverage — *arithmetic* domain appears |
| flat-facts | 270 | 100 % | 7 | 49 | 94 | 0 | Same 7 domains as no-facts but fewer per-domain assertions |
| no-repair | 315 | 100 % | 7 | 40 | 93 | 0 | More immediate assertions; LLM covers Phase 1's gap |
| full | 244 | 100 % | 6 | 45 | 132 | 0 | Baseline — missing *arithmetic* domain |

**Reading the rows:**

- **no-ast catastrophic collapse**: 244 → 6. The 6 are
  handshake + reset + credit + arithmetic stabs, no mux or
  sequential content. This mirrors CMAC but at larger scale: AST
  is load-bearing for the structural majority of assertions.
- **naive baseline**: 1 assertion at ACR = 33 %. Effectively
  useless output at 45 839 tokens.
- **no-facts produces the *broadest* coverage** (7 domains
  including arithmetic, which full misses). Token-efficiency and
  semantic coverage agree — facts are a net negative here.
- **flat-facts**: same 7 domains, 270 vs 326 assertions. Module
  scoping costs ~20 % of assertions without adding domains.
- **no-repair is the highest-domain, highest-immediate-count
  variant**: disabling Phase 1 doesn't reduce quality — the
  feedback loop re-prompts the LLM and it writes *more*
  immediate-form assertions that would otherwise be rewritten
  locally.
- **Bare-fragment count**: 9 on no-facts. The Phase 1 regex edge
  case leaks on a small number of assertions (~3 % of output).
  Still a bug worth fixing.

### 4.3 Component → quality dimension summary

| Component | What it buys (quality) | Evidence |
|---|---|---|
| **AST skeletons** | Structural coverage — mux, passthrough, reset-value, register-update. Without AST the output is a handful of generic protocol stabs. | CMAC 55→11, CACC 244→6. |
| **Feedback loop** | Lint correctness + ACR. Without it nothing compiles. | no-feedback = 0 assertions on both designs. |
| **RTL facts card** | Narrower, more register-update-focused output. Not always a quality win. | On CACC, facts *reduces* domain count from 7 to 6; on CMAC, +1 domain. |
| **Module scoping** | Marginal: ~20 % fewer assertions, same domain count. | flat-facts vs full on CACC. |
| **Phase 1 repair** | Safety net for malformed fragments; feedback loop recovers most without it, but edge cases leak (see 9 bare fragments in CACC no-facts). | no-repair: 50 → 55 on CMAC; 315 → 244 on CACC (actually *fewer* with repair on because Phase 1 sometimes drops). |
| **Naive single-prompt** | The zero baseline — fails outright on mid-to-large designs. | 0, 1, 0, 36, 0 assertions on CMAC/CACC/RVV/FifoX/RVV. |

---

## 5. Observations and limitations

### What the output is good at

- **Structural datapath coverage.** Mux selects, passthroughs, and
  register-update patterns are enumerated exhaustively. AST + the
  lint loop + signal validation make this reliable.
- **Reset semantics.** Every clocked register of interest gets a
  reset-value check. Phase 1's `disable iff` and `|->` normalizers
  make these all conform to one style.
- **Lint-legal output.** Across 5 designs and 11 ablation variants
  the only non-lint-clean artefact is a Phase 1 regex bug that
  produces 9 bare fragments out of 326 assertions on CACC
  no-facts. The rest compiles cleanly.

### What the output is weak at

- **Handshake protocol properties.** Except on RVV, the LLM does
  not systematically write `valid |-> ready` or `valid && !ready
  |-> $stable(data)` patterns. FifoX misses its Decoupled-protocol
  handshake entirely.
- **Arithmetic / overflow / saturation.** No assertion in any full
  run checks for saturation, sign-extension correctness, or nan
  handling — even on CMAC and CACC where these are explicit in the
  RTL.
- **Cross-module invariants.** The pipeline emits per-module
  assertions; cross-module properties ("if A.sent then B.received
  within 3 cycles") are not in the output on RVV or CACC.
- **Mechanical repetition at scale.** On CACC, 132 of 244
  assertions are register-update patterns like `(en[i]) |=> reg_i
  == $past(src_i)`. This reads as bulk, not insight.

### Metric caveats

- **ACR is lint-based**, not semantic. All five full runs score
  ≥ 98 %, but this measures compile-ability, not whether the
  assertion expresses the *intended* property.
- **Domain coverage is keyword-based** (see §1 taxonomy). An
  assertion that mentions `ready` gets tagged handshake even if
  it's actually a passthrough. The counts are indicative, not exact.
- **No mutation-kill measurements yet.** The cleanest quality
  number — how many injected RTL mutations does the assertion set
  catch — requires a simulator loop we haven't integrated. This is
  the single biggest gap in this evaluation.

### Follow-ups

1. Wire up fault-injection + mutation-kill scoring on CMAC and
   FifoX (smallest testbeds) before scaling to CACC.
2. Fix the Phase 1 regex edge case that lets 9 bare fragments
   through on CACC no-facts.
3. Prompt-engineering experiment: can a targeted "add handshake
   coverage" re-prompt raise FifoX's handshake count to ≥ 3?
4. Try the flipped default (`use_rtl_facts: false`) on FifoX and
   RVV to confirm the coverage gain seen on CACC generalises.
