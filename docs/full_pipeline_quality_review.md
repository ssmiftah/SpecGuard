# Full-Pipeline Assertion Quality Review (qwen3.6:35b)

_Source: `ablation_study_20260424_0549/<design>/full/run_01/sva.sv`
across the 4 designs that were run on the 35B model
(fifox / cmac / rvv / cacc). Cross-checked against the 15-rep
distribution per design. Sample is the rep-01 SVA file; classification
counts are stable across the 15 reps because 35B is highly
deterministic (CV 0–6 % on assertion counts)._

This review judges the assertions on **syntax** (does pyslang accept
them), **structural quality** (well-formed properties, real signals,
no duplicates), **complexity** (what SVA constructs they exercise),
and **semantic plausibility** (do they express something the design
actually guarantees).

---

## 1. Headline numbers

| Design | LOC | Assertions (rep 01) | 15-rep range | Lint status | Avg length (chars) | Max length |
|---|---:|---:|---:|---|---:|---:|
| FifoX | 586 | **24** | 24–24 (CV 0.0 %) | ALL_PASSED | 122 | 194 |
| CMAC | 1 574 | **54** | 54–62 (CV 3.8 %) | ALL_PASSED | 138 | 201 |
| RVV | 1 591 | **90** | 90–92 (CV 0.6 %) | 90 ok / 2 dropped | 107 | 194 |
| CACC | 30 905 | **762** | 703–902 (CV 6.3 %) | ALL_PASSED | 206 | 249 |

Across all 4 designs (180 cells × 15 reps = 2 700 generated rep-files,
sampled from rep 01):

- **Zero bare-fragment assertions** (no missing `assert property
  (@(posedge clk) disable iff (...))` wrappers).
- **Zero duplicates** within a file.
- **Zero hallucinated signals** (every reference is in the design's
  signal map; signal-validation pass dropped nothing).
- **One non-trivial lint drop**: RVV had 2 of 92 generated assertions
  rejected by pyslang, leaving 90.

---

## 2. Pattern distribution

Every assertion was tagged by the SVA construct it uses. Counts
below are for rep 01.

| Pattern | FifoX | CMAC | RVV | CACC | What it tests |
|---|---:|---:|---:|---:|---|
| Simple immediate (`assert (a == b)`) | 4 | 33 | 78 | 4 | Combinational equality / passthrough |
| Ternary multiplexer (`a == (s ? b : c)`) | 0 | 0 | 10 | **172** | Datapath mux selection |
| Implication (`!cond \|\| (val == X)`) | 0 | 32 | 0 | 0 | Case-branch enumeration |
| Reset implication (`!rst \|-> val == reset_val`) | 0 | 15 | 0 | **131** | Async reset values |
| Sequential `\|=> $past(...)` | 6 | 6 | 0 | **419** | Register update causality |
| Concurrent predicate (`cond \|-> ...`) | 0 | 0 | 0 | 0 | Spec-level invariants |
| Range / arithmetic | 5 | 0 | 24 | 30 | Counter math, bit-arithmetic |
| Bit-indexed (`x[N:M]`) | 0 | 9 | 2 | **147** | Sliced datapath checks |
| Concatenation (`{a, b}`) | 2 | 37 | 10 | 91 | Bit-pack literals |
| Compound predicate (`&&` / `\|\|`) | 0 | 32 | 16 | 0 | Multi-condition gates |

**What's missing across every cell:**

- `$stable(x)` — backpressure data-stability checks
- `$rose(x)` / `$fell(x)` — edge-detected protocol events
- `$onehot(x)` / `$onehot0(x)` — one-hot encoding invariants
- `throughout` / `until` — temporal sequence operators
- `##N` timing — multi-cycle protocol assertions
- `s_eventually` / `s_always` — liveness / safety operators

The pipeline produces a **flat-temporal** style: same-cycle
implications and one-cycle-delayed `$past` updates. It does not yet
emit assertions that span more than one clock edge, and does not
exercise the higher-level SVA temporal language.

---

## 3. Complexity scaling

Average assertion length scales with design complexity (not with
the number of assertions):

```
LOC      avg length   max length
586      122          194        (FifoX)
1574     138          201        (CMAC)
1591     107          194        (RVV)
30905    206          249        (CACC)
```

CACC's 206-character average reflects the fact that its assertions
predicate on multi-signal compound conditions (`((layer_st |
accu_valid[8]) == 1'b1) |=> accu_cnt == $past(accu_cnt_w)`) and use
3-way ternary nests for FP16 NaN/zero handling.

RVV is the most concise: short immediate equalities like `assert
(prio_en == (|req))` express genuine arithmetic invariants in 1
line.

---

## 4. Per-design quality narrative

### 4.1 FifoX — small, deterministic, but with one quality smell

**Strengths**

- 24 assertions, **identical across all 15 reps** (zero variance).
  Every `mem_N == $past(8'h0)` line is bit-identical.
- The `mcount` invariant is a real arithmetic property:
  `(ivalid | dec) |=> mcount == $past(mcount + {1'h0, ivalid ?
  icount : 3'h0} - {3'h0, dec})` — captures the FIFO occupancy
  update with valid/dec gating.
- The `count == $past(count + ivalid - ovalid)` assertion is the
  classic "next-cycle count = current + push - pop" — exactly the
  invariant a human would write.

**Weaknesses**

- Ten of the 24 assertions follow the pattern
  `assert property (@(posedge clock) disable iff (!reset) mem_N ==
  $past(8'h0))`. Decoded: "outside reset, mem_N always equals the
  past value of literal 0 (i.e. always 0)." This is **wrong for a
  FIFO storage cell** — `mem_N` is supposed to *change* on writes.
  These assertions are tautologies that lint accepts but a real
  simulator would fail the moment any data is pushed.

- **`disable iff` polarity is mixed.** Some assertions use
  `disable iff (reset)` (correct for active-high reset on FifoX);
  others use `disable iff (!reset)` (wrong polarity — disables only
  when not in reset). Both lint-pass because pyslang doesn't enforce
  reset-polarity semantics. A human reviewer would catch it.

### 4.2 CMAC — case-branch enumeration done well

**Strengths**

- 32 of 54 assertions are case-branch enumerations like
  `assert (!({is_8bit, in_code} == 4'b0001) || (out_data ==
  {~src_data[15], src_data}))`. Every Booth recoder encoding gets
  its own assertion. **This is the right shape for a combinational
  decoder**: the AST extracted the case statement, the LLM filled in
  the expected output per branch.
- 15 reset-implication assertions cover every pipeline register
  with its async-reset value.
- 6 sequential register-update assertions correctly bind
  `$past(cfg_is_int8)` to the next-cycle latched value with
  `cfg_reg_en` as the predicate.
- All 54 lint cleanly on every rep; only one rep produced an
  outlier 62-assertion file.

**Weaknesses**

- The output has **no protocol or handshake assertions** —
  CMAC has CSB request/response interfaces and a back-pressure
  `_pvld / _prdy` pair, but none of the 54 assertions check the
  handshake discipline.
- No arithmetic invariant on the multiplier itself (e.g.
  `result == src_a * src_b` for any configured precision mode).

### 4.3 RVV — most concise, real arithmetic, one lint drop

**Strengths**

- The single most-impressive assertion in the corpus:
  ```sv
  assert (grant_tmp == ({req,req} & ~({req,req} - (2*REQ_NUM)'(prio))));
  ```
  This is the textbook round-robin priority-shifted grant
  formula — duplicated request mask, subtract priority, NAND with
  the original. The model recognised the pattern and reproduced it
  exactly.
- `assert (prio_en == (|req))` — OR-reduction = "any request
  pending."
- `assert (prio_new == ({grant[REQ_NUM-2:0],grant[REQ_NUM-1]}))` —
  one-cycle priority rotation.
- 78 of 90 are immediate equalities expressing real datapath
  arithmetic; this is the highest density of "spec-grade" content
  in the benchmark.

**Weaknesses**

- 2 of 92 generated assertions failed pyslang and were dropped
  (the `EMPTY_REFINEMENT` status indicates the lint loop tried but
  couldn't fix them). This is the only design with a non-zero lint
  drop on 35B-full.
- The `assert (en == (single_push ? ({{(DEPTH-1){1'b0}},1'b1} <<
  wrPtr) : {DEPTH{1'b0}}))` line uses `DEPTH` as if it were a
  parameter — correct in context, but a tool that can't resolve
  parameter elaboration would reject it.
- No `$onehot(grant)` assertion despite the design clearly needing
  one — the LLM expressed grant correctness as an arithmetic
  identity instead of as a one-hot invariant.

### 4.4 CACC — broadest coverage, dominated by bulk register updates

**Strengths**

- **762 assertions** at CV 6.3 % — the largest output in the
  benchmark and remarkably stable.
- Three-way ternary selectors for FP16 special-value handling, e.g.
  ```sv
  assert (di_mans[35:0] == (oi_nan ? {36'b0} : di_nan ?
      {di_mant_pre[37:27],25'b0} : di_manm));
  ```
  This is the actual NaN-mantissa-mask formula from the RTL —
  the LLM correctly expressed the priority of `oi_nan` over
  `di_nan` over the default.
- Compound-predicate sequential updates like
  ```sv
  ((in_valid & ~in_nan) == 1'b1) |=> in_mant_cut == $past(in_mant_cut_nxt)
  ```
  combine two source signals into the gating predicate — the LLM
  read the real always_ff sensitivity.
- Reset to non-zero literal: `accu_channel_st == {9{1'b1}}` is
  expressed as a 9-bit replicated-1 literal, not just `9'h1FF`.
- 131 reset-implication assertions cover every clocked register's
  initial value.

**Weaknesses**

- **Bulk repetition is the dominant texture.** 419 sequential
  `$past`-style assertions and 172 ternary mux assertions are
  produced by walking through indexed signals (`calc_op0_int_16`,
  `calc_op0_int_17`, … `calc_op0_int_55`). 200+ of the 762
  assertions are essentially the same template with different
  indices. This is *correct* but *repetitive*.
- **No protocol assertions on the credit interface** despite CACC
  driving `accu2sc_credit_vld / size`. The credit-conservation
  invariant (the most important spec-level CACC property) does not
  appear in any of the 15 rep files.
- **No `$stable` backpressure check** on the `cacc2sdp_valid /
  ready` output handshake.

---

## 5. Cross-design quality patterns

### 5.1 What the model does well

1. **Combinational case enumeration.** Booth/CMAC case statements
   become per-branch assertions with the right output formula —
   accurate, complete, lint-clean.
2. **Pipeline register update with `$past`.** The model consistently
   binds `register_d1 == $past(register)` with the correct enable
   predicate.
3. **Reset-value identification.** Async resets are correctly
   modelled with `disable iff` clauses and the right reset value
   (including non-zero replicated literals like `{9{1'b1}}`).
4. **Multi-way ternary selectors for special-value handling.** FP16
   NaN/zero precedence chains are expressed correctly.
5. **Real arithmetic invariants.** RVV's round-robin grant formula
   and FifoX's count-update formula are both genuine spec-level
   properties expressed as immediate equalities.

### 5.2 What the model doesn't do

1. **No SVA temporal operators.** `$stable`, `$rose`, `$fell`,
   `$onehot`, `throughout`, `until`, `##N`, and the `s_*` operators
   are absent across the corpus. The output is "structural SVA":
   same-cycle implications and one-cycle-delay `$past` updates.
2. **Protocol-level invariants are missing.** Credit conservation
   (CACC), CSB request/response pairing (CMAC), valid-ready
   stability under back-pressure (any of them) — none of these
   appear unless the AST extractor specifically points at the
   underlying always_ff block.
3. **One-hot mode bits are checked by reset-value, not by mutual
   exclusion.** The `cfg_is_int8`/`cfg_is_int16`/`cfg_is_fp16`
   one-hot is enforced only through individual reset assertions; no
   `assert ($onehot({cfg_is_int8, cfg_is_int16, cfg_is_fp16}))`
   appears.
4. **No cross-module invariants.** Every assertion targets one top
   module's signals; properties that span hierarchy are not
   emitted.

### 5.3 Quality smells worth fixing

- **FifoX `mem_N == $past(8'h0)` tautologies** are the clearest
  semantic bug in the corpus — 10 assertions that lint-pass but
  describe wrong behaviour. The pattern came from the AST visiting
  the reset-value initialiser without realising the register has
  data inputs.
- **FifoX `disable iff` polarity inconsistency** — some assertions
  use the active-low form, some use the active-high form. Pyslang
  doesn't catch this; a real simulator with the correct reset
  semantics would.
- **CACC bulk-expansion repetition** — 200+ near-duplicate
  assertions across `calc_op0_int_16`…`_55` lower the
  signal-to-noise ratio. The "true" assertion content of CACC is
  closer to 200–250 distinct properties; the rest is mechanical.

---

## 6. Quality vs the 14B model

Same designs, same configs, different model. The 35B output:

- Has **dramatically less variance** (CV 0–6 % vs 14B's 4–26 %) —
  the 35B's `full` runs on FifoX produce **24.0 ± 0.0** assertions
  across 15 reps; the 14B was 33.3 ± 5.2.
- Is **moderately better-formed at the boundary cases** — the 14B
  occasionally produced bare fragments that needed Phase 1 repair;
  the 35B produced zero bare fragments anywhere.
- Has **higher absolute yield** on CACC (~840 assertions vs ~310
  for 14B-full), but the quality mix is similar — same 70 / 30
  split between sequential register updates and structural mux /
  reset checks; same absence of protocol-level invariants.

The 35B is producing the same *kind* of assertions as the 14B, just
*more* of them and with less noise. The qualitative gaps (no
temporal operators, no protocol invariants) are not resolved by
scaling the model — they need pipeline-level changes (e.g. a
"protocol-property prompt" pass that explicitly asks for
`$stable` / `$onehot` / `##N` constructs from the spec).

---

## 7. Summary

| Quality dimension | 35B-full verdict |
|---|---|
| Syntax / lint pass rate | 100 % on FifoX/CMAC/CACC, 98 % on RVV |
| Hallucinated signals | 0 across all designs |
| Duplicate assertions | 0 |
| Bare fragments | 0 |
| Reproducibility (15-rep CV) | 0–6.3 % on assertion count — excellent |
| Combinational case coverage | Excellent (Booth/CMAC fully enumerated) |
| Mux / passthrough coverage | Excellent (CACC's 172 ternaries, RVV's 78 simples) |
| Reset value coverage | Excellent (131 reset-impl on CACC) |
| Sequential register update | Excellent (419 `$past` on CACC) |
| Real arithmetic invariants | Good (RVV grant formula, FifoX count) |
| Protocol / handshake invariants | **Poor** (zero on FifoX/CMAC/CACC, none on RVV either) |
| Higher-level SVA operators | **Absent** ($stable, $onehot, $rose, ##N, throughout) |
| Cross-module invariants | **Absent** |
| Semantic correctness (vs lint) | **Mixed** — FifoX `mem_N` tautologies are real bugs |

**Bottom line:** the 35B full pipeline produces a deep, lint-clean,
reproducible body of structural assertions that exhaustively cover
the combinational and one-cycle-sequential surface of every design.
It does not yet produce the kind of multi-cycle, protocol-level,
temporal-operator-bearing assertions that a verification engineer
would write by hand from the spec — and increasing model size from
14B to 35B does not close that gap.
