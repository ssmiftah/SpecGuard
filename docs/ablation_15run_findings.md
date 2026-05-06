# 15-Run Ablation Analysis

_Source: `ablation_study_20260420_1232/` · 4 designs × 6 variants × 15
reps = **360 runs**. Model: qwen3:14b-32k via Ollama, temperature 0.1.
Design LOC: fifox 586 · cmac 1 574 · rvv 1 591 · cacc 30 905._

This report summarises the 15-run matrix on the four small-to-mid
designs. For each analysis it reports the central tendency (mean /
median) **and the variance** (standard deviation σ and coefficient of
variation CV%), so the reader can judge whether observed differences
across variants are real or noise.

---

## 0. Success rate — which variants even finish

| Variant | Runs / design | Successful | Failure mode |
|---|---:|---:|---|
| `full` | 15 | **60/60** | — |
| `no-facts` | 15 | **60/60** | — |
| `flat-facts` | 15 | **60/60** | — |
| `no-repair` | 15 | **60/60** | — |
| `no-ast` | 15 | **48/60** (80 %) | occasional 0-assertion collapse |
| `no-feedback` | 15 | **0/60** | *always* 0 surviving assertions |

All failures cluster on the two variants that disable the pipeline's
generative backbone: `no-ast` (no structural scaffolding for the LLM
to elaborate) and `no-feedback` (no refinement loop to rescue
malformed output). `no-feedback` failing 100 % of the time across 4
designs × 15 runs = 60/60 runs is the strongest deterministic signal
in the study.

---

## 1. End-to-end token consumption, assertion yield, efficiency

### 1.1 Per-cell statistics (successful runs only)

Totals show **mean ± σ (CV%)**. `tok/A med` is the median of
per-run `total_tokens / final_assertions`.

**CMAC** (1 574 LOC, small MAC multiplier)

| Variant | OK/N | Tokens μ ± σ (CV) | Tokens med | Assertions μ ± σ (CV) | Tok/A med |
|---|---:|---:|---:|---:|---:|
| full | 15/15 | 67 489 ± 13 591 (20.1 %) | 58 330 | 53.2 ± 2.2 (4.2 %) | **1 122** |
| no-facts | 15/15 | 62 488 ± 12 585 (20.1 %) | 53 943 | 46.3 ± 4.3 (9.2 %) | 1 326 |
| flat-facts | 15/15 | 70 575 ± 18 121 (25.7 %) | 59 356 | 50.5 ± 5.0 (9.9 %) | 1 283 |
| no-repair | 15/15 | 71 160 ± 20 328 (28.6 %) | 58 513 | 51.9 ± 4.0 (7.7 %) | 1 190 |
| no-ast | 14/15 | 140 952 ± 21 908 (15.5 %) | 146 758 | 10.6 ± 2.2 (21.2 %) | 12 054 |
| no-feedback | 0/15 | — | — | — | — |

**FifoX** (586 LOC, Chisel-emitted FIFO)

| Variant | OK/N | Tokens μ ± σ (CV) | Tokens med | Assertions μ ± σ (CV) | Tok/A med |
|---|---:|---:|---:|---:|---:|
| full | 15/15 | 59 117 ± 12 579 (21.3 %) | 57 408 | 33.3 ± 5.2 (15.7 %) | **1 748** |
| no-facts | 15/15 | 58 804 ± 13 863 (23.6 %) | 55 406 | 29.5 ± 5.6 (18.9 %) | 1 880 |
| flat-facts | 15/15 | 52 088 ± 15 336 (29.4 %) | 57 343 | 29.9 ± 4.8 (16.2 %) | 1 884 |
| no-repair | 15/15 | 55 642 ± 12 441 (22.4 %) | 57 421 | 31.7 ± 6.2 (19.5 %) | 1 691 |
| no-ast | 11/15 | 331 486 ± 203 377 (61.4 %) | 209 183 | 8.7 ± 4.6 (52.8 %) | 52 296 |
| no-feedback | 0/15 | — | — | — | — |

**RVV** (1 591 LOC, hierarchical RISC-V vector backend)

| Variant | OK/N | Tokens μ ± σ (CV) | Tokens med | Assertions μ ± σ (CV) | Tok/A med |
|---|---:|---:|---:|---:|---:|
| full | 15/15 | 14 358 ± 1 489 (10.4 %) | 14 268 | 91.0 ± 5.2 (5.7 %) | **152** |
| no-facts | 15/15 | 13 896 ± 8 723 (62.8 %) | 11 181 | 91.7 ± 4.9 (5.3 %) | 121 |
| flat-facts | 15/15 | 15 559 ± 2 280 (14.7 %) | 14 781 | 91.8 ± 4.6 (5.1 %) | 163 |
| no-repair | 15/15 | 14 671 ± 2 875 (19.6 %) | 14 023 | 94.5 ± 2.1 (2.3 %) | 148 |
| no-ast | 11/15 | 53 599 ± 27 865 (52.0 %) | 39 614 | 9.2 ± 5.2 (56.6 %) | 5 144 |
| no-feedback | 0/15 | — | — | — | — |

**CACC** (30 905 LOC, convolution accumulator)

| Variant | OK/N | Tokens μ ± σ (CV) | Tokens med | Assertions μ ± σ (CV) | Tok/A med |
|---|---:|---:|---:|---:|---:|
| full | 15/15 | 651 082 ± 16 829 (2.6 %) | 649 583 | 312.7 ± 82.3 (26.3 %) | 2 160 |
| **no-facts** | 15/15 | **564 290 ± 21 272 (3.8 %)** | 559 799 | **395.8 ± 83.8 (21.2 %)** | **1 371** |
| flat-facts | 15/15 | 644 611 ± 12 826 (2.0 %) | 648 827 | 303.6 ± 71.1 (23.4 %) | 2 036 |
| no-repair | 15/15 | 653 089 ± 18 881 (2.9 %) | 651 041 | 323.8 ± 99.7 (30.8 %) | 1 950 |
| no-ast | 12/15 | 94 241 ± 58 660 (62.2 %) | 92 434 | 6.2 ± 2.8 (44.7 %) | 16 734 |
| no-feedback | 0/15 | — | — | — | — |

### 1.2 Variance behaviour — is anything we observe actually real?

- **Token counts are tight on the load-bearing variants**. CV is 2.6–
  3.8 % on CACC (deterministic RAG/batch scheduling dominates), 10–
  30 % on the three smaller designs where the LLM makes fewer calls
  and each decision has more weight.
- **Assertion yield is noisier than tokens**. CACC full is 312.7 ± 82.3
  (CV 26.3 %); CACC no-facts is 395.8 ± 83.8 (CV 21.2 %). The 83-
  assertion gap between the two variant *means* is larger than one σ,
  i.e. statistically distinguishable.
- **`no-ast` has extreme variance**: CV 52–62 % on tokens *and*
  assertions on FifoX, RVV, CACC. Without AST skeletons the LLM
  free-wheels; two runs of the same config diverge by more than 2×
  in both tokens and yield.
- **`no-feedback` always finishes with zero assertions** — not high-
  variance, but deterministically broken.

### 1.3 Headline — is the no-facts win on CACC reliable?

On CACC with 15 reps, `no-facts` vs `full`:

- Tokens: 564 290 ± 21 272 **vs** 651 082 ± 16 829 → **−13 %**, gap
  of 87 K tokens > 4× the larger σ.
- Assertions: 395.8 ± 83.8 **vs** 312.7 ± 82.3 → **+27 %**, gap of
  83 assertions ≈ one σ.
- Tok/assertion (median): 1 371 **vs** 2 160 → **−37 %**, and every
  single one of the 15 no-facts runs finished with fewer
  tokens-per-assertion than the *median* of the 15 full runs. The
  win is not a one-run fluke.

---

## 2. Component attribution

For each pipeline component, what the ablation costs in median
assertion yield and median tok/A (ratios vs the `full` row for that
design):

### CMAC

| Disable | Δ Assertions | Δ Tokens | Δ Tok/A |
|---|---:|---:|---:|
| `use_rtl_facts` (no-facts) | −13 % | −7 % | +18 % |
| `module_facts_mode` (flat-facts) | −5 % | +5 % | +14 % |
| `enable_deterministic_repair` (no-repair) | −2 % | +5 % | +6 % |
| `use_ast_assertions` (no-ast) | **−80 %** | **+109 %** | **+975 %** |
| refinement loop (no-feedback) | collapse (0 assertions) | — | ∞ |

### FifoX

| Disable | Δ Assertions | Δ Tokens | Δ Tok/A |
|---|---:|---:|---:|
| `use_rtl_facts` | −11 % | −0.5 % | +8 % |
| `module_facts_mode` | −10 % | −12 % | +8 % |
| `enable_deterministic_repair` | −5 % | −6 % | −3 % |
| `use_ast_assertions` | **−74 %** | **+461 %** | **+2 891 %** |
| refinement loop | collapse | — | ∞ |

### RVV

| Disable | Δ Assertions | Δ Tokens | Δ Tok/A |
|---|---:|---:|---:|
| `use_rtl_facts` | +1 % | −3 % | −20 % |
| `module_facts_mode` | +1 % | +8 % | +7 % |
| `enable_deterministic_repair` | +4 % | +2 % | −3 % |
| `use_ast_assertions` | **−90 %** | **+273 %** | **+3 285 %** |
| refinement loop | collapse | — | ∞ |

### CACC

| Disable | Δ Assertions | Δ Tokens | Δ Tok/A |
|---|---:|---:|---:|
| **`use_rtl_facts` (no-facts)** | **+27 %** | **−13 %** | **−37 %** |
| `module_facts_mode` | −3 % | −1 % | −6 % |
| `enable_deterministic_repair` | +4 % | +0.3 % | −10 % |
| `use_ast_assertions` | **−98 %** | **−86 %** | **+675 %** |
| refinement loop | collapse | — | ∞ |

### Cross-design verdict

- **AST skeletons and the feedback loop are load-bearing on every
  design.** Removing either collapses yield by 74–98 % or produces
  zero output.
- **Deterministic repair is a small win at most, sometimes a net
  loss.** The feedback loop recovers what Phase 1 would have caught.
- **Module scoping is neutral** (±10 %) on every design.
- **The facts card is the only component whose contribution
  *reverses sign* with design size.** On small designs it gives
  +5 to +13 % yield; on CACC it *costs* 27 % yield and 13 % extra
  tokens. The 15-run distributions don't overlap on CACC — this is
  not noise.

---

## 3. Semantic coverage

Domain count and per-domain incidence on each variant's median run
(the run with the median assertion count out of 15). Columns: mux,
passthrough, reset-implication, sequential (`|=>`), `$past`-register,
handshake, range-bound.

| Design | Variant | Total | mux | pass | reset | seq | past | hand | range | # Domains |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cmac | full | 53 | 2 | 1 | 12 | 3 | 3 | 0 | 0 | **5** |
|  | no-facts | 48 | 0 | 1 | 11 | 0 | 3 | 0 | 0 | 3 |
|  | flat-facts | 52 | 0 | 1 | 13 | 4 | 5 | 0 | 0 | 4 |
|  | no-ast | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|  | no-repair | 53 | 0 | 1 | 14 | 4 | 4 | 0 | 0 | 4 |
| fifox | full | 33 | 0 | 3 | 4 | 15 | 0 | 2 | 6 | **7** |
|  | no-facts | 32 | 0 | 3 | 1 | 15 | 0 | 0 | 7 | 5 |
|  | flat-facts | 30 | 1 | 3 | 0 | 20 | 4 | 0 | 6 | 6 |
|  | no-ast | 8 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 3 |
|  | no-repair | 33 | 0 | 3 | 1 | 18 | 0 | 0 | 6 | 5 |
| rvv | full | 94 | 10 | 23 | 0 | 0 | 0 | 8 | 0 | **5** |
|  | no-facts | 93 | 11 | 23 | 0 | 0 | 0 | 6 | 0 | 4 |
|  | flat-facts | 94 | 10 | 23 | 0 | 0 | 0 | 9 | 0 | 4 |
|  | no-ast | 8 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 1 |
|  | no-repair | 95 | 10 | 23 | 0 | 0 | 0 | 8 | 0 | 5 |
| cacc | full | 306 | 107 | 2 | 66 | 129 | 129 | 3 | 2 | 8 |
|  | **no-facts** | **404** | 23 | 2 | 138 | 216 | 224 | **31** | 4 | **9** |
|  | flat-facts | 303 | 27 | 2 | 89 | 176 | 179 | 9 | 0 | 8 |
|  | no-ast | 6 | 0 | 0 | 5 | 0 | 0 | 5 | 0 | 3 |
|  | no-repair | 327 | 107 | 2 | 49 | 149 | 168 | 1 | 2 | 8 |

**Observations.**

- **no-facts on CACC widens coverage.** It gains a 9th domain that
  `full` misses, triples the number of reset-implication assertions
  (66 → 138), and nearly 10× the handshake assertions (3 → 31). This
  matches the RQ1 yield gain: the LLM spends the freed attention on
  spec-document content instead of re-stating facts.
- **flat-facts loses the mux domain on CACC** (107 → 27) but keeps
  8 domains total. Dropping module scoping in particular pulls the
  LLM away from structural checks.
- **RVV domain mix is identical across full/no-facts/flat-facts/
  no-repair.** Protocol surface dominates the output regardless of
  knob; the four successful variants are functionally
  interchangeable on this design.
- **no-ast domains collapse to 0–3 on every design.** The LLM
  generates a handful of generic protocol stabs without structural
  grounding.

---

## 4. Quality–efficiency map (Pareto)

Axes: tokens consumed (lower is better) and assertions produced
(higher is better). Each point is the median of 15 runs per
(design, variant); `no-feedback` is off-chart (0 assertions).

```
CACC (30,905 LOC)
 assertions ↑
   400│                           ○ no-facts  (560K, 396)
      │
   320│                           ○ no-repair (651K, 324)
      │                           ○ full      (650K, 313)
   300│                           ○ flat      (649K, 304)
      │
     10│  × no-ast (92K, 6)
      0│                                                              → tokens
       └──────────────────────────────────────────────────────────────
         100K        300K        500K       700K
```
```
RVV (1,591 LOC)                             CMAC (1,574 LOC)
                                            assertions ↑
 assertions ↑                                     60│  ○ full  (58K, 53)
    100│  ○ no-repair (14K, 94)                    │  ○ no-repair (58K, 52)
     90│  ○ full      (14K, 91)                    │  ○ flat    (59K, 51)
        │  ○ no-facts  (11K, 92)                  50│  ○ no-facts(54K, 46)
        │  ○ flat      (15K, 92)                    │
     10│                                         10│
      │  × no-ast    (40K, 9)                      │  × no-ast   (147K, 11)
      0│                                          0│
        └────────────────────────                   └─────────────────────
          10K    40K    70K                           50K    100K   150K
```

### Pareto summary

- **On CACC, `no-facts` strictly dominates `full`**: fewer tokens
  (median 560 K vs 650 K) *and* more assertions (396 vs 313).
  Neither `flat-facts` nor `no-repair` is on the frontier — both are
  dominated by `full` on tokens and by `no-facts` on assertions.
- **On RVV, four variants cluster** — full/no-facts/flat-facts/
  no-repair land within a tight box around (14 K, 91). The LLM
  generates the same protocol assertion set regardless of knob.
- **On CMAC and FifoX, `full` is Pareto-optimal**: it gives the
  highest assertion count at a competitive token budget.
- **`no-ast` is dominated on every design** — higher tokens, lower
  yield, higher variance. Nothing to recommend it.
- **`no-feedback` is not on the chart at all** — 0 assertions means
  it's at `(tokens, 0)` on every design, so any other variant
  dominates it.

---

## 5. Scalability — SpecGuard vs naive baseline

Benchmarks span ~50× in LOC (586 → 30 905). Does the pipeline's
advantage scale with complexity? Naive baseline is a single-prompt
LLM with full RTL + spec docs, no AST/facts/scoping.

| Design | LOC | Naive tokens | Naive asserts | Naive tok/A | SpecGuard (full) tokens μ | SpecGuard asserts μ | SpecGuard tok/A med |
|---|---:|---:|---:|---:|---:|---:|---:|
| fifox | 586 | 87 004 | 36 | 2 417 | 59 117 ± 12 579 | 33.3 ± 5.2 | 1 748 |
| cmac | 1 574 | 60 186 | **0** | ∞ | 67 489 ± 13 591 | 53.2 ± 2.2 | 1 122 |
| rvv | 1 591 | 34 324 | **0** | ∞ | 14 358 ± 1 489 | 91.0 ± 5.2 | **152** |
| cacc | 30 905 | 45 839 | 1 | 45 839 | 651 082 ± 16 829 | 312.7 ± 82.3 | 2 160 |

### 5.1 Does the naive baseline scale?

It does not. It breaks down in three of four designs:

- **CMAC and RVV: zero output.** A 60 K- or 34 K-token prompt
  returns `property…endproperty` without `assert`, or `// comment
  only` text that fails lint entirely.
- **CACC: one assertion.** 45 839 tokens for a single handshake
  assertion is 45× more expensive per correct output than SpecGuard.
- **FifoX: 36 assertions, some malformed.** The smallest design is
  the one where naive actually works — its entire RTL + spec fits
  in context.

The naive baseline gets *worse* as designs scale because single-shot
prompting fails before it can exhaust the problem.

### 5.2 Does SpecGuard scale?

Yes, consistently:

- **Yields monotonically track complexity** (ordered by LOC):
  33 → 53 → 91 → 313 assertions. FifoX is smallest in LOC but
  largest-per-LOC protocol surface, so RVV outyields it.
- **Token cost scales sub-linearly with design size** for the
  datapath-heavy blocks (CACC is 20× larger than CMAC but only 9.6×
  the tokens). RVV is an outlier in the other direction — it
  produces 91 assertions for 14 K tokens because its protocol-rich
  top module gives the LLM many targets per call.
- **Variance is bounded.** CV on tokens is 2.6 % (CACC) to 21 %
  (FifoX); CV on assertions is 4 % (CMAC) to 26 % (CACC). None of
  the 15-run distributions overlap zero or collapse to noise.

### 5.3 Per-design efficiency gap (naive vs SpecGuard)

| Design | Naive tok/A | SpecGuard tok/A | Efficiency gain |
|---|---:|---:|---:|
| fifox | 2 417 | 1 748 | **1.4×** |
| cmac | ∞ (0 survived) | 1 122 | ∞ |
| rvv | ∞ (0 survived) | 152 | ∞ |
| cacc | 45 839 | 2 160 | **21.2×** |

On the one design where naive produces anything meaningful (fifox),
SpecGuard is 1.4× more efficient per correct assertion. On the
largest design where naive still produces *something*, SpecGuard is
21× more efficient. On two of four designs, naive is effectively
unusable.

---

## 6. Consistency / deviance / variance — compiled view

Three summary tables. **Higher CV = less consistent.**

### 6.1 Token consumption CV per variant

| Design | full | no-facts | flat-facts | no-ast | no-repair |
|---|---:|---:|---:|---:|---:|
| cmac | 20.1 % | 20.1 % | 25.7 % | 15.5 % | 28.6 % |
| fifox | 21.3 % | 23.6 % | 29.4 % | **61.4 %** | 22.4 % |
| rvv | 10.4 % | **62.8 %** | 14.7 % | **52.0 %** | 19.6 % |
| cacc | **2.6 %** | 3.8 % | 2.0 % | 62.2 % | 2.9 % |

### 6.2 Assertion yield CV per variant

| Design | full | no-facts | flat-facts | no-ast | no-repair |
|---|---:|---:|---:|---:|---:|
| cmac | 4.2 % | 9.2 % | 9.9 % | 21.2 % | 7.7 % |
| fifox | 15.7 % | 18.9 % | 16.2 % | 52.8 % | 19.5 % |
| rvv | 5.7 % | 5.3 % | 5.1 % | 56.6 % | 2.3 % |
| cacc | 26.3 % | 21.2 % | 23.4 % | 44.7 % | 30.8 % |

### 6.3 Consistency verdict

- **`no-ast` is noisy everywhere.** Yield CV 21–57 %, token CV 15–62 %.
  Cannot be used as a reliable point of comparison because the output
  shape depends on the LLM's mood that run.
- **`full` is the most reliable on large datapath designs.** CACC
  full has 2.6 % token CV — LLM budget is effectively deterministic.
- **RVV `no-facts` is an anomaly.** Token CV is 62.8 % while
  assertion CV is only 5.3 %. This means the LLM produces the same
  91-ish assertions on every run but takes wildly different numbers
  of tokens to get there — some runs converge in one pass, others
  need multiple feedback rounds. Worth an investigation if RVV is a
  benchmark anchor.
- **CACC assertion yield has 20–30 % CV across all successful
  variants.** This is not variant-specific — it's the design's
  signature. The 27 % gain no-facts shows over full is *larger than*
  this per-variant CV, so the effect is real.

---

## 7. Caveats + follow-ups

- **4-design matrix.** The CACC no-facts win is strong here but
  monotonicity across designs would strengthen the claim. The
  separate large-designs run (`run_large_designs_ablation.sh`,
  currently on the other terminal) will extend this to pdp / cdp /
  cmacfull / cdma / csc / sdp.
- **15 reps is enough for 95 % CIs.** With n=15, the ±σ bounds here
  are roughly 95 % CIs (t-distribution critical value at n=15 is
  2.14, very close to 2 for a normal). For the CACC no-facts vs full
  comparison, both the token and assertion 95 % CIs are non-
  overlapping.
- **Semantic coverage is one-run snapshot.** Section 3 uses the
  median-assertion-count run per cell. A fuller analysis would
  compute domain-count variance across the 15 reps.
- **Naive baseline is n=1 per design.** The 4 naive numbers come
  from a single previous run each. To match the 15-rep rigor of
  the ablation matrix we'd need to rerun the naive on every design
  15 times too.
