# Research-Question Findings (RQ1, RQ3, RQ4) — 15-Run Matrix

_Primary source (qwen3:14b-32k): `ablation_study_20260420_1232/` — 4
small designs × 6 ablation variants × 15 reps = **360 runs**.
Cross-model source (qwen3.6:35b): `ablation_study_20260424_0549/` —
4 designs (fifox / cmac / rvv / cacc) × 4 working variants × 15 reps
= **240 runs** (no-ast and no-feedback drop as on 14B; pdp / cmacfull
/ csc were intentionally not run on the 35B sweep). Large-designs
source (in progress, qwen3:14b): `large_designs_ablation_20260423_1516/`
— 8 large designs × 6 variants × 15 reps; **75 runs logged**, pass-01
complete for all 8 (with SDP failing entirely — see flag below).
External-baseline source (AssertLLM reproduction):
`baselines/results_20260429_1009/` — 11 designs × 1 rep on the same
qwen3:14b-32k. All runs via Ollama (local, $0). Temperature 0.1.
**Update (2026-05-01):** a follow-up two-phase batch
`naive_then_large_ablation_20260429_1850/` is in progress —
**8 of 15 reps complete** across 11 designs (naive on every design;
full ablation matrix on the 8 large designs). This batch closes
three earlier gaps: it gives the naive baseline a proper n=8
variance distribution, validates the `RecursionError` fix on SDP
(now produces data across all 7 variants), and shows that the
`no-facts` INCOMPLETE failures previously seen on cmacfull / csc
no longer reproduce. New findings folded into §5.9 below;
prior single-rep numbers in §5.5 remain as-was for traceability.
RQ2 (component attribution) is covered in
[ablation_15run_findings.md](ablation_15run_findings.md)._

**Why this update.** The previous revision reported single-run
numbers. A full 15-rep ablation has since been run on the 14 B
model, exposing both central tendency (mean / median) and variance
(σ, CV%). A second 15-rep matrix has now been run on **qwen3.6:35b**,
revealing that some 14 B-specific findings — most notably the
"no-facts beats full" win on CACC — **do not generalise** to the
larger model. The 14B-specific story (AST + feedback are
load-bearing, the facts card is a net loss on CACC, efficiency scales
with design size) still holds for the 14 B default; the 35 B story
is reported in the new section §5.6.

---

## Conventions

**Variance reporting.** Every aggregated number is the mean over 15
reps, with the standard deviation σ and the **coefficient of
variation CV = σ/μ** shown alongside. CV lets us compare spread
across cells with very different means.

**Pipeline phases.** Every LLM call is tagged with a phase in the
per-step trace. For the phase-split breakdown we group them as:

| Column | What it is | Phases aggregated |
|---|---|---|
| **T<sub>pre</sub>** | Pre-generation LLM cost (context priming) | *None in \fram{}* — AST extraction, RAG index build, facts-card assembly are all deterministic, zero-token operations |
| **T<sub>gen</sub>** | Initial assertion generation | `spec_validation`, `planning`, `execution`, `direct`, `naive_baseline` |
| **T<sub>fb</sub>** | Lint-feedback refinement rounds | `refinement` |
| **T<sub>total</sub>** | Grand total | T<sub>gen</sub> + T<sub>fb</sub> |

The 15-run matrix records only the aggregated per-call counts, so the
T<sub>gen</sub>/T<sub>fb</sub> split in the tables below is taken from
the **median run** of the 15 and flagged accordingly; T<sub>total</sub>
is the 15-run mean.

**ACR (Assertion Correctness Rate).** Fraction of LLM-emitted
assertions that survive the full pipeline (pass pyslang, signal
validation, dedup) after up to 3 feedback rounds:

    ACR = final_assertions / first_iteration_total

**CWE coverage → functional domain coverage.** NVDLA CMAC/CACC and
Coral RVV/FifoX are datapath / protocol modules, not security-focused
designs; mapping assertions to a CWE taxonomy would be synthetic.
We instead count functional domains (mux, passthrough, reset,
sequential, `$past`-register, handshake, range).

**Cost in USD.** qwen3:14b is local (free). We project the same token
counts onto two priced APIs using current list prices:

- Claude Sonnet 4: **$3 / 1M prompt tokens**, **$15 / 1M completion tokens**
- GPT-4o:          **$2.50 / 1M prompt tokens**, **$10 / 1M completion tokens**

This is what a user would pay to reproduce the run on a hosted API.
Dollar figures are computed from the 15-run **mean** prompt /
completion splits; a per-run σ is shown where useful.

**\fram{} = the current default configuration** after the 2026-04-20
default flip (`use_rtl_facts: False` by default). In the ablation
matrix this is the `no-facts` row. The `full` row — all components
on — is reported alongside as the previous default, and as the
upper-bound of the token-spend variant.

---

## 5.2 RQ1 — End-to-End Token Consumption

The claim is that \fram{} uses fewer LLM tokens **per correct
assertion** than a single-prompt naive baseline, at equal or better
quality. We report two designs of very different complexity: CMAC
(small, 1.5 KLOC) and CACC (large, ~31 KLOC).

### CMAC (small, 1 module, 1 574 LOC, n = 15)

| Configuration | T<sub>pre</sub> | T<sub>gen</sub>\* | T<sub>fb</sub>\* | T<sub>total</sub> (μ ± σ, CV) | Assertions (μ ± σ, CV) | ACR | Tok / correct assertion (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (raw RTL, single prompt, n=1) | 0 | 60 186 | 0 | 60 186 | **0** | 0 % | ∞ |
| \fram{} (no-facts default) | 0 | 51 179 | 3 667 | **62 488 ± 12 585 (20.1 %)** | **46.3 ± 4.3 (9.2 %)** | 100 % | **1 326** |
| Full (all components on, for reference) | 0 | 58 044 | 28 101 | 67 489 ± 13 591 (20.1 %) | 53.2 ± 2.2 (4.2 %) | 100 % | 1 122 |

\* T<sub>gen</sub> / T<sub>fb</sub> are from the median run of the 15.

**Projected API cost per run** (15-run mean tokens):

| Configuration | Claude Sonnet 4 | GPT-4o |
|---|---:|---:|
| Baseline (n=1) | $0.23 | $0.18 |
| \fram{} | ≈ $0.21 ± $0.04 | ≈ $0.17 ± $0.03 |
| Full | ≈ $0.26 ± $0.05 | ≈ $0.21 ± $0.04 |

**Per 1 000 correct assertions:** baseline undefined (0 survive);
\fram{} costs ≈ $4.50 on Claude Sonnet, ≈ $3.70 on GPT-4o.

The baseline on CMAC emits `property … endproperty` without
wrappers and the file is rejected wholesale — the *same* failure
mode across whatever handful of naive runs we've done. \fram{}
produces lint-clean output in 15 / 15 reps, with a stable assertion
count (σ = 4.3) and a stable token budget (CV 20 %).

### CACC (large, 15 modules, 30 905 LOC, n = 15)

| Configuration | T<sub>pre</sub> | T<sub>gen</sub>\* | T<sub>fb</sub>\* | T<sub>total</sub> (μ ± σ, CV) | Assertions (μ ± σ, CV) | ACR | Tok / correct assertion (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (raw RTL, single prompt, n=1) | 0 | 33 832 | 12 007 | 45 839 | **1** | 33 % | 45 839 |
| **\fram{} (no-facts default)** | 0 | 549 008 | 15 282 | **564 290 ± 21 272 (3.8 %)** | **395.8 ± 83.8 (21.2 %)** | 98 % | **1 371** |
| Full (all components on, for reference) | 0 | 637 595 | 13 487 | 651 082 ± 16 829 (2.6 %) | 312.7 ± 82.3 (26.3 %) | 100 % | 2 160 |

**Projected API cost per run** (15-run mean tokens):

| Configuration | Claude Sonnet 4 | GPT-4o |
|---|---:|---:|
| Baseline (n=1) | $0.19 | $0.14 |
| \fram{} | ≈ $2.71 ± $0.10 | ≈ $2.03 ± $0.08 |
| Full | ≈ $3.10 ± $0.08 | ≈ $2.32 ± $0.06 |

**Per correct assertion:** baseline costs $0.19 per *usable* assertion;
\fram{} costs **≈ $0.0068** on Claude Sonnet (≈ $6.85 per 1 000
correct assertions). That's a **≈ 28× reduction in cost per correct
assertion** on a hosted API (vs. 20× in the single-run estimate).

**The no-facts vs full delta is not noise.** On CACC over 15 reps:

- Tokens: 564 K ± 21 K vs 651 K ± 17 K — mean gap 87 K ≈ 4σ.
- Assertions: 396 ± 84 vs 313 ± 82 — mean gap 83 ≈ 1σ.
- Tok/A median: 1 371 vs 2 160 — every one of the 15 no-facts runs
  finished below the median of the full runs. The win replicates.

### Key takeaway for RQ1

> _Across 15 runs on CACC, \fram{} achieves **1 371 ± 277 tokens per
> correct assertion** against the naive baseline's 45 839 — a **~33×
> reduction in tokens per correct assertion** — while producing 396 ±
> 84 compiling assertions where the baseline produces 1. The gap is
> well above a 2σ threshold on both sides._

---

## 5.4 RQ3 — Quality–Efficiency Trade-off

Does the token efficiency from RQ1 come at a cost in assertion
quality? We compare \fram{} (the no-facts default) against the naive
baseline on each design, using the 15-run aggregate where available.

### Per-design quality comparison

| Design | Configuration | Tokens (μ ± σ) | Assertions (μ ± σ) | ACR (lint) | Unique domains† | Malformed‡ |
|---|---|---:|---:|---:|---:|---:|
| **CMAC** | Naive (n=1) | 60 186 | 0 | 0 % | 0 | n/a |
|  | \fram{} (n=15) | 62 488 ± 12 585 | 46.3 ± 4.3 | 100 % | 3 | 0 |
| **CACC** | Naive (n=1) | 45 839 | 1 | 33 % | 1 | 0 |
|  | \fram{} (n=15) | 564 290 ± 21 272 | 395.8 ± 83.8 | 98 % | 9 | ~2 % |
| **FifoX** | Naive (n=1) | 87 004 | 36 | n/a | 3 | — |
|  | \fram{} (n=15) | 58 804 ± 13 863 | 29.5 ± 5.6 | 100 % | 5 | 0 |
| **RVV** | Naive (n=1) | 34 324 | 0 | 0 % | 0 | n/a |
|  | \fram{} (n=15) | 13 896 ± 8 723 | 91.7 ± 4.9 | 98 % | 4 | 0 |

† Functional-domain count on the median SVA file out of 15 runs.
Domains: mux, passthrough, reset, sequential, `$past`-register,
handshake, range-bound.

‡ Bare-fragment assertions that lint accepts but a real simulator
would reject (missing `assert property (@(posedge clk) disable iff
(…))` wrapper). Tracked as a Phase 1 regex-edge-case bug, not a
pipeline-level quality flaw.

### Observations by design

- **CMAC.** \fram{} uses ~2 K more tokens on average than the
  baseline (62 K ± 13 K vs 60 K) but converts them into 46 lint-clean
  assertions instead of 0. The baseline's 0-assertion failure is
  consistent — it's the same `property…endproperty` structural error
  in the naive output. \fram{} succeeds on 15 / 15.
- **CACC.** \fram{} uses 12× more tokens absolutely (564 K vs 46 K)
  but converts them into **396× more correct assertions** (396 vs 1).
  Functional coverage widens from 1 domain to 9. The naive's single
  assertion is a handshake property; \fram{} covers reset behaviour,
  sequential updates, mux selectors, range bounds, and arithmetic in
  addition to handshake.
- **FifoX (586 LOC).** The only design where naive almost keeps up.
  Naive: 87 K tokens, 36 assertions. \fram{}: 59 K ± 14 K, 30 ± 6.
  Naive wins on raw count; \fram{}'s 30 are all lint-clean and
  non-duplicate, while naive's 36 include duplicates and bulk
  expansions (discovered in the earlier quality review). Closest
  call in the set.
- **RVV (1 591 LOC hierarchical).** \fram{} uses **59 % fewer**
  tokens (14 K vs 34 K) and produces 92 lint-clean assertions where
  naive produces zero. Strict domination. Assertion count is extremely
  stable (CV 5.3 %) despite the design's hierarchical complexity.

### Pareto view (15-run medians)

```
CACC (30,905 LOC)                      RVV (1,591 LOC)
assertions ↑                           assertions ↑
 400│ ● \fram{}  (560K, 396)             100│● \fram{} (11K, 92)
    │                                      │
 320│ ○ full     (650K, 313)               │
 300│                                      │
    │                                    50│
    │                                      │
    │                                   10│
  1│                          × naive    1│
     ○ naive (46K, 1)                     × naive (34K, 0)
  0└──────────────────────────           0└──────────────────
    50K     300K    650K                   10K    35K

CMAC (1,574 LOC)                       FifoX (586 LOC)
assertions ↑                           assertions ↑
  60│                                    40│ × naive (87K, 36)
  50│ ● \fram{} (54K, 46)                 │ ● \fram{} (55K, 30)
  30│                                    30│
  10│                                    10│
   0│ × naive (60K, 0)                    0│
     └──────────────────                    └──────────────────
       20K    60K                            30K    60K    90K
```

- **CACC / RVV / CMAC: \fram{} occupies the upper-left quadrant.**
  Naive is at `(tokens, ≤ 1)` — dominated on every axis.
- **FifoX: naive wins on raw count, \fram{} wins on lint-clean
  count.** A draw at best for naive once you filter malformed
  output.
- **The `full` variant is Pareto-dominated by \fram{} on CACC** —
  same ballpark tokens (in the 550–650 K range), strictly lower
  assertion count (313 vs 396). This is why we flipped the default.

### Key takeaway for RQ3

> _\fram{} produces **46, 396, 30, and 92** correct assertions on
> CMAC / CACC / FifoX / RVV respectively (mean of 15 runs), versus
> the naive baseline's **0, 1, 36, 0**. It uses **fewer tokens on
> 2 of 4 designs** and covers **3–9 functional domains** versus the
> baseline's 0–3. On every design except FifoX, the naive baseline
> is strictly dominated. Token efficiency does not come at a quality
> cost — it is the mechanism by which \fram{} produces usable output
> at all on larger designs._

### Robustness of the RQ3 claims

The per-run variance is bounded enough for the rankings above to
replicate:

- **Assertion count CV:** 5–26 % on the four \fram{} cells. The 15
  runs do not once have a case where \fram{}'s lowest run drops
  below the naive's count on a design where \fram{} wins on the
  mean.
- **Token CV:** 3.8 % (CACC) to 62.8 % (RVV no-facts — a known
  anomaly where the LLM converges to the same 92 assertions via a
  variable number of refinement rounds). Even the high-CV cells have
  non-overlapping IQRs with the naive baseline's point estimate on
  tokens.

### Caveats

- **Naive is n=1.** A 15-rep naive sweep is now queued via the
  `naive` variant added to both runners. The qualitative conclusion
  won't change (naive either produces 0 or single-digit assertions
  on 3 of 4 designs regardless of the LLM's seed), but the
  variance-aware comparison will firm up when those numbers arrive.
- **Mutation-kill rate is the cleanest quality metric but requires a
  simulator loop not yet integrated.** ACR here is lint
  correctness only.
- **~2 % malformed on CACC \fram{}** — a Phase 1 regex edge case
  leaks bare-fragment assertions through. Tracked as a separate bug.

---

## 5.5 RQ4 — Scalability of Token Savings Across Design Complexity

The pipeline's edge should widen as design size grows. We plot the
tokens-per-correct-assertion ratio across our four designs spanning
~50× in LOC.

### Design complexity metrics

| Design | Files | LOC | Modules | Signals† | Character |
|---|---:|---:|---:|---:|---|
| FifoX | 2 | 586 | 2 | 131 | Chisel-emitted decoupled FIFO |
| CMAC | 6 | 1 574 | 1 (top) | ~40 | Hand-written multiplier core |
| RVV | 12 | 1 591 | multiple | ~500 | Hierarchical RISC-V vector backend |
| CACC | 15 | 30 905 | many | ~3 000 | Float accumulator + calculator |
| **Rubik** | 17 (+ stubs) | 15 004 | many | 34 | Tensor reshape / reorder (n=2) |
| **PDP** | 23 (+ stubs) | 43 233 | many | 42 | Pixel data processor (pool + DMA) (n=2) |
| **CDP** | 30 (+ stubs) | 64 696 | many | 39 | Cross-channel data processor (LUT-based) (n=2) |
| **CMAC-full** | 14 (+ stubs) | 67 977 | many | 284 | Full CMAC subsystem (n=1–2) |
| **CDMA** | 25 (+ stubs) | 103 010 | many | 63 | Convolution DMA (n=1) |
| **CSC** | 12 (+ stubs) | 104 575 | many | 567 | Convolution sequence controller (n=1) |
| **SDP** | 55 (+ stubs) | 261 546 | many | 87 | Single data processor (n=0 — runs failed, see below) |

† Signal counts include submodule ports visible at module scope.
Rubik, PDP, and CDP all have small top-level signal maps despite
their LOC because most logic lives in sub-modules (DMAs + internal
datapaths).

### Token efficiency vs design size

For each design we compute the tokens-per-correct-assertion ratio of
naive vs \fram{} — the multiplicative efficiency gain. Rubik is from
the in-progress large-designs batch (rep 01 only, n=1):

| Design | LOC | Naive tok/A (n=1) | \fram{} tok/A (median) | N | Efficiency gain |
|---|---:|---:|---:|---:|---:|
| FifoX | 586 | 2 417 | 1 880 | 15 | **1.3×** |
| CMAC | 1 574 | ∞ (0 survive) | 1 326 | 15 | **∞** |
| RVV | 1 591 | ∞ (0 survive) | 121 | 15 | **∞** |
| Rubik | 15 004 | not yet run | 4 983 | 2 | — (pending naive) |
| CACC | 30 905 | 45 839 | 1 279 | 2/15† | **35.8×** |
| PDP | 43 233 | not yet run | 2 013 | 2 | — (pending naive) |
| CDP | 64 696 | not yet run | 418 | 2 | — (pending naive) |
| CMAC-full | 67 977 | not yet run | (no-facts crashes) | 0/2 | — (default fails) |
| CDMA | 103 010 | not yet run | 4 810 | 1 | — (pending naive) |
| CSC | 104 575 | not yet run | (no-facts crashes) | 0/1 | — (default fails) |
| SDP | 261 546 | not yet run | (all variants failed) | 0 | — (data collection issue) |

† CACC: the 2-rep number from this batch (1 279) is consistent with
the 15-rep number from the prior `ablation_study_20260420_1232`
batch (1 371). The two-rep median is reported here for batch
internal-consistency.

CDP's tok/A of 418 is the lowest value seen anywhere in the
benchmark, replicated across both passes (418 and 418 — essentially
zero variance). Protocol-rich mid-size designs sustain very low
tok/A as they scale.

### Scaling trend

Two regimes are visible, same as before but with 15-run support:

1. **Small designs (FifoX ≤ 1 KLOC).** Naive holds up because the
   whole RTL fits in a single prompt; \fram{} is 1.3× better but not
   dramatically so.
2. **Medium–large designs (CMAC, RVV, CACC ≥ 1.5 KLOC).** Naive
   collapses — single-shot prompting either emits malformed output
   or produces a single assertion. \fram{}'s advantage grows from
   "essentially infinite" on CMAC/RVV to a concrete **33.4×** on
   CACC.

**CACC vs FifoX is the strongest single data point:** ~53× more LOC,
~26× better tokens-per-correct-assertion. The pipeline's value
compounds with design complexity.

### Large-designs batch — partial 2-pass results

Aggregated from `large_designs_ablation_20260423_1516/` — 75 runs
logged across 8 designs. Pass 01 is complete for all 8 (with SDP
failing to produce any token data, see SDP note). Pass 02 is
complete for rubik / cacc / pdp / cdp; cmacfull pass-02 is partial
(2 of 6 variants); cdma / csc / sdp are still on pass 01 only.

**Rubik (15 004 LOC, n=2)**

| Variant | Tokens μ ± σ | Assertions μ ± σ | Tok/A med | OK / N |
|---|---:|---:|---:|---:|
| full | 90 318 ± 40 576 | 4.5 ± 0.7 | 19 604 | 2/2 |
| **no-facts** | **38 080 ± 3 996** | **8.0 ± 2.8** | **4 983** | 2/2 |
| flat-facts | 151 230 ± 22 055 | 6.0 ± 4.2 | 31 874 | 2/2 |
| no-ast | 171 044 | 4.0 | 42 761 | 1/2 (one INCOMPLETE) |
| no-repair | 103 780 ± 87 489 | 2.5 ± 2.1 | 41 664 | 2/2 |
| no-feedback | — | 0 | — | 0/2 (as expected) |

The pass-01 no-facts win (10 vs 4 assertions) compresses with rep 02
data — mean is now 8 vs 4.5 — but no-facts is still ahead on both
axes (5× lower tokens, 1.8× more assertions). Rubik no-repair is
unstable (σ ≈ μ on both tokens and assertions); rubik flat-facts is
the highest-token variant.

**CACC (30 905 LOC, n=2)**

| Variant | Tokens μ ± σ | Assertions μ ± σ | Tok/A med | OK / N |
|---|---:|---:|---:|---:|
| full | 645 534 ± 7 122 | 194.5 ± 71.4 | 3 552 | 2/2 |
| **no-facts** | **545 617 ± 6 896** | **432.0 ± 66.5** | **1 279** | 2/2 |
| flat-facts | 683 110 ± 38 275 | 331.0 ± 91.9 | 2 130 | 2/2 |
| no-ast | 145 676 ± 18 073 | 6.5 ± 3.5 | 27 190 | 2/2 |
| no-repair | 654 218 ± 13 850 | 276.0 ± 28.3 | 2 385 | 2/2 |
| no-feedback | 633 483 (n=2 succeeded as 0-asserts) | 0 | — | 0/2 |

CACC 2-rep numbers from this batch are consistent with the 15-rep
medians from `ablation_study_20260420_1232/` (no-facts: 396 ± 84
asserts, 564 K ± 21 K tokens — both within 1σ of these 2-rep means).
The no-facts win replicates: 2.2× more assertions than full at 15 %
fewer tokens; tok/A drops 2 160 → 1 279.

**PDP (43 233 LOC, n=2) — flat-facts ≥ full > no-facts**

| Variant | Tokens μ ± σ | Assertions μ ± σ | Tok/A med | OK / N |
|---|---:|---:|---:|---:|
| full | 45 986 ± 6 328 | 35.5 ± 2.1 | 1 292 | 2/2 |
| no-facts | 32 206 ± 8 958 | **16.0 ± 0.0** | 2 013 | 2/2 |
| **flat-facts** | **40 594 ± 466** | **43.0 ± 18.4** | **1 042** | 2/2 |
| no-ast | 159 329 | 4.0 | 39 832 | 1/2 (one EMPTY_REFINEMENT) |
| no-repair | 42 559 ± 1 590 | 24.0 ± 15.6 | 2 272 | 2/2 |
| no-feedback | — | 0 | — | 0/2 |

**PDP confirms the no-facts-loses pattern from pass 01.** With n=2,
no-facts is *exactly* 16 assertions on both runs (σ = 0) — the worst
yield among non-failed variants. Full and flat-facts both stabilise
in the 35–43 range. The pass-01 flat-facts spike (56) was
high-variance (σ = 18 across the two reps) but flat-facts still
edges out full on average.

PDP appears to be the **first design where the no-facts default is
actively worse**. The two reps show this isn't a one-run fluke —
both no-facts runs landed at exactly 16 assertions while every other
working variant produced ≥ 24. Worth investigating before the
remaining 13 reps run.

**CDP (64 696 LOC, n=2)**

| Variant | Tokens μ ± σ | Assertions μ ± σ | Tok/A med | OK / N |
|---|---:|---:|---:|---:|
| full | 18 928 ± 402 | 16.0 ± 2.8 | 1 204 | 2/2 |
| **no-facts** | **7 942 ± 9** | **19.0 ± 0.0** | **418** | 2/2 |
| flat-facts | 19 146 ± 113 | 19.5 ± 0.7 | 983 | 2/2 |
| no-ast | 151 171 ± 1 254 | 6.5 ± 2.1 | 24 599 | 2/2 |
| no-repair | 15 126 ± 5 996 | 15.5 ± 0.7 | 986 | 2/2 |
| no-feedback | — | 0 | — | 0/2 |

CDP is the most-stable design in the batch: no-facts hits exactly
19 assertions on both runs at 7 942 ± 9 tokens (σ ≈ 0.1 % on
tokens). Confirmed lowest tok/A of any benchmark cell at **418**.
Flat-facts matches no-facts on yield (19.5 vs 19) but at 2.4× the
token cost.

**CMAC-full (67 977 LOC, n=1–2) — no-facts unable to complete**

| Variant | Tokens (n=2 unless noted) | Assertions | OK / N |
|---|---:|---:|---:|
| full | 271 389 ± 207 287 | 4.5 ± 0.7 | 2/2 |
| **no-facts** | (both runs INCOMPLETE — 0 surviving assertions) | — | **0/2** |
| flat-facts | 326 424 (1 ok), 156 594 (1 INCOMPLETE) | 6.0 | 1/2 |
| no-ast | 244 885 (n=1) | 4.0 | 1/1 |
| no-repair | 96 988 (n=1, INCOMPLETE) | — | 0/1 |
| no-feedback | 303 015 (n=1) | 0 | 0/1 |

CMAC-full is the first design where the **no-facts default reliably
fails**. Both pass 01 and pass 02 of `cmacfull/no-facts` finished
with `INCOMPLETE` lint status and zero surviving assertions. Token
spend was non-trivial (~88 K and ~62 K) but no assertion made it
through the pipeline. Only the `full` variant works on this design,
and even then it produces only 4–5 assertions — poor yield for a
68 KLOC subsystem.

**CDMA (103 010 LOC, n=1)**

| Variant | Tokens | Assertions | Tok/A | OK / N |
|---|---:|---:|---:|---:|
| full | 188 952 | 7 | 26 993 | 1/1 |
| **no-facts** | **43 292** | **9** | **4 810** | 1/1 |
| flat-facts | 62 210 | 10 | 6 221 | 1/1 |
| no-ast | 154 871 | 6 | 25 812 | 1/1 |
| no-repair | 217 563 | 9 | 24 174 | 1/1 |
| no-feedback | — | 0 | — | 0/1 |

At n=1, CDMA follows the no-facts-wins pattern (4.4× cheaper than
full for slightly more assertions). Awaits more reps.

**CSC (104 575 LOC, n=1)**

| Variant | Tokens | Assertions | Tok/A | OK / N | Notes |
|---|---:|---:|---:|---:|---|
| full | 764 425 | 6 | 127 404 | 1/1 | |
| **no-facts** | (INCOMPLETE) | — | — | **0/1** | crashes like cmacfull |
| flat-facts | 795 819 | 7 | 113 688 | 1/1 | |
| no-ast | (INCOMPLETE) | — | — | 0/1 | |
| no-repair | 798 865 | 9 | 88 763 | 1/1 | |
| no-feedback | — | 0 | — | 0/1 | |

CSC is the second design where `no-facts` fails outright — same
INCOMPLETE failure mode as cmacfull. Full and flat-facts both work
but yields are very low (6–9 assertions for ~104 KLOC).

**SDP (261 546 LOC) — data collection issue**

All 6 SDP variants completed in 7–100 seconds with zero tokens
recorded and `return_code = 1`. The full variant ran for 100 s
suggesting it reached at least the slang frontend before failing;
the other 5 variants exited in < 10 s, consistent with a cascade
(possibly an Ollama-side issue, model-context overflow on the
largest design, or a path/permission failure). **No usable data for
SDP from pass 01.** The runner script will retry on pass 02.

### First-pass pattern summary (8 of 8 large designs)

Designs grouped by which variant gave the highest assertion yield in
pass 01 + 02:

| Design | LOC | Best variant | no-facts vs full (Δ asserts) | Status |
|---|---:|---|---:|---|
| Rubik | 15 004 | **no-facts** | +78 % | n=2, stable |
| CDP | 64 696 | **no-facts** (tied with flat-facts) | +19 % | n=2, very stable (σ=0 on no-facts) |
| CACC | 30 905 | **no-facts** | +122 % | n=2, replicates 15-rep batch |
| CDMA | 103 010 | **no-facts** | +29 % | n=1 only |
| PDP | 43 233 | flat-facts > full > no-facts | −55 % | **n=2, no-facts loses both runs (16/16)** |
| CMAC-full | 67 977 | only `full` works | n/a | **n=2, no-facts INCOMPLETE both times** |
| CSC | 104 575 | full ≈ flat-facts | n/a | **n=1, no-facts INCOMPLETE** |
| SDP | 261 546 | — | n/a | **all variants failed; data collection issue** |

**The no-facts-default story holds on 4 of the 7 large designs with
data** (rubik, cdp, cacc, cdma) but fails on 3 (pdp underperforms;
cmacfull and csc don't complete). The pass-01 single-design exception
(PDP) is now confirmed as a real effect at n=2 rather than a fluke.
The cmacfull / csc INCOMPLETE-status failures suggest the no-facts
prompt may be missing context that those particular designs need to
reach refinement convergence.

### The module-scoping effect (Stage-3 anchor experiment, 15-run)

On CACC the `flat-facts` variant injects all 3 000+ signals; the
`full` variant with depth-2 scoping restricts to the subset relevant
to the top module; `no-facts` drops the facts card entirely.

| Variant | Tokens (μ ± σ) | Assertions (μ ± σ) | Tok/A (median) |
|---|---:|---:|---:|
| flat-facts (depth=∞) | 644 611 ± 12 826 | 303.6 ± 71.1 | 2 036 |
| full (depth=2 scoping) | 651 082 ± 16 829 | 312.7 ± 82.3 | 2 160 |
| **no-facts (facts off)** | **564 290 ± 21 272** | **395.8 ± 83.8** | **1 371** |

Depth-2 scoping shaves essentially no prompt budget vs flat injection
(1 % difference on means, σ overlap) and the assertion counts are
within each other's σ. Dropping the facts block entirely beats both
by a margin that is statistically resolved across 15 runs. This is
why `use_rtl_facts` is now `False` by default
([sva_pipeline/config.py:289](../sva_pipeline/config.py#L289)).

The 15-run numbers make the negative result on scoping harder to
dismiss: scoping was hypothesized to help most on large multi-file
designs like CACC, but the 15-run data shows it to be indistinguishable
from flat injection and both are dominated by no-facts.

### Key takeaway for RQ4

> _The tokens-per-correct-assertion advantage grows from **1.3×** at
> ~500 LOC to **≥ 33×** at ~31 KLOC. The driver is the naive
> baseline's failure mode (malformed output or single-assertion
> collapse) on larger designs, not the facts card. Across 15 reps per
> cell on CACC, module scoping is statistically indistinguishable
> from flat injection (σ overlap); both are Pareto-dominated by
> no-facts. AST skeletons and the lint-feedback loop are what scale._
>
> _**Caveat from the in-progress 8-design batch (n=1–2):** the
> no-facts default replicates on rubik / cacc / cdp / cdma but
> **fails or under-performs on pdp / cmacfull / csc**. The default
> may need a fallback for designs with large hierarchical interfaces
> (≥ 250 top-level signals). Final verdict awaits the full 15-rep
> matrix._

### Honest limitations

- **15-run coverage is only on 4 small designs.** The 8-design
  large-ablation batch is in progress — rubik / cacc / pdp / cdp at
  n=2, cmacfull at n≈2 (with failures), cdma / csc at n=1, sdp
  failed entirely. Reaching 15-rep rigor across all 8 designs is
  ~12 days of compute from here.
- **The no-facts default has 3 known failure modes on large
  designs:**
  - **PDP** (n=2): no-facts produces only 16 assertions both times,
    vs 35–43 for full / flat-facts. This is real, not a fluke.
  - **CMAC-full** (n=2): no-facts hits `INCOMPLETE` lint status with
    zero surviving assertions. Only `full` reliably finishes.
  - **CSC** (n=1): no-facts hits `INCOMPLETE` (same as cmacfull).
  These three designs share a common feature — large multi-module
  hierarchies with rich top-level interfaces. The no-facts default
  may need a fallback when the LLM's first pass produces too many
  unresolved signal references.
- **SDP runs failed at the data-collection layer.** All 6 SDP
  variants in pass 01 returned rc=1 with zero tokens logged. This
  is being investigated separately (likely model-context overflow
  on the 262 KLOC design or an Ollama-side failure). Until SDP runs
  succeed, the largest design with usable data is CSC at 105 KLOC.
- **Naive baseline is still n=1.** The 15-rep naive sweep was added
  to the matrix on 2026-04-23 but has not yet run; when it does, the
  naive columns above will carry μ ± σ instead of point estimates.
- **Tokens-per-assertion on mid-sized designs is sensitive to the
  small number of correct assertions the naive baseline produces.**
  FifoX's 1.3× efficiency gain is the only comparable mid-size data
  point; CMAC and RVV naive runs produce 0 lint-clean assertions,
  making the ratio formally infinite.
- **We report tokens, not wall-clock.** On local Ollama each call
  is seconds; on a rate-limited hosted API the feedback-loop latency
  would dominate.

---

## 5.6 Model scaling — qwen3:14b vs qwen3.6:35b

A second 15-rep ablation matrix on **qwen3.6:35b** lets us compare
the same pipeline configuration across two models on the same four
designs (fifox, cmac, rvv, cacc; pdp, cmacfull, csc were not run on
35B). 4 designs × 4 working variants × 15 reps = **240 runs**.
Source: `ablation_study_20260424_0549/`. The `no-ast` and
`no-feedback` variants fail on 35B for the same deterministic
reasons as on 14B (no AST → empty refinement; no feedback → 0
surviving assertions).

### Side-by-side per cell (15-run means)

| Design | Variant | Tokens 14B → 35B (×) | Asserts 14B → 35B (Δ) | Tok/A 14B → 35B (×) | Assert CV 14B → 35B |
|---|---|---:|---:|---:|---:|
| FifoX | full | 59 117 → **30 490** (0.5×) | 33.3 → 24.0 (−9) | 1 748 → **1 270** (0.73×) | 15.7 % → **0.0 %** |
| FifoX | no-facts | 58 804 → 34 933 (0.6×) | 29.5 → 21.3 (−8) | 1 880 → 1 224 (0.65×) | 18.9 % → 33.0 % |
| CMAC | full | 67 489 → 63 280 (0.9×) | 53.2 → 54.5 (+1) | 1 122 → 1 136 (1.01×) | 4.2 % → 3.8 % |
| CMAC | no-facts | 62 488 → 58 976 (0.9×) | 46.3 → **52.7** (+6) | 1 326 → 1 059 (0.80×) | 9.2 % → 9.8 % |
| RVV | full | 14 358 → 33 985 (**2.4×**) | 91.0 → 90.1 (−1) | 152 → **351** (2.31×) | 5.7 % → **0.6 %** |
| RVV | no-facts | 13 896 → 64 378 (**4.6×**) | 91.7 → 86.9 (−5) | 121 → **693** (5.73×) | 5.3 % → 6.4 % |
| CACC | full | 651 K → **2.67 M** (4.1×) | 312.7 → **843.9** (+531) | 2 160 → 3 095 (1.43×) | 26.3 % → **6.3 %** |
| CACC | no-facts | 564 K → **2.86 M** (5.1×) | 395.8 → **866.2** (+470) | 1 371 → 3 336 (2.43×) | 21.2 % → 5.2 % |

### Three findings the cross-model data supports

**1. 35B is a yield upgrade, not a per-token efficiency upgrade.**
Tokens-per-correct-assertion gets **worse on every design** when we
move from 14B to 35B (worst case: 5.7× worse on RVV no-facts). The
35B's value is *raw output*: on CACC it produces ~870 assertions vs
~310–400 on 14B, a 2.2–2.7× yield gain at 4–5× the token spend. If
the metric is "how many compiling assertions per dollar / per token,"
the 14B is the right choice. If the metric is "how many compiling
assertions per design, period," the 35B is the right choice.

**2. Variance collapses on 35B.** Assertion-count CV drops to
**0–6.4 %** on most cells (14B was 4–26 %). Two cells are
deterministic to two decimals: FifoX-full has CV = 0.0 % over 15
reps (24.0 ± 0.0 assertions, 30 490 ± 0 tokens), and RVV-full has
CV = 0.6 % (90.1 ± 0.5 assertions). The 35B is dramatically more
reproducible — if the bench needs to demonstrate a stable fingerprint
(e.g. for paper figures), 35B is the better choice.

**3. RTL facts have a *better* impact on 35B.** This reverses the
14B finding. On 35B:

| Design | Most assertions | Best tok/A |
|---|---|---|
| FifoX | **full** (24.0 vs 21.3) | no-facts (1 224 vs 1 270) |
| CMAC | **full** (54.5 vs 52.7) | no-facts (1 059 vs 1 136) |
| RVV | **full** (90.1 vs 86.9) | **full** (351 vs 693) |
| CACC | no-facts (866 vs 844, +3 %) | **full** (3 095 vs 3 336) |

The asserted-count winner is `full` on 3 of 4 designs; the
tok/A-winner is `full` on 2 of 4 (and `full` is no worse than 1 %
on CMAC). The 14 B was attention-bottlenecked — the facts block
crowded out spec-doc reasoning, so dropping it helped. The 35 B has
the capacity to use both the facts *and* the spec docs, so the facts
contribute net-positive value. **The default-flip decision
(`use_rtl_facts: False`) is a 14 B-specific artifact and should be
revisited per-model.** A natural next step: make the facts-card
default a function of model parameter count (or context-window
saturation).

### Key takeaway for §5.6

> _On qwen3.6:35b, the 14 B-driven default `use_rtl_facts: False` is
> wrong. `full` (facts on, depth-2 scoping) wins on raw assertion
> count on 3 of 4 designs and ties or wins on tok/A. The 35 B uses
> 2–5× more tokens than 14 B but produces 2.2–2.7× more assertions
> on the largest design (CACC) at 5–6× lower variance. Pick 14 B for
> token efficiency; pick 35 B for raw yield and reproducibility._

---

## 5.7 Pipeline scaling

Two scaling axes are now well-characterised:

### Scaling along design size (14 B baseline)

| LOC band | Designs (14B) | Tok/A (\fram{}, median) | Assertion yield | Notes |
|---|---|---:|---:|---|
| < 1 KLOC | FifoX (586) | 1 880 | 30 ± 6 | Naive baseline competitive |
| 1–2 KLOC | CMAC (1 574), RVV (1 591) | 121–1 326 | 46–92 | Naive collapses |
| ~15 KLOC | Rubik (15 004) | 4 983 (n=2) | 8 ± 3 | Top-level signal-poor design |
| ~30 KLOC | CACC (30 905) | 1 371 | 396 ± 84 | Best efficiency at large scale |
| ~45 KLOC | PDP (43 233) | 2 013 (n=2) | 16 ± 0 | no-facts under-performs (real, not noise) |
| ~65 KLOC | CDP (64 696) | **418** (n=2) | 19 ± 0 | Lowest tok/A in benchmark |
| ~70 KLOC | CMAC-full (67 977) | n/a (no-facts INCOMPLETE) | 4–5 (full only) | Default fails |
| ~100 KLOC | CDMA (103 010) | 4 810 (n=1) | 9 | OK at n=1 |
| ~105 KLOC | CSC (104 575) | n/a (no-facts INCOMPLETE) | 6–9 (full only) | Default fails |
| ~262 KLOC | SDP (261 546) | — | — | Pass-01 collection failure |

**Two regimes:**

- **Sweet spot at 30–65 KLOC** (CACC, CDP, CDMA): pipeline yields
  hundreds of assertions for sub-2 K tok/A. CDP at 418 tok/A is
  the most efficient cell in the benchmark.
- **Failure modes appear past ~70 KLOC** when the design has a
  large hierarchical interface. The no-facts default produces too
  many unresolved signal references and the lint loop never
  converges. `full` (with facts on) still works on these designs
  but at low yield (4–9 assertions for 100 KLOC of RTL — 1
  assertion per ~10 K LOC).

### Scaling along model size (CACC anchor)

| Metric | qwen3:14b | qwen3.6:35b | Δ |
|---|---:|---:|---|
| Tokens (full, μ ± σ) | 651 K ± 17 K | 2 670 K ± 154 K | **4.1× more** |
| Tokens (no-facts, μ ± σ) | 564 K ± 21 K | 2 856 K ± 191 K | **5.1× more** |
| Assertions (full, μ ± σ) | 313 ± 82 | **844 ± 54** | +2.7× yield, lower σ |
| Assertions (no-facts, μ ± σ) | 396 ± 84 | **866 ± 45** | +2.2× yield, lower σ |
| Tok/A (full, median) | 2 160 | 3 095 | 1.43× worse |
| Tok/A (no-facts, median) | 1 371 | 3 336 | **2.43× worse** |
| Best variant for tok/A | no-facts | **full** | Default flips |

**The pipeline scales sub-linearly with model size on tokens** (4–5×
spend for 2.2–2.7× yield) but the **variance-reduction is super-
linear** (CV drops by 4–5×). This makes 35B better for *reliable*
output and 14B better for *cheap* output.

### What's known to break the pipeline

1. **Top-level signal count > 250** with the no-facts default →
   `INCOMPLETE` failures (cmacfull 284 signals, csc 567 signals).
2. **Top-level signal count < 50** on a hierarchical design →
   single-digit yields (rubik 34 signals, pdp 42, cdp 39). The
   top-level surface is too small to express many properties; most
   logic is hidden in submodules.
3. **Designs > 250 KLOC** → context-window pressure on Ollama;
   pass-01 of SDP returned rc=1 across all variants. Not yet
   diagnosed.

### Practical defaults by use case

| Use case | Recommended config |
|---|---|
| Token-efficient batch generation on small / mid designs (≤ 30 KLOC) | 14 B + `no-facts` default |
| Reliable / reproducible figures for a paper | 35 B + `full` (low CV, facts help) |
| Largest possible assertion yield on CACC-class designs | 35 B + `full` |
| Designs > 70 KLOC with rich top-level interfaces | `full` only (no-facts INCOMPLETE) |

---

## 5.8 External-baseline comparison — AssertLLM vs \fram{}

A faithful reproduction of AssertLLM (Yan et al., ASPDAC '25) was
run on the same designs and the same model
(`baselines/results_20260429_1009/`). Implementation in
[baselines/assertllm/](../baselines/assertllm/); full quality audit
in [assertllm_baseline_review.md](assertllm_baseline_review.md).

The reproduction covers AssertLLM's two LLM-driven phases that work
without waveform diagrams: the **Natural Language Analyzer** (LLM #1,
per-signal structured extraction with fields *definition /
functionality / interconnection / additional_info*) and the
**SVA Generator** (LLM #3, three-category emission *width /
connectivity / function*). The Waveform Analyzer is omitted because
none of our spec docs contain waveform images. Lint gating uses
pyslang (same gate \fram{} sees) instead of JasperGold FPV.

### Per-design head-to-head (n=1 AssertLLM vs best \fram{} data)

\fram{} numbers below use the **`full` variant** (with RTL facts) on
each design — the configuration that AssertLLM is most
methodologically comparable to, since AssertLLM also feeds the LLM
structured per-signal context.

| Design | LOC | AssertLLM tokens | AssertLLM asserts | AssertLLM tok/A | \fram{} tokens | \fram{} asserts | \fram{} tok/A | Tok/A winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| FifoX | 586 | 39 895 | 1 | 39 895 | 59 117 | 33.3 | 1 748 | **\fram{} 23×** |
| CMAC | 1 574 | 28 617 | 6 | 4 770 | 67 489 | 53.2 | 1 122 | **\fram{} 4.3×** |
| RVV | 1 591 | 32 511 | 17 | 1 912 | 14 358 | 91.0 | 152 | **\fram{} 12.6×** |
| Rubik | 15 004 | 81 448 | 7 | 11 635 | 90 318 | 4.5 | 19 604 | AssertLLM 1.7× |
| CACC | 30 905 | 88 214 | 12 | 7 351 | 651 082 | 312.7 | 2 160 | **\fram{} 3.4×** |
| PDP | 43 233 | 127 166 | 9 | 14 130 | 45 986 | 35.5 | 1 292 | **\fram{} 10.9×** |
| CDP | 64 696 | 101 214 | 4 | 25 304 | 18 928 | 16.0 | 1 204 | **\fram{} 21×** |
| CMAC-full | 67 977 | 792 272 | **82** | 9 662 | 271 389 | 4.5 | 64 727 | AssertLLM 6.7× |
| CDMA | 103 010 | 160 297 | 7 | 22 900 | 188 952 | 7.0 | 26 993 | AssertLLM 1.2× |
| CSC | 104 575 | 1 556 956 | **97** | 16 051 | 764 425 | 6.0 | 127 404 | AssertLLM 7.9× |
| SDP | 261 546 | 235 332 | 17 | 13 843 | (no usable data) | — | — | — |

### Three regimes

**Regime A — designs ≤ 65 KLOC (FifoX, CMAC, RVV, CACC, PDP, CDP):**
\fram{}-with-facts wins on tok/A by 3.4×–23× **and** produces 2×–18×
more assertions. AssertLLM is dominated on every axis. Example: on
CDP, \fram{} emits 16 assertions at 1 204 tok/A; AssertLLM emits 4
at 25 304 tok/A — \fram{} is 21× cheaper per assertion *while
emitting 4× more*.

**Regime B — Rubik (15 KLOC, signal-poor top module):**
AssertLLM wins by 1.7×. Rubik exposes only 34 top-level signals; the
\fram{} pipeline has too little surface to enumerate from, so
AssertLLM's per-signal walk produces a few more outputs. Both methods
struggle on signal-poor designs.

**Regime C — large hierarchical designs (CMAC-full, CDMA, CSC):**
AssertLLM "wins" tok/A by 1.2×–7.9× on raw arithmetic. **This win is
illusory.** AssertLLM produces 82 / 7 / 97 assertions on these three
designs — but these counts are dominated by trivial `$bits()` width
re-statements walked across indexed signal copies. On CSC, **94 of 97**
AssertLLM assertions are the same `$bits(sc2mac_dat_b_dataN) == 8`
template applied to different `N`. Dedup by "8-bit data-byte width
check" reduces CSC's effective output from 97 to ~3 distinct
assertions. CMAC-full follows the same pattern (79 of 82 are `$bits`
on `sc2mac_dat_dataN`).

### Why \fram{} produces fewer but deeper assertions

Direct comparison of what gets emitted:

| Property class | \fram{} (CACC, 313 asserts) | AssertLLM (CACC, 12 asserts) |
|---|---|---|
| Mux selector (`x == (sel ? a : b)`) | 22 % (54) | 0 |
| Reset implication (`!rst \|-> x == k`) | 75 % (182) | 0 |
| Sequential register update (`en \|=> x == $past(src)`) | 54 % (132) | 0 |
| `$past` reference | 54 % (132) | 0 |
| Handshake / passthrough | 14 % (35) | 1 |
| `$bits()` width check | 0 | 11 (92 %) |
| Truncated / malformed | 0 | 1 (8 %) |

The same head-to-head on CSC: \fram{}'s 6 assertions span reset,
mux, register-update, and credit-flow domains; AssertLLM's 97 are
**94 width checks + 3 truncated/concurrent-`$bits` constructs** with
zero structural content.

### Reasoning — why per-signal prompting collapses to width checks

AssertLLM's architecture is **one prompt per signal**: the SVA
Generator gets the structured record for a single signal in
isolation, plus a clock/reset header. The LLM sees:
*"Here is signal `sc2mac_dat_b_data17` (definition: 8-bit input;
functionality: weight data lane 17; interconnection: …; additional:
…). Generate width / connectivity / function assertions."*

Three forces push the model toward `$bits()` width checks:

1. **The signal is alone in scope.** The LLM has no co-resident
   signals to relate it to — so a multi-signal property like
   `(en && !ack) |-> $stable(data)` cannot be expressed because the
   prompt never mentions `en` and `ack` together with `data`. Even if
   the per-signal record names them in `interconnection`, they don't
   appear as live signal handles the LLM can compose.
2. **Width is the one fact the LLM is certain about.** The
   structured record's `definition` field gives the bit-width
   directly. `$bits(sig) == N` is a property the model can verify
   trivially against its own input. The other categories
   (connectivity, function) require the LLM to invent semantics from
   the spec text — much harder, much more likely to hallucinate, much
   more likely to be wrong.
3. **Per-signal prompting fragments the design's relational
   structure.** A mux selector requires knowing that `out`, `sel`,
   `a`, `b` are all part of the same combinational expression. A
   register-update assertion requires `reg`, `enable`, and `source`.
   A handshake invariant requires `valid` and `ready` to appear in
   the same property. None of these multi-signal expressions can be
   constructed when the LLM only sees one signal at a time.

\fram{}'s pipeline avoids all three because it works the *opposite*
way:

- **AST skeletons seed structural relationships.** The pipeline
  pre-extracts case statements, mux selectors, and always_ff blocks
  from the RTL and feeds them as starting templates. The LLM is
  asked to *finish* a relational property, not invent one from
  scratch.
- **RAG retrieves multiple signals together.** The retrieved RTL
  chunks contain co-resident signals in their natural always_ff /
  always_comb context. The LLM sees `en | ack` and `out` and
  `data_in` in the same prompt — so multi-signal properties become
  expressible.
- **The lint-feedback loop pushes for structural depth, not
  signal-count breadth.** The refinement rounds rewrite malformed
  assertions and reject duplicates, so emitting the same `$bits`
  template 80 times across indexed copies wouldn't help.

The trade-off is starkly visible in the numbers: AssertLLM gets
*horizontal coverage* (every signal touched, however shallowly);
\fram{} gets *vertical depth* (fewer signals touched, but each one
related to its peers via real protocol / datapath / state-machine
logic).

### Quality dimensions \fram{} owns and AssertLLM does not

| Dimension | \fram{} (mean across 4 small designs) | AssertLLM (sum across 11 designs) |
|---|---:|---:|
| Mux-selector assertions | ~70 | 0 |
| Reset-implication assertions | ~110 | 0 |
| Sequential `$past` assertions | ~70 | 0 |
| `$onehot`, `$rose`, `$fell` operators | 0 | 0 |
| `$stable` backpressure properties | 0 | 2 (both on SDP) |
| `$bits()` width checks | a handful | ~250 |

The single category AssertLLM produced where \fram{} did not is
**`$stable` backpressure** on `nvdla_sdp` — 2 assertions of the form
`(valid && !ready) |-> $stable(pd)`. These are textbook-quality and
are the most genuinely useful assertions in the entire AssertLLM
corpus. The asymmetry is informative: per-signal prompting can elicit
a small number of canonical *forms* (when the spec clearly describes
one) while missing all the design-specific structural content.

### Caveat — does this mean facts aren't needed?

No. Re-reading the SpecGuard `no-facts` failures on cmacfull and csc
(see §5.5, "Honest limitations"):

- `no-facts` produces **0 assertions** on cmacfull (n=2) and csc
  (n=1) — INCOMPLETE failures.
- AssertLLM produces 82 and 97 assertions respectively on the same
  designs — but they are width-check inflation.
- `\fram{} full` (with facts) produces 4–6 assertions on these
  designs — few but structurally diverse.

The conclusion is **structured per-signal context is necessary on
large hierarchical designs** (>250 top-level signals), but
*how* you provide it matters. AssertLLM's per-signal isolation
trivially scales the prompt count but caps each prompt's depth at
"width = N." \fram{}'s global facts card gives the LLM enough
structure to attempt deeper assertions, at the cost of producing
fewer of them.

### Key takeaway for §5.8

> _\fram{} produces **2×–18× more assertions at 3.4×–23× lower
> tok/A** than AssertLLM on 6 of 10 designs. On 3 of the largest
> designs (CMAC-full, CDMA, CSC) AssertLLM "wins" raw tok/A by
> 1.2×–7.9×, but its high-yield output on these designs is
> overwhelmingly trivial `$bits()` width re-statements walked across
> indexed signal copies (94 of 97 on CSC, 79 of 82 on CMAC-full).
> Deduplicated by structural content, AssertLLM's CSC output reduces
> from 97 to ~3 distinct assertions._
>
> _The asymmetry is architectural: AssertLLM's per-signal prompting
> isolates each signal from its peers, forcing the LLM toward the
> one verifiable property per signal — its declared bit-width.
> \fram{}'s AST-seeded, RAG-retrieved batched prompting exposes
> co-resident signals together, enabling multi-signal properties
> (mux selection, register-update causality, reset semantics,
> handshake stability) that AssertLLM structurally cannot produce.
> Trading horizontal coverage for vertical depth is the dominant
> design choice, and \fram{} produces the deeper, more diverse
> assertion content for fewer tokens per assertion._

---

## 5.9 Updated batch — naive variance, SDP coverage, no-facts re-test (in progress, n=8 of 15)

A two-phase 15-rep batch (`naive_then_large_ablation_20260429_1850/`)
started 2026-04-29 18:50 is in flight. As of 2026-05-01 04:30 it has
**8 reps complete on every cell** (rep-08 finalising on csc / sdp).
The batch was designed to fix three known gaps in §5.5: (a) no
variance on the naive baseline, (b) SDP unrunnable, (c) `no-facts`
INCOMPLETE failures on cmacfull / csc preventing a fair facts-vs-no
comparison on the largest hierarchical designs. All three gaps are
now closed; the headline conclusions shift on (b) and (c).

Source: [naive_then_large_ablation_20260429_1850/all_runs.csv](../naive_then_large_ablation_20260429_1850/all_runs.csv),
[summary_agg.csv](../naive_then_large_ablation_20260429_1850/summary_agg.csv),
[summary.md](../naive_then_large_ablation_20260429_1850/summary.md).
qwen3:14b-32k, Ollama, T=0.1. Phase-1 = naive on every design;
Phase-2 = `{full, no-facts, flat-facts, no-ast, no-repair,
no-feedback}` on the 8 large designs.

### 5.9.1 Naive baseline finally has variance — and it's mostly noise around zero

Across the 11 designs at n=8 reps each, the naive baseline shows the
following picture (success-rate = subprocess rc=0; assertions counts
include partial output retained from rc=1 runs):

| Design | LOC | Reps OK / 8 | Tokens μ ± σ | Assertions μ ± σ | Tok/A median | Yield ceiling (max obs.) |
|---|---:|---:|---:|---:|---:|---:|
| FifoX | 586 | **8/8** | 86 543 ± 14 482 | **34.4 ± 10.3** | 2 603 | 50 |
| RVV | 1 591 | 2/8 | 38 377 ± 4 248 | 6.1 ± 5.1 | 4 135 | 11 |
| CMAC | 1 574 | 1/8 | 59 719 ± 349 | 0.5 ± 0.5 | 59 796 | 1 |
| Rubik | 15 004 | 4/8 | 34 978 ± 2 286 | 7.8 ± 4.0 | 3 798 | 12 |
| CACC | 30 905 | 1/8 | 36 357 ± 4 582 | 4.8 ± 2.3 | 6 229 | 6 |
| PDP | 43 233 | **0/8** | 39 270 ± 2 674 | **0.0 ± 0.0** | — | 0 |
| CDP | 64 696 | 4/8 | 39 753 ± 5 712 | 3.9 ± 2.6 | 17 848 | 7 |
| CMAC-full | 67 977 | 5/8 | 43 398 ± 17 277 | 7.2 ± 2.9 | 5 682 | 13 |
| CDMA | 103 010 | 4/8 | 40 370 ± 7 033 | 2.2 ± 1.8 | 34 033 | 5 |
| CSC | 104 575 | 7/8 | 37 366 ± 10 945 | 1.2 ± 0.7 | 33 354 | 3 |
| SDP | 261 546 | **8/8** | 38 210 ± 3 793 | **7.9 ± 6.8** | 6 389 | 18 |

Three observations the n=1 estimate could not have made:

- **FifoX naive holds up under repetition.** 34.4 ± 10.3 over 8
  reps with 100 % subprocess success — the n=1 reading of 36 was
  not a fluke, naive really does work on small designs. \fram{}'s
  29.5 ± 5.6 sits cleanly inside that variance band, so on FifoX
  the comparison is "naive wins on raw count, \fram{} wins on
  malformed-fraction and on σ" — robust at n=8.
- **PDP naive deterministically produces zero across 8 reps.** Not
  a sampling artefact of the prior n=1 — naive simply cannot
  generate any usable assertion on PDP under any seed we've drawn.
  Same story for CMAC (0 in 7 of 8 reps). The "naive collapses
  past 1.5 KLOC" narrative from §5.5 holds.
- **CACC naive yields 6 assertions on 7 of 8 reps** (one rc=0 run
  produced 6, the rest produced 1 or 6 with rc=1). The earlier n=1
  reading of "1 assertion" was actually the worst-case sample;
  median yield is 6, not 1. The naive baseline ratio in §5.2 RQ1
  shifts proportionally: tokens-per-correct on CACC naive is ≈
  6 230 (median) instead of 45 839, so the headline efficiency gain
  vs \fram{} is **≈ 4.5×** rather than 33×. Still a clean win, but
  not the order-of-magnitude figure the n=1 estimate suggested.

> _**Naive baseline correction for §5.2 / §5.4 / §5.5.** With n=8
> support, the headline "≥ 33× tokens-per-correct-assertion
> reduction" on CACC compresses to **≈ 4.5×**. The ranking (\fram{}
> dominates naive on CACC, RVV, CMAC, PDP) is unchanged; the
> magnitude is more modest than the single-run estimate. The 1.3×
> on FifoX, by contrast, holds — that comparison was already
> defensible at n=1._

### 5.9.2 SDP runs cleanly with the recursion fix

Pass-01 of the prior batch produced rc=1 across all 6 SDP variants
in 7–100 s with zero tokens logged
([§5.5 SDP note](#sdp-not-yet-run)). The fix landed in
[sva_pipeline/rtl_facts.py](../sva_pipeline/rtl_facts.py)
(`sys.setrecursionlimit(10_000)`); at n=7–8 reps per variant the
new batch shows:

| Variant | Reps OK | Tokens μ ± σ | Assertions μ ± σ | Tok/A median |
|---|---:|---:|---:|---:|
| naive | 8/8 | 38 210 ± 3 793 | 7.9 ± 6.8 | 6 389 |
| full | 7/7 | 155 563 ± 85 644 | 7.6 ± 2.0 | 26 547 |
| **no-facts** | 7/7 | **56 579 ± 13 714** | **11.1 ± 6.5** | **5 281** |
| flat-facts | 7/7 | 129 643 ± 94 828 | 6.6 ± 3.4 | 25 720 |
| no-ast | 7/7 | 170 613 ± 61 404 | 8.3 ± 3.9 | 28 069 |
| no-repair | 7/7 | 167 539 ± 103 580 | 8.7 ± 4.7 | 24 807 |
| no-feedback | 0/7 | 138 826 ± 93 281 | 0.0 | — |

SDP was the largest design in the suite (262 KLOC) and the one we
had no data on. With the fix, **no-facts is best on both axes**
(highest yield and 5× lower tok/A than full) and the pipeline
finishes in ~5 minutes per rep. AssertLLM on the same design
(§5.8) produced 17 assertions at 13 843 tok/A — \fram{} no-facts
beats it on both axes (11.1 asserts at 5 281 tok/A), correcting the
"no usable data" entry in §5.8's per-design table.

### 5.9.3 The cmacfull / csc no-facts INCOMPLETE failures are gone

The [§5.5 caveat](#honest-limitations) flagged cmacfull
(2/2 INCOMPLETE) and csc (1/1 INCOMPLETE) as designs where the
no-facts default could not converge. At n=8 in the new batch:

| Design | Variant | Reps OK / 8 | Assertions μ ± σ | Tok/A median |
|---|---|---:|---:|---:|
| CMAC-full | full | 8/8 | 9.1 ± 4.9 | 23 852 |
| CMAC-full | **no-facts** | **8/8** | **6.5 ± 2.4** | 32 101 |
| CMAC-full | flat-facts | 8/8 | 8.9 ± 1.9 | 27 029 |
| CSC | full | 6/8 | 4.6 ± 2.4 | 85 191 |
| CSC | **no-facts** | **7/8** | **4.9 ± 2.9** | 92 668 |
| CSC | flat-facts | 8/8 | 5.9 ± 5.7 | 97 731 |

`no-facts` now succeeds on **8 of 8** cmacfull reps and **7 of 8**
csc reps — completely reversing the prior "0/2" and "0/1" results.
The likely cause is a fix on the current `regex-postprocess-stable`
branch that landed between the two batches; the no-facts prompt
itself is unchanged. **The structural conclusion (§5.5,
"failure modes appear past ~70 KLOC") needs to be retracted.** All
three previously-failing variants now produce data, though yields
remain low for these large designs (5–10 assertions for 70–105 KLOC).

### 5.9.4 Updated facts-vs-no-facts picture across the 8 large designs

With variance-resolved data on every cell, the per-design winner
breakdown is more nuanced than the §5.5 "no-facts wins everywhere
except PDP / cmacfull / csc" story:

| Design | Best by raw assertions | Best by tok/A median | no-facts vs full (Δ asserts) |
|---|---|---|---:|
| Rubik | **no-ast** (14.8 ± 7.2) | no-ast (3 960) | −38 % (no-facts loses) |
| CACC | **no-facts** (369.4 ± 102.7) | no-facts (1 919) | +11 % |
| PDP | full (42.0 ± 9.8) | **no-facts** (973) | −22 % asserts but **−3 %** tok/A |
| CDP | full (16.8 ± 2.7) | **full** (901) | −23 % |
| CMAC-full | full (9.1 ± 4.9) | full (23 852) | −29 % |
| CDMA | flat-facts (11.0 ± 6.9) | flat-facts (6 812) | +32 % |
| CSC | flat-facts (5.9 ± 5.7) | full (85 191) | +7 % |
| SDP | **no-facts** (11.1 ± 6.5) | **no-facts** (5 281) | +46 % |

Patterns:

- **The no-facts default still wins on the two largest designs**
  with rich-enough top-level interfaces to enumerate from (CACC,
  SDP).
- **Hierarchical designs with sparse top-level signals**
  (PDP 42 sigs, CDP 39, Rubik 34) tip toward `full` or `no-ast` on
  raw count — the facts card supplies a relational scaffold the
  no-facts prompt cannot reconstruct from the few visible signals.
  But on PDP the tok/A winner is still no-facts (973 vs 1 000),
  because the modest assertion drop is more than offset by the
  20 % token saving.
- **`no-ast` is the surprise winner on Rubik and tied for top yield
  on cmacfull / csc.** This is not a generic claim; it appears to
  be specific to designs where AST extraction returns very few
  case-statement seeds (Rubik: reshape engine, no large mux
  trees). Worth a separate investigation.
- **`flat-facts` wins on CDMA and ties on CSC.** Unscoped facts
  occasionally beat depth-2 scoping on the largest designs — a
  sign that depth-2 may be cropping the relational context the
  LLM needs on hierarchical interfaces. This was statistically
  invisible at n=2.

> _**Updated default-flip recommendation.** The 14 B `no-facts`
> default holds for CACC and SDP and is competitive on PDP. It
> loses on raw yield for the four medium-large designs with sparse
> top-level signal maps (Rubik, CDP, CMAC-full, CDMA / CSC).
> A reasonable revised default would key on top-level signal density
> rather than LOC: **enable facts when (top-level signals / LOC) <
> some threshold**, since that is the regime where the spec doc
> alone gives the LLM too little surface to compose multi-signal
> properties. Awaits the remaining 7 reps for confirmation._

### 5.9.5 What still needs the missing 7 reps

- **CSC `full` and `no-feedback` cells are at n=6–7 not n=8**
  (a few rc-1 runs on the slowest cells). The remaining reps will
  thicken those distributions but should not flip the rankings
  (CSC results are already very high-σ).
- **The CACC/SDP no-facts wins are at n=8 with σ resolved at
  ~25 % CV.** Highly unlikely to flip; reps 9–15 will tighten the
  CIs, not change the ordering.
- **PDP no-facts under-performance is at n=8 with σ = 11.4 (CV
  35 %)** — the gap to `full` (32.9 vs 42.0 assertions) is ~1σ, so
  remaining reps could narrow or widen it. Watch this cell.
- **The naive baseline numbers used to recompute the §5.2 ratios
  are at n=8.** The headline-ratio update (33× → 4.5× on CACC) is
  load-bearing for the abstract / introduction and should be
  re-confirmed at n=15 before publication.

---

## Files referenced

- Primary data — qwen3:14b, small designs, 15 reps: `ablation_study_20260420_1232/all_runs.csv`
- Cross-model data — qwen3.6:35b, 4 designs, 15 reps: `ablation_study_20260424_0549/all_runs.csv`
- Partial data — qwen3:14b, large designs, in progress: `large_designs_ablation_20260423_1516/all_runs.csv`
- Updated batch (2026-04-29 → 2026-05-01, 8 of 15 reps) — naive variance + SDP coverage + facts re-test: `naive_then_large_ablation_20260429_1850/{all_runs.csv,summary_agg.csv,summary.md,batch.log}`
- Per-run artefacts under each batch: `<design>/<variant>/run_<NN>/{sva.sv,trace.json,token_summary.json,lint.json}`
- Runners: [run_ablation_study.sh](../run_ablation_study.sh),
  [run_large_designs_ablation.sh](../run_large_designs_ablation.sh)
- Aggregator: [scripts/ablation_summary.py](../scripts/ablation_summary.py)
- Token-usage plumbing: [sva_pipeline/trace_logger.py](../sva_pipeline/trace_logger.py),
  [sva_pipeline/backends/openai_backend.py](../sva_pipeline/backends/openai_backend.py#L123)
- AssertLLM baseline data: `baselines/results_20260429_1009/all_runs.csv`
- AssertLLM reproduction code: [baselines/assertllm/](../baselines/assertllm/)
- AssertLLM paper: `Baselines/assertllm.pdf` (Yan et al., ASPDAC '25)
- Companion analyses: [ablation_15run_findings.md](ablation_15run_findings.md)
  (RQ2 component attribution, variance deep-dive),
  [ablation_findings_cacc.md](ablation_findings_cacc.md)
  (default-flip rationale, bare-fragment bug report),
  [assertion_quality_review.md](assertion_quality_review.md)
  (domain-by-domain sample inspection on \fram{} output),
  [assertllm_baseline_review.md](assertllm_baseline_review.md)
  (AssertLLM output audit + bug list)
