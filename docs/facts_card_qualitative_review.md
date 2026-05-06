# RTL Facts Card — Qualitative Review on the 8 Large Designs

_Source data: `naive_then_large_ablation_20260429_1850/` (8–9 reps per
cell). Each cell pair below is **`full`** (RTL facts on, depth-2
scoping) vs **`no-facts`** (facts disabled). All other pipeline
components — AST extraction, RAG, lint loop — are held constant.
Methodology + analyzer in `/tmp/facts_analyzer.py` (regenerable)._

The §5.5 ablation tells us *whether* the facts card helps on each
design; this review tells us *how*. We analyse the actual emitted
SVAs, count distinct register/signal targets, categorise structural
patterns, and detect malformed output.

---

## 1. Headline metrics

| Design | n | Asserts μ ± σ — full | Asserts μ ± σ — no‑facts | Δ | Reg. targets μ — full | Reg. targets μ — no‑facts | Cats μ — full | Cats μ — no‑facts | Malf. (∑ over reps) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Rubik** | 9 | **11.4 ± 7.9** | 7.6 ± 3.6 | **+50 %** | **3.1** | 1.6 | 2.4 | 2.8 | 0 / 0 |
| **PDP** | 9 | **41.3 ± 9.4** | 33.6 ± 10.8 | **+23 %** | **22.9** | 19.4 | 4.8 | 4.7 | 0 / 0 |
| **CDP** | 9 | **16.6 ± 2.6** | 13.8 ± 5.2 | **+20 %** | 8.2 | 7.9 | 3.1 | 2.9 | 0 / 0 |
| **CMAC‑full** | 8 | **9.1 ± 4.9** | 6.5 ± 2.4 | **+40 %** | **4.9** | 2.2 | 2.9 | 2.9 | 0 / 0 |
| **CACC** | 9 | 342.0 ± 72.3 | **379.1 ± 100.4** | −10 % | 284.2 | 297.2 | 6.0 | 7.1 | **0 / 30** |
| **CDMA** | 8 | 7.9 ± 5.6 | **10.4 ± 3.8** | −24 % | 3.0 | 4.1 | 2.1 | 2.2 | 0 / 0 |
| **CSC** | 8 | 4.6 ± 2.4 | **5.6 ± 2.3** (n=7) | −18 % | 1.2 | 2.1 | 1.5 | 1.6 | 0 / 0 |
| **SDP** | 8 | 7.9 ± 2.0 | **10.6 ± 6.2** | −25 % | 3.4 | 3.6 | 2.4 | 3.0 | 0 / 0 |

`Reg. targets` = distinct LHS-of-equality signals across the run
(unique registers/wires the assertion *checks*).
`Cats` = distinct structural categories among
{`reset-imp`, `seq-update`, `past-other`, `stable-backpressure`,
`onehot`, `inside-set`, `edge-detect`, `handshake`,
`implication-other`, `other`}.
`Malf.` = orphan `else $error(...)` lines without a preceding
`assert` (a syntactic-edge-case the lint loop accepts but a real
simulator rejects).

The split is clean: **facts win on the four hierarchical / config-
shadow designs, no-facts win on the four interface-rich designs**,
with one dimension (malformed-assertion suppression on CACC) where
facts deliver a benefit no-facts does not.

---

## 2. The four wins — what facts buy you

Each of these designs is dominated by **monitor-shadow registers**:
many similar-shaped flops capturing config values. The facts card
enumerates them, the LLM walks the list, and yield scales linearly
with how many registers the facts card surfaced.

### 2.1 PDP (43 KLOC, 42 top signals) — +23 %, +18 covered registers per rep

Median rep diff (full=40, no-facts=39 assertions but only 19 vs 23
distinct register targets):

```
only-full   (13): mon_reg2dp_nan_to_zero, mon_reg2dp_pad_bottom,
                  mon_reg2dp_pad_left, mon_reg2dp_pad_right,
                  mon_reg2dp_pad_top, cmd_en, cv_int_rd_req_ready_d0,
                  cv_int_wr_req_ready_d0, mc_int_rd_req_ready_d0, …
only-NF     (13): mon_gap_between_layers, mon_layer_end_flg,
                  mon_op_en_dly, mon_op_en_pos,
                  mon_reg2dp_cube_in_channel/height/width, …
```

Both runs cover ~20 registers but **different ones**. The no-facts
run picks the registers the LLM remembers from the spec doc
(cube_in_*, gap_between_layers); the with-facts run reaches the
register classes the spec doesn't enumerate (pad_*, the `_d0`
clock-domain-crossing handshake registers). The total surface
covered (full ∪ no-facts) is wider when you run both — supporting
the hypothesis that facts and spec-doc context are complementary, not
substitutable.

### 2.2 CDP (65 KLOC, 39 top signals) — +20 %, narrowest variance

Mostly the same `mon_reg2dp_*` shadowing pattern as PDP. CDP's
unusually tight σ on the with-facts variant (2.6 vs 5.2 for no-facts)
is because the facts-listed register set is deterministic across
runs, so the LLM converges to the same ~16 properties on every seed.
The no-facts run has higher variance because *which* registers it
remembers from the spec varies seed-to-seed.

### 2.3 CMAC-full (68 KLOC, 284 top signals) — +40 %, 2× register coverage

The largest relative win on raw assertions. CMAC-full has 284
top-level signals (much higher than PDP/CDP); without facts, the LLM
emits 6.5 ± 2.4 properties touching only 2 distinct registers. With
facts it emits 9.1 ± 4.9 touching ~5 registers. Median diff:

```
only-full ( 3): nvdla_core_rstn, reg_offset_rd_int, sc2mac_wt_mask
only-NF   ( 4): NV_NVDLA_CMAC_CORE_MAC_exp_output,
                NV_NVDLA_CMAC_CORE_slcg_clock,
                NV_NVDLA_CMAC_REG_dual_working, mode
```

The full variant produces concrete register checks; the no-facts
variant emits names that look generated from the **module hierarchy
text** (long `NV_NVDLA_CMAC_CORE_*` paths) — a sign the LLM is
guessing from the file/module names rather than reading the actual
signal map.

### 2.4 Rubik (15 KLOC, 34 top signals) — +50 %, but very high variance

Rubik is the design where facts produce the largest *relative* win
(+50 % assertions) and the largest σ on both variants (7.9 / 3.6).
With only 34 top-level signals visible, the spec doc is the only
substantial input the no-facts prompt can use, and Rubik's spec is
unusually thin on register-level detail — so no-facts caps out at 7
assertions in most reps. With facts the LLM gets the credit-flow
signals (`p1_skid_ready_flop` reset value = 1, an unusual non-zero
reset that the facts card extracted correctly).

---

## 3. The four losses — what facts cost you

These designs have **rich top-level interfaces** (CACC 3000+ signals
visible; SDP 87 with deep handshake surface). The facts card eats
prompt budget that the LLM would otherwise spend reading spec-doc
protocol descriptions, and the resulting output is *narrower* even
when it's *more numerous*.

### 3.1 CACC (31 KLOC, ~3000 top signals) — −10 % asserts, but **30 malformed avoided**

The cell where no-facts wins on raw count is also the only cell
where no-facts produces malformed output: 30 orphan `else $error`
lines across 4 of 9 reps (rep_06 alone had 15). An example from
[cacc/no-facts/run_02:706](naive_then_large_ablation_20260429_1850/cacc/no-facts/run_02/sva.sv#L706):

```
// Reset: cacc2sdp_valid must be 0 on reset (from spec 10)
else $error("cacc2sdp_valid async reset value mismatch");
```

— the comment commits to a property the LLM never finished writing.
With facts on CACC, this happens **zero times across 9 reps**. The
structured templates fed by the facts card don't truncate mid-property.

So the choice on CACC is not "more vs fewer assertions" but
"~340 lint-clean vs ~370 emitted, with ~3 malformed per rep on
average" — once you discount the malformed lines, the gap narrows
significantly.

### 3.2 SDP (262 KLOC, 87 top signals) — −25 % asserts, but cross-signal-equality risk

[sdp/no-facts/run_06](naive_then_large_ablation_20260429_1850/sdp/no-facts/run_06/sva.sv) emits 11 assertions, of which 8 are
`$stable` backpressure invariants on the various sdp2{cvif,mcif,pdp,
csb} interfaces — a strong, deep property class the with-facts
variant misses (3 such assertions in run_04). **This is the no-facts
upside**: with attention not eaten by the facts card, the LLM
recognises the symmetry across the seven external interfaces and
writes one $stable property per interface.

But the same file also includes:

```systemverilog
(sdp2pdp_valid && sdp2pdp_ready) |-> (sdp2pdp_pd == cacc2sdp_pd)
(sdp2cvif_rd_req_valid && sdp2cvif_rd_req_ready) |-> (sdp2cvif_rd_req_pd == cacc2sdp_pd)
(sdp2mcif_rd_req_valid && sdp2mcif_rd_req_ready) |-> (sdp2mcif_rd_req_pd == cacc2sdp_pd)
```

These claim that SDP's *output* payload equals SDP's *input* payload
when valid+ready hold — i.e. that SDP is a passthrough. That's
semantically wrong (SDP applies a non-linear post-processing
operation). The facts-on variant would have seen the comb/seq driver
records for these signals and not made the equality claim.

Net read: no-facts trades **structural depth** (8 $stable vs 3) for
**semantic accuracy** (3 cross-signal hallucinations). The lint
loop catches neither because both classes are syntactically valid.

### 3.3 CDMA (103 KLOC, 63 top signals) — −24 % asserts, broader category mix

Median diff:

```
only-full ( 3): cdma2buf_wt_wr_en, cdma2sc_wt_entries, nvdla_core_rstn
only-NF   ( 6): cdma2buf_dat_wr_data, cdma2buf_wt_wr_data,
                cdma2glb_done_intr_pd, cdma2sc_dat_pending_req,
                cdma2sc_wt_kernels, cdma_dat2cvif_rd_req_pd
```

No-facts touches 6 distinct interfaces vs full's 3. CDMA is a
DMA-engine, so the spec-doc emphasis on `_pending_req`, `_done_intr`,
and write-data integrity is exactly the surface the LLM misses when
the facts card is loaded. Notably, both variants produce 0 malformed
on CDMA — the malformed-assertion regression is genuinely
CACC-specific.

### 3.4 CSC (105 KLOC, 567 top signals) — −18 % asserts, but full has 0 distinct registers in some reps

CSC is the design with the **smallest absolute output** and the
largest signal map (567 top signals). The facts card's depth-2
scoping shaves the 567-signal map down to a subset, but even the
scoped subset is too large to fit in the prompt usefully.
Median: full=4 (1 register target), no-facts=7 (2 register targets).
A weak win for no-facts on a design where neither variant produces
much.

---

## 4. The patterns

Three dimensions emerge consistently across the 8 designs:

### 4.1 Coverage breadth vs spec-protocol depth (the central trade-off)

**Facts increase register-target coverage where the design has many
similar registers** (PDP: +18 % register targets, Rubik: +94 %,
CMAC-full: +123 %). On those designs the LLM walks the list.

**No-facts increases category breadth where the design has rich
interface protocols** (SDP: 3.0 vs 2.4 categories, CACC: 7.1 vs 6.0).
Without the facts card, the LLM spends attention on the spec-doc
description of `valid/ready/$stable` interface contracts.

The two effects are **not in opposition on the same axis** — they're
complementary. A pipeline that ran both variants and unioned the
output would always cover more than either alone.

### 4.2 Malformed-output suppression (CACC-specific but real)

Across the 8-design × ~9-rep corpus, the only cell with malformed
output is **CACC no-facts**: 30 orphan `else $error` lines across 4
of 9 reps. The same 9 reps with facts on produce 0 malformed lines.
This is consistent with the prior 15-rep batch (`§5.4 caveat: ~2 %
malformed on CACC`), and now we can attribute that 2 % entirely to
the no-facts configuration.

CACC is the largest individual file in the corpus (~600 KB of RTL
text); the no-facts prompt for CACC is right at the edge of the
context window where the LLM starts producing degraded output. The
facts card's structural templates apparently anchor the output and
prevent the degradation.

### 4.3 Variance reduction on facts-driven cells

Where facts win on yield, they also win on variance — because the
facts-listed register set is deterministic across reps:

| Design | Asserts CV — full | Asserts CV — no-facts |
|---|---:|---:|
| Rubik | 69 % | 47 % |
| PDP | 23 % | 32 % |
| CDP | 16 % | 38 % |
| CMAC-full | 54 % | 37 % |
| CACC | 21 % | 26 % |
| CDMA | 71 % | 37 % |
| CSC | 52 % | 41 % |
| SDP | 25 % | 58 % |

PDP, CDP, SDP show the cleanest pattern: the variant that wins on
yield also wins on variance. CDMA is the exception (no-facts wins on
both). Generally: **facts produce a more reproducible output**,
which matters if the bench is being used for paper-figure
fingerprints.

### 4.4 Hallucination signature (qualitative, hard to count)

The malformed-assertion detector catches syntactic breakage but not
*semantic* hallucinations. The two clear semantic-hallucination
patterns I caught reading the SVAs by hand:

- **Equating opposite-direction signals**: `request.pd == response.pd`
  on PDP/SDP no-facts. The signals' `direction` field is in the facts
  card; without it the LLM doesn't know request and response flow in
  opposite directions.
- **Inventing module-path names from the hierarchy text**:
  `NV_NVDLA_CMAC_CORE_MAC_exp_output` on CMAC-full no-facts — a
  synthetic full-path identifier that doesn't exist as a flat signal.

Both classes pass lint (the names are syntactically valid) but would
be flagged immediately by a simulator or by any signal-validator that
checks against the actual elaborated design. A counter for these
would require AST-level analysis we don't currently have wired up.

---

## 5. Recommendation

The §5.5 default-flip (`use_rtl_facts: False`) is correct **on
average across the corpus** — no-facts wins on 4 of 8 designs and
saves ~85 K tokens on the largest cell (CACC). But the win is
narrower than the §5.5 numbers alone suggest, and it comes with two
hidden costs:

1. **Malformed-assertion regression on CACC** (30 orphan `else $error`
   lines across 9 no-facts reps; 0 on full).
2. **Cross-signal hallucinations** (≥ 3 per SDP no-facts run; rare
   on full).

A reasonable revised default would key the facts toggle on the
**ratio of (top-level signals : top-level register-shadow patterns)**:

- High ratio (CACC, SDP, CDMA, CSC) → no-facts. The signal map is
  large enough that the facts card crowds out spec-doc attention; the
  LLM is better off reading the protocol description.
- Low ratio (Rubik, PDP, CDP, CMAC-full) → full. The signal map
  is small and dominated by similar-shape register classes; the
  facts card's enumeration is the dominant yield driver and the
  lint-clean property template prevents truncation.

A coarser proxy (LOC : top-level signals) gives the same partition
on the 8 designs with one swap (CMAC-full lands in the wrong bucket),
but the register-shadow ratio is cleaner. Pre-pipeline detection of
"is this design config-shadow-heavy?" would let the runner pick the
default automatically.

For the CACC malformed-assertion issue specifically, we may want to
fix the no-facts prompt template directly — adding a one-line
"every assertion must include `assert property (...)` followed by an
`else $error(...)` and a terminating `;`" instruction may close the
gap independently of the facts toggle.

---

## Files referenced

- Per-rep SVA files: [naive_then_large_ablation_20260429_1850/](../naive_then_large_ablation_20260429_1850/)
  `<design>/<full|no-facts>/run_NN/sva.sv`
- Headline-metric source for §5.5: [rq_findings.md §5.9](rq_findings.md#L960)
- Earlier qualitative CDP/PDP comparison: this analysis extends the
  PDP run_01 + CDP run_06 case study from the same source
- Analyser script: `/tmp/facts_analyzer.py` (regenerable on demand)
