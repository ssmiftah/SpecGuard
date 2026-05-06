# What Makes a SystemVerilog Assertion Good?

A reference for evaluating SVA-generation methods. Written after manually
inspecting outputs from SpecGuard, AssertLLM, Assertain, and ChIRAAG and
realising that pattern-presence metrics (does the file have `$past`? does
it have a multi-signal antecedent?) credit shallow or broken output that
a human reviewer would reject in seconds.

The five dimensions below are ordered by priority. Validity is a hard gate
— without it, no other dimension matters. The remaining four are graduated
quality measures.

---

## 1. Validity (gate)

An assertion that does not compile, references hallucinated signals, or
fails lint is not a verification artefact. It is text. Validity is the
gate that separates verification work from prose.

**Sub-criteria:**

- **Compileable** — parses as SystemVerilog, all `assert property (NAME)`
  references resolve to a declared `property NAME;` in the same scope.
- **Real signals** — every identifier used in the assertion exists in the
  design's RTL. Hallucinated identifiers (LLM-invented names) make the
  assertion meaningless even if it parses.
- **Lint-clean** — no obvious anti-patterns (`disable iff X` cancelling
  the antecedent `X`, malformed orphan `else $error(...)` lines without
  a preceding `assert`, sampled-value functions outside a clocked
  context, etc.).

**What we can measure programmatically:**
- Property-name resolution: catch `assert property (truncated_name)` where
  `truncated_name` was never declared.
- Signal validity against a per-design RTL manifest (extracted via pyslang
  or regex-collected from the design's `rtl_dir`).
- Malformed-fragment count (existing `qwc.detect_malformed`).

**What we cannot easily measure programmatically:**
- Whether the assertion would actually elaborate (full pyslang parse + bind
  on the baseline output is doable but expensive).
- Race conditions, glitches, and other dynamic semantic issues.

**Examples:**
```sv
// VALID
assert property (@(posedge clk) disable iff (!rstn)
    (req_pvld && req_prdy) |=> req_data_stable);

// INVALID — truncated property reference (Assertain bug)
property p_handshake_check; ... endproperty
assert property (p_handshake_chec)   // missing trailing 'k'

// INVALID — hallucinated signal (ChIRAAG on NVDLA)
assert property (@(posedge nvdla_core_clk)
    state_machine_state == IDLE);    // `state_machine_state` not in RTL
```

---

## 2. Falsifiability

A good assertion can fail if the design is buggy. An assertion that is
either always true or always false constrains nothing — it is a no-op
embedded in the test bench.

**Sub-criteria:**

- **Non-trivial antecedent** — not `1`, `1'b1`, or a constant; not always
  false (which would silently disable the assertion).
- **Non-vacuous consequent** — not always true given the antecedent.
- **Not a static-config check** — `(int_precision == 8 || int_precision == 16) && (fp_precision == 16)`
  is either always true or always false at synthesis time. It does not
  fire in simulation.
- **Not a tautology** — `a |-> a`, `sig == sig`.

**What we can measure programmatically:**
- Detect tautological forms via regex.
- Detect single-constant-on-each-side comparisons (`(literal) == (literal)`).
- Detect `disable iff (X)` followed by an assertion whose antecedent is
  `X` (the disable kills the antecedent — dead code).
- Flag assertions whose antecedent is solely a literal.

**What we cannot easily measure programmatically:**
- Whether the antecedent is reachable in the design's actual state space
  (would require formal reachability analysis or simulation).
- Whether the consequent is logically implied by the antecedent given the
  RTL semantics (formal property checking).

**Examples:**
```sv
// FALSIFIABLE — fires if reset doesn't clear the register
assert property (@(posedge clk) !rstn |-> (out_valid == 1'b0));

// NOT FALSIFIABLE — static configuration check, no temporal aspect
assert ((int_precision == 8 || int_precision == 16) && (fp_precision == 16));

// NOT FALSIFIABLE — tautology
assert property (@(posedge clk) sig |-> sig);

// NOT FALSIFIABLE — disable iff cancels the antecedent
assert property (@(posedge clk) disable iff (!rstn)
    !rstn |-> (sig == 0));   // disable removes any cycle where !rstn is true
```

---

## 3. Specificity

A good assertion encodes a particular design behavior, not a generic
shape. Width checks (`$bits(sig) == 1`) and existence checks
(`sig inside {1'b0, 1'b1}`) are correct but trivially provable from the
signal declaration — they would pass even on an empty design that just
declares the signal.

**Sub-criteria:**

- **Datapath content** — bit slicing (`sig[7:0]`), concatenation
  (`{a, b, c}`), arithmetic (`a + b`, `a * b - c`).
- **Concrete expected values** — multi-bit literals (`17'h10000`,
  `4'b0010`) on the consequent, not just `1'b0` or `1`.
- **Signal-derived consequent** — the consequent contains identifiers
  from the design, not only literals (e.g., `out == in[7:0] + 1` is
  signal-derived; `out == 0` is literal-only).
- **Conditional truth-table coverage** — different antecedent patterns
  map to different consequent expressions (e.g., the booth encoder where
  each input pattern produces a different output formula).

**What we can measure programmatically (with caveats):**
- Bit-slice / concat / multi-bit-lit / arithmetic via regex on the
  assertion text.
- "Specificity at validity" — only count behavioral markers when the
  surrounding assertion uses ≥80% real signals (avoids crediting
  hallucinated assertions that happen to contain the right syntactic
  markers).

**What we cannot easily measure programmatically:**
- Whether the consequent encodes the *correct* expected value (would
  require an oracle / golden model).
- Truth-table coverage relative to the design's actual input space.

**Examples:**
```sv
// SPECIFIC — encodes a row of the booth encoder truth table
assert (!({is_8bit, in_code} == 4'b0001) ||
        (out_data == {~src_data[15], src_data}));

// SPECIFIC — sequential register update with concrete past value
assert property (@(posedge clk) disable iff (!rstn)
    (mon_op_en_pos) |=> mon_reg2dp_input_data_type == $past(reg2dp_input_data_type));

// NOT SPECIFIC — width check, true by signal declaration
assert ($bits(global_clk_ovr_on_sync) == 1);

// NOT SPECIFIC — generic implication shape, no design-specific content
assert property (@(posedge clk) sig_a |-> sig_b);
```

---

## 4. Domain breadth

A good assertion *suite* covers multiple categories of design behavior,
not just one. A file with 50 reset checks and nothing else has a coverage
gap in handshake protocol, datapath, and FSM transitions.

**Domain categories:**

- **Reset behavior** — register initial values, reset propagation timing.
- **Handshake protocol** — valid/ready, request/response, payload
  stability while held.
- **Datapath transformation** — bit manipulation, arithmetic invariants,
  mux selection, encoder/decoder truth tables.
- **State-machine transitions** — legal state sequences, state-encoding
  invariants (`$onehot`).
- **Backpressure / liveness** — `$stable` while not ready, eventual
  completion guarantees.

**Sub-criteria:**

- Number of distinct domains touched in a file (out of ~5–6 reasonable
  categories).
- Whether the *primary* domain matches what the design actually exposes
  (a pure datapath design without handshakes shouldn't be penalised for
  lacking handshake assertions).

**What we can measure programmatically:**
- Reuse `qwc.categorise` (already maps each assertion to one of 10 domain
  labels: `reset-imp`, `handshake`, `seq-update`, `edge-detect`,
  `stable-backpressure`, `onehot`, `inside-set`, `past-other`,
  `implication-other`, `other`).
- Collapse the 10 labels into 5 domain categories above.

**What we cannot easily measure programmatically:**
- Whether the *right* domains were covered for the design under test
  (would require knowing the design's behavioral surface).
- Whether the assertions in each domain are deep or shallow.

---

## 5. Coverage density

A good assertion suite scales its assertion count with the design's
complexity. Five assertions on a 100,000-LOC design is too thin
regardless of how good those five are.

**Sub-criteria:**

- **Complex-unique density** — `n_complex_unique / KLOC`. Targets vary
  by domain (control-heavy: ≥0.5/KLOC; pure datapath: ≥0.2/KLOC may be
  enough since one truth-table covers many cycles).
- **Output-signal coverage** — fraction of the module's output ports
  appearing on the consequent side of at least one assertion.

**What we can measure programmatically:**
- Density via `n_complex_unique / (LOC / 1000)`.
- Output coverage if we have the RTL manifest with port directions.

**What we cannot easily measure programmatically:**
- Whether each output is *meaningfully* exercised, not just touched.
- Whether assertions discriminate between "important" outputs (handshake
  signals, control outputs) and "incidental" outputs (debug, status).

---

## Why we use a gated framework

Earlier additive frameworks (8 independent dims summed to a Q score)
allowed methods to compensate. A method could fail signal validity and
still earn substantial Q_dim by stacking syntactic-richness markers
around hallucinated identifiers. ChIRAAG's NVDLA outputs exhibited
exactly this pathology.

The gated framework closes that loophole:

```
Q = 0                         if !validity
Q = falsifiability            ∈ [0, 1]
  + specificity               ∈ [0, 1]
  + domain_breadth            ∈ [0, 1]
  + coverage_density          ∈ [0, 1]
                              → Q ∈ [0, 4] for valid files;
                                Q = 0      otherwise
```

Validity does not contribute to Q numerically — it just enables the rest.
This forces ChIRAAG's NVDLA hallucinations to Q = 0 and Assertain's
broken-prop-ref outputs to Q = 0, which matches what manual inspection
shows.

---

## What this still doesn't capture

Honest disclosure of the metric's blind spots. These are real limits, not
implementation gaps:

1. **No semantic correctness check.** We don't know whether
   `out == in[7:0] + 1` is the *correct* invariant for this design or a
   plausible-looking guess. A formal property checker or simulation-based
   mutation kill rate would be needed.
2. **No notion of "the right assertions for this design."** A booth
   encoder benefits from truth-table assertions; a FIFO benefits from
   full/empty/handshake assertions. We measure breadth but not
   appropriateness.
3. **No path or state-space coverage.** Three assertions might cover the
   same state-machine path; we'd count them as 3 distinct domains if they
   use different operators, but they exercise no additional behavior.
4. **No simulation or formal verification.** All checks are static text
   analysis. An assertion that lints clean and has rich syntax can still
   be vacuous in the design's actual reachable state space.

These limits mean the metric is a **lower-bound proxy for assertion
quality**. A method that scores high on Q is probably good. A method that
scores low is definitely bad. A method that scores medium needs human
review to disambiguate.

---

## Implications for SVA-generation methods

What different methods do well and badly under this framework:

| Method | Validity | Falsifiability | Specificity | Domain breadth | Density |
|---|---|---|---|---|---|
| SpecGuard | ✓ (signal-validated, lint-gated) | ✓ (no static-config style) | ✓ (datapath truth tables on cmac, reset+seq-update on cdp) | ✓ (multi-domain per design) | ✓ (scales with LOC) |
| AssertLLM | ✓ (uses real signals) | partial (`$bits == 1` is trivially true) | ✗ (width checks only) | ✗ (one domain) | partial |
| Assertain | ✗ on most designs (broken prop refs) | n/a | n/a | n/a | n/a |
| ChIRAAG (Coral) | ✓ (small RTL, identifiers mostly real) | ✓ | partial | partial | thin |
| ChIRAAG (NVDLA) | ✗ (hallucinated identifiers) | n/a | n/a | n/a | n/a |

**The headline efficiency claim should be framed accordingly:** SpecGuard
is the only method that consistently passes the validity gate AND scores
on the four quality dimensions across the design suite. Per-rep token
cost is a secondary metric — relevant only after validity is established.
