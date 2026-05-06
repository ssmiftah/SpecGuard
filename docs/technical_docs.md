# SpecGuard Technical Documentation

> **SpecGuard** is the four-stage SVA generation system this project
> implements. The internal Python package is named `sva_pipeline/` for
> historical reasons; both names refer to the same system. See
> `docs/SpecGuard_Architecture.md` for the high-level walk-through.

Complete module-by-module reference for SpecGuard, covering every function in
`sva_pipeline/`, the data flow between stages, and the implementation
decisions behind each post-processing rule.

**Total codebase: 9,686 lines of Python across 26 files.**

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Entry Point — main.py](#2-entry-point)
3. [Configuration — config.py, config_loader.py](#3-configuration)
4. [RTL Analysis — slang_frontend.py](#4-rtl-analysis)
5. [AST Assertion Generation — ast_assertions.py](#5-ast-assertion-generation)
6. [Retrieval — rag.py](#6-retrieval)
7. [Agent — agent.py](#7-agent)
8. [Tools — tools.py](#8-tools)
9. [RTL Facts — rtl_facts.py (Stage 2-3)](#9-rtl-facts)
10. [Grammar — grammar.py (Stage 2.5)](#10-grammar)
11. [Lint Loop — lint_loop.py](#11-lint-loop)
12. [LLM Backends — backends/](#12-llm-backends)
13. [Mutation Testing — mutation/](#13-mutation-testing)
14. [Supporting Modules](#14-supporting-modules)
15. [Issues and Fixes Log](#15-issues-and-fixes-log)

---

## 1. Architecture Overview

### Data Flow

```
main.py
  ├── config_loader.load_project_config(yaml)       → PipelineConfig
  ├── slang_frontend.build_design_info(rtl, top)     → DesignInfo
  ├── config_loader.auto_detect(config, info)        → assertion style
  ├── config_loader.generate_task(...)               → task string
  │
  ├── [if small] read docs as plain text             → no heavy imports
  ├── [if large] rag.build_or_load_*_retriever()     → retrievers
  │     └── Uses code-specific embeddings for RTL
  │     └── Uses NLP embeddings for docs
  │     └── Hierarchical 2-stage for 5+ modules
  │
  ├── backends.create_backend(config)                → LLMBackend
  ├── agent.SVAAgent(config, ...)
  │     ├── ast_assertions.generate_ast_assertions() → skeletons (deterministic)
  │     ├── _llm_enrich_skeletons()                  → spec-validated (batched)
  │     │     ├── trivial → output directly
  │     │     └── complex → LLM in batches of 20
  │     ├── [fallback] plan_assertions() → execute_plan()
  │     └── [fallback] _direct_generate()
  │
  ├── lint_loop.run_lint_loop(agent, sva, config, signal_map)
  │     ├── fix_immediate_implication()              → fix |-> and ->
  │     ├── fix_double_negation()                    → simplify !(!(..)‖(..))
  │     ├── fix_condition_only_assertions()          → add output checks
  │     ├── deduplicate_assertions()                 → string-based dedup
  │     ├── remove_wrong_style_assertions()          → remove wrong style
  │     ├── validate_signals()                       → drop hallucinated signals
  │     ├── semantic_deduplicate()                   → canonical-form dedup
  │     ├── llm_self_review()                        → LLM checks its own work
  │     ├── split_assertions() → lint_all_assertions()
  │     └── agent.refine_assertions()                → fix failures
  │
  ├── write_output(sva, path)
  └── mutation.run_mutation_testing()                → mutation_report.json

agent.trace.save()                                   → trace.json + trace.csv
```

### Module Dependency Graph

```
main.py
  ├── config_loader.py  ← config.py, mutation/config.py
  ├── slang_frontend.py ← design_graph.py (fallback)
  ├── ast_assertions.py ← (no heavy deps, regex only)
  ├── rag.py            ← sentence_transformers, faiss (LAZY import)
  ├── agent.py          ← backends/, tools.py, ast_assertions.py, trace_logger.py
  ├── lint_loop.py      ← slang_frontend.py
  ├── html2md.py        ← beautifulsoup4, markdownify
  └── mutation/         ← sim_harness, testbench_gen, operators, report
```

**Critical design principle:** Heavy imports (torch, transformers, faiss,
sentence_transformers) are **lazy-loaded** — only imported when actually
needed. This prevents OOM crashes when the full RAG stack isn't required.

---

## 2. Entry Point

**File:** `main.py` (400+ lines)

### Functions

| Function | Purpose |
|----------|---------|
| `load_full_rtl_context(rtl_dir, threshold)` | Read all .v/.sv; return if < threshold chars |
| `write_output(sva_code, output_path, log_path)` | Write SVA file + append to run log |
| `main()` | Pipeline orchestrator |

### `main()` Execution Order

1. Parse CLI (single arg: YAML path)
2. Load config via `load_project_config(yaml)`
3. Setup logging from `config.log_level`
4. Build DesignInfo via `slang_frontend.build_design_info()`
5. Auto-detect assertion style, dtype
6. Generate task if not provided
7. HTML conversion if configured
8. Full context injection (small designs) or doc injection
9. **AST-only check**: skip RAG + LLM if `ast_only=True`
10. Build RAG indices if needed (lazy import)
11. Free embedding models from GPU
12. Instantiate SVAAgent (loads LLM only if `ast_only=False`)
13. `agent.generate_assertions(task)` → raw SVA
14. `run_lint_loop(agent, raw_sva, config)` → validated SVA
15. Write output
16. Mutation testing if enabled
17. Save trace

### Key Issue: Memory Management

**Problem:** Top-level imports of `torch`, `transformers`, `faiss`, and
`sentence_transformers` consumed 4+ GB at startup before any pipeline
logic ran. Combined with the LLM (~16 GB GPU), this caused OOM on
60 GB systems.

**Fix:** All heavy imports are deferred to the point they're needed:
- Line ~237: `from sva_pipeline.rag import ...` only when `need_rag=True`
- Line ~355: `from sva_pipeline.agent import SVAAgent` only at agent init
- Line ~345: `from sva_pipeline.lint_loop import run_lint_loop` only at lint time
- AST-only mode skips all heavy imports entirely

---

## 3. Configuration

### config.py (~180 lines)

`PipelineConfig` dataclass with 55+ fields organised by section:

| Section | Key Fields |
|---------|-----------|
| Design | `rtl_dir`, `top_file`, `top_module`, `docs_dir` |
| LLM | `model_id`, `backend`, `api_key`, `quantization` |
| Generation | `temperature`, `max_new_tokens`, `top_p` |
| RAG | `rtl_embedding_model`, `doc_embedding_model`, `rtl_top_k` |
| Hierarchical | `use_hierarchical_retrieval`, `hierarchical_stage1_k` |
| Hybrid | `use_hybrid_retrieval`, `rrf_k` |
| AST | `use_ast_assertions`, `ast_only`, `ast_max_case_branches` |
| Plan-Execute | `use_plan_execute`, `max_planning_steps` |
| Assertion Style | `reject_assert_property` (None=auto, True=immediate, False=concurrent) |
| Lint | `max_refinement_iterations`, `lint_failures_file` |
| HTML | `html_docs_enabled`, `html_docs_files` |
| Security | `threat_model` |
| Mutation | `mutation: MutationConfig` (nested) |
| Logging | `log_level` |

### config_loader.py (~520 lines)

| Function | Purpose |
|----------|---------|
| `load_project_config(yaml_path)` | YAML → PipelineConfig |
| `auto_detect(config, design_info)` | Fill assertion style from clock detection |
| `generate_task(...)` | Auto-generate verification task from design characteristics |
| `compute_source_checksum(dir)` | SHA-256 for index staleness detection |
| `is_index_stale(path, checksum)` | Compare stored vs current checksum |
| `_validate_yaml(data)` | Check required fields, verify paths |
| `_flatten_yaml(data)` | Map nested YAML to flat config fields |
| `_apply_smart_defaults(config)` | Derive output paths from top_module |
| `_detect_dtype()` | Auto GPU dtype from compute capability |

### YAML-to-Field Mapping

35+ mappings in `_YAML_TO_FIELD` dict. Key additions since initial version:

```python
"model.backend"           → "backend"
"model.quantization"      → "quantization"
"model.api_key"           → "api_key"
"retrieval.rtl_embedding_model" → "rtl_embedding_model"
"retrieval.use_hierarchical"    → "use_hierarchical_retrieval"
"agent.use_ast_assertions"      → "use_ast_assertions"
"agent.ast_only"                → "ast_only"
```

---

## 4. RTL Analysis — slang_frontend.py

**File:** `sva_pipeline/slang_frontend.py` (~660 lines)

Replaces both Yosys (design_graph.py) and Verible with pyslang.

### Key Functions

| Function | Purpose |
|----------|---------|
| `build_design_info(rtl_dir, top_module, top_file)` | Main entry → DesignInfo |
| `_build_with_slang(...)` | pyslang implementation |
| `_detect_clock(inst)` | Find `posedge` patterns → clock name |
| `_detect_reset(inst)` | Find `negedge` + reset naming → reset name |
| `_classify_signal_type(name, dir, width)` | Naming heuristics (clock, reset, control, data) |
| `slang_lint(sva_code, reject_assert_property)` | In-process SVA validation |

### DesignInfo Dataclass

```python
@dataclass
class DesignInfo:
    top_module: str
    modules: Dict[str, ModuleInfo]
    hierarchy_tree: Dict[str, List[str]]
    hierarchy_text: str           # replaces hierarchy.txt
    signal_map: Dict[str, Any]    # replaces signal_map.json
    graph_summary_text: str
    has_clock: bool               # auto-detected
    clock_signal: Optional[str]   # e.g. "pclk"
    reset_signal: Optional[str]   # e.g. "prstn"
```

### SVA Linting

`slang_lint()` wraps SVA in a module shell and compiles with pyslang:
- Concurrent assertions → module scope
- Immediate assertions → `always_comb` block
- Filters out `UndeclaredIdentifier` and `TypoIdentifier` (expected for wrapper)
- Semantic checks: rejects `assert property` when `reject_assert_property=True`,
  rejects `.width` attribute

---

## 5. AST Assertion Generation — ast_assertions.py

**File:** `sva_pipeline/ast_assertions.py` (~560 lines)

Deterministic assertion generation from RTL patterns — no LLM needed.

### Pattern Types Extracted

| Pattern | Regex Detection | Assertion Template |
|---------|----------------|-------------------|
| Case branch | `case(sel) val: begin out=expr; end endcase` | `assert(!(sel==val) \|\| (out==expr))` |
| Direct assign | `assign out = expr;` | `assert(out == (expr))` |
| Wire passthrough | `assign out = signal;` | `assert(out == signal)` |
| Ternary mux | `assign out = sel ? a : b;` | `assert(out == (sel ? a : b))` |
| Comb comparison | `out = (in == const);` | `assert(out == (in == const))` |
| Sequential reset (async) | `always @(posedge clk or negedge rst) if(!rst) reg<=0;` | `assert property(@clk !rst \|-> reg==0)` |
| Sequential reset (sync) | `always @(posedge clk) if(!rst) reg<=0;` | `assert property(@clk !rst \|=> reg==0)` |
| Sequential functional | `if(cond) reg <= expr;` | `assert property(@clk disable iff(!rst) cond \|=> reg==$past(expr))` |
| Mutual exclusivity | Multiple flags decoded from same source | `assert((flag_a + flag_b + flag_c) <= 1)` |

### Key Functions

| Function | Purpose |
|----------|---------|
| `extract_patterns(source, clock, reset)` | Extract all RTL patterns via regex |
| `generate_skeletons(patterns, is_combinational)` | Convert patterns to assertions |
| `generate_ast_assertions(...)` | Convenience: extract + generate |
| `format_skeletons_as_sva(skeletons)` | Format for direct output |
| `format_skeletons_for_llm(skeletons)` | Format for LLM validation prompt |
| `_extract_case_patterns(source)` | Find `case...endcase` blocks |
| `_extract_assign_patterns(source)` | Find `assign out = expr;` |
| `_extract_always_patterns(source, clk, rst)` | Find always blocks with reset/functional |
| `_generate_invariant_skeletons(patterns)` | Auto-detect mutual exclusivity groups |

### Key Issue: Default Case Contamination

**Problem:** The combinational always block extractor found blocking
assignments inside `case default:` branches and treated them as
unconditional assignments, producing `assert(out_data == 17'h10000)`.

**Fix:** Strip `case...endcase` blocks from the always body before
extracting blocking assignments:
```python
body_no_case = re.sub(r"case\s*\(.+?\).*?endcase", "", body, flags=re.DOTALL)
```

### Key Issue: Async vs Sync Reset

**Problem:** All reset assertions used `|=>` (next-cycle), but async
resets clear immediately (same cycle).

**Fix:** `RTLPattern` now carries `reset_is_async` and `reset_polarity`
fields. The skeleton generator uses `|->` for async, `|=>` for sync:
- Async detected: `negedge rst` in sensitivity list → `reset_is_async=True`
- Polarity detected: `if (!rst)` → active-low, `if (rst)` → active-high

---

## 6. Retrieval — rag.py

**File:** `sva_pipeline/rag.py` (~950 lines)

### Embedding Models

| Purpose | Model | Dimension | Size |
|---------|-------|-----------|------|
| RTL code | `jinaai/jina-embeddings-v2-base-code` | 768 | 89 MB |
| Documentation | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 22 MB |

**Improvement:** jina code model shows 106.7% better code-to-query
similarity vs MiniLM on Verilog code.

### Retriever Classes

| Class | Description |
|-------|-------------|
| `FAISSRetriever` | L2-normalised cosine via IndexFlatIP |
| `HybridRetriever` | FAISS + BM25 with Reciprocal Rank Fusion |
| `HierarchicalRetriever` | Two-stage: module summary → filtered code |

### Hierarchical Two-Stage Retrieval

```
Stage 1: Module summary index (small, ~20-50 entries)
    Query: "key expansion" → matches keyExpansion, AES_Encrypt
    │
Stage 2: Full code index (filtered to Stage 1 modules)
    Query: "key expansion" → returns keyExpansion source code
```

Activated when `use_hierarchical_retrieval=True` and design has 5+ modules.

### Context-Enriched Chunks

RTL chunks are prefixed with hierarchy metadata:
```
[Module: keyExpansion | Parent: AES_Encrypt | Ports: key(in,128), w(out,1408)]
module keyExpansion ...
```

`_build_module_prefix(module_name, design_info)` generates the prefix.

### Key Functions

| Function | Purpose |
|----------|---------|
| `_get_encoder(model_name)` | Singleton encoder cache |
| `chunk_rtl_file(path, max_chars, design_info)` | Module-aware RTL chunking with hierarchy prefix |
| `load_rtl_chunks(rtl_dir, max_chars, design_info)` | Walk directory + chunk |
| `chunk_document_file(path, size, overlap)` | Paragraph-aware doc chunking |
| `build_module_summary_chunks(design_info)` | One compact chunk per module |
| `build_or_load_hierarchical_retriever(...)` | Build both stages |
| `_tokenize_for_bm25(text)` | Verilog-aware tokenizer (camelCase, underscore split) |

---

## 7. Agent — agent.py

**File:** `sva_pipeline/agent.py` (~1250 lines)

### Generation Strategy Priority

```python
def generate_assertions(self, task):
    # 1. AST-guided (deterministic, fastest)
    if use_ast_assertions:
        skeletons = _ast_guided_generate()
        if ast_only: return skeletons
        else: return _llm_enrich_skeletons(skeletons)  # batch+filter

    # 2. Plan-then-execute (LLM-based)
    if use_plan_execute:
        plan = plan_assertions()  → execute_plan()

    # 3. Direct ReAct (last resort)
    return _direct_generate()
```

### AST + LLM Batch Processing

`_llm_enrich_skeletons()` splits skeletons:
- **Trivial** (case_branch, wire_passthrough, direct_assign) → output directly
- **Complex** (ternary, sequential, comparisons) → batched to LLM, 20 per batch

Each batch gets a fresh conversation with the validation prompt asking
the LLM to compare skeletons against the design documentation.

### System Prompt

`_build_system_prompt()` constructs the prompt with:
- Design hierarchy, graph summary, signal map
- Full RTL and docs (if context-injected)
- Tool descriptions
- Assertion style guidance (conditional on clock detection)

`_assertion_style_guidance()` returns different instructions based on
`reject_assert_property`:
- `True` → immediate assertions only, no `assert property`
- `False` → concurrent assertions with `@(posedge clk) disable iff (!rst)`

### Key Functions

| Function | Purpose |
|----------|---------|
| `_ast_guided_generate()` | AST extraction + optional LLM enrichment |
| `_llm_enrich_skeletons(skeletons, task)` | Batch+filter spec validation |
| `plan_assertions(task)` | Planning phase → JSON plan |
| `execute_plan(plan)` | Per-assertion focused generation |
| `_direct_generate(task)` | Original 40-iteration ReAct loop |
| `refine_assertions(failures)` | Fix lint failures (with `\|->` guidance) |
| `_step(messages)` | One LLM call via backend + trace logging |
| `_dispatch(call)` | Tool execution + trace logging |
| `_extract_sva(text)` | Extract SVA from code fences / raw text |

---

## 8. Tools — tools.py

**File:** `sva_pipeline/tools.py` (~600 lines)

### Five Agent Tools

| Tool | Backend | Description |
|------|---------|-------------|
| `rtl_retrieve` | FAISSRetriever / HierarchicalRetriever | Search RTL index; redirects if full context injected |
| `doc_retrieve` | FAISSRetriever / HybridRetriever | Search doc index; redirects if docs injected |
| `slang_extract` | DesignInfo graph lookup | Module ports and structure (instant) |
| `signal_map_lookup` | Dict search | Signal attributes (substring match) |
| `slang_lint` | pyslang in-process | SVA syntax + semantic validation |

### Tool Naming History

Originally named `verible_lint` and `yosys_extract`. Renamed to
`slang_lint` and `slang_extract` when Slang replaced both tools.
The dispatch handler accepts both old and new names for backward
compatibility.

---

## 9. RTL Facts — rtl_facts.py (Stage 2-3)

**File:** `sva_pipeline/rtl_facts.py`

Single source of structured RTL facts extracted via pyslang. Used by
the agent (prompt augmentation) and post-processors (validation).

### RTLFacts Dataclass

| Field | Type | Description |
|-------|------|-------------|
| `signal_definitions` | `Dict[str, str]` | Signal → defining expression |
| `case_selectors` | `Dict[str, Set[str]]` | Signal → case selector expressions |
| `combinational_signals` | `Set[str]` | LHS of continuous assigns / comb always |
| `signal_drive_kind` | `Dict[str, str]` | Signal → "comb", "seq", or "mixed" |
| `constant_signal_pairs` | `Dict[str, Set[str]]` | Literal → owning signals |
| `all_signals` | `Set[str]` | Every declared signal name |
| `signal_widths` | `Dict[str, int]` | Signal → bit width (from Compilation) |
| `signal_frequencies` | `Dict[str, int]` | Signal → IdentifierName reference count |
| `clock_signals` | `Set[str]` | All clock signals (MCD-aware) |
| `reset_signals` | `Set[str]` | All reset signals |
| `reset_polarity` | `Dict[str, str]` | Reset → "low" or "high" |
| `clock_reset_pairs` | `Set[Tuple]` | (clock, reset) domain pairs |
| `reset_values` | `Dict[str, Tuple]` | Register → (reset_signal, reset_value) |
| `signal_to_module` | `Dict[str, str]` | Signal → module name (Stage 3 lazy) |
| `per_module` | `Dict[str, RTLFacts]` | Module → scoped RTLFacts (Stage 3 full) |
| `module_hierarchy` | `Dict[str, int]` | Module → depth from top_module |

### Key Functions

| Function | Purpose |
|----------|---------|
| `extract_rtl_facts(rtl_dir, signal_map, top_module, module_facts_mode)` | Single entry point — parse + extract all facts |
| `format_facts_for_prompt(facts, ...)` | Format facts as Markdown for system prompt (6 sections, two-tier loading, token budget) |
| `format_facts_reminders(facts, ...)` | Compact reminders for end of prompt (instruction sandwich) |
| `format_batch_facts(facts, signal_names)` | Per-batch signal-specific facts (Stage 2b) |
| `format_already_covered(prior_sva)` | Compact list of prior-batch assertions (tested counterproductive, unused) |
| `RTLFacts.for_module(name, depth)` | Return module-scoped facts (tries per_module → lazy filter → flat fallback) |

### Prompt Formatter Sections (in order)

1. Clock/reset pairs (always, ~30 tokens)
2. Reset values — unusual only (core) or full (extended)
3. Signal widths + drive-kind annotations, width-grouped compact format
4. Case-driven signal selectors
5. Constant ownership (negative phrasing)
6. Submodule signal timing (Fix 6, when module-scoped)
7. Signal name constraints + hallucination denylist
8. Common SVA mistakes (generic bad patterns)

### Module Scoping (Stage 3)

Two modes controlled by `module_facts_mode`:
- **"lazy"** (Option G): `_build_signal_to_module()` maps signals to modules
  via Compilation hierarchy. `RTLFacts._filter_for_module()` filters flat
  facts at format time. Fast, lossy on name collisions.
- **"full"** (Option F): `_split_by_module()` finds ModuleDeclaration
  boundaries. `_extract_per_module_facts()` calls existing extractors per
  module via `_roots()` adapter. Collision-proof.

Depth-limited scoping (`module_scope_depth`): `_build_module_hierarchy()`
maps modules to their depth from `top_module`. `for_module(name, depth=2)`
includes all modules within 2 levels.

---

## 10. Grammar — grammar.py (Stage 2.5)

**File:** `sva_pipeline/grammar.py`

Generates a GBNF grammar that constrains LLM output so signal identifiers
must come from `RTLFacts.all_signals`. Gated behind `use_grammar_constraints`.

| Function | Purpose |
|----------|---------|
| `generate_sva_grammar(all_signals, clock_signals, reset_signals)` | Build GBNF with signal allowlist |
| `generate_sva_grammar_from_facts(facts)` | Convenience wrapper from RTLFacts |

Backend support: vLLM (`extra_body.guided_grammar`), local HF (outlines
`CFGLogitsProcessor`). Ollama not supported via OpenAI API.

---

## 11. Lint Loop — lint_loop.py

**File:** `sva_pipeline/lint_loop.py`

### Full Post-Processing and Validation Chain

```
raw_sva
  Phase 1 — TRANSFORM:
  → fix_bare_property_fragments()    # wrap bare property as assert
  → fix_immediate_implication()      # |-> and -> to !(c)||(r)
  → fix_double_negation()            # !(!(..)‖(..))‖(..) simplification
  → fix_immediate_and_form()         # AND-form to implication
  → fix_condition_only_assertions()  # extract output from error message
  → fix_next_cycle_on_combinational()# |=> to |-> for comb signals (RTLFacts)

  Phase 2 — REMOVE WRONG:
  → remove_trivial_assertions()      # drop trivially true, width mismatches
  → remove_wrong_style_assertions()  # remove assert property on comb designs
  → verify_constant_signal_pairs()   # drop wrong constant-signal pairings
  → validate_signal_widths()         # drop out-of-range bit selects
  → validate_reset_values()          # drop wrong reset values (MCD-aware)
  → check_case_selector_mismatch()   # drop wrong case selectors (optional)

  Phase 3 — VALIDATE & DEDUPLICATE:
  → validate_signals()               # drop hallucinated signals + log denylist
  → deduplicate_assertions()         # string-based exact dedup
  → semantic_deduplicate()           # canonical-form semantic dedup
  → remove_subsumed_and_contradicting() # logical subsumption rules

  Phase 4 — LLM Self-Review (optional):
  → llm_self_review()                # LLM checks its own assertions

  Phase 5 — Lint Loop:
  → split_assertions()               # one per assert statement
  → lint_all_assertions()            # pyslang check each
  → [failures] → agent.refine_assertions()
  → repeat up to 3x
```

### Phase 1: Syntax Post-Processors

**`fix_immediate_implication()`**: Handles both `|->` and `->` in
immediate assertions. Detects via regex, splits on the operator,
reconstructs as `!(cond) || (consequence)`. Skips `assert property`
(where these operators are valid).

**`fix_double_negation()`**: Detects `assert (!( !(COND) || (RESULT) ) || (RESULT))`
and simplifies to `assert (!(COND) || (RESULT))`. The LLM sometimes
wraps AST skeletons in an extra negation layer.

**`fix_condition_only_assertions()`**: Detects assertions where the
error message describes the expected output but the assertion only
checks the condition. Extracts `out_data = EXPR` from the error
message and restructures to `!(cond) || (out_data == EXPR)`.

**`deduplicate_assertions()`**: String-based deduplication. Normalises
whitespace and strips error messages, then compares assertion bodies.
Keeps the first occurrence (AST assertions come first).

**`remove_wrong_style_assertions()`**: Removes `assert property` on
combinational designs (no clock), including the preceding comment line.

### Phase 2: Signal Existence Validation

**`validate_signals(sva_code, signal_map)`**: Cross-references signal
names in each assertion against the design's `signal_map`.

- `_extract_signal_names(assertion)` extracts identifiers, filtering
  out SV keywords, system functions (`$bits`, `$error`), and bit-literal
  fragments (`b011`, `hff`).
- Uses a **soft threshold**: drops an assertion only if **>50%** of its
  signal references are unknown. This avoids false positives from internal
  signals not in the port-level signal map.
- Effectively catches LLM hallucinations (invented signal names).

### Phase 3: Semantic Deduplication

**`semantic_deduplicate(sva_code)`**: Goes beyond string-based dedup
by normalising assertion bodies to a canonical form.

`_normalise_assertion_semantic(assertion)` produces a canonical form:
- Strips error messages, comments, clock edges, `disable iff` clauses
- Removes all parentheses
- Normalises negation spacing (`! x` → `!x`)
- Sorts `||` clauses alphabetically
- Converts to lowercase

This catches cases where the LLM regenerates an AST assertion with
different formatting, error messages, or operand ordering. For example:
```
AST:  assert (!(code == 3'b011) || (out_data == 16'h0)) else $error("case 011");
LLM:  assert ( !( code==3'b011 )  ||  ( out_data==16'h0 ) ) else $error("booth +2x");
→ Both normalise to: "!code == 3'b011 || out_data == 16'h0"
→ Second is removed as duplicate
```

### Phase 4: LLM Self-Review

**`llm_self_review(agent, sva_code, signal_map)`**: Asks the LLM to
review and correct its own generated assertions in a second pass.

The LLM receives:
- All current assertions (both AST and LLM-generated)
- The signal map (valid signal names, directions, and widths)
- Instructions to check: signal existence, width correctness,
  logical correctness, and usefulness

The LLM is told to:
1. Fix assertions with wrong signal names or widths
2. Fix logically incorrect assertions
3. Remove trivially true assertions (e.g., `assert(x == x)`)
4. Keep all good assertions unchanged
5. **Not add new assertions** — only fix or remove existing ones

Skipped when `ast_only=True` (no LLM loaded) or when no signal map
is available. Falls back to original assertions if the review returns
empty output.

### Refinement Prompt

When the lint loop sends failures to the LLM for repair, it checks
for the common `|->` error and adds specific fix guidance:

```
IMPORTANT: The |-> operator is ONLY valid inside 'assert property'.
In immediate assertions, replace:
  assert (COND |-> RESULT) else $error(...);
with:
  assert (!(COND) || (RESULT)) else $error(...);
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `split_assertions(sva_code)` | Split SVA into individual assertion entries |
| `lint_single_assertion(assertion)` | Lint one assertion via pyslang |
| `lint_all_assertions(assertions)` | Lint all, return passed + failures |
| `fix_immediate_implication(sva)` | Fix `\|->` and `->` in immediate assertions |
| `fix_double_negation(sva)` | Simplify `!(!(..) \|\| (..)) \|\| (..)` |
| `fix_condition_only_assertions(sva)` | Add output check from error message |
| `deduplicate_assertions(sva)` | String-based exact dedup |
| `remove_wrong_style_assertions(sva, config)` | Remove wrong assertion style |
| `validate_signals(sva, signal_map)` | Drop assertions with hallucinated signals |
| `semantic_deduplicate(sva)` | Canonical-form semantic dedup |
| `llm_self_review(agent, sva, signal_map)` | LLM reviews its own assertions |
| `reassemble_assertions(passed)` | Join passing assertions into SVA string |
| `run_lint_loop(agent, sva, config, signal_map)` | Full post-generation pipeline |

---

## 12. LLM Backends — backends/

### base.py

`LLMBackend` Protocol — single method:
```python
def generate(messages, tools) -> Tuple[str, List[Dict]]
```

### local.py

HuggingFace local model with quantization support:
- `none` → bfloat16/float16 (full precision)
- `int8` → 8-bit via bitsandbytes
- `int4` → 4-bit NF4 via bitsandbytes (best quality/size ratio)

Loads to `cuda:0` with `low_cpu_mem_usage=True`.
Skipped entirely when `ast_only=True`.

Stage 2.5: `set_grammar(gbnf)` creates an outlines `CFGLogitsProcessor`
that constrains token generation to the signal-allowlist grammar. Requires
`pip install outlines`. Grammar and tool-calling are mutually exclusive.

### openai_backend.py

OpenAI-compatible API. Tool definitions passed directly (already in
OpenAI format). Works with vLLM, Ollama via `api_base`.

Stage 2.5: `set_grammar(gbnf)` stores a GBNF grammar. When active
(and no tools in the request), passes it via `extra_body={"guided_grammar": ...}`
for vLLM/SGLang. Ollama ignores `extra_body` through the OpenAI API.

Stage 2a: Prefix cache warming — `_warm_prefix_cache()` sends a minimal
request with the system prompt so the server caches the prefix before
the first real batch. Benefits Ollama, vLLM, SGLang.

### anthropic_backend.py

Anthropic Claude. Converts tool schema from OpenAI to Anthropic format.
Handles message format differences (separate system, tool_result blocks,
consecutive role merging). No grammar constraint support.

---

## 13. Mutation Testing — mutation/

### Mutation Operators (7 types)

| Operator | Example |
|----------|---------|
| OP_REPLACE | `&` → `\|`, `==` → `!=` |
| CONST_REPLACE | `1'b0` → `1'b1` |
| SIGNAL_SWAP | Swap same-width signals |
| BITSLICE_MUT | `[17:2]` → `[16:1]` |
| COND_NEGATE | `!rst` → `rst` |
| ASSIGN_DELETE | Comment out assignment |
| SENSITIVITY_MUT | `posedge` → `negedge` |

### SVA Injection for Simulation

`_inject_sva_into_dut()` separates assertions by type:
- Concurrent (`assert property`) → module scope
- Immediate (`assert (...)`) → inside `always_comb begin...end`

This was required because Verilator only allows `assert property` at
module scope and immediate assertions in procedural blocks.

### Simulators

| Simulator | Backend Function | SVA Support |
|-----------|-----------------|-------------|
| Verilator | `_run_verilator()` | Limited (no double `\|=>`) |
| xsim | `_run_xsim()` | Full IEEE 1800-2017 |

---

## 14. Supporting Modules

### trace_logger.py (~180 lines)

Records every pipeline step to JSON + CSV:
- Model output text and length
- Tool calls and observations
- Lint iteration results
- Mutation testing score
- Phase tracking (extraction, generation, planning, execution, refinement, lint)

### html2md.py (~410 lines)

Converts HTML spec files to Markdown with image extraction:
- Strips non-content (nav, scripts, footers)
- Extracts images to `_images/` subdirectory (data URIs, local files, remote URLs)
- Cached by mtime

### json2csv.py (~320 lines)

Converts pipeline JSON reports to CSV:
- Lint reports → `_lint_summary.csv` + `_lint_failures.csv`
- Mutation reports → `_mutation_summary.csv` + `_operators.csv` + `_mutants.csv`
- Trace reports → `_trace.csv`

### design_graph.py (~440 lines)

Yosys-based fallback when pyslang is not installed.

---

## 15. Issues and Fixes Log

### Memory Issues

**Issue: 51 GB RSS → OOM kill during model loading**
- Root cause: `AutoModelForCausalLM.from_pretrained()` with `device_map="auto"`
  stages the full model in CPU RAM before transferring to GPU
- Fix: Changed to `device_map="cuda:0"` with `low_cpu_mem_usage=True`
- File: `sva_pipeline/backends/local.py`

**Issue: OOM even with direct GPU loading**
- Root cause: Top-level imports of torch, faiss, sentence_transformers at startup
  consume 4+ GB before any pipeline logic runs
- Fix: All heavy imports deferred to point of use (lazy import pattern)
- Files: `main.py` (lines 237, 355)

**Issue: Two SentenceTransformer instances double GPU memory**
- Root cause: RTL and doc retrievers each created their own encoder
- Fix: Singleton `_ENCODER_CACHE` dict shares one instance per model name
- File: `sva_pipeline/rag.py` (lines 36-43)

**Issue: VSCode extension kills pipeline process (exit code 144)**
- Root cause: VSCode process manager has execution time limits
- Fix: `run_pipeline.sh` runs pipeline via `nohup` with memory monitoring
- File: `run_pipeline.sh`

### Assertion Syntax Issues

**Issue: `|->` in immediate assertions fails linting**
- Root cause: `|->` is only valid inside `assert property`, not `assert (...)`
- Fix: `fix_immediate_implication()` converts to `!(cond) || (result)`
- File: `sva_pipeline/lint_loop.py` (line 307)

**Issue: `->` (plain arrow) also fails in immediate assertions**
- Root cause: Same as above, different operator syntax
- Fix: Updated regex to catch both `|->` and standalone `->`
- File: `sva_pipeline/lint_loop.py` (line 328)

**Issue: Double negation `!(!(...) || (...)) || (...)`**
- Root cause: LLM wraps AST skeleton's `!(cond) || (result)` in extra negation
- Fix: `fix_double_negation()` simplifies by stripping outer negation
- File: `sva_pipeline/lint_loop.py` (line 365)

**Issue: Condition-only assertions (no output check)**
- Root cause: LLM writes expected output in error message but not in assertion
- Fix: `fix_condition_only_assertions()` parses error message and restructures
- File: `sva_pipeline/lint_loop.py` (line 412)

### Assertion Quality Issues

**Issue: Only width checks generated (no functional assertions)**
- Root cause: 8B model can't compose `condition |-> consequence` assertions
- Fix: AST extraction generates structural assertions deterministically
- File: `sva_pipeline/ast_assertions.py`

**Issue: Wrong mutual exclusivity logic**
- Root cause: LLM uses convoluted negation instead of one-hot check
- Fix: AST auto-detects decode groups and generates `(sum <= 1)`
- File: `sva_pipeline/ast_assertions.py`, `_generate_invariant_skeletons()`

**Issue: Unconditional output value assertions**
- Root cause: LLM over-generalises case-specific zero-pattern checks
- Fix: Prompt instructs LLM to skip unconditional output checks
- File: `sva_pipeline/agent.py`, validation prompt

**Issue: Assertions from RTL not spec (circular verification)**
- Root cause: AST-only verifies "RTL does what RTL does"
- Fix: LLM validates AST skeletons against documentation (ground truth)
- File: `sva_pipeline/agent.py`, `_llm_enrich_skeletons()`

**Issue: LLM output truncated at 4096 tokens**
- Root cause: 95 skeletons too many for single LLM call
- Fix: Batch+filter — trivial direct, complex in batches of 20
- File: `sva_pipeline/agent.py`, `_llm_enrich_skeletons()`

### Reset Assertion Issues

**Issue: Async reset uses `|=>` (next cycle) instead of `|->` (same cycle)**
- Root cause: Template didn't distinguish sync vs async reset
- Fix: `RTLPattern` carries `reset_is_async` detected from sensitivity list
- File: `sva_pipeline/ast_assertions.py`

**Issue: Active-high reset uses wrong polarity in `disable iff`**
- Root cause: Template assumed all resets are active-low
- Fix: `RTLPattern` carries `reset_polarity` detected from `if(!rst)` pattern
- File: `sva_pipeline/ast_assertions.py`

### Simulation Issues

**Issue: All mutants stillborn in Verilator**
- Root cause: Immediate assertions at module scope (Verilator requires procedural)
- Fix: `_inject_sva_into_dut()` separates concurrent (module scope) and
  immediate (`always_comb`) assertions
- File: `sva_pipeline/mutation/sim_harness.py`

**Issue: Verilator internal fault on double `|=>` chains**
- Root cause: `a |=> b |=> c` not supported in Verilator 5.020
- Fix: Added xsim backend (`simulator: "xsim"`) with full SVA support
- File: `sva_pipeline/mutation/sim_harness.py`

### LLM Assertion Quality Issues (Post-Generation)

**Issue: LLM hallucinated signal names**
- Root cause: LLM invents signals not present in the design (e.g., `fake_ready`,
  `internal_state`) when it can't find the right signal
- Fix: `validate_signals()` extracts signal identifiers from each assertion and
  cross-references against `design_info.signal_map`. Drops assertions where
  more than 50% of signals are unknown
- File: `sva_pipeline/lint_loop.py`, `validate_signals()`

**Issue: LLM duplicates AST assertions with different formatting**
- Root cause: LLM regenerates the same check AST already produced, but with
  different whitespace, error messages, or operand ordering. String-based
  `deduplicate_assertions()` misses these because the text differs
- Fix: `semantic_deduplicate()` normalises to a canonical form (strips parens,
  error messages, clock edges; sorts `||` clauses; lowercases) before comparison
- File: `sva_pipeline/lint_loop.py`, `semantic_deduplicate()`

**Issue: LLM generates logically incorrect assertions**
- Root cause: Wrong widths, wrong signal references, trivially true checks
  (e.g., `assert(x == x)`). These pass syntax linting but are semantically wrong
- Fix: `llm_self_review()` sends all assertions + signal map back to the LLM
  in a review-focused prompt. The LLM checks signal existence, width correctness,
  logical correctness, and removes unfixable assertions
- File: `sva_pipeline/lint_loop.py`, `llm_self_review()`

### Default Case Contamination

**Issue: `assert(out_data == 17'h10000)` as unconditional assertion**
- Root cause: Combinational always block extractor found assignments inside
  `case default:` and treated them as standalone
- Fix: Strip `case...endcase` blocks before extracting blocking assignments
- File: `sva_pipeline/ast_assertions.py`, `_extract_always_patterns()`
