# SpecGuard — Architecture & Developer Guide

> **SpecGuard** is a token-efficient four-stage process for generating
> SystemVerilog Assertions (SVA) from RTL and design specifications. Throughout
> this document the system is referred to as SpecGuard; legacy code/comments may
> still use the working name "SoC-LLM" or "the pipeline".

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Layout](#2-repository-layout)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Component Deep-Dive](#4-component-deep-dive)
   - 4.1 [config.py — Central Configuration](#41-configpy--central-configuration)
   - 4.2 [rag.py — Dual FAISS Retrieval Layer](#42-ragpy--dual-faiss-retrieval-layer)
   - 4.3 [tools.py — Agent Tool Implementations](#43-toolspy--agent-tool-implementations)
   - 4.4 [agent.py — Spec-Validation Agent Loop](#44-agentpy--spec-validation-agent-loop)
   - 4.5 [lint_loop.py — Post-Generation Lint Feedback Loop](#45-lint_looppy--post-generation-lint-feedback-loop)
   - 4.6 [main.py — Entry Point & CLI](#46-mainpy--entry-point--cli)
5. [Data Flow: End-to-End Walk-Through](#5-data-flow-end-to-end-walk-through)
6. [Tool Reference](#6-tool-reference)
7. [Qwen3 Tool-Calling Protocol](#7-qwen3-tool-calling-protocol)
8. [Spec-Validation Loop State Machine](#8-spec-validation-loop-state-machine)
9. [Lint Feedback Loop State Machine](#9-lint-feedback-loop-state-machine)
10. [FAISS Index Details](#10-faiss-index-details)
11. [Known Limitations & Future Work](#11-known-limitations--future-work)
12. [Running SpecGuard](#12-running-specguard)
13. [Debugging & Troubleshooting](#13-debugging--troubleshooting)

---

## 1. Overview

SpecGuard's central objective is **token-efficient SVA generation**: every cycle
of LLM inference must spend its budget on what only an LLM can solve — the
semantic claim each assertion makes about the design — and not on work a
deterministic tool can do for free. The means is **design-knowledge weaving**:
at each step we choose what to extract ahead of time, in what compact form to
deliver it, and through which channel (system prompt, retrieval, or tool call)
to make it available, so the model receives the strongest grounding at the
lowest possible token cost.

The four stages of the process (see Figure in the paper, or
`Latex Docs/Architecture.tex`):

1. **Stage 1 — Design Analysis.** pyslang parser, AST extractor with
   cluster-and-compact, RTL facts card. Produces LLM-bound assertion skeletons,
   trivial-pattern direct-emission output, and a structured facts card.
2. **Stage 2 — Spec-Validation Agent.** Qwen3-14B (Q4_K_M) over Ollama's
   OpenAI-compatible endpoint, batched at 40 skeletons per call, with a hybrid
   FAISS+BM25 RAG retriever (code-specific embedder for RTL, NLP embedder for
   spec docs) and `slang_lint` available as a tool call.
3. **Stage 3 — Post-Processing & Refinement.** Three deterministic phases
   (regex repair, structural check, semantic filter) followed by a lint
   feedback loop (≤3 refinement cycles).
4. **Stage 4 — Security & Output.** Optional CWE-tagged security pass reusing
   the Stage 2 endpoint, optional mutation testing (Verilator/xsim), and the
   final verified SVA file.

The benchmark suite spans 11 designs from NVDLA and Google Coral NPU
(119–261,546 LOC). The system was originally prototyped on AES-Verilog; the
AES configuration is preserved as a small smoke-test and `aes_project.yaml`
remains a useful "hello world" example.

### Goals

| Goal | How it is met |
|------|---------------|
| Assertions grounded in actual RTL | `rtl_retrieve` + `yosys_extract` tools look up real signal names and widths |
| Assertions consistent with spec | `doc_retrieve` tool searches the design specification |
| Syntactically valid output | `verible_lint` tool validates every assertion before the final answer |
| Correct assertion style | System prompt and `verible_lint` both enforce immediate `assert (...)` syntax for combinational designs |
| Reproducible, configurable | All parameters live in `PipelineConfig`; CLI overrides every field |

---

## 2. Repository Layout

```
SoC-LLM/
├── main.py                        # CLI entry point
├── signal_map.json                # Manually authored signal metadata (AES)
├── sva_output.sv                  # Generated assertions (output)
├── sva_pipeline_log.txt           # Append-only run log
│
├── sva_pipeline/                  # Core pipeline package
│   ├── __init__.py                # Package marker & docstring
│   ├── config.py                  # PipelineConfig dataclass
│   ├── rag.py                     # FAISS-based RAG layer
│   ├── tools.py                   # Tool implementations + TOOL_DEFINITIONS
│   ├── agent.py                   # SVAAgent (ReAct loop, model loading)
│   └── lint_loop.py               # Post-generation lint feedback loop
│
├── indices/                       # Persisted FAISS indices (auto-created)
│   ├── rtl_index.faiss
│   ├── rtl_index.pkl
│   ├── doc_index.faiss
│   └── doc_index.pkl
│
├── docs/                          # Design documentation fed into RAG
│   ├── AES_spec.txt               # FIPS vectors, port specs, timing
│   └── AES_README.md              # Module-level README
│
├── RTL Cases/
│   └── AES-Verilog/
│       ├── SourceCode/            # RTL sources (.v files)
│       └── tb/                    # Testbench (not indexed by the pipeline)
│
└── OtherTools/
    └── hierarchy.txt              # Design hierarchy (injected into system prompt)
```

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              main.py                                    │
│  CLI → PipelineConfig → load hierarchy + signal_map → build indices     │
│  → extract RTL facts (pyslang) → instantiate SVAAgent → generate → lint │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │          SVAAgent            │
                    │  (agent.py)                  │
                    │                              │
                    │  Qwen3-14B (int4, ~7 GB)     │
                    │  ReAct loop (max 40 iters)   │
                    │  System prompt with:         │
                    │    • RTL Facts context card  │
                    │      (Stage 2 — widths,      │
                    │       drive kinds, resets,   │
                    │       case selectors, MCD)   │
                    │    • full hierarchy          │
                    │    • signal map table        │
                    │    • workflow instructions   │
                    │    • critical reminders      │
                    │      (instruction sandwich)  │
                    └──────┬───────────────────────┘
                           │ tool calls
          ┌────────────────┼─────────────────────────────┐
          │                │                             │
    ┌─────▼──────┐  ┌──────▼──────┐  ┌───────────────────▼───────────────┐
    │ rtl_       │  │ doc_        │  │ yosys_extract  signal_map_lookup  │
    │ retrieve   │  │ retrieve    │  │ verible_lint                      │
    │            │  │             │  │                                   │
    │ FAISS RTL  │  │ FAISS Doc   │  │ Yosys subprocess  JSON dict       │
    │ index      │  │ index       │  │ Verible subprocess                │
    └─────┬──────┘  └──────┬──────┘  └───────────────────┬───────────────┘
          │                │                             │
    ┌─────▼────────────────▼─────────────────────────────▼────────────────┐
    │              RTL source files (.v)    docs (.md/.txt)               │
    │              signal_map.json          hierarchy.txt                 │
    └─────────────────────────────────────────────────────────────────────┘
```

The pipeline is a **single-agent design** — one LLM instance (Qwen3-14B or
similar) acts as both "RTL analyst" and "spec analyst" by routing its queries
to the appropriate RAG index through different tools.

RTL facts are extracted via pyslang before the agent starts and injected into
the system prompt as a structured context card (Stage 2). This gives the LLM
ground-truth signal information (widths, drive kinds, reset values, clock
domains) that reduces post-processor corrections by ~93% on tested designs.
For multi-module designs, facts can be scoped to the top module's hierarchy
(Stage 3) to focus the LLM's context.

---

## 4. Component Deep-Dive

### 4.1 `config.py` — Central Configuration

**File:** [sva_pipeline/config.py](../sva_pipeline/config.py)

A single `@dataclass` (`PipelineConfig`) holds every tunable parameter. No
value is hard-coded anywhere else in the codebase; all modules accept a
`PipelineConfig` instance.

#### Key fields

| Field | Default | Purpose |
|-------|---------|---------|
| `model_id` | `"Qwen/Qwen3-8B"` | HuggingFace model to load |
| `dtype` | `"bfloat16"` | Weight dtype (halves VRAM vs float32) |
| `device_map` | `"auto"` | Lets accelerate spread layers across devices |
| `enable_thinking` | `False` | Disables Qwen3's internal `<think>` scratchpad for faster, crisper tool calls |
| `max_new_tokens` | `1024` | Per-step token budget (thought + action) |
| `temperature` | `0.1` | Near-deterministic — keeps tool calls structured |
| `embedding_model` | `"all-MiniLM-L6-v2"` | ~80 MB, fast, adequate semantic quality |
| `rtl_top_k` / `doc_top_k` | `5` | Chunks returned per RAG query |
| `doc_chunk_size` | `1000` | Max chars per documentation chunk |
| `rtl_max_chunk_chars` | `8000` | Max chars before a module is sub-split |
| `rtl_index_path` | `"indices/rtl_index"` | FAISS index prefix for RTL corpus |
| `doc_index_path` | `"indices/doc_index"` | FAISS index prefix for docs corpus |
| `force_rebuild_index` | `False` | Rebuild indices even if cached files exist |
| `verible_bin` | `/home/user/Verible/bin/verible-verilog-syntax` | Full path to Verible binary |
| `yosys_bin` | `"yosys"` | Yosys binary (assumed on PATH) |
| `max_iterations` | `40` | Hard cap on ReAct loop iterations |
| `max_refinement_iterations` | `3` | Lint-fix-relint cycles after generation |
| `lint_failures_file` | `"./lint_failures.json"` | JSON report of lint failures |
| `max_refinement_react_steps` | `15` | ReAct steps per refinement call |

---

### 4.2 `rag.py` — Dual FAISS Retrieval Layer

**File:** [sva_pipeline/rag.py](../sva_pipeline/rag.py)

Provides two distinct retrieval corpora, each with a purpose-built chunking
strategy, embedded with the same sentence-transformer, and stored in a FAISS
flat inner-product index.

#### RTL Chunking (`chunk_rtl_file`)

```
Verilog file
    │
    ▼
Regex: module…endmodule  (re.DOTALL)
    │
    ├── Module fits within rtl_max_chunk_chars (8000)?
    │       YES → single chunk
    │       NO  → sliding window sub-chunks (8000 chars, 200-char overlap)
    │
    ▼
Chunk: { "text": <module body>, "metadata": { source, module, type, chunk_index } }
```

**Why module-level splitting?** SVA assertions reference exact signal names
within a module's port list and `always` blocks. Splitting mid-module would
sever port declarations from logic, misleading the embeddings.

The AES design produces **16 RTL chunks** across its 15 Verilog files (the
`keyExpansion` module is large enough to be split into 2 sub-chunks).

#### Documentation Chunking (`chunk_document_file`)

```
Markdown / plain-text file
    │
    ▼
Sliding window (1000 chars, 200-char overlap)
    │
    ├── Snap boundary to \n\n (paragraph break) if available
    ├── Fall back to \n (line break)
    └── Fall back to hard character cut
    │
    ▼
Chunk: { "text": <passage>, "metadata": { source, type, chunk_index, char_start } }
```

The AES documentation produces **15 doc chunks** across `AES_spec.txt` and
`AES_README.md`.

#### `FAISSRetriever`

| Method | Description |
|--------|-------------|
| `build(chunks)` | Encodes all chunk texts with SentenceTransformer, L2-normalises vectors, inserts into `IndexFlatIP` |
| `save(path_prefix)` | Writes `<prefix>.faiss` (binary index) + `<prefix>.pkl` (chunk metadata) |
| `load(path_prefix)` | Restores from disk — fast path on subsequent runs |
| `retrieve(query, k)` | Embeds query, L2-normalises, searches index, returns top-k `{text, metadata, score}` dicts |

**L2 normalisation** converts inner-product search into cosine similarity, which
is the correct metric for sentence embeddings whose directional component
carries the semantic signal.

#### `build_or_load_retriever`

Convenience function called by `main.py`. Checks whether `<prefix>.faiss` and
`<prefix>.pkl` both exist; loads from disk if so (skipping re-embedding), or
builds and saves them on the first run.

---

### 4.3 `tools.py` — Agent Tool Implementations

**File:** [sva_pipeline/tools.py](../sva_pipeline/tools.py)

Implements five tools that the agent can call, plus `dispatch_tool` (the
router) and `TOOL_DEFINITIONS` (the schema passed to the Qwen3 chat template).

Every tool function returns a plain `str` — the *observation* the agent reads
before deciding its next action. Errors are returned as descriptive strings
(not raised as exceptions) so the agent can read them and adapt.

#### Tool 1: `rtl_retrieve` (dispatched via `FAISSRetriever`)

Searches the RTL FAISS index. Returns up to `k` chunks formatted with source
file name, module name, and cosine similarity score so the agent knows
provenance.

#### Tool 2: `doc_retrieve` (dispatched via `FAISSRetriever`)

Searches the documentation FAISS index. Same format as `rtl_retrieve` with
source file name and score.

#### Tool 3: `yosys_extract(module_name, rtl_dir, yosys_bin)`

1. Walks `rtl_dir` for all `.v`/`.sv` files.
2. Builds a Yosys script: `read_verilog -sv "<file>"` for each source (paths
   are quoted to handle spaces in directory names), then `hierarchy -check -top
   <module>`, `proc`, `write_json <tmp>.json`.
3. Runs Yosys as a subprocess (120-second timeout).
4. Parses the JSON netlist; extracts port names, directions, and bit widths
   for the requested module.
5. Returns a formatted port table, e.g.:

```
Module: AES_Encrypt
Ports:
  input   [127:0]  in
  input   [127:0]  key
  output  [127:0]  out
```

#### Tool 4: `signal_map_lookup(signal_name, signal_map)`

Case-insensitive substring search over the pre-loaded `signal_map.json` dict.
Returns all matching signal entries with their full attribute set (module,
direction, width, type, description). Supports partial matching so the agent
can search `"key"` to find all key-related signals at once.

#### Tool 5: `verible_lint(sva_code, verible_bin)`

1. Wraps the SVA snippet in a minimal module shell (with `clk` and `rst_n`
   declared to avoid spurious undeclared-identifier errors from concurrent
   assertion references).
2. Writes to a temp `.sv` file.
3. Runs `verible-verilog-syntax <file>` (30-second timeout).
4. **Additional semantic checks** (applied even when Verible returns exit 0):
   - Rejects `assert property (...)` — concurrent assertion form, invalid for
     combinational designs.
   - Rejects `<signal>.width` — not a valid SV runtime attribute; must use
     `$bits(<signal>)` instead.
5. Returns `"PASS: …"` or `"FAIL: …"` with actionable error text.

#### `dispatch_tool`

Single routing function called by the agent loop. Maps `tool_name` strings to
the correct implementation, injects the necessary context (retrievers,
signal_map, paths), and catches any unexpected exceptions to return a safe
error string.

#### `TOOL_DEFINITIONS`

OpenAI-compatible JSON schema list passed to `tokenizer.apply_chat_template(tools=...)`.
Qwen3 uses these descriptions to decide when to invoke each tool and what
arguments to supply.

---

### 4.4 `agent.py` — ReAct Agent Loop

**File:** [sva_pipeline/agent.py](../sva_pipeline/agent.py)

#### Model Loading

```python
self.model = AutoModelForCausalLM.from_pretrained(
    config.model_id,
    dtype=torch.bfloat16,   # ~16 GB for 8B params
    device_map=config.device_map,
    trust_remote_code=True,
)
```

Qwen3-8B in bfloat16 occupies ~16 GB VRAM, leaving ~16 GB headroom on an
RTX 5090 32 GB for KV cache during long ReAct traces.

#### System Prompt (`_build_system_prompt`)

The system prompt is constructed once at `__init__` time and injected as the
first message in every conversation. It contains:

1. **Design hierarchy** — full content of `hierarchy.txt`, always available
   without a tool call.
2. **Signal map summary** — compact table (`_summarise_signal_map`) of all
   signals with module, direction, width, and description.
3. **Tool descriptions** — what each of the five tools does and when to use it.
4. **Workflow instructions** — six-step process: spec → RTL → signal
   cross-check → generate → validate → emit.
5. **Critical constraints**:
   - Immediate assertions only (`assert (...) else $error(...)` — no
     `assert property`).
   - Use `$bits(signal)` not `signal.width`.
   - Do not make up signal names.

#### `generate_assertions(task)` — ReAct Loop

```
messages = [system_prompt, user_task]

for iteration in range(max_iterations):
    response = _step(messages)           # one model forward pass
    tool_calls = _parse_tool_calls(response)

    if tool_calls:
        messages.append(assistant: response)
        for call in tool_calls:
            observation = dispatch_tool(call)
            messages.append(tool: observation)
        # loop continues
    else:
        final_answer = response          # no tool call → done
        break

else:  # max_iterations reached
    messages.append(user: "No more tools. Write SVA now. <<SVA_COMPLETE>>")
    final_answer = _step(messages)       # one forced final pass

return _extract_sva(final_answer)
```

**Forced final answer:** If the agent exhausts its iteration budget while still
making tool calls, a synthetic user turn is appended that explicitly forbids
further tool use and demands an immediate assertion code block. This ensures
the pipeline always produces output rather than timing out silently.

#### `_step(messages)`

Applies the Qwen3 chat template with `tools=TOOL_DEFINITIONS` and
`enable_thinking=False`, tokenises, runs `model.generate()`, and decodes only
the newly generated tokens (`outputs[0][prompt_len:]`) with
`skip_special_tokens=False` (to preserve `<tool_call>` markers).

#### `_parse_tool_calls(text)`

Extracts all `<tool_call>…</tool_call>` blocks using a `re.DOTALL` regex,
parses each as JSON, and normalises `name`/`arguments` keys (Qwen3 variants
may use `function`/`parameters` instead).

#### `refine_assertions(failures) -> str`

Post-generation refinement method called by `lint_loop.py`.  Starts a
**fresh** conversation (system prompt + focused refinement prompt) rather
than reusing the generation conversation history, which is 30-40 turns
long and near context limits.

The refinement prompt lists every failing assertion with its Verible error
message and asks the model to fix them.  The model runs a shorter ReAct
loop (max `config.max_refinement_react_steps` = 15 iterations) and still
has access to all five tools -- particularly `verible_lint` for
self-validation.

#### `_extract_sva(text)`

Pulls the final SVA code out of the model's response in three stages:

1. Closed code fence: ` ```systemverilog ... ``` `
2. Unclosed code fence: ` ```systemverilog ... ` (model was cut off)
3. Final `verible_lint` tool call: extracts `sva_code` argument (handles the
   case where the agent was mid-validation when the iteration limit hit)
4. Fallback: returns the full response text verbatim

---

### 4.5 `rtl_facts.py` — RTL Facts Extraction & Prompt Formatting (Stage 2-3)

**File:** [sva_pipeline/rtl_facts.py](../sva_pipeline/rtl_facts.py)

Single source of structured RTL facts extracted via pyslang. Serves two
consumers: the agent (prompt augmentation) and the post-processors
(validation against RTL ground truth).

#### Design decisions

- **No caching** — facts are re-extracted every run (~0.5s for CMAC, ~1s
  for nvdla_mul). Avoids staleness bugs.
- **Flat fields for post-processors** — existing validators use flat dicts
  (`signal_widths`, `combinational_signals`, etc.). Module scoping only
  affects the prompt formatter, not the validators.
- **Two-tier prompt loading** — core sections always emit; extended
  sections fill the remaining budget greedily in priority order.
- **Instruction sandwich** — compact reminders at the end of the system
  prompt reinforce the highest-leverage rules (drive kinds, MCD, unusual
  reset values, formatting).

#### Module scoping (Stage 3)

Two modes, swappable via `module_facts_mode` config:
- **"lazy"** (Option G): builds `signal_to_module` mapping from pyslang
  Compilation hierarchy; filters flat facts at format time. Fast, lossy
  on name collisions.
- **"full"** (Option F): splits syntax trees by `ModuleDeclaration`
  boundaries; calls existing extractors per module via `_roots()` adapter.
  Collision-proof.

Depth-limited scoping (`module_scope_depth`, default 2) controls how many
hierarchy levels are included. Based on 6-way A/B testing on nvdla_mul,
depth=2 is empirically optimal (224 assertions, +39% vs flat).

Fix 6: when module scoping removes signals from the detailed widths table,
a compact "Submodule signal timing" section lists combinational/sequential
signal names from deeper modules (~30-50 tokens) so the LLM still uses
correct `|->` vs `|=>`.

#### A/B validated results

| Design | Config | Assertions | Post-proc actions |
|--------|--------|-----------|-------------------|
| CMAC | no facts | 110 | 57 |
| CMAC | facts ON | 122 | 4 (**-93%**) |
| nvdla_mul | flat facts | 161 | 10 |
| nvdla_mul | depth=2 | 224 (**+39%**) | 13 |

---

### 4.6 `lint_loop.py` — Post-Generation Validation & Lint Feedback

**File:** [sva_pipeline/lint_loop.py](../sva_pipeline/lint_loop.py)

After the ReAct agent generates SVA, this module runs a multi-phase
validation pipeline followed by a deterministic lint-fix-relint loop that
ensures every assertion in the final output is both semantically valid
and syntactically correct.

#### Why a separate loop?

The agent has a `slang_lint` tool it *can* call during generation, but it
often runs out of its 40-iteration budget before validating everything.
Moving validation into a post-generation phase makes it **deterministic** —
every single assertion is guaranteed to be checked, regardless of how the
agent spent its iteration budget.

#### Phase 1: TRANSFORM (syntax fixes)

Seven fixers run in order before any validation:
1. `fix_bare_property_fragments()` — wrap bare `(cond) |-> (result)` as `assert property`
2. `fix_immediate_implication()` — `|->` and `->` to `!(c) || (r)` in immediate assertions
3. `fix_double_negation()` — `!(!(..) || (..)) || (..)` simplification
4. `fix_immediate_and_form()` — AND-form to implication
5. `fix_condition_only_assertions()` — extract output from error message
6. `fix_next_cycle_on_combinational()` — `|=>` to `|->` for combinational signals (uses `RTLFacts.combinational_signals`)
7. `fix_same_cycle_past_on_sequential()` — `|->` to `|=>` when the consequence uses `$past(...)` and the LHS is sequential. Prevents the timing-off-by-one where `(en) |-> q == $past(d)` asserts q-now equals d-previous (wrong for a flop that captures d when en=1). Correct form is `(en) |=> q == $past(d)`.

#### Phase 2: REMOVE WRONG (structural validation)

Six validators remove assertions that are structurally broken:
1. `remove_trivial_assertions()` — trivially true checks, width-mismatched literals
2. `remove_wrong_style_assertions()` — `assert property` on combinational designs
3. `verify_constant_signal_pairs()` — wrong constant-signal pairings (uses `RTLFacts`)
4. `validate_signal_widths()` — out-of-range bit selects, width-mismatched comparisons (uses `RTLFacts`)
5. `validate_reset_values()` — wrong reset values, cross-domain reset errors (MCD-aware, uses `RTLFacts`)
6. `check_case_selector_mismatch()` — wrong case selector expressions (optional, uses `RTLFacts`)

#### Phase 3: VALIDATE & DEDUPLICATE

Four validators ensure signal correctness and remove redundancy:
1. `validate_signals()` — drop assertions with hallucinated signal names (>50% unknown threshold). Also logs rejected names to the hallucination denylist (per design/model JSON file).
2. `deduplicate_assertions()` — string-based exact dedup
3. `semantic_deduplicate()` — canonical-form semantic dedup (normalises whitespace, error messages, operand ordering)
4. `remove_subsumed_and_contradicting()` — logical subsumption rules (groups by signal set, removes weaker assertions)

#### Phase 4: LLM Self-Review (optional)

`llm_self_review(agent, sva_code, signal_map)` sends all assertions back
to the LLM with the signal map. Skipped when `ast_only=True` or
`use_self_review=False` (default OFF).

#### Phase 5: Lint-Fix-Relint Loop

```
all_passed = []
remaining = validated_sva

for each refinement iteration (max 3):
    assertions = split_assertions(remaining)
    passed, failures = lint_all_assertions(assertions)
    all_passed.extend(passed)
    save JSON report

    if no failures → break
    remaining = agent.refine_assertions(failures)

return reassemble_assertions(all_passed)
```

#### `split_assertions(sva_code) -> List[Dict]`

Line-based splitter that walks the SVA string and groups each assertion with
its preceding `//` comment lines.  Handles multi-line assertions by tracking
parenthesis depth and string literal state.

**Truncation guard:** if a new `assert` line appears while still accumulating
an unterminated previous assertion, the splitter checks whether the previous
buffer is balanced (`;` terminator, paren depth zero). If balanced it is
committed; otherwise it is discarded and a fresh buffer starts from the new
line. Prevents a single LLM-truncated header (e.g., `max_new_tokens` cutoff)
from swallowing every subsequent assertion into one blob.

**Orphan comment sweep** (part of `remove_trivial_assertions`): any `//`
line whose next non-blank line is another comment (or EOF) is dropped.
Removes both prefixed stubs (`// Assertion 1:`) and free-form descriptive
headers the LLM emitted without bodies.

#### `run_lint_loop(agent, raw_sva, config, signal_map) -> str`

Main orchestration function. Runs all five phases in order. The `signal_map`
parameter is passed from `main.py` via `design_info.signal_map`.

#### `run_lint_loop(agent, raw_sva, config, signal_map) -> str`

Main orchestration function (see Phase 5 above for the lint-fix-relint
portion). The full pipeline runs all five phases in sequence, with each
phase filtering or fixing assertions before the next.

---

### 4.7 `main.py` — Entry Point & CLI

**File:** [main.py](../main.py)

Orchestrates the pipeline in six steps:

```
1. parse_args()              → argparse Namespace
2. load_hierarchy()          → str (hierarchy.txt content)
3. load_signal_map()         → dict (signal_map.json, _comment keys filtered)
4. build_or_load_retriever() → FAISSRetriever × 2 (RTL + doc)
5. SVAAgent().generate_assertions(task) → str (raw SVA code)
6. run_lint_loop(agent, raw_sva, config) → str (validated SVA code)
7. write_output()            → sva_output.sv + sva_pipeline_log.txt
```

`load_signal_map` filters out any JSON key starting with `"_"` (e.g.
`_comment`, `_comment2`) so the agent only sees real signal entries.

All `PipelineConfig` fields can be overridden from the CLI:

```bash
python main.py \
  --rtl-dir "./RTL Cases/AES-Verilog/SourceCode" \
  --docs-dir ./docs \
  --output sva_output.sv \
  --max-iterations 40 \
  --rebuild-index
```

---

## 5. Data Flow: End-to-End Walk-Through

```
main.py startup
    │
    ├─ Read hierarchy.txt ──────────────────────────► injected into system prompt
    ├─ Read signal_map.json (filter _* keys) ───────► injected into system prompt
    │
    ├─ Walk RTL Cases/AES-Verilog/SourceCode/
    │    └─ chunk_rtl_file() per .v file
    │         └─ regex split on module…endmodule
    │              └─ sub-split if > 8000 chars
    │                   └─ 16 RTL chunks total
    │
    ├─ Walk docs/
    │    └─ chunk_document_file() per .md/.txt
    │         └─ paragraph-aware sliding window
    │              └─ 15 doc chunks total
    │
    ├─ FAISSRetriever.build() × 2
    │    └─ SentenceTransformer.encode() → L2-norm → IndexFlatIP.add()
    │    └─ save() → indices/rtl_index.{faiss,pkl}, doc_index.{faiss,pkl}
    │
    ├─ extract_rtl_facts(rtl_dir, signal_map, top_module, module_facts_mode)
    │    └─ pyslang: signal widths, drive kinds, resets, case selectors,
    │       clock/reset pairs, signal frequencies, module hierarchy
    │    └─ → RTLFacts dataclass (consumed by agent + post-processors)
    │
    └─ SVAAgent.__init__(facts=rtl_facts)
         ├─ format_facts_for_prompt() → RTL Facts context card in system prompt
         ├─ format_facts_reminders() → instruction sandwich reminders
         ├─ _warm_prefix_cache() (Ollama/vLLM only)
         ├─ AutoModelForCausalLM.from_pretrained(Qwen3-14B, int4)
         │
         └─ generate_assertions(task)
                   │
                   ReAct loop (up to 40 iterations):
                   │
                   ├─ _step() → Qwen3 generates tool call or final answer
                   │
                   ├─ <tool_call>{"name":"doc_retrieve","arguments":{...}}</tool_call>
                   │       └─ FAISSRetriever.retrieve() → top-5 doc chunks
                   │
                   ├─ <tool_call>{"name":"rtl_retrieve","arguments":{...}}</tool_call>
                   │       └─ FAISSRetriever.retrieve() → top-5 RTL chunks
                   │
                   ├─ <tool_call>{"name":"yosys_extract","arguments":{...}}</tool_call>
                   │       └─ Yosys subprocess → JSON netlist → port table
                   │
                   ├─ <tool_call>{"name":"signal_map_lookup","arguments":{...}}</tool_call>
                   │       └─ dict substring search → signal attributes
                   │
                   ├─ <tool_call>{"name":"verible_lint","arguments":{...}}</tool_call>
                   │       └─ Verible subprocess + semantic checks → PASS/FAIL
                   │
                   └─ No tool call → _extract_sva() → raw SVA string
                        │
              ┌─────────▼───────────────────────────────────────┐
              │  run_lint_loop(agent, raw_sva, config,          │
              │                facts, signal_map, ...)          │
              │    Phase 1: TRANSFORM (6 syntax fixers)         │
              │    Phase 2: REMOVE WRONG (6 structural          │
              │             validators using RTLFacts)          │
              │    Phase 3: VALIDATE & DEDUPLICATE (signal      │
              │             validation + denylist logging +     │
              │             dedup + subsumption removal)        │
              │    Phase 4: LLM self-review (optional)          │
              │    Phase 5: lint → refine → repeat up to 3×     │
              │    → reassemble passing assertions              │
              │    → save lint_failures.json                    │
              └─────────┬───────────────────────────────────────┘
                        │
                        └─ sva_output.sv
```

---

## 6. Tool Reference

| Tool | Inputs | Returns | Typical use |
|------|--------|---------|-------------|
| `rtl_retrieve` | `query: str`, `k: int=5` | Up to k RTL chunks with source/module/score | Find how a signal or logic block is implemented |
| `doc_retrieve` | `query: str`, `k: int=5` | Up to k doc chunks with source/score | Find what a module must do per spec |
| `yosys_extract` | `module_name: str` | Port table (name, direction, width) | Confirm exact port names before writing assertions |
| `signal_map_lookup` | `signal_name: str` | Full signal attributes (partial match) | Look up width/direction for a specific signal |
| `verible_lint` | `sva_code: str` | `"PASS: …"` or `"FAIL: …"` | Validate SVA syntax and combinational-design rules |

---

## 7. Qwen3 Tool-Calling Protocol

Qwen3 is primed to emit tool invocations by passing `TOOL_DEFINITIONS` to
`tokenizer.apply_chat_template(tools=...)`. The model outputs:

```
<tool_call>
{"name": "verible_lint", "arguments": {"sva_code": "assert (out == 128'h69c4...) else $error(...);"}}
</tool_call>
```

After each tool call the agent appends two messages to the conversation:

```python
# The assistant turn that contained the call:
{"role": "assistant", "content": "<tool_call>...</tool_call>"}

# The tool observation:
{"role": "tool", "name": "verible_lint", "content": "PASS: SVA syntax is valid."}
```

The updated conversation is then fed back into `_step()` for the next
iteration. Qwen3 reads the observation and decides whether to call another
tool or emit its final answer.

`enable_thinking=False` suppresses the internal `<think>…</think>` scratchpad,
making outputs shorter and tool calls more direct.

---

## 8. ReAct Loop State Machine

```
               ┌──────────────┐
               │   START      │
               │  messages =  │
               │  [sys, user] │
               └──────┬───────┘
                      │
              ┌───────▼────────┐
         ┌───►│  _step()       │◄──────────────────┐
         │    │  model forward │                   │
         │    └───────┬────────┘                   │
         │            │                            │
         │    ┌───────▼────────┐                   │
         │    │ _parse_tool_   │                   │
         │    │ calls(text)    │                   │
         │    └───┬────────────┘                   │
         │        │                                │
         │   tool │                 no tool call   │
         │   call │                 (final answer) │
         │        │                                │
         │    ┌───▼────────────┐   ┌───────────────┴──┐
         │    │ dispatch_tool()│   │  _extract_sva()  │
         │    │ → observation  │   │  → sva_output.sv │
         │    └───┬────────────┘   └──────────────────┘
         │        │                          ▲
         │    append assistant + tool        │
         │    messages to history            │
         │        │                          │
         └────────┘         if max_iters     │
                            reached ─────────┘
                            (forced final answer)
```

---

## 9. Post-Generation Validation & Lint State Machine

```
               ┌──────────────────┐
               │  raw SVA string  │
               │  from agent      │
               └────────┬─────────┘
                        │
               ┌────────▼───────────────────┐
               │  Phase 1: SYNTAX FIXES     │
               │  fix_immediate_implication │
               │  fix_double_negation       │
               │  fix_condition_only        │
               │  deduplicate_assertions    │
               │  remove_wrong_style        │
               └────────┬───────────────────┘
                        │
               ┌────────▼──────────────────┐
               │  Phase 2: SIGNAL CHECK    │
               │  validate_signals()       │
               │  drop >50% unknown sigs   │
               └────────┬──────────────────┘
                        │
               ┌────────▼──────────────────┐
               │  Phase 3: SEMANTIC DEDUP  │
               │  semantic_deduplicate()   │
               │  canonical-form matching  │
               └────────┬──────────────────┘
                        │
               ┌────────▼──────────────────┐
               │  Phase 4: LLM SELF-REVIEW │
               │  llm_self_review()        │
               │  (skipped if ast_only)    │
               └────────┬──────────────────┘
                        │
               ┌────────▼─────────┐
          ┌───►│ split_assertions │
          │    └────────┬─────────┘
          │             │
          │    ┌────────▼──────────────┐
          │    │ lint_all_assertions   │
          │    │ (pyslang × N)         │
          │    └──┬────────────────┬───┘
          │       │                │
          │    passed           failures
          │       │                │
          │    ┌──▼───┐    ┌───────▼────────────┐
          │    │ acc- │    │ failures > 0 AND   │
          │    │ umu- │    │ iteration < max?   │
          │    │ late │    └──┬─────────────┬───┘
          │    └──┬───┘       │             │
          │       │          YES           NO
          │       │           │             │
          │       │    ┌──────▼─────────┐   │
          │       │    │ refine_        │   │
          │       │    │ assertions()   │   │
          │       │    │ (agent fix)    │   │
          │       │    └──────┬─────────┘   │
          │       │           │             │
          │       │     fixed SVA           │
          │       │           │             │
          └───────────────────┘             │
                  │                         │
           ┌──────▼─────────────────────────▼──┐
           │  reassemble_assertions(passed)    │
           │  → sva_output.sv                  │
           └───────────────────────────────────┘
```

Each iteration's results (passed count, failed count, failure details) are
persisted to `lint_failures.json` after every cycle, enabling post-mortem
inspection even if the pipeline crashes mid-loop.

---

## 10. FAISS Index Details

| Property | RTL Index | Doc Index |
|----------|-----------|-----------|
| Corpus | `.v`/`.sv` files in `rtl_dir` | `.md`/`.txt` files in `docs_dir` |
| Chunking strategy | Module-level (`module…endmodule`) | Paragraph-aware sliding window |
| Chunk size limit | 8000 chars (sub-split with 200-char overlap) | 1000 chars (200-char overlap) |
| Chunks (AES design) | 16 | 15 |
| Embedding model | `all-MiniLM-L6-v2` (384-dim) | same |
| FAISS index type | `IndexFlatIP` (exact, cosine via L2-norm) | same |
| Persistence | `indices/rtl_index.{faiss,pkl}` | `indices/doc_index.{faiss,pkl}` |
| Cache behaviour | Loaded on subsequent runs unless `--rebuild-index` | same |

Switching to `IndexIVFFlat` would be appropriate for corpora of tens of
thousands of chunks; the flat index is sufficient for typical RTL designs.

---

## 11. Known Limitations & Future Work

### Current Limitations

1. **Key schedule width assertions are contradictory** (AES-specific).
   The generated assertions check `$bits(fullkeys) == 1408`, then `== 1664`,
   then `== 1920`. Since `fullkeys` has a single width at elaboration time,
   at most one can pass. Correct approach: use `generate` blocks per instance.

2. **`fullkeys` scope** (AES-specific).
   `fullkeys` is internal to `keyExpansion`. Accessing it requires
   hierarchical references or `bind` statements.

3. **LLM non-determinism on GGUF models.**
   Ollama's Q4_K_M quantisation produces variable output across runs —
   assertion counts and post-processor action counts fluctuate by 10-20%
   even with identical prompts and temperature=0.1. HF int4 (bitsandbytes)
   is more stable but slower without prefix caching.

4. **Module scoping too aggressive on shallow hierarchies.**
   Module-scoped facts with depth < 2 can remove drive-kind annotations
   for submodule signals the LLM references. Fix 6 (submodule comb list)
   mitigates this, but hierarchical scoping with depth=2 is needed for
   best results on deep designs.

5. **Grammar constraints not tested in production.**
   Stage 2.5 GBNF grammar is built but untested on real runs. Ollama
   doesn't support it via the OpenAI API; requires vLLM or outlines.

### Future Work

- **Grammar constraints (Stage 2.5)** — validate signal-allowlist GBNF on
  vLLM backend to prevent hallucinated signals at the token level.
- **Hierarchical scoping refinement** — auto-detect optimal depth per design
  based on module count and signal density, instead of fixed depth=2.
- **Coverage-driven iteration** — run assertions in a simulator, collect
  coverage data, and feed back into the agent.
- **Bind-context-aware output** — wrap assertions in `bind` blocks for
  hierarchical signal references.
- **Per-batch dynamic fact retrieval** — FAISS-based retrieval of relevant
  facts per batch (Stage 2b infrastructure exists, needs further testing).
- **Hallucination denylist auto-tuning** — automatically enable denylist
  injection after N runs accumulate sufficient data.
- **Fine-tuning / LoRA** on (RTL facts, assertion) pairs from successful
  runs, eliminating prompt overhead entirely.

---

## 12. Running the Pipeline

### Prerequisites

```bash
# Python venv with dependencies
python -m venv .venv && source .venv/bin/activate
pip install torch transformers sentence-transformers faiss-cpu accelerate

# Yosys (for structural extraction)
sudo apt install yosys          # or build from source

# Verible (for SVA syntax checking)
# Install to /home/user/Verible/bin/ and update config.verible_bin if needed
```

### Basic run (AES design, all defaults)

```bash
python main.py
```

### Force rebuild of FAISS indices

```bash
python main.py --rebuild-index
```

### Custom RTL directory

```bash
python main.py \
  --rtl-dir "./RTL Cases/NVDLA_hw/vmod" \
  --docs-dir "./docs/nvdla" \
  --output sva_nvdla.sv \
  --max-iterations 40
```

### Custom verification task

```bash
python main.py --task "Generate SVA for the keyExpansion module only"
```

### Outputs

| File | Description |
|------|-------------|
| `sva_output.sv` | Generated SVA assertions (overwritten each run) |
| `sva_pipeline_log.txt` | Append-only log of every run's output |
| `lint_failures.json` | JSON report of lint failures and refinement history |
| `indices/rtl_index.{faiss,pkl}` | Cached RTL FAISS index |
| `indices/doc_index.{faiss,pkl}` | Cached documentation FAISS index |

---

## 13. Debugging & Troubleshooting

### CUDA out-of-memory

Kill any lingering Python processes from previous runs before starting a new
one. Qwen3-8B occupies ~16 GB; a zombie process from a failed run can consume
the same amount, leaving insufficient VRAM.

```bash
nvidia-smi   # check for stale processes
kill <PID>   # if found
```

### FAISS index stale after adding new RTL files

```bash
python main.py --rebuild-index
```

### Verible not found

Check that `verible_bin` in `config.py` (or `--verible-bin` on the CLI) points
to the correct absolute path:

```bash
/home/user/Verible/bin/verible-verilog-syntax --version
```

### Yosys fails on module extraction

- Ensure Yosys is installed and on `PATH`.
- Check that the module name matches exactly (case-sensitive).
- Check that the RTL directory path is correct and contains `.v` files.

### Agent keeps making tool calls until iteration limit

Increase `--max-iterations` or inspect the log to see which tool calls are
repeating. Common cause: the agent is calling `signal_map_lookup` for every
signal individually rather than using the summary already in the system prompt.
The system prompt already advises against this; if it persists, the model may
need a more explicit constraint.

### Generated assertions use `assert property`

The `verible_lint` tool returns `FAIL` for `assert property (...)` and provides
a corrective message. If this still appears in the output it means the forced
final-answer step bypassed linting. Check that `_extract_sva` is not
accidentally picking up a `verible_lint` tool call argument instead of the
final code block.
