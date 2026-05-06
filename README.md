# SpecGuard: Token-Efficient SystemVerilog Assertion Generation

**SpecGuard** is a four-stage process for generating, validating, and (optionally)
mutation-testing SystemVerilog Assertions (SVA) for hardware designs. Its central
objective is **token-efficient SVA generation**: every cycle of LLM inference is
spent on what only an LLM can solve — the semantic claim each assertion makes
about the design — and not on work a deterministic tool can do for free. The
means is *design-knowledge weaving*: at each step we choose what to extract
ahead of time, in what compact form to deliver it, and through which channel
(system prompt, retrieval, or tool call) to make it available, so the model
receives the strongest grounding at the lowest possible token cost.

```
python main.py project.yaml
```

## How It Works

```
       ┌──────────────┐
       │ project.yaml │   <- 2 lines minimum
       └──────┬───────┘
              ▼
┌─────────────────────────────┐    ┌─────────────────────────────┐
│  Stage 1: Design Analysis    │    │  Stage 2: Spec-Validation   │
│  (deterministic, no LLM)     │    │            Agent             │
│  • pyslang parser            │    │  • LLM Agent (Qwen3-14B,    │
│  • AST extractor + cluster   │───▶│    Q4_K_M GGUF, Ollama)     │
│    & compact (multi-clause   │    │  • Hybrid RAG (FAISS+BM25), │
│    properties)               │    │    code + NLP embedders     │
│  • RTL facts card            │    │  • Batched: 40 skel/call    │
│  • wrapper-detect fallback   │    │  • slang_lint tool calls    │
└─────────────────────────────┘    └──────────────┬──────────────┘
                                                   ▼
┌─────────────────────────────┐    ┌─────────────────────────────┐
│  Stage 4: Security & Output │    │  Stage 3: Post-Processing & │
│  • Security pass (CWE-      │    │            Refinement       │
│    tagged scenarios, opt.)  │◀───│  • Phase 1 Deterministic    │
│  • Mutation testing         │    │    Repair                   │
│    (Verilator / xsim, opt.) │    │  • Phase 2 Structural Check │
│  • Verified SVA output      │    │  • Phase 3 Semantic Filter  │
│                              │    │  • Lint feedback (≤3 cyc.)  │
└─────────────────────────────┘    └─────────────────────────────┘

OUTPUTS:  sva_output.sv  ·  lint_failures.json  ·  trace.{json,csv}
          token_summary.json  ·  mutation_report.json (optional)
```

The architecture mirrors the four-stage figure in the paper. See
[`docs/SpecGuard_Architecture.md`](docs/SpecGuard_Architecture.md) for the
detailed component-level walk-through.

## Quick Start

**1. Install:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pyslang bitsandbytes
```

**2. Set up the LLM backend (choose one):**

```bash
# Option A: Ollama (recommended — easy setup, prefix caching)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:14b

# Option B: Local HuggingFace (no server needed, uses GPU directly)
# Just configure model.backend: "local" in your YAML (see below)
```

**3. Create a project config:**
```yaml
# project.yaml
design:
  rtl_dir: "./path/to/verilog"
  top_module: "MyModule"

# For Ollama:
model:
  backend: "openai"
  id: "qwen3:14b"
  api_base: "http://localhost:11434/v1"
  api_key: "ollama"

# Or for local HuggingFace:
# model:
#   backend: "local"
#   id: "Qwen/Qwen3-14B"
#   quantization: "int4"
```

**4. Run:**
```bash
python main.py project.yaml
```

## Three Assertion Generation Modes

| Mode | Config | Speed | Quality | Use Case |
|------|--------|-------|---------|----------|
| **AST-only** | `ast_only: true` | <1 sec | Structural (case branches, assigns, resets) | Regression testing, CI/CD |
| **AST + LLM** | `ast_only: false` | ~2 min | Structural + spec-derived invariants | Design verification |
| **LLM-only** | `use_ast_assertions: false` | ~5 min | Varies by model | Exploration, protocol assertions |

### AST-Only (No LLM Needed)

Extracts RTL patterns via regex and generates assertions deterministically:
- Case statement branches → one assertion per case value
- Continuous assignments → `assert (out == expr)`
- Sequential resets → async (`|->`) or sync (`|=>`) based on sensitivity list
- Decoded flag groups → mutual exclusivity checks (`sum <= 1`)

```yaml
agent:
  use_ast_assertions: true
  ast_only: true
```

### AST + LLM Spec Validation (Recommended)

AST extracts structure, then the LLM validates each skeleton against
the design documentation:
- Trivial assertions (case branches, wires) → output directly
- Complex assertions (sequential, ternary, multi-signal) → batched to LLM (40 per batch)
- LLM adds: protocol invariants, mode-dependent constraints, edge cases

```yaml
agent:
  use_ast_assertions: true
  ast_only: false
```

## Supported Models

| Backend | Config | VRAM | Best For |
|---------|--------|------|----------|
| **Qwen3-14B via Ollama** (recommended) | `backend: "openai"`, `api_base: "http://localhost:11434/v1"` | ~10 GB (Q4_K_M) | Prefix caching, easy model swapping |
| Qwen3-14B int4 (local HF) | `backend: "local"`, `quantization: "int4"` | ~7 GB | No server needed, stable output |
| Qwen3-8B (local HF) | `backend: "local"`, `id: "Qwen/Qwen3-8B"` | ~16 GB (bf16) or ~4 GB (int4) | Lighter GPU requirement |
| vLLM / SGLang server | `backend: "openai"`, `api_base: "http://localhost:8000/v1"` | Varies | Grammar constraints (Stage 2.5), high throughput |
| GPT-4o (API) | `backend: "openai"`, `api_key: "sk-..."` | 0 | Large designs, no local GPU |
| Claude Sonnet (API) | `backend: "anthropic"`, `api_key: "sk-ant-..."` | 0 | 200K context |

## Project Structure

```
SoC-LLM/
├── main.py                     Entry point
├── run_pipeline.sh             Runner with memory monitoring
├── json2csv.py                 Report converter
│
├── sva_pipeline/               Core pipeline
│   ├── config.py               Configuration dataclass
│   ├── config_loader.py        YAML loading + auto-detection
│   ├── slang_frontend.py       RTL analysis + SVA linting (pyslang)
│   ├── ast_assertions.py       AST pattern extraction + skeleton gen
│   ├── agent.py                LLM agent (AST+LLM, plan-execute, ReAct)
│   ├── rtl_facts.py            RTL facts extraction + prompt formatter (Stage 2-3)
│   ├── grammar.py              GBNF grammar generator (Stage 2.5)
│   ├── rag.py                  FAISS + BM25 + hierarchical retrieval
│   ├── tools.py                5 agent tools
│   ├── lint_loop.py            Post-processing + lint feedback
│   ├── trace_logger.py         Step-by-step trace recording
│   ├── html2md.py              HTML-to-Markdown converter
│   ├── design_graph.py         Yosys fallback
│   │
│   ├── backends/               LLM backends
│   │   ├── local.py            HuggingFace + quantization + grammar (outlines)
│   │   ├── openai_backend.py   OpenAI / vLLM / Ollama + grammar (GBNF)
│   │   └── anthropic_backend.py  Claude
│   │
│   └── mutation/               Mutation testing
│       ├── operators.py        7 mutation operators
│       ├── mutant_generator.py RTL mutant creation
│       ├── testbench_gen.py    Auto testbench generation
│       ├── sim_harness.py      Verilator / xsim simulation
│       └── report.py           JSON/CSV reporting
│
├── ab_compare.py               A/B test comparison tool
├── examples/                   Example YAML configs
└── docs/                       Documentation
    ├── user_manual.md          How to use the tool
    ├── technical_docs.md       Module-by-module reference
    ├── pipeline_architecture.md  Design decisions
    └── rtl_facts_findings.md   Stage 2-3 A/B test results
```

## Key Issues Encountered and Solutions

### Memory Management

| Issue | Cause | Solution |
|-------|-------|----------|
| 51 GB RSS → OOM kill | `from_pretrained` with `device_map="auto"` stages full model in CPU RAM | Changed to `device_map="cuda:0"` with `low_cpu_mem_usage=True` |
| OOM even with cuda:0 | Top-level imports of torch/faiss/transformers loaded at startup | Lazy imports — heavy libraries only loaded when RAG or LLM is actually needed |
| Embedding model + LLM compete for GPU | Two SentenceTransformer instances loaded | Singleton encoder cache (`_ENCODER_CACHE`) shares one instance |
| VSCode extension kills process | Exit code 144 — VSCode process manager timeout | `run_pipeline.sh` with `nohup` runs outside VSCode |

### Assertion Quality

| Issue | Cause | Solution |
|-------|-------|----------|
| Only width checks generated | 8B model can't compose conditional assertions | AST extraction generates structural assertions deterministically |
| `assert(condition)` without output check | Model writes expected output in error message only | Post-processor extracts output from error message and restructures |
| `\|->` in immediate assertions | Invalid in `assert()`, only valid in `assert property` | Post-processor converts to `!(cond) \|\| (result)` |
| `->` in immediate assertions | Same issue, different operator | Post-processor handles both `\|->` and `->` |
| Double negation `!(!(..) \|\| (..)) \|\| (..)` | LLM wraps AST skeleton in extra negation | Post-processor simplifies to `!(..) \|\| (..)` |
| Wrong mutual exclusivity logic | LLM uses convoluted negation instead of one-hot check | AST auto-detects decode groups and generates `sum <= 1` |
| Unconditional output value assertions | LLM over-generalises case-specific checks | Prompt instructs LLM to skip unconditional output checks |
| Assertions from RTL not spec (circular) | AST-only verifies "RTL does what RTL does" | LLM validates AST skeletons against documentation (ground truth) |
| LLM hallucinated signal names | LLM invents signals not in the design | Signal existence validation drops assertions where >50% of signals are unknown |
| LLM duplicates AST assertions | LLM regenerates what AST already produced, with different formatting | Semantic deduplication normalises assertion bodies to canonical form before comparison |
| LLM generates logically incorrect assertions | Wrong widths, wrong logic, trivially true checks | LLM self-review pass re-checks all assertions against signal map and removes/fixes broken ones |
| LLM uses `\|=>` on combinational signals | Doesn't know which signals are combinational vs sequential | RTL facts extract drive-kind (comb/seq) from pyslang and inject into prompt (Stage 2). Reduced fix_next_cycle from 33 to 0 on CMAC. |
| LLM misuses constants/widths | Pairs wrong literal with wrong signal, or uses wrong bit widths | RTL facts inject constant ownership (negative phrasing) and signal widths into prompt |
| Noisy prompt on large designs | 311 signals dumped flat confuses the LLM | Module-scoped facts (Stage 3) narrow to top module + 2 levels. Increased assertions by 39% on nvdla_mul. |

### Assertion Syntax

| Issue | Cause | Solution |
|-------|-------|----------|
| `assert property` on combinational design | No clock → concurrent assertions invalid | Auto-detect clock presence, reject `assert property` when no clock |
| Async reset uses `\|=>` (next cycle) | Should be `\|->` (same cycle) for async reset | AST detects `negedge rst` in sensitivity list → uses `\|->` |
| Sync reset uses `\|->` (same cycle) | Should be `\|=>` (next cycle) for sync reset | AST distinguishes sync vs async from sensitivity list |
| Active-high reset with wrong polarity | `disable iff (!rst)` when reset is active-high | AST detects polarity from `if(!rst)` vs `if(rst)` patterns |

### Simulation and Mutation Testing

| Issue | Cause | Solution |
|-------|-------|----------|
| All mutants stillborn (Verilator) | Immediate assertions at module scope | SVA injection separates concurrent (module scope) and immediate (`always_comb`) |
| Verilator internal fault | Double `\|=>` chains not supported in Verilator 5.020 | Added xsim backend (`simulator: "xsim"`) with full SVA support |
| Verilator can't find internal signals | Assertions reference DUT internal wires | SVA injected directly into DUT source before `endmodule` |
| `assert property` in immediate wrapper | `always_comb` can't contain concurrent assertions | SVA injection auto-classifies and places accordingly |

### Token Truncation

| Issue | Cause | Solution |
|-------|-------|----------|
| LLM output truncated at 4096 tokens | 95 skeletons too many for single LLM call | Batch+filter: trivial assertions direct, complex in batches of 20 |
| Warning "text generation exceeded" | HuggingFace reminder about max_new_tokens | Non-critical — pipeline falls back to AST skeletons |

## Outputs

| File | Description |
|------|-------------|
| `*_sva.sv` | Generated SVA assertions |
| `*_lint.json` | Lint iteration history |
| `*_mutation_report.json` | Mutation test results |
| `*_trace.json` | Full pipeline trace (every LLM call) |
| `*_trace.csv` | Human-readable trace |

Convert reports to CSV: `python json2csv.py --all`

## Documentation

- **[User Manual](docs/user_manual.md)** — How to use the tool, config reference, troubleshooting
- **[Technical Documentation](docs/technical_docs.md)** — Module-by-module reference with every function
- **[SpecGuard Architecture](docs/SpecGuard_Architecture.md)** — Four-stage workflow, design-knowledge weaving, component deep-dive
- **[RTL Facts Findings](docs/rtl_facts_findings.md)** — Stage 1 facts-card A/B results
- **[VRAM Budget](docs/vram_budget.md)** — How we fit a 14B model + embedder on a limited-VRAM GPU (Ollama out-of-process backend, activation-budget tuning)
