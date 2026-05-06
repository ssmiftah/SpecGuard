# SpecGuard User Manual

How to use **SpecGuard** to generate SystemVerilog assertions from RTL and
design specifications, without needing to understand the four-stage internal
architecture (covered in `docs/SpecGuard_Architecture.md`).

> SpecGuard's central objective is *token-efficient SVA generation*: the system
> chooses what to extract from the design ahead of time, in what compact form,
> and through which channel (system prompt, retrieval, or tool call), so the
> LLM gets the strongest grounding at the lowest possible token cost. As an
> end user you only see one CLI: `python main.py project.yaml`.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Your First Run](#2-your-first-run)
3. [The Config File](#3-the-config-file)
4. [Assertion Generation Modes](#4-assertion-generation-modes)
5. [Choosing a Model](#5-choosing-a-model)
6. [Understanding the Outputs](#6-understanding-the-outputs)
7. [Writing Good Specifications](#7-writing-good-specifications)
8. [Mutation Testing](#8-mutation-testing)
9. [HTML Documentation Conversion](#9-html-documentation-conversion)
10. [Security Assertions](#10-security-assertions)
11. [Trace Files and Debugging](#11-trace-files-and-debugging)
12. [Converting Reports to CSV](#12-converting-reports-to-csv)
13. [Troubleshooting](#13-troubleshooting)
14. [Full Config Reference](#14-full-config-reference)

---

## 1. Installation

### Prerequisites

- Python 3.10 or later
- NVIDIA GPU with 8+ GB VRAM (for local models; not needed for API backends)
- Verilator 5.x or Vivado xsim (optional, for mutation testing)

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pyslang bitsandbytes
```

### Optional: API backends

```bash
pip install openai      # for GPT-4, vLLM, ollama
pip install anthropic   # for Claude
```

### Verify

```bash
python -c "from sva_pipeline.config import PipelineConfig; print('OK')"
```

---

## 2. Your First Run

### Step 1: Create a config file

```yaml
# my_project.yaml
design:
  rtl_dir: "./path/to/verilog"
  top_module: "MyModule"
```

That's it — two fields. Everything else is auto-detected.

### Step 2: Run

```bash
./run_pipeline.sh my_project.yaml
```

Use `run_pipeline.sh` instead of `python main.py` — it monitors memory
and captures the full log. This is important because the LLM uses
significant GPU memory and can cause system instability if run inside
an IDE terminal.

### Step 3: Check output

Generated assertions are in `./MyModule_sva.sv`.

---

## 3. The Config File

### What the pipeline auto-detects (you don't configure these)

| Feature | How it's detected |
|---------|-------------------|
| Clock signal | `always @(posedge ...)` in RTL |
| Reset signal | `negedge rst` patterns + `!rst_n` conditions |
| Reset type | Async (in sensitivity list) vs sync |
| Reset polarity | Active-low (`!rst`) vs active-high (`rst`) |
| Assertion style | Clock → concurrent; no clock → immediate |
| Context strategy | RTL < 50K chars → inject; else → RAG |
| Index staleness | Source file checksums |
| Output paths | From top module name |
| Task description | From signal patterns and design type |

### Adding documentation (recommended)

```yaml
design:
  rtl_dir: "./rtl"
  top_module: "my_module"
  docs_dir: "./docs"
```

Place `.md` or `.txt` spec files in `docs_dir`. The pipeline uses them
to validate assertions against the design specification.

### Specifying the top file

For multi-file designs, tell the pipeline which file contains the top module:

```yaml
design:
  rtl_dir: "./rtl"
  top_file: "my_top.v"
  top_module: "my_top"
```

---

## 4. Assertion Generation Modes

### AST-Only (fastest, no LLM needed)

Extracts RTL patterns via regex and generates assertions deterministically.
No GPU needed, completes in under 1 second.

```yaml
agent:
  use_ast_assertions: true
  ast_only: true
```

**Produces:** case-branch assertions, assignment checks, reset behaviour,
mutual exclusivity invariants.

**Does NOT produce:** protocol assertions, spec-derived invariants,
semantic descriptions.

**Best for:** regression testing, CI/CD pipelines, quick structural checks.

### AST + LLM Spec Validation (recommended)

AST extracts structure, LLM validates against documentation:

```yaml
agent:
  use_ast_assertions: true
  ast_only: false
```

**How it works:**
- Trivial assertions (case branches, wires) → output directly, no LLM
- Complex assertions (sequential, ternary, multi-signal compacted) → sent to LLM in batches of 40
- LLM compares each skeleton against the spec documentation
- LLM adds protocol invariants and spec-derived properties

**Best for:** design verification with specification documents.

### LLM-Only (legacy)

Disable AST extraction and use the LLM for everything:

```yaml
agent:
  use_ast_assertions: false
  use_plan_execute: true
```

**Best for:** exploring unusual designs, protocol-heavy modules.

### RTL Facts Prompt Augmentation (Stage 2)

When enabled, the pipeline injects a structured "RTL Facts" block into
the LLM system prompt. The block is extracted from the RTL via pyslang
(authoritative, not inferred) and contains:

- Clock and reset domains (multi-clock-domain aware)
- Reset values (with non-zero values flagged as unusual)
- Signal widths annotated with drive kind (`(c)`=combinational use `|->`,
  `(s)`=sequential use `|=>`, `(m)`=mixed driver)
- Case-driven signal selectors (prevents the LLM from using the wrong
  selector expression in assertions)
- Constant ownership (negative phrasing: which signal each multi-bit
  literal belongs to)
- Generic SVA mistakes to avoid (width mismatch, wrong implication,
  conjunction-vs-implication, out-of-range bit selects, missing wrappers)

```yaml
agent:
  use_rtl_facts: true              # default: false
  rtl_facts_soft_budget: 1500      # token target — try to stay under
  rtl_facts_hard_budget: 2400      # absolute ceiling, never exceed
```

**Two-tier loading:** core sections (clock/reset pairs, unusual reset
values, constraints, bad patterns) are always emitted. Extended sections
(full reset table, full signal widths, case selectors, constant pairs)
fill the remaining budget greedily in priority order. The formatter
trims gracefully on small designs and small budgets.

**A/B testing:** run your project once with `use_rtl_facts: false` and
once with `use_rtl_facts: true`, then compare:

- LLM-generated assertion count before vs after post-processing
- Number of post-processor removals (`*_lint.json`)
- Final correctness rate (manual review or mutation testing)

If facts help, you should see fewer post-processor removals (the LLM
makes fewer mistakes upstream) and similar or better final correctness.

**Hallucination knowledgebase (optional):** when the pipeline runs,
`validate_signals` records every dropped hallucinated signal name to a
per-(design, model) JSON file under `indices/hallucinations/`. Logging
is always-on. To inject the top-N most frequent hallucinations as a
"do NOT use these names" reminder in the system prompt, enable:

```yaml
agent:
  use_hallucination_denylist: true       # default: false
  hallucination_denylist_top_n: 5
  hallucination_denylist_dir: "indices/hallucinations"
```

The denylist is filtered on load: any name that has since become a real
signal (e.g., the user added it to the RTL) is automatically dropped, so
stale entries don't poison future runs. Use this once you've accumulated
data from a few runs of the same design+model combination.

**Best for:** local models (Qwen3-8B, Llama-3-8B) where prompt quality
has high leverage. The block is roughly 1300-2300 tokens for typical
designs, so make sure your model has at least 8K context window (with
Ollama, override `num_ctx` as shown in section 5).

### Module-Scoped Facts (Stage 3)

For designs with many modules, the flat facts block can be noisy (e.g.,
nvdla_mul has 311 signals across 19 modules). Module scoping narrows the
facts to the top module and its nearby submodules, giving the LLM a more
focused context.

```yaml
agent:
  module_facts_mode: "lazy"    # "off" (default), "lazy", or "full"
  module_scope_depth: 2        # 0=top only, 1=+submodules, 2=+2 levels, -1=all
```

**Modes:**
- `"off"` — flat facts (all signals merged, no module boundaries). Default.
- `"lazy"` — builds a signal-to-module mapping from the Compilation
  hierarchy, filters flat facts at format time. Fast, zero extractor
  changes, but lossy on signal name collisions across modules.
- `"full"` — extracts facts per-module via ModuleDeclaration syntax tree
  boundaries. Collision-proof but slightly slower.

**Depth parameter:** controls how many levels of submodule hierarchy are
included in the scoped facts view. Based on A/B testing on nvdla_mul
(19 modules, 311 signals):

| Depth | Signals | Assertions | Post-proc actions |
|-------|---------|-----------|-------------------|
| flat  | 311     | 161       | 10                |
| 0     | 36      | 199       | 16                |
| 1     | 50      | 218       | 19                |
| **2** | **238** | **224**   | **13**            |
| 3     | 260     | 188       | 20                |
| 4     | 280     | 211       | 23                |

Depth=2 produced the most assertions (224, +39% vs flat) with the
second-lowest post-processor actions (13). Too narrow (depth=0) loses
submodule context; too broad (depth=3-4) adds noise and increases
width-mismatch errors.

When module scoping is active, a "Submodule signal timing" section is
automatically appended listing combinational/sequential signals from
deeper submodules so the LLM still uses the correct implication operator
(`|->` vs `|=>`).

**When to enable:** designs with 10+ modules and 200+ signals benefit
most. Small designs (CMAC, 5 modules, 124 signals) see no improvement
from scoping — the flat view is already clean.

---

## 5. Choosing a Model

### Local models

```yaml
model:
  backend: "local"
  id: "Qwen/Qwen3-8B"
  quantization: "none"      # "none", "int8", "int4"
```

| Model | Quantization | VRAM | Quality |
|-------|-------------|------|---------|
| Qwen3-8B | none | 16 GB | Protocol checks, mode flags |
| Qwen3-14B | int4 (NF4) | 7 GB | Functional case-branch assertions |
| Qwen3-14B | int8 | 14 GB | Sometimes worse than int4 for instruction-following |
| Qwen3-14B | none | 28 GB | Best local (needs 32+ GB VRAM) |

**Note:** int4 NF4 quantization often produces better results than int8
for instruction-following tasks because NF4 uses a non-uniform quantization
grid optimised for weight distributions.

### API models

**OpenAI:**
```yaml
model:
  backend: "openai"
  id: "gpt-4o"
  api_key: "sk-..."         # or set OPENAI_API_KEY env var
```

**Anthropic Claude:**
```yaml
model:
  backend: "anthropic"
  id: "claude-sonnet-4-20250514"
  api_key: "sk-ant-..."     # or set ANTHROPIC_API_KEY env var
```

**Local server (Ollama, vLLM, llama.cpp, SGLang):**

Any server that exposes an OpenAI-compatible chat completions endpoint
works with `backend: "openai"`. Set `api_base` to the server's URL.

#### Ollama (recommended for ease of use)

Ollama gives you automatic prefix caching across requests, which makes
the second batch onwards in a pipeline run noticeably faster — the static
system prompt (with hierarchy, signal map, and any RTL facts block) is
re-used from cache instead of being re-tokenised.

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull the model
ollama pull qwen3:8b

# 3. Server starts automatically on port 11434
```

```yaml
model:
  backend: "openai"
  id: "qwen3:8b"                          # Ollama tag, NOT HF id
  api_base: "http://localhost:11434/v1"
  api_key: "ollama"                       # ignored by Ollama, required by SDK
```

⚠ **Context window gotcha:** Ollama's default `num_ctx` is 2048 tokens,
which is too small for the SVA pipeline (the system prompt alone is often
3000+ tokens). Override it with a Modelfile:

```
# qwen3-32k.Modelfile
FROM qwen3:8b
PARAMETER num_ctx 32768
```

```bash
ollama create qwen3:8b-32k -f qwen3-32k.Modelfile
# then use id: "qwen3:8b-32k" in your YAML
```

#### vLLM (best raw throughput)

```bash
pip install vllm
vllm serve Qwen/Qwen3-8B --enable-prefix-caching --port 8000
```

```yaml
model:
  backend: "openai"
  id: "Qwen/Qwen3-8B"
  api_base: "http://localhost:8000/v1"
  api_key: "dummy"
```

### Generation parameters

```yaml
model:
  temperature: 0.1         # lower = more deterministic (recommended)
  max_new_tokens: 4096     # increase for complex designs
```

---

## 6. Understanding the Outputs

### `*_sva.sv` — Generated Assertions

Three types of assertions:

```sv
// Immediate (combinational design):
assert (!(code == 3'b011) || (out_data == {~src_data[15], src_data[14:0], 1'b0}))
  else $error("Booth +2x case failed");

// Concurrent (clocked design, async reset):
assert property (@(posedge clk) !rst_n |-> reg == 1'b0)
  else $error("async reset value mismatch");

// Concurrent (clocked design, functional):
assert property (@(posedge clk) disable iff (!rst_n)
  (cond) |=> reg == $past(expr))
  else $error("functional update mismatch");
```

### Post-Generation Validation

Before linting, assertions go through several validation phases:

| Phase | What Happens |
|-------|-------------|
| Syntax fixes | Fix `\|->` in immediate assertions, double negation, condition-only assertions |
| Signal validation | Drop assertions referencing signals not in the design (>50% unknown = dropped) |
| Semantic dedup | Remove logically equivalent assertions (catches formatting-only differences) |
| LLM self-review | LLM reviews its own output for signal/width/logic errors |
| Lint loop | pyslang syntax validation with LLM repair, up to 3 iterations |

### `*_lint.json` — Lint Report

```json
{
  "iterations": [
    {"iteration": 1, "passed": 95, "failed": 2},
    {"iteration": 2, "passed": 2, "failed": 0}
  ],
  "final_status": "ALL_PASSED"
}
```

### `*_trace.json` / `*_trace.csv` — Pipeline Trace

Records every step. Open the CSV in a spreadsheet:

| Step | Phase | Tools Used | Output Preview |
|------|-------|------------|----------------|
| 1 | ast_extraction | - | Extracted 95 skeletons |
| 2 | spec_validation | slang_lint | assert !(code==...) ... |
| 3 | lint | - | passed=93, failed=2 |
| 4 | refinement | slang_lint | assert !(code==...) ... |

---

## 7. Writing Good Specifications

The quality of spec-validated assertions depends on your documentation.

### What works well

Describe **behaviour and invariants**, not RTL signal names:

```
The booth recoder has five operations: zero (0x), positive multiply (+1x),
negative multiply (-1x), double (+2x), and negative double (-2x).

For positive operations, the inversion flag must be clear.
For negative operations, it must be set.

In 8-bit mode, only the lower 8 bits of the multiplicand are used.
The upper bits of the output must always be zero.
```

### What doesn't work

Don't write assertion code in the spec:
```
// BAD — this is just telling the LLM what to type:
assert (code == 3'b011 |-> out_data == {~src_data[15], src_data[14:0], 1'b0})
```

Don't be too vague:
```
// BAD — the LLM generates width checks only:
Verify the booth recoder works correctly.
```

### The middle ground

Describe the structure of what to verify with enough detail to be
actionable:

```
For EACH booth code value (000 through 111), verify that the output
matches the expected operation (0x, +1x, -1x, +2x, -2x).

The zero pattern is different between 8-bit and 16-bit modes.

When the sign flag is zero, the internal code must equal the input code.
```

### Few-shot examples

If the LLM struggles with assertion syntax, add examples from a
**different** design (not the one being verified):

```
Example from a 2-bit ALU:
  assert (!(op == 2'b00) || (result == A + B)) else $error("add failed");
  assert (!(op == 2'b01) || (result == A - B)) else $error("sub failed");
```

---

## 8. Mutation Testing

### Enable

```yaml
mutation_testing:
  enabled: true
  simulator: "xsim"       # or "verilator"
  sim_cycles: 200
```

### Choosing a simulator

| Simulator | Speed | SVA Support | Config |
|-----------|-------|-------------|--------|
| Verilator | ~1s/mutant | Limited (no double `\|=>`) | `simulator: "verilator"` |
| xsim (Vivado) | ~3s/mutant | Full IEEE 1800-2017 | `simulator: "xsim"` |

Use **xsim** when assertions contain complex temporal operators.
Use **Verilator** for faster runs with simple assertions.

### Specifying DUT and support files

```yaml
mutation_testing:
  dut_files:
    - "my_top.v"          # only this file gets mutated
  support_files:
    - "submodule1.v"      # compiled but not mutated
```

### Reading the report

```bash
python json2csv.py mutation_report.json
```

Key metric: **mutation score = killed / (total - stillborn)**

| Score | Meaning |
|-------|---------|
| 100% | Every injected bug caught |
| 70-99% | Good coverage, check surviving mutants |
| < 70% | Significant gaps — review survivors |

---

## 9. HTML Documentation Conversion

```yaml
html_docs:
  enabled: true
  files:
    - "./specs/architecture.html"
  output_dir: "./docs"
```

Converts HTML to Markdown, extracts images (timing diagrams, block
diagrams). Cached by file modification time.

---

## 10. Security Assertions

```yaml
security:
  threat_model: "./threats/threat_model.md"
```

The pipeline indexes the threat model and adds a SECURITY category
to the verification task.

---

## 11. Trace Files and Debugging

Every run produces trace files:
- `*_trace.json` — full detail
- `*_trace.csv` — spreadsheet-friendly

### Phases in the trace

| Phase | What happens |
|-------|-------------|
| `ast_extraction` | Regex pattern extraction from RTL |
| `spec_validation` | LLM validates skeletons against docs |
| `planning` | LLM creates JSON assertion plan |
| `execution` | Per-assertion focused generation |
| `direct` | Direct ReAct loop |
| `refinement` | LLM fixes lint failures |
| `self_review` | LLM reviews its own assertions for correctness |
| `lint` | Per-assertion linting results |
| `mutation_testing` | Mutation score |

---

## 12. Converting Reports to CSV

```bash
python json2csv.py --all              # convert all reports
python json2csv.py report.json        # convert one file
python json2csv.py report.json -o out # custom prefix
```

---

## 13. Troubleshooting

### System hangs / OOM kill (exit code 137)

Your system ran out of memory. Solutions:
- Use `ast_only: true` (no LLM loaded, ~500 MB)
- Use `quantization: "int4"` (7 GB for 14B model)
- Use API backend (0 VRAM)
- Run via `./run_pipeline.sh` (not from IDE terminal)
- Close Firefox and other heavy applications

### Process killed by VSCode (exit code 144)

The VSCode extension kills long-running processes. Always run from
a **separate terminal**, not the VSCode integrated terminal:

```bash
./run_pipeline.sh project.yaml
```

### Only width checks generated

The model isn't producing functional assertions. Try:
1. Enable AST extraction: `use_ast_assertions: true`
2. Use 14B model with int4: `quantization: "int4"`
3. Write better spec docs (see Section 7)
4. Use API backend (Claude, GPT-4)

### All mutants stillborn

Assertions have syntax issues that the simulator can't compile.
- Switch to `simulator: "xsim"` (broader SVA support)
- Check `lint_failures.json` for syntax errors

### Token truncation warning

"This is a friendly reminder - the current text generation call
has exceeded the max_new_tokens..."

Non-critical. The pipeline falls back to AST skeletons. To avoid:
- Increase `max_new_tokens: 8192`
- The batch+filter approach handles this automatically

### Too many assertions removed by signal validation

Signal validation uses a soft threshold (>50% unknown signals = drop).
If legitimate assertions are being removed:
- Check that your design's ports are being detected correctly by
  looking at the signal map in the trace/log output
- Internal signals not in the port-level signal map won't be
  recognized — the 50% threshold is designed to handle this, but
  deeply internal designs may need adjustment

### LLM self-review removes good assertions

The self-review pass is conservative and may occasionally remove
valid assertions. Check the log for "Self-review: X → Y assertions"
to see how many were affected. If too aggressive:
- Use `ast_only: true` to skip the LLM entirely
- Or use a stronger API model (Claude, GPT-4) for better review quality

### "No RTL source available" in AST mode

The AST extractor can't find the RTL files. Ensure:
- `rtl_dir` exists and contains `.v`/`.sv` files
- `top_file` matches an actual filename in `rtl_dir`

---

## 14. Full Config Reference

```yaml
# ═══════════════════════════════════════════
# REQUIRED
# ═══════════════════════════════════════════
design:
  rtl_dir: "./path/to/verilog"
  top_module: "ModuleName"

# ═══════════════════════════════════════════
# OPTIONAL — Design
# ═══════════════════════════════════════════
  top_file: "top.v"
  docs_dir: "./docs"

# ═══════════════════════════════════════════
# OPTIONAL — Model
# ═══════════════════════════════════════════
model:
  backend: "local"             # "local", "openai", "anthropic"
  id: "Qwen/Qwen3-8B"
  quantization: "none"         # "none", "int8", "int4"
  api_base: ""
  api_key: ""
  dtype: "bfloat16"
  temperature: 0.1
  max_new_tokens: 4096
  top_p: 1.0
  enable_thinking: false

# ═══════════════════════════════════════════
# OPTIONAL — Output
# ═══════════════════════════════════════════
output:
  sva_file: "./<top_module>_sva.sv"
  log_file: "./<top_module>_log.txt"
  lint_report: "./lint_failures.json"

# ═══════════════════════════════════════════
# OPTIONAL — Task
# ═══════════════════════════════════════════
task: |
  Custom verification task...

# ═══════════════════════════════════════════
# OPTIONAL — HTML Docs
# ═══════════════════════════════════════════
html_docs:
  enabled: false
  files: []
  output_dir: ""

# ═══════════════════════════════════════════
# OPTIONAL — Security
# ═══════════════════════════════════════════
security:
  threat_model: ""

# ═══════════════════════════════════════════
# OPTIONAL — Logging
# ═══════════════════════════════════════════
logging:
  level: "INFO"

# ═══════════════════════════════════════════
# ADVANCED — Retrieval
# ═══════════════════════════════════════════
retrieval:
  rtl_embedding_model: "jinaai/jina-embeddings-v2-base-code"
  doc_embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  rtl_top_k: 5
  doc_top_k: 5
  context_threshold: 50000
  use_hybrid: true
  rrf_k: 60
  use_hierarchical: true
  hierarchical_stage1_k: 5
  hierarchical_stage2_k: 5
  force_rebuild: false

# ═══════════════════════════════════════════
# ADVANCED — Agent
# ═══════════════════════════════════════════
agent:
  use_ast_assertions: true
  ast_only: false
  ast_max_case_branches: 50
  max_iterations: 40
  max_planning_steps: 15
  max_execution_steps: 5
  max_refinement_iterations: 3
  use_plan_execute: true
  use_self_review: false
  use_dataflow_check: false

  # Stage 2: RTL facts prompt augmentation (see section 4)
  use_rtl_facts: true             # default: true
  rtl_facts_soft_budget: 1500
  rtl_facts_hard_budget: 2400
  use_per_batch_facts: false
  use_grammar_constraints: false
  use_hallucination_denylist: false
  hallucination_denylist_top_n: 5
  hallucination_denylist_dir: "indices/hallucinations"

  # Stage 3: Module-scoped facts (see section 4)
  module_facts_mode: "off"        # "off", "lazy", or "full"
  module_scope_depth: 2           # 0=top only, 2=recommended, -1=all

# ═══════════════════════════════════════════
# OPTIONAL — Mutation Testing
# ═══════════════════════════════════════════
mutation_testing:
  enabled: false
  dut_files: []
  support_files: []
  testbench: ""
  operators:
    - OP_REPLACE
    - CONST_REPLACE
    - SIGNAL_SWAP
    - BITSLICE_MUT
    - COND_NEGATE
    - ASSIGN_DELETE
    - SENSITIVITY_MUT
  simulator: "verilator"       # or "xsim"
  sim_cycles: 500
  sim_timeout_sec: 30
  max_mutants: 200
  max_workers: 4
  report_file: "./mutation_report.json"
```
