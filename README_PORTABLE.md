# SVA Pipeline — Portable Ablation Benchmark

## Setup

1. **Create a Python venv and install deps:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Install and serve Ollama with qwen3:14b-32k:**
   ```bash
   # Install: https://ollama.com/download
   ollama pull qwen3:14b
   # Create a 32K-context Modelfile
   cat > Modelfile <<'EOF'
   FROM qwen3:14b
   PARAMETER num_ctx 32768
   EOF
   ollama create qwen3:14b-32k -f Modelfile
   ollama serve &   # if not already running as a service
   ```

3. **Verify:**
   ```bash
   curl http://localhost:11434/v1/models | grep qwen3
   ```

## Run the full ablation matrix

```bash
./run_all_ablations.sh
```

This executes **7 designs × 6 variants × 15 repetitions = 630 runs**. Expect
~230 h (~10 days) wall-clock on an RTX 4090.

### Subset for quicker validation

```bash
REPS=3 DESIGNS="cmac fifox" ./run_all_ablations.sh   # smoke test, ~30 min
```

Env vars:
- `DESIGNS` — space-separated list from: `cmac cacc fifox rvv pdp cmacfull csc`
- `VARIANTS` — from: `full no-facts flat-facts no-ast no-repair no-feedback`
- `REPS` — integer, default 15

## Output layout

```
ablation_study_<TIMESTAMP>/
├── <design>/<variant>/run_<NN>/
│   ├── sva.sv                  # generated assertions
│   ├── lint.json               # pyslang lint results
│   ├── token_summary.json      # prompt/completion/total tokens
│   ├── trace.{csv,json}        # per-step LLM trace
│   ├── log.txt                 # stdout
│   └── stderr.txt
├── all_runs.csv                # one row per run (for plotting)
├── summary.md                  # aggregated mean/median/std per (design,variant)
├── summary_agg.csv
└── batch.log
```

## Designs in the benchmark

| Design | Source | LOC | Character |
|---|---|---:|---|
| cmac | NVDLA MAC multiplier submodule | 1 574 | Small, single-file |
| fifox | Google Coral NPU (Chisel-emitted) | 586 | Decoupled FIFO |
| rvv | Google Coral NPU RISC-V Vector backend | 1 591 | Hierarchical |
| cacc | NVDLA Convolution Accumulator | 30 905 | Large datapath |
| pdp | NVDLA Pixel Data Processor | 43 233 | DMA + pool |
| cmacfull | NVDLA full CMAC subsystem | 67 977 | MAC array + regs |
| csc | NVDLA Convolution Sequence Controller | 104 575 | Control plane |

## Ablation variants

| Variant | Knob |
|---|---|
| full | all components on (baseline) |
| no-facts | `use_rtl_facts: false` |
| flat-facts | `module_facts_mode: "off"` (facts on, scoping off) |
| no-ast | `use_ast_assertions: false` |
| no-repair | `enable_deterministic_repair: false` |
| no-feedback | repair off **and** `max_refinement_iterations: 0` |

## Troubleshooting

- **Ollama context overflow** — verify `num_ctx 32768` in the Modelfile.
- **`jina` embedder OOM** — drop `retrieval.encoder_batch_size` to 4 in the
  YAML of the failing design.
- **pyslang parse errors** — the NVDLA designs expect `allow_pyslang_warnings:
  true` which is already set in every config.
- **Single run failing** — check `ablation_study_<TS>/<design>/<variant>/run_NN/stderr.txt`
  for the specific error, then re-run just that variant with
  `DESIGNS=<design> VARIANTS=<variant> REPS=1`.
