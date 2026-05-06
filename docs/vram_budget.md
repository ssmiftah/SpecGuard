# VRAM Budget & Model Fitting

This document records the steps we took to run the SVA generation pipeline
on a single workstation GPU. The target model is a 14B-parameter Qwen3
derivative with a 32K context window; the embedder is a ~560M-param Jina
model used for RAG. Both must coexist with Python, PyTorch, and FAISS on a
GPU that does not have the headroom to host the 14B model in native
precision.

## The budget problem

A 14B model loaded in `bfloat16` needs roughly **28 GB** just for weights.
Add the KV cache for a 32K context (tens of GB at native precision) and
activation memory for generation, and the naive configuration exceeds
80 GB. Our target box has well under that available to Python once the
embedder, FAISS indices, and framework overhead are subtracted. Several
earlier runs on NVDLA CACC OOM'd at 27–28 GB during weight load, before
any forward pass.

## What we tried and where it led

Each step below records a thing we changed, why, and what we saw. The
final configuration is a combination of the ones marked ✅.

### 1. Native HuggingFace Transformers (bfloat16) — ❌

- **What:** Load Qwen3-14B via `transformers.AutoModelForCausalLM` at
  `torch_dtype=torch.bfloat16`.
- **Observation:** OOM at weight load on CACC. Never reached generation.
- **Why it failed:** 28 GB weights + PyTorch allocator overhead +
  embedder + framework state exceeded the physical budget.

### 2. 4-bit quantization via `bitsandbytes` (`load_in_4bit`) — ❌

- **What:** Keep HF, add `quantization_config=BitsAndBytesConfig(load_in_4bit=True)`.
  Weights drop to ~7–8 GB.
- **Observation:** Load succeeded; still OOM on CACC during generation
  at `max_new_tokens=4096` with `rtl_top_k=5`.
- **Why it failed:** KV cache for 32K context plus activations at larger
  top-k retrieval pushed the transient peak over the budget. Quantized
  weights reduce the *static* footprint, not the *activation* footprint.

### 3. Tighten the activation budget — ✅

Before abandoning in-process loading, we cut everything that scales with
sequence length:

- **`max_new_tokens: 4096 → 2048`** — halves generation-time KV growth.
- **`rtl_top_k: 5 → 3`**, **`doc_top_k: 5 → 3`** — smaller prompts into
  the LLM. Trade: less retrieved context per batch. On CACC this was
  acceptable because RTL facts already carry most of the load-bearing
  information.
- **`rtl_facts_hard_budget: 2500 → 1500`** — caps the facts block in
  the prompt.
- **`rtl_max_chunk_chars: 2500 → 1500`** — smaller FAISS chunks; also
  helps the jina embedder avoid OOM on very large single files
  (CACC calculator is 610 KB in one file).
- **`encoder_batch_size: 32 → 8`** — the jina embedder is the peak
  consumer during RAG build. Smaller batches push its peak down at the
  cost of indexing wall-time.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — mitigates
  fragmentation that otherwise reserves memory we cannot reuse.

All of the above are plain YAML/env changes. They moved us from
"OOM before first token" to "OOM after a few batches" — progress, but
still not viable for long runs.

### 4. Move the LLM out-of-process: Ollama backend — ✅

This is the step that actually unblocked the pipeline.

- **What:** Switch `model.backend` from `huggingface` to `openai`, point
  `api_base` at a local Ollama server (`http://localhost:11434/v1`),
  use `api_key: "ollama"` (Ollama ignores the value but the OpenAI
  client requires one), and load the model via an Ollama `Modelfile`
  that sets `num_ctx 32768`.
- **Why it works:** Ollama runs the model in its own process with its
  own GGUF-quantized weights (typically Q4_K_M) and its own memory
  arena. The Python process only needs the embedder, FAISS, and
  framework state. We hand the LLM batches off over HTTP and get
  assertions back; there is no PyTorch tensor for the 14B model in our
  address space at all.
- **Practical effect:** weights pressure drops to ~0 GB in the Python
  process. The embedder (~1.5–2 GB) and FAISS fit comfortably. Ollama
  handles its own budget independently and streams tokens back.
- **Side benefit:** the same code path talks to any OpenAI-compatible
  endpoint (e.g., vLLM, TGI), so swapping the backend for A/B runs is
  a one-line YAML change.

Example (from [examples/nvdla_cacc_full.yaml](../examples/nvdla_cacc_full.yaml)):

```yaml
model:
  backend: "openai"
  id: "qwen3:14b-32k"
  api_base: "http://localhost:11434/v1"
  api_key: "ollama"
  temperature: 0.1
  max_new_tokens: 2048

retrieval:
  rtl_max_chunk_chars: 1500
  encoder_batch_size: 8
  rtl_top_k: 3
  doc_top_k: 3
```

The `qwen3:14b-32k` tag points at an Ollama model built with a
Modelfile that overrides `num_ctx` to 32768:

```
FROM qwen3:14b
PARAMETER num_ctx 32768
```

Register it once with `ollama create qwen3:14b-32k -f Modelfile`.

## Per-component budget (Ollama configuration, approximate)

| Component | Where | Footprint |
|---|---|---|
| Qwen3-14B weights (Q4_K_M) | Ollama process | ~8–9 GB |
| KV cache @ 32K ctx, active prompt | Ollama process | a few GB, grows with prompt |
| Jina embedder (bf16) | Python process | ~1.5–2 GB |
| FAISS indices (CPU-side) | RAM, not VRAM | — |
| PyTorch / framework overhead | Python process | ~1 GB |

The two process arenas do not contend directly. If either side grows,
the other still gets its budget. This separation is the single biggest
reason the pipeline is now stable on long runs.

## Things worth knowing

- **GGUF models behave differently from HF.** The same 14B Qwen3 checkpoint
  under GGUF quantization is not token-identical to bf16. Aggressive
  prompt hints that worked on bf16 (e.g., `"IMPORTANT: ..."`) degrade
  Q4_K_M output; gentle examples work better. See the
  [backend hint feedback memory](../.claude-style-hints).
- **`num_ctx` must be set on the Ollama side.** The client can only
  *request* up to what the Modelfile allows. A client asking for 32K
  against a model compiled at 4K silently truncates.
- **`num_ctx` costs memory.** Doubling the context roughly doubles the
  KV cache. We set 32K because our CACC prompts plus RTL facts can run
  to ~15–20K tokens; 16K would truncate. If you are working on a
  smaller design, build a 16K variant and save a few GB.
- **Batching is still our lever.** Each generation batch sends ~20 AST
  skeletons plus facts. Shrinking the batch reduces per-call peak
  without touching the model configuration.
- **Single-file designs hurt the embedder, not the LLM.** CACC's 610 KB
  calculator file was the trigger for tightening `rtl_max_chunk_chars`
  and `encoder_batch_size`. The LLM never saw the whole file at once.
- **Deprecated env var name.** PyTorch now prefers `PYTORCH_ALLOC_CONF`
  over `PYTORCH_CUDA_ALLOC_CONF`; the old name still works with a warning.

## If you have more VRAM

If the box grows, the reasonable order to re-enable things is:

1. `rtl_top_k` and `doc_top_k` back to 5 — more retrieved context per batch.
2. `max_new_tokens` back to 4096 — fewer truncation-mid-assertion events.
3. `encoder_batch_size` back up — faster RAG build.
4. Only then consider an in-process backend again. Ollama has proven
   stable enough that giving up the process isolation is rarely worth it.
