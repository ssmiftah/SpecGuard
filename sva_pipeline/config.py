"""
config.py
---------
Central configuration for the SVA generation pipeline.

All paths, model settings, and tunable hyperparameters live here so that
the pipeline modules never hard-code values.  Users typically don't interact
with this file directly — they write a YAML project file (see config_loader.py)
and the loader maps YAML fields to these dataclass fields.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .mutation.config import MutationConfig


@dataclass
class PipelineConfig:
    # -----------------------------------------------------------------------
    # Design settings (REQUIRED — must come from YAML or CLI)
    # -----------------------------------------------------------------------
    # Path to the directory containing Verilog/SystemVerilog source files.
    # The pipeline searches this directory recursively for .v/.sv files.
    rtl_dir: str = ""

    # Path to the specific file containing the top-level module.
    # This is the entry point for compilation tools.  When provided,
    # pyslang/Yosys starts elaboration from this file.  Other files in
    # rtl_dir are included as supporting sources.
    top_file: str = ""

    # Top-level module name for hierarchy elaboration.
    top_module: str = ""

    # Optional path to documentation files (.md, .txt) for RAG retrieval.
    docs_dir: str = ""

    # -----------------------------------------------------------------------
    # LLM settings
    # -----------------------------------------------------------------------
    model_id: str = "Qwen/Qwen3-8B"
    device_map: str = "auto"
    dtype: str = "bfloat16"
    enable_thinking: bool = False

    # Backend selection: "local" (HuggingFace), "openai", or "anthropic".
    backend: str = "local"
    api_base: str = ""   # OpenAI API base URL (for vLLM/ollama/Azure)
    api_key: str = ""    # API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var)

    # -----------------------------------------------------------------------
    # Quantization (local backend only)
    # -----------------------------------------------------------------------
    # "none"  — full precision (bfloat16/float16, ~2 bytes per param)
    # "int8"  — 8-bit quantization via bitsandbytes (~1 byte per param)
    # "int4"  — 4-bit quantization via bitsandbytes (~0.5 bytes per param)
    # Qwen3-8B:  none=~16GB, int8=~8GB, int4=~4GB
    # Qwen3-14B: none=~28GB, int8=~14GB, int4=~7GB
    quantization: str = "none"

    # -----------------------------------------------------------------------
    # Generation hyperparameters
    # -----------------------------------------------------------------------
    max_new_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 1.0

    # -----------------------------------------------------------------------
    # RAG / embedding settings
    # -----------------------------------------------------------------------
    # Legacy field — used as fallback when rtl/doc models aren't set.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Code-aware embedding model for RTL indexing.
    rtl_embedding_model: str = "jinaai/jina-embeddings-v2-base-code"

    # NLP embedding model for documentation indexing.
    doc_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    rtl_top_k: int = 5
    doc_top_k: int = 5
    doc_chunk_size: int = 1000
    doc_chunk_overlap: int = 200
    rtl_max_chunk_chars: int = 8000

    # Batch size for RAG encoder. Lower this for large designs where the
    # embedder OOMs on dense RTL chunks (e.g. CACC: reduce to 8 or 16).
    # Default 64 works for small designs; reduce proportionally to design size.
    rag_encoder_batch_size: int = 64

    # -----------------------------------------------------------------------
    # Persisted index paths
    # -----------------------------------------------------------------------
    rtl_index_path: str = "indices/rtl_index"
    doc_index_path: str = "indices/doc_index"
    force_rebuild_index: bool = False

    # -----------------------------------------------------------------------
    # Assertion style
    # -----------------------------------------------------------------------
    # None = auto-detect from design (clock present → concurrent, else immediate).
    # True = force immediate assertions only (combinational design).
    # False = allow concurrent assertions (clocked design).
    reject_assert_property: Optional[bool] = None

    # -----------------------------------------------------------------------
    # External tool binaries (fallback — only used if pyslang is unavailable)
    # -----------------------------------------------------------------------
    verible_bin: str = "verible-verilog-syntax"
    yosys_bin: str = "yosys"

    # When pyslang compilation has warnings (non-fatal issues like sign
    # conversion, width expansion), the pipeline stops unless this is True.
    # Fatal diagnostics (unknown module, undeclared identifier) ALWAYS stop
    # the pipeline regardless of this flag.
    #
    # Default False (strict): stops on warnings, forcing users to either
    # fix the RTL or explicitly opt in. In a TTY session, the user is
    # prompted Y/n before the hard stop. Set True for fully automated runs
    # where warnings are acceptable.
    allow_pyslang_warnings: bool = False

    # Filename patterns to exclude from AST extraction. Any file whose
    # basename matches any of these glob patterns is skipped by the AST
    # walker. pyslang still reads them (for structural compilation) but
    # the AST assertion extractor ignores them. Intended for stub files
    # (DesignWare stubs, RAM stubs) whose signals shouldn't be asserted
    # on — the stubs aren't functionally the real hardware.
    ast_exclude_patterns: List[str] = field(default_factory=lambda: [
        "DW_*.v",           # DesignWare stubs
        "*_stub.v",         # single-module stub files
        "*_stubs.v",        # grouped stub files
        "nv_ram_stubs.v",   # our NVDLA RAM stubs
    ])

    # When True, AST-generated passthrough/combinational assertions are
    # filtered to only those whose LHS signal is in ``signal_map`` OR is
    # a port of the top module. Internal-only signals have their AST
    # assertions skipped (they're RTL restatements with no verification
    # value). Does NOT affect LLM-generated assertions, or non-trivial
    # assertions (sequential resets, case branches, mutex invariants).
    ast_skip_trivial_internal: bool = True

    # Out-of-scope signal patterns. Assertions referencing any signal
    # whose name matches ANY of these regex patterns are dropped in
    # Phase 2 of the lint loop. Used to enforce "do not assert on X"
    # rules from the spec (e.g., SLCG signals, gated-clock variants,
    # SRAM power management).
    # Example for NVDLA:
    #   out_of_scope_patterns:
    #     - '^dla_clk_ovr_on_sync$'
    #     - '^global_clk_ovr_on_sync$'
    #     - '^tmc2slcg_.*'
    #     - '^slcg_.*_en$'
    #     - '^nvdla_cell_(gated_)?clk_.*'
    #     - '^nvdla_op_gated_clk_.*'
    #     - '^pwrbus_ram_pd$'
    out_of_scope_patterns: List[str] = field(default_factory=list)

    # Structural detection of out-of-scope signals via pyslang. When True,
    # the pipeline analyzes module instantiation hierarchy + port
    # connections to identify signals that only feed clock-gating / DFT /
    # power sinks, then transitively expands the set via assignment chains.
    # Design-agnostic — relies on module instance structure, not signal
    # naming conventions.
    detect_out_of_scope_structural: bool = True

    # Module-name patterns (regex) used to identify "sink" modules during
    # structural out-of-scope detection. Empty list => use the built-in
    # defaults (covers SLCG, clock-gates, DFT, power cells across designs).
    out_of_scope_sink_module_patterns: List[str] = field(default_factory=list)

    # Max iterations for out-of-scope propagation through assignment
    # chains. Real designs converge in 2-4 iterations; this is a safety
    # cap to prevent runaway in pathological cases.
    out_of_scope_propagation_max_iterations: int = 10

    # -----------------------------------------------------------------------
    # Analysis exports
    # -----------------------------------------------------------------------
    # When True, the pipeline writes a suite of human-readable diagnostic
    # artifacts alongside the SVA output: module hierarchy, RTL facts JSON,
    # raw AST patterns/skeletons, a per-step assertion trace CSV, and a
    # per-module assertion count CSV. Default True — exports are cheap
    # (~100ms) and invaluable for debugging.
    export_analysis: bool = True

    # Directory for analysis artifacts. Empty = derive from output_sva_file
    # basename (e.g., ``./nvdla_cacc_test/cacc_analysis/``).
    analysis_dir: str = ""

    # Drop "sig[N:M] == other[N:M]" and "sig == other[N:M]" assertions
    # where LHS and RHS are slices/signals with matching bit widths.
    # These are AST-generated bus-slice restatements that add no
    # verification value — pure RTL-to-SVA syntax translation.
    # Default True: aggressive dedup of structural noise.
    remove_bus_slice_restatements: bool = True

    # -----------------------------------------------------------------------
    # Output paths
    # -----------------------------------------------------------------------
    output_sva_file: str = "./sva_output.sv"
    log_file: str = "./sva_pipeline_log.txt"

    # -----------------------------------------------------------------------
    # Verification task description
    # -----------------------------------------------------------------------
    # If empty, auto-generated from design characteristics.
    task: str = ""

    # -----------------------------------------------------------------------
    # ReAct agent loop settings
    # -----------------------------------------------------------------------
    max_iterations: int = 40

    # -----------------------------------------------------------------------
    # Post-generation lint refinement loop
    # -----------------------------------------------------------------------
    max_refinement_iterations: int = 3
    lint_failures_file: str = "./lint_failures.json"
    max_refinement_react_steps: int = 15

    # When False, Phase 1 deterministic-repair transforms are skipped:
    # fix_bare_property_fragments, fix_immediate_implication,
    # fix_double_negation, fix_immediate_and_form,
    # fix_condition_only_assertions, fix_next_cycle_on_combinational,
    # fix_same_cycle_past_on_sequential. Used by the token-efficiency
    # ablation to measure the cost of routing every failed assertion to
    # the LLM via the lint feedback loop instead of fixing them locally.
    enable_deterministic_repair: bool = True

    # -----------------------------------------------------------------------
    # Full context injection
    # -----------------------------------------------------------------------
    use_full_context_injection: bool = True
    context_injection_threshold: int = 50000

    # -----------------------------------------------------------------------
    # Hierarchical two-stage retrieval
    # -----------------------------------------------------------------------
    # Stage 1: module summary index (coarse) → find relevant modules.
    # Stage 2: full code index (fine) → search within those modules.
    # Skipped for small designs (< 5 modules) or when context injection is active.
    use_hierarchical_retrieval: bool = True
    hierarchical_stage1_k: int = 5     # modules from Stage 1
    hierarchical_stage2_k: int = 5     # chunks from Stage 2
    module_summary_index_path: str = "indices/module_summary_index"

    # -----------------------------------------------------------------------
    # Hybrid BM25 + dense retrieval
    # -----------------------------------------------------------------------
    use_hybrid_retrieval: bool = True
    rrf_k: int = 60

    # -----------------------------------------------------------------------
    # AST-guided assertion generation
    # -----------------------------------------------------------------------
    # Extract RTL patterns (case branches, assignments, etc.) via regex
    # and generate assertion skeletons deterministically. The LLM can then
    # enrich the skeletons with semantic descriptions and protocol assertions.
    use_ast_assertions: bool = True
    ast_only: bool = False           # True = skip LLM entirely (pure deterministic)
    ast_max_case_branches: int = 50  # skip per-branch for large case statements
    use_self_review: bool = False    # LLM reviews its own assertions post-generation
    use_dataflow_check: bool = False # RTL data-flow analysis for case selector mismatches

    # -----------------------------------------------------------------------
    # AST clustering + compaction (Policy B — uniform LLM spec validation)
    # -----------------------------------------------------------------------
    # When enabled, AST-extracted patterns are grouped by abstract template
    # before LLM enrichment.  Bit-replicated families are packed into
    # concat-and-mux assertions verified deterministically by symbolic
    # substitution; same-value reset families pack into a single
    # concat-equality assertion; only one representative per cluster is
    # sent to the LLM for spec-vs-RTL conformance.  Cluster siblings
    # emit directly with a back-reference comment.
    #
    # The legacy trivial/complex split is bypassed when this flag is on,
    # producing a uniform "every unique template is spec-checked"
    # methodology (paper-defensible) at a more controllable token cost
    # on bit-walk-heavy designs.
    ast_use_clustering: bool = False
    ast_enable_compaction: bool = True       # arithmetic concat-mux compaction
    ast_enable_value_clustering: bool = True # seq_reset value-level grouping
    ast_cluster_min_compact_size: int = 5    # min cluster size to attempt arithmetic compaction
    ast_cluster_max_depth: int = 4           # recursion-depth bound for the partition algorithm
    ast_value_cluster_min_size: int = 2      # min value-cluster size to pack into one assertion

    # Naive-baseline mode. When True, the agent skips AST extraction,
    # RTL facts, module scoping, and skeleton batching; it sends the
    # full RTL + docs to the LLM in a single prompt. Used to measure
    # token cost of a "one-shot" LLM-for-SVA approach against the
    # full pipeline. All post-processors still run on the output so
    # assertion quality is scored identically.
    naive_baseline: bool = False

    # -----------------------------------------------------------------------
    # Stage 2: RTL facts prompt augmentation
    # -----------------------------------------------------------------------
    # When enabled, format_facts_for_prompt() injects a structured "RTL
    # Facts" block into the LLM system prompt. The block contains
    # design-specific clock/reset pairs, reset values, signal widths
    # annotated with drive kind, case selectors, constant ownership,
    # and a generic "common SVA mistakes" section.
    #
    # Default OFF. The CACC ablation (2026-04-20) showed the facts card
    # increases total tokens by ~18% *and* cuts assertion yield by ~25%
    # on both CMAC and CACC. See docs/ablation_findings_cacc.md for the
    # full analysis. Configs that still want facts must opt in explicitly.
    use_rtl_facts: bool = False

    # Soft target for the facts block size (token estimate). The greedy
    # fill stops adding extended sections once this is reached.
    rtl_facts_soft_budget: int = 1500

    # Hard cap for the facts block. Sections that would push past this
    # are dropped entirely. Default 2400 leaves headroom in a 32K context
    # window after accounting for examples and the user query.
    rtl_facts_hard_budget: int = 2400

    # Hallucination denylist: when validate_signals removes assertions,
    # the rejected signal names are logged to a per-design, per-model file.
    # On subsequent runs, the top-N most frequent names are injected into
    # the negative constraints section of the prompt as "do NOT use these".
    # Logging is always-on (cheap, builds the knowledgebase). Injection is
    # gated on this flag so it can be disabled for ablation runs.
    # Per-batch signal-specific fact injection. Adds a compact "Signals in
    # this batch" section to each LLM batch prompt with widths, drive kinds,
    # and reset values for the specific signals in that batch. Reinforces
    # drive-kind annotations (|-> vs |=>) but adds token pressure that can
    # increase bare-fragment errors on GGUF models. Default OFF — tested
    # neutral on CMAC (2 batches), may help on larger designs with more batches.
    use_per_batch_facts: bool = False

    # Stage 2.5: Grammar-constrained generation. When enabled, a GBNF
    # grammar is generated from RTLFacts that constrains signal identifiers
    # to the design's actual signal list. Prevents hallucinated signal names
    # at the token level. Requires a backend that supports grammar constraints:
    # - vLLM: via extra_body.guided_grammar (OpenAI-compatible API)
    # - SGLang: similar to vLLM
    # - Local HF: via outlines LogitsProcessor (requires `pip install outlines`)
    # - Ollama: NOT supported via the OpenAI-compatible API
    # Default OFF — adds 5-15% inference latency and may degrade quality
    # on heavily quantized models.
    use_grammar_constraints: bool = False

    # Stage 3: Module-scoped facts for prompt formatting.
    # When top_module is set and this is enabled, the prompt formatter
    # shows facts scoped to the top module instead of the flat merged view.
    # Two modes:
    #   "off"   — disabled, use flat facts (backward compat)
    #   "lazy"  — Option G: build signal→module mapping from Compilation,
    #             filter flat facts at format time. Zero extractor changes,
    #             lossy on signal name collisions across modules.
    #   "full"  — Option F: per-module extraction via ModuleDeclaration
    #             boundary split. Collision-proof but re-walks subtrees.
    module_facts_mode: str = "off"

    # Depth limit for module scoping. Controls how many levels of
    # submodule hierarchy are included in the scoped facts view:
    #   0 = top module only (most aggressive scoping)
    #   1 = top module + direct submodule signals
    #   2 = top module + 2 levels of submodules
    #  -1 = unlimited (include entire hierarchy = equivalent to flat)
    # Only used when module_facts_mode != "off".
    module_scope_depth: int = 2

    use_hallucination_denylist: bool = False
    hallucination_denylist_top_n: int = 5
    hallucination_denylist_dir: str = "indices/hallucinations"

    # -----------------------------------------------------------------------
    # Plan-then-execute agent structure
    # -----------------------------------------------------------------------
    use_plan_execute: bool = True
    max_planning_steps: int = 15
    max_execution_steps_per_assertion: int = 5

    # -----------------------------------------------------------------------
    # HTML documentation conversion
    # -----------------------------------------------------------------------
    # Convert HTML spec files to Markdown before RAG indexing.
    # Listed files are converted and placed in html_docs_output_dir
    # (defaults to docs_dir).  Cached by mtime — skips if .md is newer.
    html_docs_enabled: bool = False
    html_docs_files: List[str] = field(default_factory=list)
    html_docs_output_dir: str = ""  # defaults to docs_dir if empty

    # -----------------------------------------------------------------------
    # Security / threat model
    # -----------------------------------------------------------------------
    # Optional path to a threat model document (.txt or .md).  When provided,
    # the pipeline indexes it alongside the design docs and generates
    # security-focused assertions (e.g. key leakage, fault injection
    # resistance).  When absent, security assertions are skipped entirely.
    threat_model: str = ""

    # When True (and threat_model is non-empty), run the security pass after
    # the lint loop. The pass parses the threat model into one (cwe_id,
    # scenario, signals) tuple per CWE section and asks the LLM to propose
    # a CWE-tagged SVA property per scenario. Outputs flow through the same
    # lint + signal validator gates and are appended to the SVA file.
    security_pass_enabled: bool = False
    # Cap on number of CWE scenarios to process per run (0 = no cap).
    # Useful for smoke tests and to bound LLM spend.
    security_pass_max_scenarios: int = 0

    # -----------------------------------------------------------------------
    # Mutation testing
    # -----------------------------------------------------------------------
    mutation: MutationConfig = field(default_factory=MutationConfig)

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    log_level: str = "INFO"
