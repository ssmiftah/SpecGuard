"""
config_loader.py
----------------
YAML-based configuration loading with auto-detection.

Provides a single entry point — ``load_project_config()`` — that resolves
configuration from three sources in priority order:

    defaults  →  YAML file  →  CLI overrides  →  auto-detection

The user only needs to provide a minimal YAML file with ``design.rtl_dir``
and ``design.top_module``.  Everything else has smart defaults or is
auto-detected from the design.

Example minimal YAML (``project.yaml``):
    design:
      rtl_dir: "./RTL Cases/NVDLA_hw/vmod/nvdla/apb2csb"
      top_module: "NV_NVDLA_apb2csb"

    output:
      sva_file: "./nvdla_sva.sv"
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .config import PipelineConfig
from .mutation.config import MutationConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YAML key → PipelineConfig field mapping
# ---------------------------------------------------------------------------
# Nested YAML keys (dot-separated) map to flat dataclass field names.

_YAML_TO_FIELD: Dict[str, str] = {
    # design section
    "design.rtl_dir":           "rtl_dir",
    "design.top_file":          "top_file",
    "design.top_module":        "top_module",
    "design.docs_dir":          "docs_dir",

    # model section
    "model.id":                 "model_id",
    "model.backend":            "backend",
    "model.api_base":           "api_base",
    "model.api_key":            "api_key",
    "model.dtype":              "dtype",
    "model.quantization":       "quantization",
    "model.temperature":        "temperature",
    "model.max_new_tokens":     "max_new_tokens",
    "model.top_p":              "top_p",
    "model.enable_thinking":    "enable_thinking",

    # output section
    "output.sva_file":          "output_sva_file",
    "output.log_file":          "log_file",
    "output.lint_report":       "lint_failures_file",

    # task (top-level string)
    "task":                     "task",

    # html_docs section
    "html_docs.enabled":        "html_docs_enabled",
    "html_docs.files":          "html_docs_files",
    "html_docs.output_dir":     "html_docs_output_dir",

    # security section
    "security.threat_model":    "threat_model",
    "security.pass_enabled":    "security_pass_enabled",
    "security.max_scenarios":   "security_pass_max_scenarios",

    # logging section
    "logging.level":            "log_level",

    # retrieval section
    "retrieval.embedding_model":     "embedding_model",
    "retrieval.rtl_embedding_model": "rtl_embedding_model",
    "retrieval.doc_embedding_model": "doc_embedding_model",
    "retrieval.use_hierarchical":    "use_hierarchical_retrieval",
    "retrieval.hierarchical_stage1_k": "hierarchical_stage1_k",
    "retrieval.hierarchical_stage2_k": "hierarchical_stage2_k",
    "retrieval.rtl_top_k":           "rtl_top_k",
    "retrieval.doc_top_k":           "doc_top_k",
    "retrieval.context_threshold":   "context_injection_threshold",
    "retrieval.rtl_max_chunk_chars": "rtl_max_chunk_chars",
    "retrieval.doc_chunk_size":      "doc_chunk_size",
    "retrieval.encoder_batch_size":  "rag_encoder_batch_size",
    "retrieval.use_hybrid":          "use_hybrid_retrieval",
    "retrieval.rrf_k":               "rrf_k",
    "retrieval.force_rebuild":       "force_rebuild_index",

    # agent section
    "agent.max_iterations":                  "max_iterations",
    "agent.max_planning_steps":              "max_planning_steps",
    "agent.max_execution_steps":             "max_execution_steps_per_assertion",
    "agent.max_refinement_iterations":       "max_refinement_iterations",
    "agent.use_ast_assertions":              "use_ast_assertions",
    "agent.ast_only":                        "ast_only",
    "agent.ast_max_case_branches":           "ast_max_case_branches",
    "agent.use_self_review":                 "use_self_review",
    "agent.use_dataflow_check":              "use_dataflow_check",
    "agent.use_plan_execute":                "use_plan_execute",
    "agent.naive_baseline":                  "naive_baseline",
    "agent.enable_deterministic_repair":     "enable_deterministic_repair",

    # Stage 2: RTL facts prompt augmentation
    "agent.use_rtl_facts":                   "use_rtl_facts",
    "agent.use_per_batch_facts":             "use_per_batch_facts",
    "agent.use_grammar_constraints":         "use_grammar_constraints",
    "agent.module_facts_mode":               "module_facts_mode",
    "agent.module_scope_depth":              "module_scope_depth",
    "agent.reject_assert_property":          "reject_assert_property",
    "agent.allow_pyslang_warnings":          "allow_pyslang_warnings",
    "agent.ast_exclude_patterns":            "ast_exclude_patterns",
    "agent.ast_skip_trivial_internal":       "ast_skip_trivial_internal",
    # AST clustering + compaction (Policy B — uniform LLM spec validation).
    "agent.ast_use_clustering":              "ast_use_clustering",
    "agent.ast_enable_compaction":           "ast_enable_compaction",
    "agent.ast_enable_value_clustering":     "ast_enable_value_clustering",
    "agent.ast_cluster_min_compact_size":    "ast_cluster_min_compact_size",
    "agent.ast_cluster_max_depth":           "ast_cluster_max_depth",
    "agent.ast_value_cluster_min_size":      "ast_value_cluster_min_size",
    "agent.out_of_scope_patterns":                          "out_of_scope_patterns",
    "agent.detect_out_of_scope_structural":                 "detect_out_of_scope_structural",
    "agent.out_of_scope_sink_module_patterns":              "out_of_scope_sink_module_patterns",
    "agent.out_of_scope_propagation_max_iterations":        "out_of_scope_propagation_max_iterations",
    "agent.remove_bus_slice_restatements":                  "remove_bus_slice_restatements",
    "output.analysis_dir":                                  "analysis_dir",
    "output.export_analysis":                               "export_analysis",
    "agent.rtl_facts_soft_budget":           "rtl_facts_soft_budget",
    "agent.rtl_facts_hard_budget":           "rtl_facts_hard_budget",
    "agent.use_hallucination_denylist":      "use_hallucination_denylist",
    "agent.hallucination_denylist_top_n":    "hallucination_denylist_top_n",
    "agent.hallucination_denylist_dir":      "hallucination_denylist_dir",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_project_config(yaml_path: str) -> PipelineConfig:
    """
    Load pipeline configuration from a YAML project file.

    Resolution order: PipelineConfig defaults → YAML values → smart defaults.
    The YAML file is the single source of truth — no CLI overrides.

    Parameters
    ----------
    yaml_path : str
        Path to the project YAML file.

    Returns
    -------
    PipelineConfig
    """
    yaml_data = _load_yaml(yaml_path)
    _validate_yaml(yaml_data)

    # Flatten nested YAML into PipelineConfig field names.
    config_dict = _flatten_yaml(yaml_data)
    logger.info("Loaded config from %s (%d fields).", yaml_path, len(config_dict))

    # Build PipelineConfig — unset fields get dataclass defaults.
    config = PipelineConfig(**config_dict)

    # Apply smart defaults for derived fields (output paths, dtype, etc.).
    config = _apply_smart_defaults(config)

    return config


# ---------------------------------------------------------------------------
# Auto-detection (runs after design info is built)
# ---------------------------------------------------------------------------

def auto_detect(
    config: PipelineConfig,
    design_info: Any,  # DesignInfo from slang_frontend
) -> PipelineConfig:
    """
    Fill in auto-detected values based on design characteristics.

    Called after ``build_design_info()`` so we have clock/reset info.

    Parameters
    ----------
    config : PipelineConfig
        Partially filled config.
    design_info : DesignInfo
        Design analysis results from slang_frontend.

    Returns
    -------
    PipelineConfig
        Updated config with auto-detected values.
    """
    # Auto-detect assertion style from clock presence.
    if config.reject_assert_property is None:
        config.reject_assert_property = not design_info.has_clock
        logger.info(
            "Auto-detected assertion style: %s (clock %s)",
            "immediate" if config.reject_assert_property else "concurrent",
            "not found" if config.reject_assert_property else f"= {design_info.clock_signal}",
        )

    return config


def generate_task(
    top_module: str,
    signal_map: Dict[str, Any],
    has_clock: bool,
    reset_signal: Optional[str],
    module_count: int,
    has_threat_model: bool = False,
) -> str:
    """
    Auto-generate a verification task description from design characteristics.

    Analyses the signal map to determine the design type and selects
    appropriate property categories.

    Parameters
    ----------
    top_module : str
        Top-level module name.
    signal_map : dict
        Auto-generated signal map.
    has_clock : bool
        Whether the design has a clock signal.
    reset_signal : str or None
        Detected reset signal name.
    module_count : int
        Number of modules in the design.

    Returns
    -------
    str
        A structured verification task description.
    """
    signal_count = len(signal_map)

    # Infer design type from signal patterns.
    sig_names = " ".join(signal_map.keys()).lower()
    if any(kw in sig_names for kw in ["encrypt", "decrypt", "cipher", "key", "sbox"]):
        design_type = "cryptographic engine"
    elif any(kw in sig_names for kw in ["apb", "axi", "csb", "ahb", "wishbone"]):
        design_type = "bus protocol bridge/controller"
    elif any(kw in sig_names for kw in ["addr", "data", "write", "read", "ready", "valid"]):
        design_type = "interface controller"
    elif any(kw in sig_names for kw in ["fifo", "push", "pop", "full", "empty"]):
        design_type = "FIFO/buffer"
    else:
        design_type = "digital logic module"

    # Assertion style guidance.
    if has_clock:
        style_note = (
            "This is a CLOCKED design — use concurrent assertions with "
            f"@(posedge <clock>)"
        )
        if reset_signal:
            style_note += f" disable iff (!{reset_signal})"
        style_note += "."
    else:
        style_note = (
            "This is a COMBINATIONAL design — use immediate assertions only: "
            "assert (...) else $error(...);"
        )

    # Build property categories — functional first, interface last.
    categories = []

    if has_clock:
        categories.append(
            "FUNCTIONAL CORRECTNESS — for each case/branch in the RTL, "
            "verify that the output matches the expected value.  Write one "
            "assertion per case branch with the input condition as the guard"
        )
        categories.append(
            "PROTOCOL / TIMING — verify handshake protocols, sequencing, "
            "state transitions"
        )
    else:
        categories.append(
            "FUNCTIONAL CORRECTNESS — for each case/branch in the RTL, "
            "verify that the output matches the expected value.  Write one "
            "assertion per case branch with the input condition as the guard"
        )

    # Check for specific signal patterns and add relevant categories.
    if any(s.get("type") == "control" for s in signal_map.values()):
        categories.append(
            "CONTROL SIGNAL LOGIC — verify enable gating, ready/valid "
            "handshakes, select lines"
        )

    if any(s.get("type") == "data" and s.get("width", 0) > 1 for s in signal_map.values()):
        categories.append(
            "DATA INTEGRITY — verify data forwarding, width preservation, "
            "no bit corruption"
        )

    if reset_signal:
        categories.append(
            f"RESET BEHAVIOUR — verify all state registers are properly "
            f"cleared when {reset_signal} is asserted"
        )

    if has_threat_model:
        categories.append(
            "SECURITY — use the threat model document (available via "
            "doc_retrieve) to generate assertions for: key/secret leakage "
            "prevention, fault injection resistance, side-channel "
            "countermeasure verification, and any other threats identified "
            "in the threat model"
        )

    categories.append(
        "INTERFACE COMPLIANCE — verify port widths and directions"
    )

    # Format the categories.
    cat_text = "\n".join(
        f"  {i}. {cat}" for i, cat in enumerate(categories, 1)
    )

    # Documentation guidance.
    doc_note = ""
    if has_threat_model:
        doc_note += (
            "\n⚠ IMPORTANT: A threat model document is available in your context. "
            "Read it carefully and derive security assertions from the threats described.\n"
        )

    return f"""Generate comprehensive SystemVerilog Assertions (SVA) for the {top_module} \
module ({design_type}).

The design has {module_count} module(s) and {signal_count} signal(s).
{style_note}

Read the design documentation in your context carefully, then read the RTL
source code.  For every case branch, conditional assignment, and mux select
in the RTL, write an assertion that checks the output value for that
specific input condition.
{doc_note}
Cover all of the following:

{cat_text}

Validate every assertion with the linting tool before submitting.
"""


# ---------------------------------------------------------------------------
# Index staleness detection
# ---------------------------------------------------------------------------

def compute_source_checksum(source_dir: str, extensions: set = {".v", ".sv"}) -> str:
    """
    Compute a fast checksum of all source files in a directory.

    Uses file paths + modification times (not file contents) for speed.
    Returns a hex digest string that changes whenever a file is added,
    removed, or modified.
    """
    entries: List[str] = []
    for root, _, files in os.walk(source_dir):
        for fname in sorted(files):
            if Path(fname).suffix in extensions:
                fpath = os.path.join(root, fname)
                mtime = os.path.getmtime(fpath)
                entries.append(f"{fpath}:{mtime}")

    content = "\n".join(sorted(entries))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def is_index_stale(index_path: str, current_checksum: str) -> bool:
    """
    Check if the persisted index is stale by comparing checksums.

    Returns True if the index should be rebuilt.
    """
    checksum_file = index_path + ".checksum"
    if not os.path.exists(checksum_file):
        return True

    try:
        with open(checksum_file, "r") as fh:
            stored = fh.read().strip()
        return stored != current_checksum
    except OSError:
        return True


def save_checksum(index_path: str, checksum: str) -> None:
    """Save a checksum alongside the index files."""
    checksum_file = index_path + ".checksum"
    Path(checksum_file).parent.mkdir(parents=True, exist_ok=True)
    with open(checksum_file, "w") as fh:
        fh.write(checksum)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> Dict[str, Any]:
    """Load and parse a YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(data).__name__}")

    return data


def _validate_yaml(data: Dict[str, Any]) -> None:
    """
    Validate that required YAML fields are present.

    Raises ValueError with a clear message if validation fails.
    """
    design = data.get("design", {})

    if not design.get("rtl_dir"):
        raise ValueError(
            "Missing required field: design.rtl_dir\n"
            "Your YAML config must specify the RTL source directory:\n\n"
            "  design:\n"
            '    rtl_dir: "./path/to/verilog"\n'
            '    top_module: "ModuleName"'
        )

    if not design.get("top_module"):
        raise ValueError(
            "Missing required field: design.top_module\n"
            "Your YAML config must specify the top-level module name:\n\n"
            "  design:\n"
            '    rtl_dir: "./path/to/verilog"\n'
            '    top_module: "ModuleName"'
        )

    # Verify rtl_dir exists.
    rtl_dir = design["rtl_dir"]
    if not os.path.isdir(rtl_dir):
        raise ValueError(
            f"design.rtl_dir does not exist: {rtl_dir}\n"
            "Please check the path in your YAML config."
        )

    # Verify top_file exists if provided.
    top_file = design.get("top_file", "")
    if top_file and not os.path.exists(top_file):
        # Try resolving relative to rtl_dir.
        resolved = os.path.join(rtl_dir, top_file)
        if not os.path.exists(resolved):
            raise ValueError(
                f"design.top_file not found: {top_file}\n"
                f"Also tried: {resolved}\n"
                "Please check the path in your YAML config."
            )


def _flatten_yaml(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested YAML into PipelineConfig field names using the mapping.

    The ``mutation_testing`` section is handled specially: its contents are
    parsed into a ``MutationConfig`` dataclass rather than flat fields.
    """
    result: Dict[str, Any] = {}

    # Handle mutation_testing section as a nested dataclass.
    mut_data = data.get("mutation_testing", {})
    if mut_data:
        # Map YAML keys to MutationConfig field names (1:1 in this case).
        mut_kwargs = {}
        for key, value in mut_data.items():
            # Convert YAML key names to Python field names.
            field_name = key  # YAML keys match dataclass field names
            if hasattr(MutationConfig, field_name):
                mut_kwargs[field_name] = value
        result["mutation"] = MutationConfig(**mut_kwargs)

    def _walk(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                # Skip mutation_testing — handled above.
                if full_key == "mutation_testing":
                    continue
                # Check if this dotted key maps to a config field.
                if full_key in _YAML_TO_FIELD:
                    result[_YAML_TO_FIELD[full_key]] = value
                else:
                    # Recurse into nested dicts.
                    _walk(value, full_key)

    _walk(data)
    return result


def _apply_smart_defaults(config: PipelineConfig) -> PipelineConfig:
    """
    Apply smart defaults for fields that depend on other field values.
    """
    # Default output paths based on top_module name.
    if not config.output_sva_file or config.output_sva_file == "./sva_output.sv":
        if config.top_module:
            config.output_sva_file = f"./{config.top_module}_sva.sv"

    if not config.log_file or config.log_file == "./sva_pipeline_log.txt":
        if config.top_module:
            config.log_file = f"./{config.top_module}_log.txt"

    # Default index paths.
    if config.rtl_index_path == "indices/rtl_index":
        config.rtl_index_path = f"indices/{config.top_module}_rtl"
    if config.doc_index_path == "indices/doc_index":
        config.doc_index_path = f"indices/{config.top_module}_doc"
    if config.module_summary_index_path == "indices/module_summary_index":
        config.module_summary_index_path = f"indices/{config.top_module}_summary"

    # Auto-detect dtype from GPU if set to "auto".
    if config.dtype == "auto":
        config.dtype = _detect_dtype()

    return config


def _detect_dtype() -> str:
    """
    Auto-detect the best dtype based on GPU capabilities.

    Ampere+ (compute >= 8.0) → bfloat16
    Turing  (7.5)            → float16
    No GPU                   → float32
    """
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 8:
                return "bfloat16"
            elif cap[0] >= 7:
                return "float16"
        return "float32"
    except ImportError:
        return "float32"
