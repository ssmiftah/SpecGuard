"""
agent.py
--------
ReAct agent that generates SystemVerilog Assertions (SVA) for a hardware design.

Architecture
------------
* Model  : Qwen3-8B loaded in bfloat16 via HuggingFace Transformers.
* Tools  : Five tools (rtl_retrieve, doc_retrieve, yosys_extract,
           signal_map_lookup, slang_lint) defined in tools.py.
* Loop   : ReAct (Reasoning + Acting).  Each iteration the model emits
           either a <tool_call> block (→ observation is appended and the
           loop continues) or plain text (→ treated as the final answer).

Qwen3 tool-calling protocol
----------------------------
When `tools` are passed to `tokenizer.apply_chat_template()`, Qwen3 is
primed to output tool invocations as:

    <tool_call>
    {"name": "tool_name", "arguments": {"arg": "value"}}
    </tool_call>

After each tool call we append:
  - The assistant turn that contained the call.
  - A "tool" role message with the observation.

The loop then calls the model again with the updated history until the
model stops emitting tool calls.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import PipelineConfig
from .tools import TOOL_DEFINITIONS, dispatch_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex to extract every <tool_call>…</tool_call> block from model output.
# ---------------------------------------------------------------------------
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

# Marker the model is asked to output when it has finished generating SVA.
_DONE_SENTINEL = "<<SVA_COMPLETE>>"

# Verilog/SV submodule instantiation:  `module_name u_inst_name (`
# (anchored at line start, conventional `u_*` instance prefix).
# Used by the AST wrapper-detection fallback to spot top files that are
# mostly submodule instantiations rather than behavioural logic.
_SUBMODULE_INST_RE = re.compile(
    r'^\s*[A-Za-z_]\w*\s+u_\w+\s*\(', re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Assertion style guidance (design-aware)
# ---------------------------------------------------------------------------

def _assertion_style_guidance(reject_assert_property: bool) -> str:
    """
    Return the assertion-style instructions for the system prompt.

    For combinational designs (no clock): immediate assertions only.
    For clocked designs: concurrent assertions with @(posedge clk) are preferred.
    """
    if reject_assert_property:
        return """  ⚠ CRITICAL — IMMEDIATE ASSERTIONS ONLY (the slang_lint tool will REJECT
  any code that uses "assert property"):

  FORBIDDEN syntax (concurrent assertion — ALWAYS wrong here):
    assert property (out == expected) else $error("...");
    assert property (@posedge clk) (out == expected) |-> ...;

  REQUIRED syntax (immediate assertion — correct for combinational logic):
    assert (out == expected) else $error("...");

  The ONLY difference is removing the word "property".
  "assert property" is a concurrent form even without @(posedge clk).
  slang_lint checks for this and will FAIL until you remove "property"."""
    else:
        return """  This is a CLOCKED design.  Use CONCURRENT assertions with a clock edge:

  PREFERRED syntax (concurrent assertion with clock):
    assert property (@(posedge clk) disable iff (!rst_n)
      condition |-> consequence
    ) else $error("...");

  For simple checks that do not need temporal reasoning, you may also use
  immediate assertions inside always blocks:
    assert (condition) else $error("...");

  Examples:
    // Protocol: valid must be deasserted after ready handshake
    assert property (@(posedge pclk) disable iff (!prstn)
      csb2nvdla_valid && csb2nvdla_ready |=> !csb2nvdla_valid || psel
    ) else $error("spurious valid after handshake");

    // Data forwarding
    assert property (@(posedge pclk)
      csb2nvdla_wdat == pwdata
    ) else $error("write data mismatch");"""


# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

def _build_system_prompt(
    hierarchy: str,
    signal_map_summary: str,
    graph_summary_text: str = "",
    full_rtl_context: Optional[str] = None,
    reject_assert_property: bool = True,
    rtl_facts_block: str = "",
    rtl_facts_reminders: str = "",
    backend_hint: str = "",
) -> str:
    """
    Construct the system prompt that is injected at position 0 in every
    conversation.  The hierarchy, signal map, and optionally the full RTL
    source and design graph summary are always present so the model has
    structural context without needing to call a tool for it.

    Parameters
    ----------
    hierarchy : str
        Full content of hierarchy.txt.
    signal_map_summary : str
        A human-readable summary of signal_map.json.
    graph_summary_text : str
        Compact design graph summary from Yosys (modules, ports, hierarchy).
    full_rtl_context : str or None
        When provided, the complete RTL source code is injected directly.
    rtl_facts_block : str
        Pre-formatted RTL facts block from format_facts_for_prompt(). When
        non-empty it is injected as a "context card" right after the
        persona setup, before the design hierarchy section. Empty string
        disables this feature (current default).
    rtl_facts_reminders : str
        Compact "Critical reminders" block from format_facts_reminders().
        Injected at the END of the system prompt (instruction sandwich)
        so the most important rules benefit from recency bias.
    backend_hint : str
        Backend-specific formatting guidance. For OpenAI/Ollama backends,
        includes extra emphasis on single-line assertion formatting.
    """
    # Conditionally build the graph section.
    graph_section = ""
    if graph_summary_text:
        graph_section = f"""
═══════════════════════════════════════════════════════════
DESIGN GRAPH (always available — no tool call needed)
═══════════════════════════════════════════════════════════
{graph_summary_text}
"""

    # Conditionally build the full RTL section.
    rtl_section = ""
    rtl_tool_note = ""
    if full_rtl_context:
        rtl_section = f"""
═══════════════════════════════════════════════════════════
COMPLETE RTL SOURCE CODE (all files — no retrieval needed)
═══════════════════════════════════════════════════════════
{full_rtl_context}
"""
        rtl_tool_note = (
            "   → NOTE: The full RTL is already in your context above. "
            "You should not need this tool for this design.\n"
        )

    # Optional RTL facts context card (Stage 2 prompt augmentation).
    facts_section = ""
    if rtl_facts_block:
        facts_section = f"""
═══════════════════════════════════════════════════════════
RTL FACTS (parser-extracted, authoritative for this design)
═══════════════════════════════════════════════════════════
{rtl_facts_block}
"""

    return f"""You are an expert hardware verification engineer specialising in \
SystemVerilog Assertions (SVA).

Your task is to generate comprehensive, syntactically correct SVA assertions \
for the hardware design described below.
{facts_section}

═══════════════════════════════════════════════════════════
DESIGN HIERARCHY (always available — no tool call needed)
═══════════════════════════════════════════════════════════
{hierarchy}
{graph_section}
═══════════════════════════════════════════════════════════
SIGNAL MAP SUMMARY (always available — no tool call needed)
═══════════════════════════════════════════════════════════
{signal_map_summary}
{rtl_section}
═══════════════════════════════════════════════════════════
TOOLS AVAILABLE
═══════════════════════════════════════════════════════════
You have five tools:

1. rtl_retrieve(query, k=5)
   → Search the RTL source index for Verilog/SV code snippets.
   → Use this to inspect module internals, logic, and signal usage.
{rtl_tool_note}
2. doc_retrieve(query, k=5)
   → Search the documentation index for specification passages.
   → Use this to find what a module is required to do.

3. yosys_extract(module_name)
   → Extract exact port names, directions, and bit-widths from the RTL
     using Yosys.  Use this before finalising any signal reference.

4. signal_map_lookup(signal_name)
   → Look up signal attributes (width, direction, module, description).
   → Supports partial matching (e.g. "key" finds all key-related signals).
   → NOTE: The signal map summary above already lists all signals. Only call
     this tool when you need the full detail of a specific signal not clear
     from the summary. Do NOT call it for every signal individually.

5. slang_lint(sva_code)
   → Validate SVA syntax with verible-verilog-syntax.
   → ALWAYS call this before emitting any final assertion.
   → If it returns FAIL, read the error, fix the assertion, and retry.

═══════════════════════════════════════════════════════════
WORKFLOW (follow this order)
═══════════════════════════════════════════════════════════
Step 1 — Understand the specification
  Use doc_retrieve to find what properties the design must satisfy.

Step 2 — Understand the implementation
  Use rtl_retrieve + yosys_extract to find how those properties are
  implemented: signal names, widths, module boundaries.

Step 3 — Cross-check signals
  Use signal_map_lookup to confirm exact names and bit-widths before
  referencing them in assertions.

Step 4 — Generate assertions
{_assertion_style_guidance(reject_assert_property)}

  Also FORBIDDEN: signal.width — '.width' is not a valid SV signal attribute.
    WRONG: assert (fullkeys.width == 1408) else $error(...);
    RIGHT: assert ($bits(fullkeys) == 1408) else $error(...);
  Note: $bits() only works for signals visible in the current scope.
  For internal signals of submodules that are not directly accessible,
  document the width constraint as a comment rather than an assertion.

  Write SVA assertions covering:
  • Functional correctness  (output == expected given input)
  • Interface compliance    (valid signal combinations, protocol handshakes)
  • Data integrity          (no bit corruption across pipeline stages)
  • Timing / sequencing     (correct state transitions)

═══════════════════════════════════════════════════════════
COMMON MISTAKES TO AVOID — STRICT
═══════════════════════════════════════════════════════════
These four patterns silently produce useless assertions. They parse
cleanly and use real signals, but they encode no enforceable property.
Reject them at write time:

1. NEVER combine `disable iff (!rstn)` with an antecedent of `!rstn`.
   The disable cancels the antecedent — the assertion never fires.
   PICK ONE:
     • Reset value check:  assert property (@(posedge clk) !rstn |-> sig == reset_val);
     • Hardened invariant: assert property (@(posedge clk) disable iff (!rstn) cond |-> claim);
   Do not write:
     WRONG: assert property (@(posedge clk) disable iff (!rstn) (!rstn) |-> sig == 0);

2. NEVER write `assert (cond1 && cond2)` to express implication.
   That asserts BOTH conditions are ALWAYS true on every cycle — almost never the intent.
   USE:
     • Immediate:   assert (!cond1 || cond2) else $error("...");
     • Concurrent:  assert property (cond1 |-> cond2) else $error("...");
   Do not write:
     WRONG: assert ((req_valid && !req_ready) && !fifo_pop);
     RIGHT: assert (!(req_valid && !req_ready) || !fifo_pop);

3. NEVER write a bare `assert property (... (sig == LITERAL))` with no
   antecedent and no implication for a non-parameter signal. That asserts
   the signal equals the literal on EVERY clock cycle — datapath signals
   change every cycle, so this fires constantly.
   Do not write:
     WRONG: assert property (@(posedge clk) disable iff (!rstn) (mac2accu_data0 == 176'b0));
     RIGHT: assert property (@(posedge clk) !rstn |-> (mac2accu_data0 == 176'b0));

4. NEVER write `(valid && !ready) |-> ... ##1 (valid && ready)` to express
   handshake completion. Real handshakes allow ready to arrive 1+ cycles
   later, not exactly 1.
   USE:
     • Eventually:  assert property ((valid && !ready) |-> ##[1:$] (valid && ready));
     • Next cycle:  assert property ((valid && !ready) |=> (valid && ready));    // only if you know exactly 1

Step 5 — Validate every assertion
  Call slang_lint on each assertion.  Fix any failures before proceeding.

Step 6 — Final output
  When all assertions pass lint, emit your final answer containing ONLY
  the validated SVA code block, followed by {_DONE_SENTINEL}.

Do NOT make up signal names — always confirm them with yosys_extract or
signal_map_lookup first.

═══════════════════════════════════════════════════════════
RESPONSE STYLE — STRICT
═══════════════════════════════════════════════════════════
- DO NOT narrate your reasoning. NEVER reply with phrases like
  "Let me look at…", "I'll start by…", "Let me first…", "I'll gather more
  information", "Now I'll…", or any other meta-commentary about what you
  are about to do.
- Every assistant turn must be EITHER a tool call (calling
  rtl_retrieve / doc_retrieve / yosys_extract / signal_map_lookup /
  slang_lint) OR an answer that contains at least one `assert` statement.
- A response with no tool call and no `assert` keyword is treated as a
  protocol violation and you will be re-prompted until you commit.
- If the design is signal-poor or hierarchical (most logic in submodules),
  emit assertions on whatever surface signals you can see — bit-width
  checks, reset values, port passthroughs, valid/ready handshakes —
  rather than asking for "more details." There are no more details. Use
  the design hierarchy and signal-map summary already given.
{backend_hint}{rtl_facts_reminders}
"""


# ---------------------------------------------------------------------------
# Planning prompt template (Improvement 4)
# ---------------------------------------------------------------------------

_PLAN_SENTINEL = "<<PLAN_COMPLETE>>"


def _build_planning_prompt(
    hierarchy: str,
    signal_map_summary: str,
    graph_summary_text: str = "",
    full_rtl_context: Optional[str] = None,
    rtl_facts_block: str = "",
    rtl_facts_reminders: str = "",
    backend_hint: str = "",
) -> str:
    """
    Build a system prompt for the planning phase.

    Reuses the same context sections (hierarchy, graph, signal map, RTL) but
    replaces the WORKFLOW with planning-specific instructions that ask the
    model to output a JSON plan rather than SVA code.
    """
    # Conditionally build the graph section.
    graph_section = ""
    if graph_summary_text:
        graph_section = f"""
═══════════════════════════════════════════════════════════
DESIGN GRAPH (always available — no tool call needed)
═══════════════════════════════════════════════════════════
{graph_summary_text}
"""

    # Conditionally build the full RTL section.
    rtl_section = ""
    if full_rtl_context:
        rtl_section = f"""
═══════════════════════════════════════════════════════════
COMPLETE RTL SOURCE CODE (all files — no retrieval needed)
═══════════════════════════════════════════════════════════
{full_rtl_context}
"""

    # Optional RTL facts context card (Stage 2 prompt augmentation).
    facts_section = ""
    if rtl_facts_block:
        facts_section = f"""
═══════════════════════════════════════════════════════════
RTL FACTS (parser-extracted, authoritative for this design)
═══════════════════════════════════════════════════════════
{rtl_facts_block}
"""

    return f"""You are an expert hardware verification engineer specialising in \
SystemVerilog Assertions (SVA).

Your task is to CREATE A PLAN for SVA assertion generation.  Do NOT write \
any SVA code yet — only produce a structured plan.
{facts_section}

═══════════════════════════════════════════════════════════
DESIGN HIERARCHY
═══════════════════════════════════════════════════════════
{hierarchy}
{graph_section}
═══════════════════════════════════════════════════════════
SIGNAL MAP SUMMARY
═══════════════════════════════════════════════════════════
{signal_map_summary}
{rtl_section}
═══════════════════════════════════════════════════════════
TOOLS AVAILABLE
═══════════════════════════════════════════════════════════
You have five tools: rtl_retrieve, doc_retrieve, yosys_extract,
signal_map_lookup, and slang_lint.  Use them to gather information
about the design before producing your plan.

═══════════════════════════════════════════════════════════
PLANNING INSTRUCTIONS
═══════════════════════════════════════════════════════════
1. Use doc_retrieve to understand what properties the design must satisfy.
2. Use rtl_retrieve / yosys_extract to identify exact signal names and widths.
3. Output a JSON array inside a ```json ... ``` code fence.

Each entry in the array should be:
{{
  "id": <integer>,
  "category": "<functional_correctness|interface_compliance|data_integrity|key_schedule>",
  "property": "<human-readable description of what to assert>",
  "relevant_signals": ["signal1", "signal2", ...],
  "relevant_modules": ["ModuleName", ...],
  "assertion_sketch": "<rough draft of the assert statement>"
}}

IMPORTANT:
- Use ONLY immediate assertions: assert (...) else $error("...");
- Do NOT use "assert property" — this is a combinational design with no clock.
- Aim for 10-20 planned assertions covering all four categories.

After the JSON array, write {_PLAN_SENTINEL}.
{backend_hint}{rtl_facts_reminders}
"""


# ---------------------------------------------------------------------------
# SVAAgent
# ---------------------------------------------------------------------------

class SVAAgent:
    """
    ReAct agent that drives the SVA generation loop.

    Parameters
    ----------
    config : PipelineConfig
        All path and model configuration.
    rtl_retriever : FAISSRetriever or HybridRetriever
        Pre-built index over RTL source files.
    doc_retriever : FAISSRetriever or HybridRetriever
        Pre-built index over documentation files.
    hierarchy : str
        Full text of hierarchy.txt (injected into system prompt).
    signal_map : dict
        Parsed content of signal_map.json.
    design_graph : DesignGraph, optional
        Pre-built Yosys design graph for instant lookups.
    graph_summary_text : str
        Compact text summary of the design graph for the system prompt.
    full_rtl_context : str, optional
        When provided, all RTL source is injected into the system prompt.
    """

    def __init__(
        self,
        config: PipelineConfig,
        rtl_retriever,
        doc_retriever,
        hierarchy: str,
        signal_map: Dict[str, Any],
        design_graph: Optional[Any] = None,
        graph_summary_text: str = "",
        full_rtl_context: Optional[str] = None,
        facts: Optional[Any] = None,
    ):
        self.config = config
        self.rtl_retriever = rtl_retriever

        # Trace logger — records every model interaction for post-mortem analysis.
        trace_path = str(Path(config.output_sva_file).with_suffix("")) + "_trace.json"
        from .trace_logger import TraceLogger
        self.trace = TraceLogger(trace_path)
        self._current_phase = "init"
        self._global_step = 0
        self.doc_retriever = doc_retriever
        self.signal_map = signal_map
        self.design_graph = design_graph
        self.full_rtl_context = full_rtl_context
        self.facts = facts

        # Clock/reset from DesignInfo (for AST-guided assertion generation).
        self._clock_signal = None
        self._reset_signal = None

        # Build a compact plain-text summary of the signal map so it can live
        # in the system prompt without being enormous.
        signal_map_summary = self._summarise_signal_map(signal_map)

        # Stage 2: format the RTL facts context card if enabled.
        # Default off so existing pipelines stay byte-identical to baseline.
        rtl_facts_block = ""
        rtl_facts_reminders_block = ""
        if getattr(config, "use_rtl_facts", False) and facts is not None:
            from .rtl_facts import format_facts_for_prompt
            denylist: Optional[List[str]] = None
            if getattr(config, "use_hallucination_denylist", False):
                denylist = self._load_hallucination_denylist()

            # Stage 3: use module-scoped facts for the prompt if available.
            prompt_facts = facts
            module_mode = getattr(config, "module_facts_mode", "off")
            if module_mode != "off" and config.top_module:
                scope_depth = getattr(config, "module_scope_depth", 0)
                scoped = facts.for_module(config.top_module, depth=scope_depth)
                if scoped is not facts:
                    prompt_facts = scoped
                    logger.info(
                        "Using module-scoped facts for prompt: %s "
                        "(depth=%d, %d signals)",
                        config.top_module, scope_depth,
                        len(scoped.all_signals),
                    )

            rtl_facts_block = format_facts_for_prompt(
                prompt_facts,
                soft_token_budget=config.rtl_facts_soft_budget,
                hard_token_budget=config.rtl_facts_hard_budget,
                denylist=denylist,
                signal_map=signal_map,
                # Fix 6: pass unscoped facts so the formatter can add
                # submodule drive-kind annotations when module-scoped.
                full_facts=facts if prompt_facts is not facts else None,
            )
            if rtl_facts_block:
                logger.info(
                    "RTL facts injected into system prompt (~%d tokens)",
                    len(rtl_facts_block) // 4,
                )

            # Instruction sandwich: compact reminders for the END of the
            # prompt. Reinforces the highest-leverage rules via recency bias.
            from .rtl_facts import format_facts_reminders
            rtl_facts_reminders_block = format_facts_reminders(
                facts, denylist=denylist,
            )

        # Backend-aware formatting hint. Disabled after A/B testing showed
        # that aggressive "IMPORTANT: You MUST..." phrasing paradoxically
        # increased bare-fragment errors on GGUF models (8 → 18). The
        # gentler bad-pattern example in the facts block is sufficient.
        backend_hint = ""

        # System prompt is constant across all turns.
        self.system_prompt = _build_system_prompt(
            hierarchy, signal_map_summary,
            graph_summary_text=graph_summary_text,
            full_rtl_context=full_rtl_context,
            reject_assert_property=config.reject_assert_property,
            rtl_facts_block=rtl_facts_block,
            rtl_facts_reminders=rtl_facts_reminders_block,
            backend_hint=backend_hint,
        )

        # Planning prompt (Improvement 4) — same context, different workflow.
        self.planning_prompt = _build_planning_prompt(
            hierarchy, signal_map_summary,
            graph_summary_text=graph_summary_text,
            full_rtl_context=full_rtl_context,
            rtl_facts_block=rtl_facts_block,
            rtl_facts_reminders=rtl_facts_reminders_block,
            backend_hint=backend_hint,
        )

        # Create the LLM backend — skip if ast_only (no LLM needed).
        if config.ast_only and config.use_ast_assertions:
            self.backend = None
            logger.info("AST-only mode — skipping LLM loading.")
        else:
            from .backends import create_backend
            self.backend = create_backend(config)

            # Stage 2.5: grammar-constrained generation.
            # Generate a GBNF grammar from facts and pass it to the backend.
            if (getattr(config, "use_grammar_constraints", False)
                    and facts is not None
                    and hasattr(self.backend, "set_grammar")):
                from .grammar import generate_sva_grammar_from_facts
                gbnf = generate_sva_grammar_from_facts(facts)
                if gbnf:
                    self.backend.set_grammar(gbnf)

            # Cache warming: for OpenAI-compatible backends (Ollama, vLLM,
            # SGLang), send a minimal request with the system prompt so the
            # server's prefix cache is warm before the first real batch.
            # The local HF backend doesn't benefit from this (no server).
            if config.backend == "openai" and rtl_facts_block:
                self._warm_prefix_cache()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate_assertions(self, task: str) -> str:
        """
        Generate SVA assertions for the given verification task.

        When ``config.use_plan_execute`` is True, runs a two-phase approach:
          1. Planning phase — short ReAct loop that outputs a JSON plan.
          2. Execution phase — per-assertion generation with focused prompts.

        Falls back to direct (single-loop) generation if planning fails or
        the feature is disabled.

        Returns
        -------
        str
            The final SVA code.
        """
        # Strategy 0: Naive baseline — single LLM call with full RTL + docs.
        # Used to measure the token cost of a one-shot LLM-for-SVA approach
        # against the optimised pipeline. All AST extraction, facts
        # injection, and skeleton batching are skipped.
        if getattr(self.config, "naive_baseline", False):
            return self._naive_generate(task)

        # Strategy 1: AST-guided (deterministic, fastest, most reliable).
        if self.config.use_ast_assertions:
            ast_result = self._ast_guided_generate(task)
            if ast_result:
                return ast_result

        # If ast_only and AST produced nothing, we can't proceed without LLM.
        if self.backend is None:
            logger.error(
                "AST-only mode but AST extraction found no patterns. "
                "No LLM available for fallback."
            )
            return ""

        # Strategy 2: Plan-then-execute (LLM-based).
        if self.config.use_plan_execute:
            plan = self.plan_assertions(task)
            if plan:
                return self.execute_plan(plan)
            logger.warning(
                "Planning phase returned empty plan — "
                "falling back to direct generation."
            )

        # Strategy 3: Direct ReAct loop (last resort).
        return self._direct_generate(task)

    # ------------------------------------------------------------------
    # AST-guided assertion generation (deterministic)
    # ------------------------------------------------------------------

    def _ast_guided_generate(self, task: str) -> str:
        """
        Generate assertions deterministically by extracting RTL patterns
        via regex and applying assertion templates.

        If ast_only is True, returns the skeletons directly.
        Otherwise, passes them to the LLM for semantic enrichment.
        """
        from .ast_assertions import (
            generate_ast_assertions,
            format_skeletons_as_sva,
            format_skeletons_for_llm,
        )

        self._current_phase = "ast_extraction"
        logger.info("=== AST-guided assertion generation ===")

        # Get RTL source text — from context injection or by reading files.
        source = self.full_rtl_context
        if not source and self.config.rtl_dir:
            import os
            import fnmatch
            from pathlib import Path

            exclude_patterns = getattr(
                self.config, "ast_exclude_patterns", []
            )

            def _is_excluded(fname: str) -> bool:
                for pat in exclude_patterns:
                    if fnmatch.fnmatch(fname, pat):
                        return True
                return False

            skipped: List[str] = []

            def _read_all_rtl(filter_to_top: bool) -> List[str]:
                """Walk rtl_dir and return file contents.  When
                ``filter_to_top`` is True, only the configured
                ``top_file`` is read; otherwise every ``.v``/``.sv``
                file is included (still honouring exclusions)."""
                acc: List[str] = []
                for root, _, files in os.walk(self.config.rtl_dir):
                    for fname in sorted(files):
                        if Path(fname).suffix not in {'.v', '.sv'}:
                            continue
                        if filter_to_top and top_file and \
                                fname != top_file and \
                                not top_file.endswith(fname):
                            continue
                        if _is_excluded(fname):
                            skipped.append(fname)
                            continue
                        fpath = os.path.join(root, fname)
                        try:
                            with open(fpath, 'r', encoding='utf-8',
                                      errors='ignore') as fh:
                                acc.append(fh.read())
                        except OSError:
                            pass
                return acc

            top_file = getattr(self.config, 'top_file', '')
            top_file_was_used = bool(top_file)
            parts = _read_all_rtl(filter_to_top=True)
            if not parts:
                # top_file filter matched nothing on disk — fall back to
                # the full directory.  The post-extract wrapper-detect
                # below covers the orthogonal case where top_file does
                # exist but is a structural wrapper with no logic.
                parts = _read_all_rtl(filter_to_top=False)
                top_file_was_used = False
            source = "\n\n".join(parts)
            if source:
                logger.info("AST: loaded %d chars from RTL files.", len(source))
            if skipped:
                logger.info(
                    "AST: excluded %d stub file(s) from extraction: %s",
                    len(skipped), ", ".join(sorted(set(skipped))),
                )

        if not source:
            logger.warning("AST: no RTL source available.")
            return ""

        # Build the "allowed signals" set for trivial-restatement filtering.
        # Includes: signal_map keys (documented interface signals) + any
        # signals from facts.all_signals that were also in signal_map.
        # This focuses trivial AST assertions on signals the LLM/user
        # cares about, not internal implementation details.
        allowed_signals: Optional[Set[str]] = None
        if getattr(self.config, "ast_skip_trivial_internal", False):
            allowed_signals = set(self.signal_map.keys()) if self.signal_map else set()
            # Also include the bare last component of hierarchical names.
            for name in list(allowed_signals):
                if "." in name:
                    allowed_signals.add(name.split(".")[-1])

        # Extract raw patterns and skeletons separately so we can export
        # the raw patterns to the analysis directory for debugging.
        from .ast_assertions import extract_patterns
        ast_patterns = extract_patterns(
            source,
            clock=getattr(self, '_clock_signal', None),
            reset=getattr(self, '_reset_signal', None),
        )

        # Wrapper-detection fallback: if the user configured a top_file
        # that turned out to be a structural wrapper (mostly submodule
        # instantiations, little/no behavioural logic), re-scan the
        # entire rtl_dir so the rest of the pipeline has skeletons to
        # work with. Two trigger conditions:
        #   (a) the top file produced *zero* patterns — e.g.,
        #       NV_NVDLA_cdma.v / rubik.v / csc.v (pure structural
        #       wrappers).
        #   (b) the top file produced a *small but non-zero* count of
        #       patterns AND has many submodule instantiations AND its
        #       pattern density is low — e.g., NV_NVDLA_cdp.v (19 pat,
        #       9 instances) and NV_NVDLA_pdp.v (58 pat, 8 instances)
        #       which previously slipped through (a) because their
        #       monitor-register patterns put them above 0.
        # Without this, those designs ship 3 LLM-bound skeletons and
        # 5–6 final assertions across thousands of design events.
        n_submod_inst = len(_SUBMODULE_INST_RE.findall(source))
        density_per_10k = (
            len(ast_patterns) * 10_000.0 / max(len(source), 1)
        )
        is_zero = (not ast_patterns)
        is_low_density_wrapper = (
            len(ast_patterns) > 0
            and n_submod_inst >= 4
            and density_per_10k < 10.0
        )
        if ((is_zero or is_low_density_wrapper)
                and top_file_was_used
                and self.config.rtl_dir):
            reason = (
                "0 patterns" if is_zero
                else (f"{len(ast_patterns)} patterns + "
                      f"{n_submod_inst} submodule instances "
                      f"(density={density_per_10k:.2f}/10K chars)")
            )
            logger.warning(
                "AST: top_file %s looks like a structural wrapper "
                "(%s). Re-scanning full rtl_dir.",
                getattr(self.config, 'top_file', ''), reason,
            )
            parts = _read_all_rtl(filter_to_top=False)
            source = "\n\n".join(parts)
            if source:
                logger.info(
                    "AST: re-loaded %d chars from full rtl_dir.",
                    len(source),
                )
                ast_patterns = extract_patterns(
                    source,
                    clock=getattr(self, '_clock_signal', None),
                    reset=getattr(self, '_reset_signal', None),
                )
                logger.info(
                    "AST: re-extraction yielded %d patterns.",
                    len(ast_patterns),
                )

        skeletons = generate_ast_assertions(
            source=source,
            clock=getattr(self, '_clock_signal', None),
            reset=getattr(self, '_reset_signal', None),
            is_combinational=self.config.reject_assert_property or False,
            max_case_branches=self.config.ast_max_case_branches,
            allowed_signals=allowed_signals,
            skip_trivial_internal=getattr(
                self.config, "ast_skip_trivial_internal", False,
            ),
        )

        # Expose to main.py for analysis export.
        self._ast_patterns = ast_patterns
        self._ast_skeletons = skeletons

        # Write analysis artifacts for this AST extraction.
        tracer = getattr(self, "_analysis_tracer", None)
        try:
            from .analysis_export import (
                resolve_analysis_dir, export_ast_patterns, export_ast_skeletons,
            )
            _analysis_dir = resolve_analysis_dir(self.config)
            if _analysis_dir:
                export_ast_patterns(_analysis_dir, ast_patterns)
                export_ast_skeletons(_analysis_dir, skeletons)
        except Exception as exc:
            logger.debug("AST analysis export failed: %s", exc)

        # Record AST extraction and skeleton counts in the tracer.
        if tracer is not None:
            try:
                tracer.record(
                    "0-extract", "ast_pattern_extraction",
                    action="extract", delta=len(ast_patterns),
                    description="Raw AST patterns from RTL",
                )
                skel_delta = len(skeletons) - len(ast_patterns)
                tracer.record(
                    "0-extract", "skeleton_generation",
                    action="add" if skel_delta > 0 else (
                        "remove" if skel_delta < 0 else "transform"
                    ),
                    delta=skel_delta,
                    description="Patterns -> assertion skeletons",
                )
            except Exception:
                pass

        if not skeletons:
            logger.warning("AST extraction found no patterns.")
            return ""

        self.trace.log_step(
            phase="ast_extraction",
            step=self._global_step + 1,
            model_output=f"Extracted {len(skeletons)} assertion skeletons",
            assertions_generated=len(skeletons),
            notes=f"patterns: {len(skeletons)}",
        )

        if self.config.ast_only:
            logger.info(
                "AST-only mode: returning %d skeletons directly.",
                len(skeletons),
            )
            return format_skeletons_as_sva(skeletons)

        # Enrich with LLM — single focused call.
        return self._llm_enrich_skeletons(skeletons, task)

    def _llm_enrich_skeletons(
        self,
        skeletons: list,
        task: str,
    ) -> str:
        """
        Validate AST-generated skeletons against the design documentation.

        Splits skeletons into:
        - TRIVIAL (case branches, wire passthroughs, simple assigns) →
          output directly without LLM validation
        - COMPLEX (ternary muxes, sequential logic, comparisons) →
          send to LLM in batches of ~20 for spec validation

        This avoids token limit issues on large designs while ensuring
        complex logic is validated against the documentation.
        """
        from .ast_assertions import format_skeletons_for_llm, format_skeletons_as_sva

        self._current_phase = "spec_validation"

        # Cluster-and-compact path: when enabled, templates are grouped,
        # bit-replicated families are compacted into packed assertions,
        # and only one representative per cluster goes through LLM
        # spec-validation. Siblings emit directly with a back-reference
        # tag. Defaults preserve the original trivial/complex split.
        use_clustering = bool(getattr(
            self.config, "ast_use_clustering", False
        ))

        if use_clustering and getattr(self, "_ast_patterns", None):
            from .ast_clustering import cluster_and_compact, units_to_skeletons
            cluster_min = int(getattr(
                self.config, "ast_cluster_min_compact_size", 5
            ))
            cluster_depth = int(getattr(
                self.config, "ast_cluster_max_depth", 4
            ))
            value_min = int(getattr(
                self.config, "ast_value_cluster_min_size", 2
            ))
            enable_compaction = bool(getattr(
                self.config, "ast_enable_compaction", True
            ))
            enable_value = bool(getattr(
                self.config, "ast_enable_value_clustering", True
            ))

            # Cluster the SAME filtered pattern set that
            # generate_ast_assertions uses to produce skeletons —
            # otherwise we'd pack hallucinated internal signals into
            # the compacted-mux assertions. Mirrors the filter logic
            # at ast_assertions.py:683-733 (case-overflow drop +
            # skip_trivial_internal).
            patterns_for_cluster = list(self._ast_patterns)
            max_cb = int(getattr(self.config,
                                 "ast_max_case_branches", 50))
            if max_cb > 0:
                from collections import Counter as _Cnt
                cb_counts: Dict[str, int] = {}
                for p in patterns_for_cluster:
                    if p.pattern_type == "case_branch" and p.selector:
                        cb_counts[p.selector] = cb_counts.get(
                            p.selector, 0) + 1
                patterns_for_cluster = [
                    p for p in patterns_for_cluster
                    if not (p.pattern_type == "case_branch"
                            and p.selector
                            and cb_counts.get(p.selector, 0) > max_cb)
                ]
            if getattr(self.config, "ast_skip_trivial_internal", False):
                # Mirror _ast_guided_generate's allowed_signals build —
                # signal_map keys + their bare-identifier last component.
                local_allowed: Set[str] = set()
                if self.signal_map:
                    local_allowed = set(self.signal_map.keys())
                    for name in list(local_allowed):
                        if "." in name:
                            local_allowed.add(name.split(".")[-1])
                if local_allowed:
                    _TRIV = {"direct_assign", "wire_passthrough",
                             "comb_comparison"}
                    kept = []
                    for p in patterns_for_cluster:
                        if p.pattern_type in _TRIV:
                            base = re.sub(r"\[.*?\]|\{.*?\}", "",
                                          p.lhs).strip()
                            base = base.split(",")[0].strip()
                            if base and base not in local_allowed:
                                continue
                        kept.append(p)
                    patterns_for_cluster = kept
            logger.info(
                "AST clustering input: %d patterns (filtered from %d raw).",
                len(patterns_for_cluster), len(self._ast_patterns),
            )

            result = cluster_and_compact(
                patterns_for_cluster,
                min_compact_size=cluster_min,
                max_depth=cluster_depth,
                enable_compaction=enable_compaction,
                enable_value_clustering=enable_value,
                value_cluster_min_size=value_min,
            )
            llm_skeletons, direct_skeletons = units_to_skeletons(result)
            logger.info(
                "=== Skeleton routing (clustered): "
                "%d to LLM (compacted+representative+individual), "
                "%d direct (siblings) ===",
                len(llm_skeletons), len(direct_skeletons),
            )
            # Direct path emits siblings with their tag prepended.
            all_assertions = [format_skeletons_as_sva(direct_skeletons)] \
                if direct_skeletons else []
            complex_skeletons = llm_skeletons
            if not complex_skeletons:
                logger.info("No skeletons left for LLM after clustering.")
                return "\n\n".join(all_assertions)
        else:
            # Legacy path: split into trivial (direct output) and
            # complex (LLM validation).
            trivial_types = {"case_branch", "wire_passthrough", "direct_assign"}
            trivial = [s for s in skeletons if s.pattern_type in trivial_types]
            complex_skeletons = [s for s in skeletons
                                 if s.pattern_type not in trivial_types]
            logger.info(
                "=== Skeleton split: %d trivial (direct) + %d complex (LLM) ===",
                len(trivial), len(complex_skeletons),
            )
            # Start with trivial assertions as-is.
            all_assertions = [format_skeletons_as_sva(trivial)] if trivial else []
            if not complex_skeletons:
                logger.info("No complex skeletons — skipping LLM validation.")
                return "\n\n".join(all_assertions)

        # Batch the complex skeletons for LLM validation.  Bumped from
        # 20 to 40 so designs whose clustering yields hundreds of
        # LLM-bound skeletons (cdma's wrapper-fallback recovers ~931)
        # finish in roughly half the LLM round-trips.  At ~5 lines per
        # skeleton, 40 fits the qwen3:14b 32K context with headroom for
        # the system prompt and per-batch facts injection.
        batch_size = 40
        batches = [
            complex_skeletons[i:i + batch_size]
            for i in range(0, len(complex_skeletons), batch_size)
        ]

        logger.info(
            "Sending %d complex skeleton(s) to LLM in %d batch(es).",
            len(complex_skeletons), len(batches),
        )

        for batch_idx, batch in enumerate(batches, 1):
            logger.info("--- Batch %d/%d (%d skeletons) ---", batch_idx, len(batches), len(batch))

            skeleton_text = format_skeletons_for_llm(batch)

            # Stage 2b: per-batch fact injection.
            # Only signal-specific facts are injected. The "already-covered"
            # list was tested and found counterproductive — it suppressed
            # LLM output (104 → 70 assertions) and added token pressure
            # that increased bare-fragment errors. Removed after A/B test.
            batch_context = ""
            if (self.facts is not None
                    and getattr(self.config, "use_rtl_facts", False)
                    and getattr(self.config, "use_per_batch_facts", False)):
                from .rtl_facts import format_batch_facts
                batch_signals = self._extract_batch_signals(batch)
                batch_facts = format_batch_facts(self.facts, batch_signals)

                if batch_facts:
                    batch_context = batch_facts + "\n\n"

            validation_prompt = (
                f"{batch_context}"
                "You have TWO sources of information:\n\n"
                "SOURCE 1 — DESIGN DOCUMENTATION (ground truth):\n"
                "The documentation in your context describes what the design "
                "SHOULD do.\n\n"
                "SOURCE 2 — AST-EXTRACTED SKELETONS (from the RTL):\n"
                f"{skeleton_text}\n\n"
                "For EACH skeleton:\n"
                "  a) Find the corresponding spec requirement\n"
                "  b) If RTL matches spec → keep, improve $error message\n"
                "  c) If RTL doesn't match spec → modify to match SPEC, "
                "add comment '// POTENTIAL RTL BUG'\n"
                "  d) If no spec requirement → keep with "
                "'// structural check'\n\n"
                "Also add new assertions for spec requirements NOT covered.\n"
                "But do NOT add unconditional output value assertions — "
                "case branch assertions already handle per-value checks.\n"
                "Focus on: mutual exclusivity, protocol invariants, "
                "mode-dependent behaviour, and cross-signal constraints.\n\n"
                "Output assertions in a ```systemverilog``` fence.\n"
                f"End with {_DONE_SENTINEL}."
            )

            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": validation_prompt},
            ]

            max_steps = self.config.max_execution_steps_per_assertion * 2
            batch_text = ""

            for step in range(1, max_steps + 1):
                logger.info("  Enrichment step %d/%d", step, max_steps)
                response_text, tool_calls = self._step(messages)

                if tool_calls:
                    messages.append({"role": "assistant", "content": response_text})
                    for call in tool_calls:
                        observation = self._dispatch(call)
                        messages.append({
                            "role": "tool",
                            "name": call.get("name", ""),
                            "content": observation,
                        })
                else:
                    batch_text = response_text
                    break

                if _DONE_SENTINEL in response_text:
                    batch_text = response_text
                    break
            else:
                messages.append({
                    "role": "user",
                    "content": (
                        "Output the assertions now in a "
                        "```systemverilog``` block. "
                        f"End with {_DONE_SENTINEL}."
                    ),
                })
                batch_text, _ = self._step(messages)

            batch_sva = self._extract_sva(batch_text)
            if batch_sva.strip():
                all_assertions.append(batch_sva)
                n_batch_asserts = batch_sva.count("assert")
                logger.info(
                    "  Batch %d: %d assertions from LLM.",
                    batch_idx, n_batch_asserts,
                )
                _t = getattr(self, "_analysis_tracer", None)
                if _t is not None:
                    try:
                        _t.record(
                            "1-generation", f"llm_batch_{batch_idx}",
                            action="add", delta=n_batch_asserts,
                            description=f"LLM enriched {len(batch)} skeletons "
                                        f"-> {n_batch_asserts} assertions",
                        )
                    except Exception:
                        pass
            else:
                # LLM produced nothing — use raw skeletons for this batch.
                logger.warning("  Batch %d: LLM returned empty — using AST skeletons.", batch_idx)
                all_assertions.append(format_skeletons_as_sva(batch))

        return "\n\n".join(all_assertions)

    # ------------------------------------------------------------------
    # Two-pass functional assertion generation (legacy, kept as fallback)
    # ------------------------------------------------------------------

    def _extract_case_branches(self, task: str) -> str:
        """
        Pass 1: Ask the model to list all case branches, conditional
        assignments, and mux selects as a table — no assertions yet.

        This is easier than composing assertions because the model only
        needs to read and summarise the RTL, not write SV syntax.

        Returns the branch table as a string, or empty string on failure.
        """
        extraction_prompt = (
            "Read the RTL source code in your context carefully.  "
            "List ALL case branches and conditional assignments in a table.\n\n"
            "For each branch, write one row:\n"
            "  CONDITION | OUTPUT_SIGNAL = EXPECTED_VALUE | NOTES\n\n"
            "Example format:\n"
            "  {is_8bit, in_code} == 4'b0001 | out_data = {~src_data[15], src_data} | +1x, 16-bit\n"
            "  {is_8bit, in_code} == 4'b0011 | out_data = {~src_data[15], src_data[14:0], 1'b0} | +2x, 16-bit\n\n"
            "Also note the out_inv value for each branch.\n"
            "List EVERY branch in the case statement.  Do not write any "
            "assertions — just the table.\n"
            f"End with {_DONE_SENTINEL}."
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": extraction_prompt},
        ]

        self._current_phase = "extraction"
        logger.info("=== Pass 1: Extracting case branches ===")

        max_steps = self.config.max_planning_steps
        final_text = ""
        for step in range(1, max_steps + 1):
            logger.info("--- Extraction step %d/%d ---", step, max_steps)
            response_text, tool_calls = self._step(messages)

            if tool_calls:
                messages.append({"role": "assistant", "content": response_text})
                for call in tool_calls:
                    observation = self._dispatch(call)
                    messages.append({
                        "role": "tool",
                        "name": call.get("name", ""),
                        "content": observation,
                    })
            else:
                final_text = response_text
                break

            if _DONE_SENTINEL in response_text:
                final_text = response_text
                break
        else:
            messages.append({
                "role": "user",
                "content": f"Output the branch table now. End with {_DONE_SENTINEL}.",
            })
            final_text, _ = self._step(messages)

        final_text = final_text.replace(_DONE_SENTINEL, "").strip()

        if "|" in final_text and len(final_text) > 100:
            logger.info("Pass 1 extracted %d chars of branch data.", len(final_text))
            return final_text
        else:
            logger.warning("Pass 1 did not produce a valid branch table.")
            return ""

    def _generate_from_branches(self, branches: str, task: str) -> str:
        """
        Pass 2: Given the branch table from pass 1, ask the model to
        convert each row into a proper assertion using |-> implication.

        This is easier because the model already has the conditions and
        expected values — it just needs to format them as SVA.
        """
        conversion_prompt = (
            "You previously extracted the following case branches from the RTL:\n\n"
            f"{branches}\n\n"
            "Now convert EACH row into a SystemVerilog immediate assertion.\n"
            "Use this exact pattern for every assertion:\n\n"
            "  assert (CONDITION |-> (OUTPUT == EXPECTED_VALUE)) "
            'else $error("description");\n\n'
            "Where |-> means 'implies' — if the condition is true, the "
            "output must equal the expected value.\n\n"
            "Also add:\n"
            "- Width checks: assert ($bits(signal) == N) for each port\n"
            "- out_inv checks: for each branch, check out_inv is 0 (positive) or 1 (negative)\n\n"
            "Write ONLY the assertions, no other text.\n"
            "Validate each with slang_lint.\n"
            f"End with {_DONE_SENTINEL}."
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": conversion_prompt},
        ]

        self._current_phase = "generation"
        logger.info("=== Pass 2: Generating assertions from branches ===")

        max_steps = self.config.max_execution_steps_per_assertion * 3
        final_text = ""
        for step in range(1, max_steps + 1):
            logger.info("--- Generation step %d/%d ---", step, max_steps)
            response_text, tool_calls = self._step(messages)

            if tool_calls:
                messages.append({"role": "assistant", "content": response_text})
                for call in tool_calls:
                    observation = self._dispatch(call)
                    messages.append({
                        "role": "tool",
                        "name": call.get("name", ""),
                        "content": observation,
                    })
            else:
                final_text = response_text
                break

            if _DONE_SENTINEL in response_text:
                final_text = response_text
                break
        else:
            messages.append({
                "role": "user",
                "content": (
                    "Output all assertions now in a ```systemverilog``` block. "
                    f"End with {_DONE_SENTINEL}."
                ),
            })
            final_text, _ = self._step(messages)

        result = self._extract_sva(final_text)
        if result.strip():
            logger.info("Pass 2 produced %d chars of assertions.", len(result))
        else:
            logger.warning("Pass 2 produced no assertions.")
        return result

    # ------------------------------------------------------------------
    # Plan-then-execute (Improvement 4)
    # ------------------------------------------------------------------

    def plan_assertions(self, task: str) -> List[Dict[str, Any]]:
        """
        Planning phase: short ReAct loop that outputs a JSON plan.

        The model uses tools to gather design information, then emits a
        JSON array describing what assertions to generate.  Each entry has:
        id, category, property, relevant_signals, relevant_modules,
        assertion_sketch.

        Returns
        -------
        list of dict
            The parsed plan, or an empty list if parsing fails.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.planning_prompt},
            {"role": "user",   "content": task},
        ]

        max_steps = self.config.max_planning_steps
        self._current_phase = "planning"
        logger.info("=== Planning phase (max %d steps) ===", max_steps)

        final_text = ""
        for step in range(1, max_steps + 1):
            logger.info("--- Planning step %d/%d ---", step, max_steps)

            response_text, tool_calls = self._step(messages)

            if tool_calls:
                messages.append({"role": "assistant", "content": response_text})
                for call in tool_calls:
                    observation = self._dispatch(call)
                    messages.append({
                        "role": "tool",
                        "name": call.get("name", ""),
                        "content": observation,
                    })
            else:
                final_text = response_text
                break

            if _PLAN_SENTINEL in response_text:
                final_text = response_text
                break
        else:
            # Force the plan output.
            messages.append({
                "role": "user",
                "content": (
                    "Step limit reached. Output the JSON plan now inside "
                    f"a ```json``` fence. End with {_PLAN_SENTINEL}."
                ),
            })
            final_text, _ = self._step(messages)  # ignore tool_calls for forced output

        return self._parse_plan(final_text)

    def execute_plan(self, plan: List[Dict[str, Any]]) -> str:
        """
        Execution phase: generate one assertion per plan entry.

        For each entry, starts a fresh short conversation with a focused
        prompt for that single assertion.  The model can still use tools
        (especially slang_lint) to validate its output.

        Returns
        -------
        str
            Concatenated SVA code for all planned assertions.
        """
        all_assertions: List[str] = []
        max_steps = self.config.max_execution_steps_per_assertion

        self._current_phase = "execution"
        logger.info(
            "=== Execution phase: %d planned assertion(s), "
            "max %d steps each ===",
            len(plan), max_steps,
        )

        for entry in plan:
            entry_id = entry.get("id", "?")
            category = entry.get("category", "unknown")
            prop = entry.get("property", "")
            signals = entry.get("relevant_signals", [])
            modules = entry.get("relevant_modules", [])
            sketch = entry.get("assertion_sketch", "")

            logger.info(
                "--- Executing assertion %s: %s ---", entry_id, category
            )

            # Stage 2b: per-entry fact injection for execution phase.
            # Only signal-specific facts — no already-covered list (see
            # AST-guided path comment for rationale).
            exec_context = ""
            if (self.facts is not None
                    and getattr(self.config, "use_rtl_facts", False)
                    and getattr(self.config, "use_per_batch_facts", False)):
                from .rtl_facts import format_batch_facts
                entry_signals = set(signals) if signals else set()
                # Also extract signals from the sketch text.
                if sketch:
                    sketch_ids = set(re.findall(r'\b([a-zA-Z_]\w*)\b', sketch))
                    entry_signals |= sketch_ids
                entry_facts = format_batch_facts(self.facts, entry_signals)

                if entry_facts:
                    exec_context = entry_facts + "\n\n"

            exec_task = (
                f"{exec_context}"
                f"Generate and validate ONE SVA assertion.\n\n"
                f"  Category: {category}\n"
                f"  Property: {prop}\n"
                f"  Signals: {', '.join(signals) if signals else 'see context'}\n"
                f"  Modules: {', '.join(modules) if modules else 'see context'}\n"
                f"  Sketch: {sketch}\n\n"
                "Write the assertion using ONLY immediate assert syntax:\n"
                "  assert (...) else $error(\"...\");\n"
                "Do NOT use 'assert property'.\n"
                "Validate with slang_lint, fix if needed, then return ONLY "
                f"the final assertion. End with {_DONE_SENTINEL}."
            )

            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": exec_task},
            ]

            final_text = ""
            for step in range(1, max_steps + 1):
                response_text, tool_calls = self._step(messages)

                if tool_calls:
                    messages.append({"role": "assistant", "content": response_text})
                    for call in tool_calls:
                        observation = self._dispatch(call)
                        messages.append({
                            "role": "tool",
                            "name": call.get("name", ""),
                            "content": observation,
                        })
                else:
                    final_text = response_text
                    break

                if _DONE_SENTINEL in response_text:
                    final_text = response_text
                    break
            else:
                # Force output.
                messages.append({
                    "role": "user",
                    "content": (
                        "Step limit reached. Output the assertion now. "
                        f"End with {_DONE_SENTINEL}."
                    ),
                })
                final_text, _ = self._step(messages)  # ignore tool_calls for forced output

            sva = self._extract_sva(final_text)
            if sva.strip():
                # Prepend a category comment.
                comment = f"// {category}: {prop}"
                full_entry = f"{comment}\n{sva.strip()}"
                all_assertions.append(full_entry)
                logger.info("  -> assertion generated (%d chars)", len(sva))
            else:
                logger.warning(
                    "  -> empty output for assertion %s", entry_id
                )

        return "\n\n".join(all_assertions)

    def _naive_generate(self, task: str) -> str:
        """
        Single-prompt baseline: send the full RTL + docs to the LLM and
        ask for assertions in one shot. No AST, no facts, no scoping,
        no skeletons, no tool calls.

        If the response is truncated at ``max_new_tokens``, a single
        follow-up call is made to collect more assertions (all cost
        counted). The combined output goes through the same lint /
        dedup / validation pipeline as the optimised path.
        """
        self._current_phase = "naive_baseline"
        logger.info("=== Naive baseline generation (single-prompt) ===")

        if self.backend is None:
            logger.error("naive_baseline requires an LLM backend.")
            return ""

        # Gather RTL source. Prefer the full_rtl_context already loaded
        # by main.py; otherwise read files ourselves.
        rtl_text = self.full_rtl_context or ""
        if not rtl_text and self.config.rtl_dir:
            import os
            parts: List[str] = []
            for root, _, files in os.walk(self.config.rtl_dir):
                for fn in sorted(files):
                    if fn.endswith((".v", ".sv", ".svh")):
                        path = os.path.join(root, fn)
                        try:
                            with open(path, "r", encoding="utf-8",
                                      errors="ignore") as fh:
                                parts.append(
                                    f"// === {fn} ===\n" + fh.read()
                                )
                        except OSError:
                            continue
            rtl_text = "\n\n".join(parts)

        # Docs: the main.py context-injection path prepends them to
        # full_rtl_context. If that path wasn't taken, read doc files
        # here directly (naive baseline bypasses RAG by design).
        docs_text = getattr(self, "full_docs_context", "") or ""
        if not docs_text and getattr(self.config, "docs_dir", ""):
            import os
            doc_parts: List[str] = []
            for root, _, files in os.walk(self.config.docs_dir):
                for fn in sorted(files):
                    if fn.endswith((".md", ".txt")):
                        path = os.path.join(root, fn)
                        try:
                            with open(path, "r", encoding="utf-8",
                                      errors="ignore") as fh:
                                doc_parts.append(
                                    f"// === {fn} ===\n" + fh.read()
                                )
                        except OSError:
                            continue
            docs_text = "\n\n".join(doc_parts)

        # Build the single prompt. Keep it simple and representative of
        # what a researcher would write with no pipeline assistance.
        system_msg = (
            "You are an expert in SystemVerilog Assertions (SVA). "
            "Given an RTL module and its documentation, emit a "
            "comprehensive set of SystemVerilog assertions covering "
            "reset values, protocol correctness, data-path integrity, "
            "and functional properties. Output each assertion on its "
            "own line inside a ```systemverilog``` code fence. Do not "
            "call any tools. Do not emit explanatory prose outside the "
            "code fence."
        )

        user_parts = [f"Verification task:\n{task}\n"]
        if docs_text.strip():
            user_parts.append(f"\nDesign documentation:\n{docs_text}\n")
        if rtl_text.strip():
            user_parts.append(f"\nRTL source:\n{rtl_text}\n")
        user_parts.append(
            "\nGenerate the SVA assertions now. Output only the code "
            "fence; no commentary."
        )
        user_msg = "".join(user_parts)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        # Call the LLM (no tool definitions — baseline is one-shot).
        self._global_step += 1
        text, _, usage = self.backend.generate(messages, [])
        self.trace.log_step(
            phase=self._current_phase,
            step=self._global_step,
            model_output=text,
            usage=usage,
            notes="naive_baseline single prompt",
        )

        accumulated = [text]

        # If the response hit max_tokens, give the model one chance to
        # continue. This is still counted as naive-baseline cost.
        max_new = getattr(self.config, "max_new_tokens", 2048)
        if len(text) > 0 and usage.get("completion_tokens", 0) >= max_new - 8:
            logger.info(
                "Naive baseline: first call hit token limit "
                "(%d tokens) — requesting continuation.",
                usage.get("completion_tokens", 0),
            )
            follow_messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content":
                    "Continue. Emit additional SVA assertions inside a "
                    "```systemverilog``` code fence. Do not repeat any "
                    "assertion you already produced."},
            ]
            self._global_step += 1
            text2, _, usage2 = self.backend.generate(follow_messages, [])
            self.trace.log_step(
                phase=self._current_phase,
                step=self._global_step,
                model_output=text2,
                usage=usage2,
                notes="naive_baseline continuation",
            )
            accumulated.append(text2)

        combined = "\n\n".join(accumulated)
        extracted = self._extract_sva(combined)

        logger.info(
            "Naive baseline produced %d char(s) of raw SVA.",
            len(extracted),
        )
        return extracted

    def _direct_generate(self, task: str) -> str:
        """
        Original single-loop ReAct generation (fallback when plan-execute
        is disabled or when the planning phase returns an empty plan).
        """
        # Initialise conversation with system prompt + first user message.
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": task},
        ]

        final_answer = ""
        iteration = 0
        stall_count = 0  # consecutive turns with neither tool calls nor assertions

        self._current_phase = "direct"
        logger.info("Starting direct ReAct loop (max_iterations=%d) …", self.config.max_iterations)

        while iteration < self.config.max_iterations:
            iteration += 1
            logger.info("--- ReAct iteration %d ---", iteration)

            # ---- Generate one model response ----
            response_text, tool_calls = self._step(messages)
            logger.debug("Model output:\n%s", response_text)

            if tool_calls:
                stall_count = 0
                messages.append({
                    "role": "assistant",
                    "content": response_text,
                })

                for call in tool_calls:
                    observation = self._dispatch(call)
                    messages.append({
                        "role": "tool",
                        "name": call.get("name", ""),
                        "content": observation,
                    })

            else:
                # Stall detector: a model output with no tool call and no
                # `assert` keyword is the qwen3.6:35b-on-rubik failure
                # signature ("Let me look at…", "I'll gather more info…").
                # Don't accept it as a final answer — push back and re-ask.
                # Only treat empty-of-assertions output as "final" once we
                # have either an `assert` keyword or have exhausted
                # patience (3 consecutive stalls).
                has_assertion = bool(re.search(r"\bassert\b", response_text))
                if not has_assertion:
                    stall_count += 1
                    is_last_iter = iteration >= self.config.max_iterations
                    if stall_count <= 3 and not is_last_iter:
                        logger.warning(
                            "Stall detected (iter %d, stall %d/3): no tool call and "
                            "no `assert` keyword. Re-prompting for SVA output.",
                            iteration, stall_count,
                        )
                        messages.append({"role": "assistant", "content": response_text})
                        messages.append({
                            "role": "user",
                            "content": (
                                "STOP. Your previous reply contained no tool call and "
                                "no `assert` statement. Do NOT narrate your reasoning. "
                                "You have all the information you need from the design "
                                "hierarchy, signal-map summary, and any retrievals "
                                "already shown above.\n\n"
                                "Output SVA assertions NOW. Each emitted line must "
                                "begin with `assert` or `assert property`. Use only "
                                "signals from the signal map above. Cover bit-width "
                                "checks ($bits), reset-value checks, port-pass-through "
                                "equalities, and any valid/ready handshake invariants "
                                "you can identify.\n\n"
                                f"End your response with {_DONE_SENTINEL}."
                            ),
                        })
                        continue

                # Either contains an assertion, or we've stalled too many times.
                logger.info(
                    "Treating response as final answer (has_assertion=%s, stall_count=%d).",
                    has_assertion, stall_count,
                )
                final_answer = response_text
                break

            if _DONE_SENTINEL in response_text:
                final_answer = response_text
                break

        else:
            logger.warning(
                "Max iterations (%d) reached — forcing final answer.",
                self.config.max_iterations,
            )
            messages.append({
                "role": "user",
                "content": (
                    "You have reached the tool-call limit. "
                    "Do NOT call any more tools — including slang_lint. "
                    "Using only the information already gathered, output the final "
                    "SystemVerilog Assertion code block now, inside a "
                    "```systemverilog ... ``` fence. "
                    "Use only immediate assertions: assert (...) else $error(...); "
                    f"End your response with {_DONE_SENTINEL}."
                ),
            })
            final_answer, _ = self._step(messages)  # ignore tool_calls

        return self._extract_sva(final_answer)

    # ------------------------------------------------------------------
    # Post-generation refinement
    # ------------------------------------------------------------------

    def refine_assertions(
        self,
        failures: List[Dict[str, Any]],
    ) -> str:
        """
        Fix assertions that failed Verible linting.

        Starts a **fresh** conversation (system prompt + focused refinement
        prompt) rather than reusing the generation conversation.  This is
        deliberate: by the time generation finishes the context is 30-40
        turns long and near effective limits.  A fresh, focused prompt
        produces better fixes with fewer tokens.

        The model still has access to all five tools during refinement —
        in particular slang_lint, so it can self-validate before
        answering.

        Parameters
        ----------
        failures : list of dict
            Each dict has "assertion", "error", and optionally "comment"
            keys.  Produced by lint_all_assertions() in lint_loop.py.

        Returns
        -------
        str
            SVA code containing corrected assertions, extracted via
            _extract_sva().
        """
        # Build the structured refinement prompt listing every failure.
        failure_blocks: List[str] = []
        for i, f in enumerate(failures, 1):
            block = (
                f"Failure {i}:\n"
                f"  Assertion: {f['assertion']}\n"
                f"  Error: {f['error']}"
            )
            if f.get("comment"):
                block += f"\n  Context comment: {f['comment']}"
            failure_blocks.append(block)

        # Detect if the common |-> error is present to add specific guidance.
        has_implication_error = any(
            "|->" in f.get("assertion", "") for f in failures
        )
        implication_fix = ""
        if has_implication_error:
            implication_fix = (
                "\n  IMPORTANT: The |-> operator is ONLY valid inside "
                "'assert property'.\n"
                "  In immediate assertions, replace:\n"
                "    assert (COND |-> RESULT) else $error(...);\n"
                "  with:\n"
                "    assert (!(COND) || (RESULT)) else $error(...);\n"
                "  This is logically equivalent: 'if COND then RESULT'.\n"
            )

        refinement_task = (
            "The following SVA assertions failed linting.  For each "
            "one the error is shown below.\n\n"
            "Fix every assertion so it passes lint.  Rules:\n"
            "  - Use ONLY immediate assertions: assert (...) else $error(...);\n"
            "  - Do NOT use 'assert property'.\n"
            "  - Do NOT use '<signal>.width'; use $bits(<signal>) instead.\n"
            + implication_fix +
            "  - Validate each fixed assertion with slang_lint before "
            "answering.\n\n"
            + "\n\n".join(failure_blocks)
            + "\n\nReturn ONLY the corrected assertions inside a "
            "```systemverilog``` code fence, one assertion per line.  "
            "Include the original comment above each assertion if one was "
            f"provided.  End your response with {_DONE_SENTINEL}."
        )

        self._current_phase = "refinement"
        logger.info(
            "Refinement: %d failing assertion(s) to fix.", len(failures)
        )

        # Fresh conversation with the same system prompt.
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": refinement_task},
        ]

        final_answer = ""
        max_steps = self.config.max_refinement_react_steps

        for step in range(1, max_steps + 1):
            logger.info("--- Refinement step %d/%d ---", step, max_steps)

            response_text, tool_calls = self._step(messages)

            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": response_text,
                })
                for call in tool_calls:
                    logger.info(
                        "Refinement tool call: %s(%s)",
                        call.get("name", ""), call.get("arguments", {}),
                    )
                    observation = self._dispatch(call)
                    messages.append({
                        "role": "tool",
                        "name": call.get("name", ""),
                        "content": observation,
                    })
            else:
                # No tool call — final answer.
                final_answer = response_text
                break

            if _DONE_SENTINEL in response_text:
                final_answer = response_text
                break
        else:
            # Exhausted refinement steps — force a final answer.
            logger.warning(
                "Refinement step limit (%d) reached — forcing final answer.",
                max_steps,
            )
            messages.append({
                "role": "user",
                "content": (
                    "You have reached the step limit.  Do NOT call any more "
                    "tools.  Output the corrected assertions now inside a "
                    "```systemverilog``` code fence.  Use only immediate "
                    "assertions: assert (...) else $error(...);"
                    f"  End with {_DONE_SENTINEL}."
                ),
            })
            final_answer, _ = self._step(messages)  # ignore tool_calls

        return self._extract_sva(final_answer)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step(self, messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Run one forward pass through the LLM backend.

        Returns
        -------
        response_text : str
            The model's text response.
        tool_calls : list of dict
            Parsed tool calls (each has "name" and "arguments" keys).
        """
        self._global_step += 1
        response_text, tool_calls, usage = self.backend.generate(
            messages, TOOL_DEFINITIONS
        )

        # Log this step to the trace (token usage is recorded when
        # available; defaults to zeros otherwise).
        self.trace.log_step(
            phase=self._current_phase,
            step=self._global_step,
            model_output=response_text,
            tool_calls=tool_calls,
            usage=usage,
        )

        return response_text, tool_calls

    def _dispatch(self, call: Dict[str, Any]) -> str:
        """
        Convenience wrapper around dispatch_tool that passes all agent state.
        Also logs the tool observation to the trace.

        This avoids repeating the long argument list in generate, plan, execute,
        and refine methods.
        """
        tool_name = call.get("name", "")
        observation = dispatch_tool(
            tool_name=tool_name,
            tool_args=call.get("arguments", {}),
            rtl_retriever=self.rtl_retriever,
            doc_retriever=self.doc_retriever,
            signal_map=self.signal_map,
            rtl_dir=self.config.rtl_dir,
            rtl_top_k=self.config.rtl_top_k,
            doc_top_k=self.config.doc_top_k,
            yosys_bin=self.config.yosys_bin,
            verible_bin=self.config.verible_bin,
            design_graph=self.design_graph,
            full_rtl_injected=self.full_rtl_context is not None,
            reject_assert_property=self.config.reject_assert_property,
        )

        # Log tool observation to the trace.
        self.trace.log_step(
            phase=self._current_phase,
            step=self._global_step,
            tool_observations=[{
                "tool": tool_name,
                "result": observation[:500],
            }],
            notes=f"tool_result:{tool_name}",
        )

        return observation

    def _parse_plan(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract a JSON plan array from the model's planning response.

        Tries multiple extraction strategies:
        1. JSON code fence (```json ... ```)
        2. Raw JSON array search (first [ ... ] block)
        3. Returns empty list on failure.
        """
        # Remove the plan sentinel if present.
        text = text.replace(_PLAN_SENTINEL, "").strip()

        # Strategy 1: JSON code fence.
        fence_re = re.compile(
            r"```json\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
        )
        match = fence_re.search(text)
        if match:
            try:
                plan = json.loads(match.group(1).strip())
                if isinstance(plan, list):
                    logger.info("Parsed plan: %d entries (from code fence).", len(plan))
                    return plan
            except json.JSONDecodeError as exc:
                logger.warning("JSON parse failed in code fence: %s", exc)

        # Strategy 2: raw JSON array.
        bracket_re = re.compile(r"\[.*\]", re.DOTALL)
        match = bracket_re.search(text)
        if match:
            try:
                plan = json.loads(match.group(0))
                if isinstance(plan, list):
                    logger.info("Parsed plan: %d entries (from raw JSON).", len(plan))
                    return plan
            except json.JSONDecodeError as exc:
                logger.warning("JSON parse failed in raw array: %s", exc)

        logger.warning("Could not parse plan from model output.")
        return []

    def _parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all <tool_call>…</tool_call> blocks from `text` and parse
        them as JSON.

        Returns
        -------
        list of dict
            Each dict has "name" (str) and "arguments" (dict) keys.
            Returns an empty list if no valid tool calls are found.
        """
        tool_calls = []
        for raw_json in _TOOL_CALL_RE.findall(text):
            try:
                parsed = json.loads(raw_json.strip())
                # Normalise: Qwen3 may use "name"/"arguments" or
                # "function"/"parameters" depending on fine-tune variant.
                if "function" in parsed:
                    name = parsed["function"].get("name", "")
                    args = parsed["function"].get("parameters", {}) or parsed["function"].get("arguments", {})
                else:
                    name = parsed.get("name", "")
                    args = parsed.get("arguments", parsed.get("parameters", {}))
                if name:
                    tool_calls.append({"name": name, "arguments": args})
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse tool call JSON: %s\nRaw: %s", exc, raw_json[:200])

        return tool_calls

    def _extract_sva(self, text: str) -> str:
        """
        Pull the SVA code block out of the model's final response.

        Looks for a ```systemverilog … ``` or ```sv … ``` fenced block first.
        Falls back to returning the full response if no code fence is found,
        so we never silently drop content.

        Parameters
        ----------
        text : str
            Raw final response from the model.

        Returns
        -------
        str
            The extracted SVA code, or the full text as a fallback.
        """
        # Remove the done sentinel if present.
        text = text.replace(_DONE_SENTINEL, "").strip()

        # Try a properly closed code fence first (``` ... ```).
        closed_fence_re = re.compile(
            r"```(?:systemverilog|sv|verilog)?\s*\n(.*?)```",
            re.DOTALL | re.IGNORECASE,
        )
        match = closed_fence_re.search(text)
        if match:
            return match.group(1).strip()

        # Handle unclosed fence — opening marker present but no closing ```.
        # This happens when the model is cut off at max_new_tokens mid-block.
        open_fence_re = re.compile(
            r"```(?:systemverilog|sv|verilog)\s*\n(.*)",
            re.DOTALL | re.IGNORECASE,
        )
        match = open_fence_re.search(text)
        if match:
            return match.group(1).strip()

        # If the response is a slang_lint tool call, the model was trying to
        # validate its final code when it ran out of iterations.  Extract the
        # sva_code argument — that IS the final answer.
        for raw_json in _TOOL_CALL_RE.findall(text):
            try:
                parsed = json.loads(raw_json.strip())
                name = parsed.get("name", "")
                args = parsed.get("arguments", parsed.get("parameters", {}))
                if name == "slang_lint":
                    code = args.get("sva_code", "")
                    if code:
                        logger.info(
                            "Extracting SVA from final slang_lint call "
                            "(%d chars).", len(code)
                        )
                        return code.strip()
            except json.JSONDecodeError:
                pass

        # No code fence at all — return the whole response as-is.
        return text

    @staticmethod
    def _extract_batch_signals(batch) -> Set[str]:
        """
        Extract signal-like identifiers from a batch of assertion skeletons.

        Scans both ``assertion_text`` and ``source_text`` of each skeleton
        for identifiers, filtering out SV keywords and literals.

        Returns a set of signal names likely referenced by this batch.
        """
        _SV_KEYWORDS = {
            "assert", "property", "else", "error", "if", "iff", "disable",
            "posedge", "negedge", "or", "and", "not", "begin", "end",
            "always", "always_comb", "always_ff", "module", "endmodule",
            "input", "output", "wire", "reg", "logic", "assign",
            "case", "endcase", "default", "for", "generate",
            "sequence", "endsequence", "cover", "assume",
        }
        signals: Set[str] = set()
        for skel in batch:
            for text in (skel.assertion_text, skel.source_text):
                if not text:
                    continue
                # Strip error messages.
                clean = re.split(r'\belse\s+\$\w+\s*\(', text, maxsplit=1)[0]
                clean = re.sub(r'\$\w+', '', clean)
                ids = set(re.findall(r'\b([a-zA-Z_]\w*)\b', clean))
                ids -= _SV_KEYWORDS
                ids = {s for s in ids if not s.isdigit()}
                ids = {
                    s for s in ids
                    if not re.match(r'^[bhdo][0-9a-fA-F_]+$', s)
                }
                signals |= ids
        return signals

    def _warm_prefix_cache(self) -> None:
        """
        Send a minimal request to the OpenAI-compatible server so it caches
        the system prompt prefix. Subsequent requests with the same prefix
        skip prompt re-tokenisation (Ollama, vLLM, SGLang all do this).

        The request asks for a single token (max_tokens=1) and discards the
        response. If it fails (server not ready, timeout), we log and move on.
        """
        try:
            logger.info("Warming prefix cache (system prompt → server) …")
            self.backend.generate(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "Ready."},
                ],
                tools=[],
            )
            logger.info("Prefix cache warmed.")
        except Exception as exc:
            logger.debug("Prefix cache warming failed (non-fatal): %s", exc)

    def _load_hallucination_denylist(self) -> List[str]:
        """
        Load the per-(design, model) hallucination denylist from disk.

        The denylist is a JSON dict ``{name: count}`` written by
        ``validate_signals`` after each pipeline run. Stale entries
        (signals that now exist in the RTL) are filtered out at load.

        Returns the names sorted by count descending. Returns an empty
        list if the file doesn't exist or has no usable entries.
        """
        import json

        config = self.config
        deny_dir = Path(config.hallucination_denylist_dir)
        # Per-(design, model) keying: use the rtl_dir basename and a
        # filesystem-safe model id.
        design_key = Path(config.rtl_dir).name or "unknown_design"
        model_key = (
            (config.model_id or "unknown_model")
            .replace("/", "_")
            .replace(":", "_")
        )
        deny_path = deny_dir / f"{design_key}__{model_key}.json"

        if not deny_path.exists():
            return []

        try:
            with open(deny_path, "r") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read denylist %s: %s", deny_path, exc)
            return []

        # Filter out names that are now real signals in the RTL.
        real_signals: set = set()
        if self.facts is not None:
            real_signals = set(self.facts.all_signals)

        filtered: List[Tuple[str, int]] = []
        for name, count in raw.items():
            if name in real_signals:
                continue
            try:
                count_int = int(count)
            except (TypeError, ValueError):
                count_int = 0
            filtered.append((name, count_int))

        filtered.sort(key=lambda kv: (-kv[1], kv[0]))
        return [name for name, _ in filtered]

    @staticmethod
    def _summarise_signal_map(signal_map: Dict[str, Any]) -> str:
        """
        Convert signal_map.json into a compact table for the system prompt.

        Only the key attributes (module, direction, width, description) are
        included.  Full details can be fetched via the signal_map_lookup tool.
        """
        if not signal_map:
            return "No signal map provided."

        lines = [
            f"{'Signal':<25} {'Module':<20} {'Dir':<8} {'Width':<8} Description",
            "-" * 85,
        ]
        for name, info in sorted(signal_map.items()):
            module = info.get("module", "?")
            direction = info.get("direction", "?")
            width = str(info.get("width", "?"))
            desc = info.get("description", "")
            lines.append(f"{name:<25} {module:<20} {direction:<8} {width:<8} {desc}")

        return "\n".join(lines)
