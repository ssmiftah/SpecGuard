"""
rtl_facts.py
------------
Single source of structured RTL facts extracted via pyslang.

Used by post-processors in ``lint_loop.py`` (and, in Stage 2, by the agent
for prompt augmentation) to validate LLM-generated SVA assertions against
the actual RTL structure.

NO CACHING — facts are re-extracted on every pipeline run to avoid
staleness bugs. Pyslang parsing is fast enough that this isn't a
bottleneck (CMAC: ~0.5s, AES: ~0.3s, nvdla_mul: ~1s).

Architecture
------------
- ``RTLFacts`` is a small dataclass with one field per category.
- ``extract_rtl_facts(rtl_dir, signal_map)`` is the single entry point.
- Per-category extractors are private functions (``_extract_*``) that take
  a list of parsed syntax trees and return their category's data.
- Adding a new fact category means: add a field to ``RTLFacts``, write a
  ``_extract_*`` function, and call it from ``extract_rtl_facts``.

Failure modes
-------------
If pyslang isn't installed or files fail to parse, ``RTLFacts.is_complete``
is set to ``False`` and ``parse_warnings`` lists the issues. Consumers
should treat facts as advisory when ``is_complete == False``.
"""

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Pyslang syntax trees on huge multi-file designs (e.g. NVDLA SDP at
# ~262 KLOC across 55 files) recurse deeper than Python's default 1000-
# frame stack inside `_walk_syntax`. Every SDP variant on the
# `large_designs_ablation_20260428_2214` remote batch crashed with
# `RecursionError: maximum recursion depth exceeded` at line 318. Bumping
# the limit to 10 000 covers the deepest tree we've measured with margin;
# cost is a few extra MB of reserved stack.
sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))

try:
    import pyslang
    SLANG_AVAILABLE = True
except ImportError:
    SLANG_AVAILABLE = False
    logger.warning("pyslang not available — RTLFacts will be empty.")


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class RTLFacts:
    """
    Structured facts about an RTL design extracted via pyslang.

    All fields default to empty so consumers can call gracefully even when
    extraction fails. Check ``is_complete`` before treating facts as
    authoritative.
    """
    # signal_name -> defining expression text
    # e.g., "in_code" -> "{3{sign}} ^ code"
    signal_definitions: Dict[str, str] = field(default_factory=dict)

    # signal_name -> set of selector expressions for case statements
    # that drive this signal
    # e.g., "out_inv" -> {"{is_8bit, in_code}"}
    case_selectors: Dict[str, Set[str]] = field(default_factory=dict)

    # All signals that are LHS of a continuous assign or a blocking
    # assignment inside a combinational always block (including case
    # branches). These signals update in the same cycle as their inputs.
    combinational_signals: Set[str] = field(default_factory=set)

    # signal_name -> drive kind: "comb", "seq", or "mixed".
    # - "comb":  only ever assigned in continuous assigns or comb always blocks
    # - "seq":   only ever assigned in posedge/negedge always blocks
    # - "mixed": appears as LHS in BOTH comb and seq drivers (rare; usually
    #            indicates a bit-split register, multi-driver bug, or a
    #            cross-module name collision in flat extraction)
    # Signals not in this map have no detected driver (inputs, intermediate
    # wires we never saw assigned, hierarchical references).
    # Used by the prompt formatter to tell the LLM whether to use |-> or |=>.
    signal_drive_kind: Dict[str, str] = field(default_factory=dict)

    # constant_literal -> set of signals assigned this literal
    # e.g., "32'h55005500" -> {"res_a_gate"}
    constant_signal_pairs: Dict[str, Set[str]] = field(default_factory=dict)

    # Every signal name declared anywhere in the RTL files. Used by
    # validate_signals to drop assertions referencing hallucinated names.
    all_signals: Set[str] = field(default_factory=set)

    # signal_name -> bit width (declared)
    # e.g., "out_data" -> 17, "cfg_is_int8_d1" -> 1
    # Used by validate_signal_widths to catch out-of-range bit selects and
    # width-mismatched comparisons.
    signal_widths: Dict[str, int] = field(default_factory=dict)

    # signal_name -> number of IdentifierName references in the parsed RTL.
    # Includes both LHS and RHS occurrences. Used by the prompt formatter
    # to rank signals when truncating sections (high-frequency signals are
    # more likely to appear in assertions, so they survive truncation).
    signal_frequencies: Dict[str, int] = field(default_factory=dict)

    # signal_name -> set of other signals it co-occurs with in any single
    # RTL statement (continuous assign or single assignment inside an
    # always block).  Symmetric: if A is in cooc[B], then B is in cooc[A].
    # Clock and reset signals are excluded from the graph (they would
    # inflate every signal's neighborhood, since most statements involve
    # them implicitly).
    #
    # Dual-use:
    #   • Preprocessing — surfaced in the prompt as "functionally-related
    #     signals" for each signal in scope, helping the LLM avoid
    #     mixing signals from unrelated subsystems.
    #   • Post-processing — ``validate_semantic_affinity`` (lint_loop.py)
    #     flags assertions whose signals never co-occur.
    signal_cooccurrence: Dict[str, Set[str]] = field(default_factory=dict)

    # signal_name -> {"upstream": set, "downstream": set}
    #
    # Causal / dataflow relationships extracted from continuous assigns
    # and always-block assignments.  For each statement ``LHS = RHS``:
    #   • every signal in RHS is recorded as "upstream" of LHS
    #     (it drives LHS, directly or via combinational logic);
    #   • every signal in LHS is recorded as "downstream" of every RHS
    #     signal (it is driven by them).
    # Sequential always blocks (posedge/negedge) record the relationship
    # with a "delay=1" attribute via ``signal_dataflow_delay`` below.
    #
    # Dual-use:
    #   • Preprocessing — annotates signal lists with "drives X / Y / Z"
    #     and "driven by P / Q / R" hints so the LLM writes implications
    #     in the correct direction (`req |-> grant`, not the reverse).
    #   • Post-processing — ``validate_implication_direction`` flags
    #     `A |-> B` where B is structurally upstream of A in the RTL.
    signal_dataflow: Dict[str, Dict[str, Set[str]]] = field(
        default_factory=dict
    )

    # (driver_signal, driven_signal) -> minimum cycle delay (0 for
    # combinational, ≥1 for paths that pass through at least one flop).
    # Populated alongside ``signal_dataflow``.  Used by the post-processor
    # to distinguish same-cycle vs next-cycle relationships when checking
    # implication direction, and by the prompt formatter to label hints
    # as "comb" vs "1-cycle" / "≥1-cycle".
    signal_dataflow_delay: Dict[Tuple[str, str], int] = field(
        default_factory=dict
    )

    # Memoised pipeline-depth queries.  Lazily filled by
    # ``pipeline_depth(facts, src, dst)`` — direct precomputation of
    # all-pairs shortest paths is O(V·E) and infeasible on 50K-signal
    # designs.  Maps ``(src, dst)`` -> minimum cumulative cycle delay
    # along the shortest path through ``signal_dataflow_delay``.  A
    # value of ``None`` means "no path within the bounded search depth".
    pipeline_depth_cache: Dict[Tuple[str, str], Optional[int]] = field(
        default_factory=dict
    )

    # Module-boundary information per signal.  For each signal:
    #   {
    #     "direction":   "input" | "output" | "inout" | "internal",
    #     "port_module": name of the module declaring this signal as a
    #                    port (None if direction == "internal"),
    #     "connections": [(parent_module, instance_name, instance_type,
    #                      port_name), ...]
    #                    Every place this signal is wired into an
    #                    instance's port — used to detect cross-module
    #                    handshakes and to back the port-connection
    #                    tier of the affinity check.
    #   }
    #
    # Dual-use:
    #   • Preprocessing — annotate signal lists with port direction so
    #     the LLM can reason about input vs output assertions.
    #   • Post-processing — adds Tier 4 to ``_affinity_link_ratio``
    #     (signals connected to the same instance are linked) and
    #     powers ``validate_cross_module_assertions``.
    signal_port_info: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # signal_name -> categorical role string.  Roles:
    #   • "clock", "reset"             — from clock_signals / reset_signals
    #   • "handshake_init"            — initiator side of a detected pair
    #   • "handshake_resp"            — responder side of a detected pair
    #   • "state"                      — selector signal of a detected FSM
    #   • "address"                    — width > 1 AND name suggests addr
    #   • "data"                       — width > 1 AND name suggests data
    #   • "control"                    — 1-bit, neither clock/reset/handshake
    #   • "status"                     — name ends with "_status" / "_done"
    #   • "unknown"                    — none of the above
    #
    # Dual-use:
    #   • Preprocessing — tag signals in width / port-direction blocks
    #     so the LLM knows which signals are control vs data and can
    #     pick appropriate property templates.
    #   • Post-processing — informational only; could power role-aware
    #     property suggestions in the lint feedback loop later.
    signal_roles: Dict[str, str] = field(default_factory=dict)

    # Documented properties extracted from design docs.  Each entry::
    #
    #     {
    #       "signals":      [signal names mentioned in the sentence],
    #       "modal":        "must" | "shall" | "should" | ...,
    #       "raw_sentence": original text (truncated),
    #       "source_file":  path to the doc file,
    #       "source_line":  1-based line number where the sentence starts,
    #     }
    #
    # Dual-use:
    #   • Preprocessing — list documented properties as a focused
    #     reference list; the LLM is explicitly directed to encode
    #     each one as an SVA property.
    #   • Post-processing — ``compute_doc_coverage`` reports which
    #     documented properties have NO matching assertion in the
    #     output (informational: no removal).
    documented_properties: List[Dict[str, Any]] = field(default_factory=list)

    # Detected state machines.  Each entry::
    #
    #     {
    #       "state_signal":  registered selector signal name,
    #       "module":        module containing the case statement
    #                        (None if unscoped),
    #       "states":        ordered list of state-pattern identifiers
    #                        observed in case items (e.g.,
    #                        ["IDLE", "BUSY", "DONE"]),
    #       "literal_values": list of literal patterns observed (e.g.,
    #                        ["2'b00", "2'b01", "2'b10"]),
    #       "encoding":      best-effort {state_name: literal} map
    #                        derived from localparam declarations,
    #       "n_states":      total distinct case-pattern count,
    #     }
    #
    # Dual-use:
    #   • Preprocessing — list FSMs with state names and encoding so
    #     the LLM can write meaningful state-related properties (state
    #     reachability, valid-state range, transition guards).
    #   • Post-processing — ``validate_state_assertions`` flags
    #     comparisons against unknown state literals (catches
    #     hallucinated state values).
    state_machines: List[Dict[str, Any]] = field(default_factory=list)

    # Detected handshake pairs.  Each entry::
    #
    #     {
    #       "initiator":  name of the request/valid signal,
    #       "responder":  name of the ack/ready signal,
    #       "protocol":   "valid_ready" | "req_ack" | "push_pop"
    #                     | "start_done",
    #       "modules":    set of modules where both signals appear,
    #       "confidence": "high" if the pair is confirmed by either
    #                     same-module port-list co-residence or a
    #                     direct dataflow link; "medium" if matched
    #                     only by name-pattern.
    #     }
    #
    # Dual-use:
    #   • Preprocessing — list detected pairs in the prompt with
    #     standard property templates (persistence, no-double-grant,
    #     stable-payload, etc.) so the LLM emits well-shaped handshake
    #     assertions instead of guessing the protocol.
    #   • Post-processing — ``validate_handshake_correctness`` flags
    #     same-cycle persistence (``X && !Y |-> X`` — wrong) vs the
    #     correct next-cycle form (``X && !Y |=> X``).
    handshake_pairs: List[Dict[str, Any]] = field(default_factory=list)

    # ---- Clock / reset structural detection (multi-domain aware) -------

    # All distinct clock signals across all sequential always blocks.
    # In an MCD design this contains every clock the design uses.
    clock_signals: Set[str] = field(default_factory=set)

    # All distinct reset signals across all sequential always blocks.
    # In an MCD design this contains every reset the design uses.
    reset_signals: Set[str] = field(default_factory=set)

    # reset_signal_name -> "low" or "high"
    # "low" means the reset is active when the signal is 0 (negedge sens,
    #       if (!rst) → reset branch).
    # "high" means active when the signal is 1.
    reset_polarity: Dict[str, str] = field(default_factory=dict)

    # Distinct ``(clock, reset)`` tuples — one entry per detected clock
    # domain. ``reset`` is None for synchronous-only blocks (no async
    # reset in the sensitivity list).
    clock_reset_pairs: Set[Tuple[str, Optional[str]]] = field(
        default_factory=set
    )

    # register_name -> (reset_signal_name, reset_value_text)
    # Tracks both WHICH reset clears each register and WHAT value it
    # resets to. ``reset_signal_name`` is None for registers without an
    # async reset branch.
    reset_values: Dict[str, Tuple[Optional[str], str]] = field(
        default_factory=dict
    )

    # ---- Stage 3: module scoping ----------------------------------------

    # signal_name -> module_name (Option G: lazy filtering).
    # Built from Compilation hierarchy. Used by the prompt formatter to
    # filter flat facts to a specific module at format time.
    signal_to_module: Dict[str, str] = field(default_factory=dict)

    # module_name -> RTLFacts (Option F: full per-module extraction).
    # Each value is a self-contained RTLFacts with signals scoped to one
    # module. Only populated when module_facts_mode == "full".
    per_module: Dict[str, Any] = field(default_factory=dict)
    # Note: type is Any to avoid forward-reference issues; runtime values
    # are RTLFacts instances.

    # module_name -> depth from top_module (Fix 4: depth-limited scoping).
    # Depth 0 = top_module, 1 = direct submodules, etc.
    module_hierarchy: Dict[str, int] = field(default_factory=dict)

    # Signals that are out-of-scope for functional assertions because they
    # only fan out to clock-gating / DFT / power-management cells (sinks).
    # Detected structurally via pyslang port-connection analysis +
    # transitive propagation through assignment chains.
    out_of_scope_signals: Set[str] = field(default_factory=set)

    # True only if every RTL file parsed successfully and pyslang is
    # available. False means consumers should treat facts as advisory.
    is_complete: bool = True

    # Human-readable warnings produced during parsing.
    parse_warnings: List[str] = field(default_factory=list)

    def for_module(
        self, module_name: str, depth: int = 0,
    ) -> "RTLFacts":
        """
        Return facts scoped to a specific module and its submodules.

        Parameters
        ----------
        module_name : str
            The top-level module to scope to.
        depth : int
            How many levels of submodule hierarchy to include.
            0 = top module only, 1 = +direct submodules, -1 = unlimited.

        Checks ``per_module`` first (Option F, collision-proof). Falls back
        to lazy filtering via ``signal_to_module`` (Option G). If neither
        is available, returns ``self`` (flat facts, backward compat).
        """
        # Determine which modules are in scope based on depth.
        modules_in_scope: Optional[Set[str]] = None
        if self.module_hierarchy and depth >= 0:
            modules_in_scope = {
                mod for mod, d in self.module_hierarchy.items()
                if d <= depth
            }
        elif self.module_hierarchy and depth < 0:
            # Unlimited depth — all modules in hierarchy.
            modules_in_scope = set(self.module_hierarchy.keys())

        # Option F: merge per-module facts for all in-scope modules.
        if self.per_module and modules_in_scope:
            in_scope_facts = [
                mf for mod, mf in self.per_module.items()
                if mod in modules_in_scope
            ]
            if in_scope_facts:
                return self._merge_module_facts(in_scope_facts)

        # Option G: lazy filtering with depth-aware module set.
        if self.signal_to_module:
            return self._filter_for_module(
                module_name, modules_in_scope=modules_in_scope,
            )

        # Fallback: return flat facts unchanged.
        return self

    @staticmethod
    def _merge_module_facts(facts_list) -> "RTLFacts":
        """Merge multiple per-module RTLFacts into a single instance."""
        merged = RTLFacts()
        for mf in facts_list:
            merged.signal_definitions.update(mf.signal_definitions)
            for k, v in mf.case_selectors.items():
                merged.case_selectors.setdefault(k, set()).update(v)
            merged.combinational_signals |= mf.combinational_signals
            merged.signal_drive_kind.update(mf.signal_drive_kind)
            for lit, sigs in mf.constant_signal_pairs.items():
                merged.constant_signal_pairs.setdefault(lit, set()).update(sigs)
            merged.all_signals |= mf.all_signals
            merged.signal_widths.update(mf.signal_widths)
            merged.signal_frequencies.update(mf.signal_frequencies)
            merged.clock_signals |= mf.clock_signals
            merged.reset_signals |= mf.reset_signals
            merged.reset_polarity.update(mf.reset_polarity)
            merged.clock_reset_pairs |= mf.clock_reset_pairs
            merged.reset_values.update(mf.reset_values)
        merged.is_complete = True
        return merged

    def _filter_for_module(
        self,
        module_name: str,
        modules_in_scope: Optional[Set[str]] = None,
    ) -> "RTLFacts":
        """
        Build a filtered RTLFacts using signal_to_module.

        This is the Option G lazy path: no re-extraction, just dict
        comprehensions over the flat fields. Fast but lossy on collisions.

        When ``modules_in_scope`` is provided (Fix 4: depth-limited),
        includes signals from all modules in that set. Otherwise includes
        only signals from ``module_name``.
        """
        if modules_in_scope:
            mine = {
                sig for sig, mod in self.signal_to_module.items()
                if mod in modules_in_scope
            }
        else:
            mine = {
                sig for sig, mod in self.signal_to_module.items()
                if mod == module_name
            }
        if not mine:
            return self  # no signals found for this module — use flat

        def _filt_dict(d):
            return {k: v for k, v in d.items() if k in mine}

        def _filt_set(s):
            return s & mine

        def _filt_const_pairs(d):
            result = {}
            for lit, sigs in d.items():
                filtered = sigs & mine
                if filtered:
                    result[lit] = filtered
            return result

        return RTLFacts(
            signal_definitions=_filt_dict(self.signal_definitions),
            case_selectors={
                k: v for k, v in self.case_selectors.items() if k in mine
            },
            combinational_signals=_filt_set(self.combinational_signals),
            signal_drive_kind=_filt_dict(self.signal_drive_kind),
            constant_signal_pairs=_filt_const_pairs(self.constant_signal_pairs),
            all_signals=_filt_set(self.all_signals),
            signal_widths=_filt_dict(self.signal_widths),
            signal_frequencies=_filt_dict(self.signal_frequencies),
            clock_signals=_filt_set(self.clock_signals),
            reset_signals=_filt_set(self.reset_signals),
            reset_polarity={
                k: v for k, v in self.reset_polarity.items() if k in mine
            },
            clock_reset_pairs={
                (c, r) for c, r in self.clock_reset_pairs
                if c in mine or (r is not None and r in mine)
            },
            reset_values={
                k: v for k, v in self.reset_values.items() if k in mine
            },
            signal_to_module={},  # not needed in filtered view
            per_module={},
            is_complete=self.is_complete,
            parse_warnings=[],
        )


# ---------------------------------------------------------------------------
# Pyslang traversal helpers (shared by all extractors)
# ---------------------------------------------------------------------------

def _walk_syntax(node):
    """Yield every syntax-tree node recursively (depth-first)."""
    yield node
    try:
        for child in node:
            yield from _walk_syntax(child)
    except TypeError:
        pass


def _kind_name(node) -> str:
    """Return the bare SyntaxKind name (e.g., 'CaseStatement') for a node."""
    return str(getattr(node, 'kind', '')).split('.')[-1]


def _lhs_base(lhs_text: str) -> Optional[str]:
    """
    Extract the base identifier from an LHS expression.

    Strips bit selects: ``sig[7:0]`` -> ``sig``.
    Returns None if the result isn't a valid identifier.
    """
    base = lhs_text.strip().split('[')[0].strip()
    return base if base.isidentifier() else None


def _parse_rtl_files(
    rtl_dir: str,
    facts: RTLFacts,
) -> List[Tuple[str, Any]]:
    """
    Parse every ``.v``/``.sv`` file under ``rtl_dir``.

    Records parse failures on ``facts.parse_warnings`` and sets
    ``facts.is_complete = False`` on any failure.

    Returns a list of ``(file_path, syntax_tree)`` tuples for successfully
    parsed files. Files that fail to parse are skipped.
    """
    if not SLANG_AVAILABLE:
        facts.is_complete = False
        facts.parse_warnings.append("pyslang not installed")
        return []

    if not Path(rtl_dir).exists():
        facts.is_complete = False
        facts.parse_warnings.append(f"rtl_dir not found: {rtl_dir}")
        return []

    trees: List[Tuple[str, Any]] = []
    for root, _, files in os.walk(rtl_dir):
        for fname in sorted(files):
            if Path(fname).suffix not in {".v", ".sv"}:
                continue
            path = os.path.join(root, fname)
            try:
                tree = pyslang.SyntaxTree.fromFile(path)
                trees.append((path, tree))
            except Exception as exc:
                facts.is_complete = False
                facts.parse_warnings.append(f"parse error {path}: {exc}")

    return trees


# ---------------------------------------------------------------------------
# Stage 3: Module-scoping helpers
# ---------------------------------------------------------------------------


def _build_signal_to_module(
    trees: List[Tuple[str, Any]],
) -> Dict[str, str]:
    """
    Option G: build a ``{signal_name: module_name}`` mapping using the
    pyslang Compilation hierarchy.

    Walks ``topInstances`` → ``InstanceSymbol.body`` members, recording
    which module each signal is declared in. On name collisions, the
    FIRST module seen wins (same as the flat extractors).

    Returns an empty dict if compilation fails.
    """
    if not SLANG_AVAILABLE or not trees:
        return {}

    mapping: Dict[str, str] = {}
    try:
        comp = pyslang.Compilation()
        for _path, tree in trees:
            comp.addSyntaxTree(tree)

        def _walk_inst(inst, depth=0):
            try:
                body = inst.body
                mod_name = body.name
            except AttributeError:
                return
            for member in body:
                tname = type(member).__name__
                if tname in {'NetSymbol', 'VariableSymbol', 'PortSymbol'}:
                    if member.name not in mapping:
                        mapping[member.name] = mod_name
                elif tname == 'InstanceSymbol':
                    _walk_inst(member, depth + 1)

        for inst in comp.getRoot().topInstances:
            _walk_inst(inst)
    except Exception as exc:
        logger.debug("signal_to_module mapping failed: %s", exc)

    return mapping


def _build_module_hierarchy(
    trees: List[Tuple[str, Any]],
    top_module: str,
) -> Dict[str, int]:
    """
    Fix 4: build a ``{module_name: depth}`` mapping from the Compilation
    hierarchy, rooted at ``top_module``.

    Depth 0 = top_module itself, depth 1 = its direct submodules, etc.
    Used to implement depth-limited scoping.

    Returns an empty dict if compilation fails or top_module isn't found.
    """
    if not SLANG_AVAILABLE or not trees or not top_module:
        return {}

    hierarchy: Dict[str, int] = {}
    try:
        comp = pyslang.Compilation()
        for _path, tree in trees:
            comp.addSyntaxTree(tree)

        def _walk_inst(inst, depth: int, found_top: bool):
            try:
                body = inst.body
                mod_name = body.name
            except AttributeError:
                return

            if mod_name == top_module:
                found_top = True
                depth = 0

            if found_top:
                # Record the shallowest depth for this module name.
                if mod_name not in hierarchy or depth < hierarchy[mod_name]:
                    hierarchy[mod_name] = depth

            for member in body:
                if type(member).__name__ == 'InstanceSymbol':
                    _walk_inst(
                        member,
                        depth + 1 if found_top else depth,
                        found_top,
                    )

        for inst in comp.getRoot().topInstances:
            _walk_inst(inst, 0, False)
    except Exception as exc:
        logger.debug("module hierarchy build failed: %s", exc)

    return hierarchy


# ---------------------------------------------------------------------------
# Out-of-scope signal detection (structural, design-agnostic)
# ---------------------------------------------------------------------------
#
# A signal is "out of scope" for functional assertions when its only
# downstream consumers are:
#   - Clock-gating cells (SLCG, ICG, *_CG, *_clk_gate*)
#   - DFT scaffolding (testpoint_*, *_BLKBOX_*, *DFT*)
#   - Power-management cells (*_pwr_*, *_ret_*)
#
# Detection is purely structural — based on pyslang's compiled hierarchy.
# No design-specific signal-name patterns are required.
#
# Algorithm:
#   1. Identify "sink" submodule instances by:
#      a. Module name matching default sink patterns (CG, gate, slcg, etc.)
#      b. Port signature: 1 output port whose name contains "gated_clk",
#         "clk_gated", or "clk_out_gated" (clock-gate cells)
#   2. Walk every InstanceSymbol's portConnections; classify each
#      connected signal as "sink-connected" or "functional-connected"
#      based on whether the destination instance is a sink.
#   3. Seed = signals that are sink-connected only (never functional).
#   4. Propagate: if a signal X is defined as ``X = f(Y)`` and Y is in
#      the out-of-scope set, X also becomes out-of-scope. Iterate until
#      the set stabilises (or hits ``max_iterations``).


# Default sink module name patterns (regex). These catch common idioms
# across designs without naming conventions specific to NVDLA.
DEFAULT_SINK_MODULE_PATTERNS = [
    r".*[Ss][Ll][Cc][Gg].*",       # SLCG / *slcg* / NV_SLCG
    r".*_[Cc][Gg]$",                # *_CG (Synopsys integrated clock gate)
    r"^[Cc][Gg]_.*",                # CG_*
    r".*[Cc]lk_?[Gg]ate.*",         # clk_gate / ClkGate / clkgate
    r".*[Gg]ated_?[Cc]lk.*",        # gated_clk / GatedClk
    r"^DFT_.*",                      # DFT_*
    r".*[Tt]estpoint.*",            # testpoint
    r".*BLKBOX.*",                   # NVDLA / Synopsys blackbox
    r".*_pwr_.*",                    # power-management
    r".*_ret_(cell|reg).*",         # retention cells
]

# Sink port-signature heuristics (output port name contains any of these)
_SINK_OUTPUT_NAME_PATTERNS = [
    r"gated_?clk",
    r"clk_?gated",
    r"clk_?out",
]


def _looks_like_clock_gate_signature(inst) -> bool:
    """
    Return True if the instance's ports look like a clock-gating cell.

    Heuristic: at least one output port whose name matches a clock-gate
    output pattern (``gated_clk``, ``clk_gated``, ``clk_out``). This
    catches gates that don't follow naming conventions on the module
    name itself.
    """
    try:
        body = inst.body
    except AttributeError:
        return False
    for member in body:
        if type(member).__name__ != "PortSymbol":
            continue
        direction = str(getattr(member, "direction", "")).lower()
        if not direction.endswith(".out"):
            continue
        name = (getattr(member, "name", "") or "").lower()
        for pat in _SINK_OUTPUT_NAME_PATTERNS:
            if re.search(pat, name):
                return True
    return False


def _is_sink_module(inst, sink_name_patterns) -> bool:
    """
    Return True if this submodule instance is a clock-gating / DFT /
    power-management sink.

    Combines name-pattern matching with port-signature heuristics.
    """
    try:
        mod_name = inst.body.name
    except AttributeError:
        return False
    if not mod_name:
        return False
    for pat in sink_name_patterns:
        try:
            if re.search(pat, mod_name):
                return True
        except re.error:
            continue
    if _looks_like_clock_gate_signature(inst):
        return True
    return False


def _signals_in_expression(expr) -> Set[str]:
    """
    Extract all referenced signal names from a pyslang Expression.

    Handles NamedValue (direct ``.symbol``), ElementSelect / RangeSelect
    (``.value`` points to the base signal), Concatenation, and other
    compound expressions by recursing through syntax-tree identifiers.
    """
    result: Set[str] = set()
    if expr is None:
        return result

    # 1. Direct symbol reference (NamedValue)
    sym = getattr(expr, "symbol", None)
    if sym is not None:
        name = getattr(sym, "name", None)
        if name and name.isidentifier():
            result.add(name)

    # 1b. getSymbolReference() — catches Assignment expressions on output
    #     ports (where the expression wraps the driven signal).
    try:
        sr = expr.getSymbolReference() if hasattr(expr, "getSymbolReference") else None
        if sr is not None:
            name = getattr(sr, "name", None)
            if name and name.isidentifier():
                result.add(name)
    except Exception:
        pass

    # 2. Semantic recursion via .value (ElementSelect, RangeSelect,
    #    MemberAccess) and .left (Assignment — output port connections)
    for attr in ("value", "left"):
        base = getattr(expr, attr, None)
        if base is not None and base is not expr:
            try:
                result |= _signals_in_expression(base)
            except Exception:
                pass

    # 3. Walk the expression's syntax for any Identifier / IdentifierName
    #    nodes. This catches concatenations, conversions, assignments,
    #    and anything else we don't special-case.
    syn = getattr(expr, "syntax", None)
    if syn is not None:
        for n in _walk_syntax(syn):
            k = _kind_name(n)
            if k in ("IdentifierName", "Identifier"):
                try:
                    text = str(n).strip()
                    if text and text.isidentifier():
                        # Exclude common SV keywords that might appear
                        # as Identifier tokens in expressions.
                        if text not in {"posedge", "negedge", "null",
                                        "and", "or", "not"}:
                            result.add(text)
                except Exception:
                    pass

    return result


def _sink_internal_signals(inst) -> Set[str]:
    """
    Return the names of wires/variables DECLARED inside a sink module's
    body (not ports). These are internal DFT/SLCG/power scaffolding
    signals — they never cross the module boundary, so the port-walker
    can't see them, but they do appear in assertions generated from the
    module's internal ``assign``/``always`` blocks.

    Excludes signals whose names match the sink module's port list. This
    matters because pyslang exposes each input port as BOTH a PortSymbol
    AND a NetSymbol with the same name; without the exclusion we'd mark
    the sink module's clock/reset inputs (e.g., nvdla_core_clk) as
    out-of-scope even though they're shared clocks used across the
    entire design.
    """
    result: Set[str] = set()
    try:
        body = inst.body
    except AttributeError:
        return result

    port_names: Set[str] = set()
    for m in body:
        if type(m).__name__ == "PortSymbol":
            n = getattr(m, "name", "")
            if n:
                port_names.add(n)

    for m in body:
        tname = type(m).__name__
        if tname in {"NetSymbol", "VariableSymbol"}:
            name = getattr(m, "name", "")
            if (name and name.isidentifier()
                    and name not in port_names):
                result.add(name)
    return result


def _seed_out_of_scope_signals(
    comp,
    sink_name_patterns,
) -> Set[str]:
    """
    Walk the FULL instance hierarchy and seed the out-of-scope set.

    Three sources feed the seed:

    1. Output of a sink module (e.g., gated clock out of an SLCG cell)
       → ALWAYS out-of-scope.
    2. Input to a sink module → out-of-scope UNLESS the signal is also
       consumed as an input to some non-sink module (then it's shared).
    3. Internal nets declared inside a sink module's body → out-of-scope.
       These are DFT testpoint aliases, gate-enable counters, etc. They
       never cross the module boundary but appear in the module's
       ``always``/``assign`` logic, so they leak into AST-generated
       assertions without this filter.

    Only INPUT ports to non-sink modules count as "functional consumers".
    Outputs are just drivers.

    The walker recurses through every InstanceSymbol at every hierarchy
    depth (not just the top).
    """
    if not SLANG_AVAILABLE:
        return set()

    sink_outputs: Set[str] = set()
    sink_inputs: Set[str] = set()
    sink_internals: Set[str] = set()
    functional_consumed: Set[str] = set()

    visited: Set[int] = set()

    def _walk_inst(inst):
        iid = id(inst)
        if iid in visited:
            return
        visited.add(iid)
        try:
            is_self_sink = _is_sink_module(inst, sink_name_patterns)
            if is_self_sink:
                sink_internals.update(_sink_internal_signals(inst))

            for sub in inst.body:
                if type(sub).__name__ != "InstanceSymbol":
                    continue
                is_sink = _is_sink_module(sub, sink_name_patterns)
                try:
                    conns = sub.portConnections
                except AttributeError:
                    conns = []
                for c in conns:
                    expr = getattr(c, "expression", None)
                    sigs = _signals_in_expression(expr)
                    if not sigs:
                        continue
                    port = getattr(c, "port", None)
                    port_dir = str(getattr(port, "direction", "")).lower()
                    is_output = port_dir.endswith(".out")
                    is_input = port_dir.endswith(".in")

                    if is_sink:
                        if is_output:
                            sink_outputs.update(sigs)
                        elif is_input:
                            sink_inputs.update(sigs)
                    else:
                        if is_input:
                            functional_consumed.update(sigs)
                # Recurse into every child instance, whether sink or not.
                _walk_inst(sub)
        except (AttributeError, TypeError):
            pass

    try:
        for top in comp.getRoot().topInstances:
            _walk_inst(top)
    except Exception as exc:
        logger.debug("out-of-scope seed walk failed: %s", exc)

    return (
        sink_outputs
        | (sink_inputs - functional_consumed)
        | sink_internals
    )


def _build_signal_dataflow(trees) -> Dict[str, Set[str]]:
    """
    Build a ``{lhs: set_of_rhs_signals}`` map covering ALL assignments in
    the design — continuous assigns, blocking & non-blocking assignments
    inside always blocks (sequential and combinational).

    Used by out-of-scope propagation to trace signals through both
    combinational paths and register pipelines.
    """
    dataflow: Dict[str, Set[str]] = {}
    id_re = re.compile(r"\b([a-zA-Z_]\w*)\b")
    sv_keywords = {
        "and", "or", "not", "xor", "if", "else", "begin", "end",
        "case", "endcase", "default", "module", "endmodule",
        "input", "output", "wire", "reg", "logic", "assign",
        "always", "always_comb", "always_ff", "posedge", "negedge",
        "iff", "disable", "assert", "property", "sequence",
        "for", "generate", "endgenerate", "signed", "unsigned",
    }

    def _collect_refs(text: str) -> Set[str]:
        ids = set(id_re.findall(text))
        ids -= sv_keywords
        ids = {x for x in ids if not x.isdigit()}
        ids = {x for x in ids if not re.match(r"^[bhdo][0-9a-fA-F_]+$", x)}
        return ids

    for _path, tree in trees:
        for node in _walk_syntax(tree.root):
            k = _kind_name(node)
            if k in {"AssignmentExpression",
                     "NonblockingAssignmentExpression"}:
                try:
                    lhs_text = str(node.left).strip()
                    rhs_text = str(node.right).strip()
                except AttributeError:
                    continue
                base = _lhs_base(lhs_text)
                if not base:
                    continue
                refs = _collect_refs(rhs_text)
                # Self-references don't count (registers holding their value).
                refs.discard(base)
                if base in dataflow:
                    dataflow[base] |= refs
                else:
                    dataflow[base] = refs

    return dataflow


def _propagate_out_of_scope(
    seed: Set[str],
    dataflow: Dict[str, Set[str]],
    max_iterations: int,
) -> Set[str]:
    """
    Expand the seed by propagating through the assignment graph in both
    directions until stable.

    FORWARD (cause → effect): if ``X = f(Y, Z, ...)`` and ALL of
      ``Y, Z, ...`` are out-of-scope, then ``X`` is out-of-scope.
      (A register sourced only from out-of-scope signals is also
       out-of-scope.)

    BACKWARD (effect → cause): if signal ``Y`` appears only as an input
      to assignments whose LHS is already out-of-scope, then ``Y`` is
      out-of-scope too.
      (A register whose only consumer is another out-of-scope register
       is itself out-of-scope — e.g., the pipeline staging `_d3`,
       `_d2`, `_d1` flops of a gate-enable signal.)

    Iterates forward and backward until both passes stabilize or
    ``max_iterations`` is reached.
    """
    if not seed or not dataflow:
        return set(seed)

    result = set(seed)

    # Build the reverse graph: signal -> set of LHS signals that reference it.
    fan_out: Dict[str, Set[str]] = {}
    for lhs, refs in dataflow.items():
        for r in refs:
            fan_out.setdefault(r, set()).add(lhs)

    for it in range(max_iterations):
        added_fwd = 0
        added_bwd = 0

        # Forward pass.
        for lhs, refs in dataflow.items():
            if lhs in result or not refs:
                continue
            if refs.issubset(result):
                result.add(lhs)
                added_fwd += 1

        # Backward pass: a signal whose consumers are ALL out-of-scope
        # (and that has at least one consumer) becomes out-of-scope.
        # We skip signals that have no LHS definition — they're ports
        # or primary inputs, and we can't prove them unused just because
        # their in-design consumers are out-of-scope.
        for sig, consumers in fan_out.items():
            if sig in result:
                continue
            if sig not in dataflow:
                # Primary input — skip to avoid false positives on
                # signals that may be externally consumed.
                continue
            if not consumers:
                continue
            if consumers.issubset(result):
                result.add(sig)
                added_bwd += 1

        if added_fwd == 0 and added_bwd == 0:
            logger.debug(
                "out-of-scope propagation converged in %d iteration(s) "
                "(fwd+bwd).", it + 1,
            )
            break

    return result


def _detect_out_of_scope_signals(
    trees,
    sink_name_patterns,
    max_iterations: int,
) -> Set[str]:
    """
    Top-level orchestrator for structural out-of-scope detection.

    Returns a set of signal names determined to be functionally
    out-of-scope — only fan out to clock-gating / DFT / power sinks, plus
    any signals transitively derived from them through assignment chains
    (both continuous and sequential).
    """
    if not SLANG_AVAILABLE or not trees:
        return set()

    try:
        comp = pyslang.Compilation()
        for _path, tree in trees:
            comp.addSyntaxTree(tree)
        seed = _seed_out_of_scope_signals(comp, sink_name_patterns)
        if not seed:
            return set()
        dataflow = _build_signal_dataflow(trees)
        expanded = _propagate_out_of_scope(seed, dataflow, max_iterations)
        return expanded
    except Exception as exc:
        logger.debug("out-of-scope detection failed: %s", exc)
        return set()


def _roots(trees_or_nodes):
    """
    Option F adapter: yield ``(path, root_node)`` from either format.

    Accepts:
    - ``[(path, SyntaxTree)]`` — standard from ``_parse_rtl_files``,
      yields ``(path, tree.root)``
    - ``[(path, node)]`` — module-scoped subtrees from
      ``_split_by_module``, yields ``(path, node)`` directly
    """
    for path, item in trees_or_nodes:
        if hasattr(item, 'root'):
            yield path, item.root
        else:
            yield path, item


def _extract_module_name(module_node) -> Optional[str]:
    """
    Extract the module name from a ``ModuleDeclaration`` syntax node.

    The name is in the ``ModuleHeader`` child as an ``Identifier`` node
    (not ``IdentifierName`` — pyslang uses different node kinds for
    declaration names vs expression references).
    """
    for child in module_node:
        if _kind_name(child) == 'ModuleHeader':
            for hchild in child:
                hk = _kind_name(hchild)
                if hk in ('Identifier', 'IdentifierName'):
                    name = str(hchild).strip()
                    if name and name.isidentifier():
                        return name
            break
    return None


def _split_by_module(
    trees: List[Tuple[str, Any]],
) -> Dict[str, List[Tuple[str, Any]]]:
    """
    Option F: split parsed trees into per-module subtrees.

    Walks each tree's root for ``ModuleDeclaration`` nodes (top-level
    only, not nested). Returns ``{module_name: [(path, module_node), ...]}``.

    Each ``module_node`` can be passed through ``_roots()`` and then to
    any extractor that calls ``_walk_syntax(root)`` — the extractor walks
    the module's subtree instead of the full file.
    """
    modules: Dict[str, List[Tuple[str, Any]]] = {}
    for path, tree in trees:
        # Scan the top two levels: CompilationUnit → SyntaxList → children.
        # ModuleDeclaration may be a direct child of root OR nested one
        # level inside a SyntaxList container. We scan both but don't
        # recurse deeper (which would find generate-block modules).
        try:
            for child in tree.root:
                if _kind_name(child) == 'ModuleDeclaration':
                    name = _extract_module_name(child)
                    if name:
                        modules.setdefault(name, []).append((path, child))
                elif _kind_name(child) == 'SyntaxList':
                    try:
                        for grandchild in child:
                            if _kind_name(grandchild) == 'ModuleDeclaration':
                                name = _extract_module_name(grandchild)
                                if name:
                                    modules.setdefault(name, []).append(
                                        (path, grandchild)
                                    )
                    except TypeError:
                        pass
        except TypeError:
            pass
    return modules


def _extract_per_module_facts(
    trees: List[Tuple[str, Any]],
    module_map: Dict[str, List[Tuple[str, Any]]],
) -> Dict[str, Any]:
    """
    Option F: extract a full RTLFacts for each module.

    Calls the existing flat extractors once per module, passing the
    module-scoped subtrees through ``_roots()``. The extractors work
    unchanged because ``_roots()`` adapts the node format.

    Returns ``{module_name: RTLFacts}``.
    """
    per_module: Dict[str, Any] = {}

    for mod_name, mod_trees in module_map.items():
        # Syntax-level extractors: use module subtrees via _roots().
        sig_defs = _extract_signal_definitions(mod_trees)
        case_sels = _extract_case_selectors(mod_trees)
        comb, drive_kinds = _extract_drive_kinds(mod_trees)
        const_pairs = _extract_constant_signal_pairs(mod_trees)
        all_sigs = _extract_all_signals(mod_trees)
        freqs = _extract_signal_frequencies(mod_trees)

        # Clock/reset detection (also syntax-level).
        (
            clk_sigs, rst_sigs, rst_pol, clk_rst_pairs, rst_vals,
        ) = _detect_clock_reset_pairs_and_resets(mod_trees)

        # Widths: use the full compilation, but filter to this module's
        # signals. The compilation-level widths are in the flat facts
        # (populated by the caller); we can't easily re-run the
        # compilation per module. Instead, the caller passes full widths
        # and we filter here.
        # (Widths are added by the caller after this function returns.)

        per_module[mod_name] = RTLFacts(
            signal_definitions=sig_defs,
            case_selectors=case_sels,
            combinational_signals=comb,
            signal_drive_kind=drive_kinds,
            constant_signal_pairs=const_pairs,
            all_signals=all_sigs,
            signal_widths={},  # filled by caller from compilation
            signal_frequencies=freqs,
            clock_signals=clk_sigs,
            reset_signals=rst_sigs,
            reset_polarity=rst_pol,
            clock_reset_pairs=clk_rst_pairs,
            reset_values=rst_vals,
            signal_to_module={},
            per_module={},
            is_complete=True,
            parse_warnings=[],
        )

    return per_module


# ---------------------------------------------------------------------------
# Per-category extractors
# ---------------------------------------------------------------------------

def _extract_signal_definitions(trees: List[Tuple[str, Any]]) -> Dict[str, str]:
    """
    Extract the defining expression for each assigned signal.

    Sources:
    - Continuous assignments: ``assign sig = expr;``
    - Blocking assignments inside combinational always blocks
      (always @(*), always_comb), but NOT inside case branches.

    The first definition encountered wins (so the unconditional
    definition takes precedence over a case-branch override).
    """
    defs: Dict[str, str] = {}

    for path, root in _roots(trees):
        for node in _walk_syntax(root):
            kind = _kind_name(node)

            # Continuous assigns: walk for AssignmentExpression directly.
            if kind == 'ContinuousAssign':
                for inner in _walk_syntax(node):
                    if _kind_name(inner) == 'AssignmentExpression':
                        try:
                            lhs_text = str(inner.left).strip()
                            rhs_text = str(inner.right).strip()
                        except AttributeError:
                            continue
                        base = _lhs_base(lhs_text)
                        if base:
                            defs.setdefault(base, rhs_text)

            # Combinational always blocks only — skip if sequential.
            elif kind == 'AlwaysBlock':
                block_text = str(node)
                if 'posedge' in block_text or 'negedge' in block_text:
                    continue

                # Walk for AssignmentExpression but skip those inside
                # CaseStatement (case branches are case-conditioned and
                # don't represent the unconditional definition).
                def _walk_filtered(n, in_case: bool = False):
                    nk = _kind_name(n)
                    if nk == 'CaseStatement':
                        in_case = True
                    if nk == 'AssignmentExpression' and not in_case:
                        try:
                            lhs_text = str(n.left).strip()
                            rhs_text = str(n.right).strip()
                            base = _lhs_base(lhs_text)
                            if base:
                                defs.setdefault(base, rhs_text)
                        except AttributeError:
                            pass
                    try:
                        for child in n:
                            _walk_filtered(child, in_case)
                    except TypeError:
                        pass

                _walk_filtered(node)

    return defs


def _extract_case_selectors(trees: List[Tuple[str, Any]]) -> Dict[str, Set[str]]:
    """
    Find ``case (selector) ... endcase`` blocks and the signals they assign.

    Handles nested case statements, multi-line selectors with concatenation,
    and case/casex/casez variants — all via pyslang's syntax tree.
    """
    selectors: Dict[str, Set[str]] = {}

    for _path, root in _roots(trees):
        for node in _walk_syntax(root):
            if _kind_name(node) != 'CaseStatement':
                continue

            try:
                selector_text = str(node.expr).strip()
            except AttributeError:
                continue

            try:
                items = node.items
            except AttributeError:
                continue

            for item_node in _walk_syntax(items):
                if _kind_name(item_node) != 'AssignmentExpression':
                    continue
                try:
                    lhs_text = str(item_node.left).strip()
                except AttributeError:
                    continue
                base = _lhs_base(lhs_text)
                if base:
                    selectors.setdefault(base, set()).add(selector_text)

    return selectors


def _extract_drive_kinds(
    trees: List[Tuple[str, Any]],
) -> Tuple[Set[str], Dict[str, str]]:
    """
    Single-pass extraction of combinational signals AND signal drive kinds.

    Walks every continuous assign and every always block once. For each
    signal that appears as LHS, records whether it is driven by:
    - a continuous assign or a combinational always block (``comb`` bucket)
    - a sequential always block with ``posedge``/``negedge`` (``seq`` bucket)

    Returns
    -------
    (combinational_signals, signal_drive_kind)
        - ``combinational_signals``: backward-compatible set used by
          ``fix_next_cycle_on_combinational`` (everything in the comb
          bucket, even if it also appears in the seq bucket).
        - ``signal_drive_kind``: name -> ``"comb"`` | ``"seq"`` | ``"mixed"``.
          A signal in both buckets is classified as ``mixed`` (rare; usually
          a bit-split register or multi-driver bug). Signals never seen as
          LHS are absent from the map (inputs, intermediate wires).
    """
    comb: Set[str] = set()
    seq: Set[str] = set()

    for _path, root in _roots(trees):
        for node in _walk_syntax(root):
            kind = _kind_name(node)

            if kind == 'ContinuousAssign':
                for inner in _walk_syntax(node):
                    if _kind_name(inner) == 'AssignmentExpression':
                        try:
                            lhs_text = str(inner.left).strip()
                        except AttributeError:
                            continue
                        base = _lhs_base(lhs_text)
                        if base:
                            comb.add(base)

            elif kind == 'AlwaysBlock':
                block_text = str(node)
                is_sequential = (
                    'posedge' in block_text or 'negedge' in block_text
                )
                bucket = seq if is_sequential else comb

                # Both blocking (=) and non-blocking (<=) assignments —
                # the latter only appear in sequential blocks but we accept
                # both kinds in either bucket for robustness.
                for inner in _walk_syntax(node):
                    ik = _kind_name(inner)
                    if ik in {
                        'AssignmentExpression',
                        'NonblockingAssignmentExpression',
                    }:
                        try:
                            lhs_text = str(inner.left).strip()
                        except AttributeError:
                            continue
                        base = _lhs_base(lhs_text)
                        if base:
                            bucket.add(base)

    drive_kind: Dict[str, str] = {}
    for name in comb | seq:
        in_comb = name in comb
        in_seq = name in seq
        if in_comb and in_seq:
            drive_kind[name] = "mixed"
        elif in_comb:
            drive_kind[name] = "comb"
        else:
            drive_kind[name] = "seq"

    return comb, drive_kind


# Filter for "interesting" Verilog literals: at least 2 width digits
# (e.g., 32'h..., 16'b...) or hex with 2+ payload chars (e.g., 4'hff).
# Excludes single-bit constants like 1'b0, 1'b1, 2'b00 which are too
# common to be useful for signal disambiguation.
_INTERESTING_LITERAL_RE = re.compile(
    r"^\d{2,}'[bhdo][0-9a-fA-F_]+$|^\d+'h[0-9a-fA-F_]{2,}$"
)


def _extract_constant_signal_pairs(
    trees: List[Tuple[str, Any]],
) -> Dict[str, Set[str]]:
    """
    Map ``{constant_literal: set(signals_assigned_this_literal)}``.

    Walks every ``AssignmentExpression`` and looks for Verilog integer
    vector literals (``IntegerVectorExpression`` nodes) on the RHS.
    Multi-bit literals only — single-bit constants are skipped.

    Used by ``verify_constant_signal_pairs`` to detect LLM assertions
    that pair a constant with the wrong signal.
    """
    pairs: Dict[str, Set[str]] = {}

    for _path, root in _roots(trees):
        for node in _walk_syntax(root):
            if _kind_name(node) != 'AssignmentExpression':
                continue
            try:
                lhs_text = str(node.left).strip()
                rhs_node = node.right
            except AttributeError:
                continue

            base = _lhs_base(lhs_text)
            if not base:
                continue

            # Walk the RHS subtree for integer vector literals.
            for sub in _walk_syntax(rhs_node):
                sk = _kind_name(sub)
                if 'IntegerVector' not in sk:
                    continue
                lit_text = str(sub).strip()
                if _INTERESTING_LITERAL_RE.match(lit_text):
                    pairs.setdefault(lit_text, set()).add(base)

    return pairs


def _extract_signal_widths(trees: List[Tuple[str, Any]]) -> Dict[str, int]:
    """
    Extract ``{signal_name: bit_width}`` for every declared signal.

    Uses pyslang's compilation symbols (``NetSymbol``, ``VariableSymbol``,
    ``PortSymbol``) which expose ``.type.bitWidth`` directly.

    A single ``Compilation`` is built from all parsed trees so that
    cross-file references resolve. Returns an empty dict if elaboration
    fails or no top instances are found.
    """
    if not SLANG_AVAILABLE or not trees:
        return {}

    widths: Dict[str, int] = {}
    try:
        comp = pyslang.Compilation()
        for _path, tree in trees:
            comp.addSyntaxTree(tree)

        def _walk_instance(inst):
            try:
                body = inst.body
            except AttributeError:
                return
            for member in body:
                tname = type(member).__name__
                if tname in {'NetSymbol', 'VariableSymbol', 'PortSymbol'}:
                    try:
                        ty = member.type
                        bw = getattr(ty, 'bitWidth', None)
                        if bw is not None and member.name not in widths:
                            widths[member.name] = bw
                    except AttributeError:
                        pass
                elif tname == 'InstanceSymbol':
                    # Recurse into submodule instances.
                    _walk_instance(member)

        for inst in comp.getRoot().topInstances:
            _walk_instance(inst)
    except Exception as exc:
        logger.debug("signal_widths extraction failed: %s", exc)

    return widths


def _detect_block_clock_and_reset(
    block_node: Any,
) -> Tuple[List[str], List[str]]:
    """
    For a single ``AlwaysBlock`` node, return ``(posedge_signals, negedge_signals)``
    by walking its ``SignalEventExpression`` children.

    Combinational blocks (``always @(*)``) return ``([], [])``.
    """
    posedge: List[str] = []
    negedge: List[str] = []

    for sub in _walk_syntax(block_node):
        if _kind_name(sub) != 'SignalEventExpression':
            continue
        edge = None
        sig = None
        for child in sub:
            ck = _kind_name(child)
            if ck == 'PosEdgeKeyword':
                edge = 'pos'
            elif ck == 'NegEdgeKeyword':
                edge = 'neg'
            elif ck == 'IdentifierName':
                sig = str(child).strip()
        if edge == 'pos' and sig:
            posedge.append(sig)
        elif edge == 'neg' and sig:
            negedge.append(sig)

    return posedge, negedge


def _classify_block(
    posedge: List[str],
    negedge: List[str],
    posedge_global_freq: Dict[str, int],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Classify a sequential always block into ``(clock, reset, polarity)``.

    Heuristic:
    - If both posedge and negedge are present:
        clock = first posedge signal
        reset = first negedge signal
        polarity = "low"
    - If only posedge with 2 signals (active-high reset on `or`):
        clock = the signal with higher global posedge frequency (the
                signal that drives the most always blocks is the clock)
        reset = the other signal
        polarity = "high"
    - If only posedge with 1 signal:
        clock = that signal
        reset = None (synchronous-only block)
        polarity = None

    Returns ``(None, None, None)`` for combinational blocks.
    """
    if not posedge:
        return None, None, None

    # Case 1: posedge clk + negedge rst — async active-low reset.
    if negedge:
        return posedge[0], negedge[0], "low"

    # Case 2: only posedge, single signal — sync-only block.
    if len(posedge) == 1:
        return posedge[0], None, None

    # Case 3: only posedge, multiple signals — active-high reset.
    # Pick the signal with higher global frequency as the clock; the
    # other is the reset. Ties broken by lexical order.
    sorted_by_freq = sorted(
        posedge,
        key=lambda s: (-posedge_global_freq.get(s, 0), s),
    )
    return sorted_by_freq[0], sorted_by_freq[1], "high"


def _detect_clock_reset_pairs_and_resets(
    trees: List[Tuple[str, Any]],
) -> Tuple[
    Set[str],                                # clock_signals
    Set[str],                                # reset_signals
    Dict[str, str],                          # reset_polarity
    Set[Tuple[str, Optional[str]]],          # clock_reset_pairs
    Dict[str, Tuple[Optional[str], str]],    # reset_values
]:
    """
    Walk every sequential ``AlwaysBlock`` once and produce:

    1. The set of clock signals
    2. The set of reset signals
    3. The polarity of each reset (active-low or active-high)
    4. Distinct (clock, reset) clock-domain pairs
    5. ``{register_name: (reset_signal_name, reset_value_text)}`` for every
       register assigned in any reset branch.

    This is multi-clock-domain aware: each block's (clock, reset) pair is
    detected independently from its sensitivity list, so designs with
    multiple clock domains record all pairs and tag each register with
    its driving reset.
    """
    # Pass 1: count posedge frequency across all sequential blocks.
    # Used to disambiguate "which posedge signal is the clock" when a
    # block has multiple posedge signals (active-high reset case).
    posedge_global_freq: Dict[str, int] = {}
    block_classes: List[Tuple[Any, Optional[str], Optional[str], Optional[str]]] = []

    for _path, root in _roots(trees):
        for node in _walk_syntax(root):
            if _kind_name(node) != 'AlwaysBlock':
                continue
            posedge, negedge = _detect_block_clock_and_reset(node)
            if not posedge:
                continue  # combinational
            for s in posedge:
                posedge_global_freq[s] = posedge_global_freq.get(s, 0) + 1
            block_classes.append((node, posedge, negedge))

    # Pass 2: classify each block now that we know global frequencies.
    clock_signals: Set[str] = set()
    reset_signals: Set[str] = set()
    reset_polarity: Dict[str, str] = {}
    clock_reset_pairs: Set[Tuple[str, Optional[str]]] = set()
    reset_values: Dict[str, Tuple[Optional[str], str]] = {}

    for node, posedge, negedge in block_classes:
        clock, reset, polarity = _classify_block(
            posedge, negedge, posedge_global_freq,
        )
        if clock is None:
            continue

        clock_signals.add(clock)
        clock_reset_pairs.add((clock, reset))

        if reset is None:
            continue

        reset_signals.add(reset)
        reset_polarity.setdefault(reset, polarity)

        # Walk this block to extract reset values.
        # The reset branch is identified by a ConditionalStatement whose
        # predicate references this reset signal with the right polarity.
        _extract_reset_values_from_block(
            node, reset, polarity, reset_values,
        )

    return (
        clock_signals,
        reset_signals,
        reset_polarity,
        clock_reset_pairs,
        reset_values,
    )


def _predicate_matches_reset(
    pred_text: str,
    reset_signal: str,
    polarity: str,
) -> bool:
    """
    Check whether a conditional predicate matches a reset condition.

    Active-low: ``!rst`` (negation of the reset signal)
    Active-high: ``rst`` (bare reset signal)
    """
    pred_text = pred_text.strip()
    # Strip outer parens.
    while pred_text.startswith('(') and pred_text.endswith(')'):
        inner = pred_text[1:-1].strip()
        # Only strip if balanced.
        depth = 0
        balanced = True
        for ch in inner:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth < 0:
                    balanced = False
                    break
        if balanced and depth == 0:
            pred_text = inner
        else:
            break

    if polarity == "low":
        return pred_text == f"!{reset_signal}" or pred_text == f"! {reset_signal}"
    else:  # high
        return pred_text == reset_signal


def _extract_reset_values_from_block(
    block_node: Any,
    reset_signal: str,
    polarity: str,
    reset_values: Dict[str, Tuple[Optional[str], str]],
) -> None:
    """
    Find the reset branch inside a single sequential always block and
    extract ``(register, value)`` pairs into ``reset_values``.

    The reset branch is the THEN-clause of the FIRST conditional whose
    predicate matches the reset signal with the expected polarity.
    """
    for cond in _walk_syntax(block_node):
        if _kind_name(cond) != 'ConditionalStatement':
            continue
        try:
            pred_text = str(cond.predicate).strip()
        except AttributeError:
            continue

        if not _predicate_matches_reset(pred_text, reset_signal, polarity):
            continue

        try:
            then_branch = cond.statement
        except AttributeError:
            continue
        if then_branch is None:
            continue

        # Walk the then-branch for assignments. Stop at any nested
        # ConditionalStatement (else-if branches aren't reset values).
        def _walk_until_nested_if(n):
            nk = _kind_name(n)
            if nk == 'ConditionalStatement' and n is not cond:
                return
            if nk in {'AssignmentExpression',
                      'NonblockingAssignmentExpression'}:
                try:
                    lhs_text = str(n.left).strip()
                    rhs_text = str(n.right).strip()
                except AttributeError:
                    return
                base = _lhs_base(lhs_text)
                if base and base not in reset_values:
                    reset_values[base] = (reset_signal, rhs_text)
            try:
                for child in n:
                    _walk_until_nested_if(child)
            except TypeError:
                pass

        _walk_until_nested_if(then_branch)
        break  # only the first matching conditional in this block


def _extract_signal_frequencies(
    trees: List[Tuple[str, Any]],
) -> Dict[str, int]:
    """
    Count ``IdentifierName`` references across every parsed tree.

    Returns a flat ``{signal_name: count}`` mapping. Used by the prompt
    formatter to rank signals during truncation: signals referenced more
    often in the RTL are more likely to appear in assertions, so they
    survive when sections are truncated to fit a token budget.

    This is a structural count (only ``IdentifierName`` nodes), so it
    excludes keywords, literals, and operator tokens. It DOES include
    every reference (LHS, RHS, expression operands, sensitivity lists,
    instance port hookups), so a signal heavily used in expressions will
    rank higher than one declared but rarely referenced.
    """
    freqs: Dict[str, int] = {}
    for _path, root in _roots(trees):
        for node in _walk_syntax(root):
            if _kind_name(node) != 'IdentifierName':
                continue
            try:
                name = str(node).strip()
            except Exception:
                continue
            if name.isidentifier():
                freqs[name] = freqs.get(name, 0) + 1
    return freqs


def _extract_signal_cooccurrence(
    trees: List[Tuple[str, Any]],
    skip_signals: Optional[Set[str]] = None,
    valid_signals: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """Build a signal co-occurrence graph at the RTL-statement level.

    Two signals A and B are recorded as co-occurring iff they both
    appear (as IdentifierName references) in the same single statement
    — either a ``ContinuousAssign`` body or a single
    ``AssignmentExpression`` inside an always block.  We also propagate
    the surrounding *condition* of an always-block assignment (the
    selector of an enclosing ``CaseStatement`` or the predicate of an
    enclosing ``ConditionalStatement``) into the co-occurrence set,
    because conditional gating is a real functional relationship.

    Returns a symmetric dict mapping each signal to the set of other
    signals it co-occurs with.

    Parameters
    ----------
    trees
        Parsed pyslang syntax trees from ``_parse_rtl_files``.
    skip_signals
        Signals to exclude entirely from the graph (typically clocks
        and resets — they appear in nearly every block and would
        dominate the affinity calculation).
    valid_signals
        If provided, restrict the graph to identifiers that appear in
        this set.  Filters out spurious tokens from the AST walker
        (e.g., parameter names, generate-loop indices).

    Notes
    -----
    Self-pairs are not recorded.  The graph is built per statement, so
    its complexity is O(N) in the number of identifier references —
    i.e., proportional to the size of the parsed RTL.  On the largest
    NVDLA module we have (~262 KLOC) this finishes in a few seconds.
    """
    skip = skip_signals or set()
    valid = valid_signals
    cooc: Dict[str, Set[str]] = {}

    def _add_pair(a: str, b: str) -> None:
        if a == b or a in skip or b in skip:
            return
        if valid is not None and (a not in valid or b not in valid):
            return
        cooc.setdefault(a, set()).add(b)
        cooc.setdefault(b, set()).add(a)

    def _idents_in(node) -> Set[str]:
        """Collect all valid IdentifierName tokens reachable from `node`."""
        out: Set[str] = set()
        try:
            for n in _walk_syntax(node):
                if _kind_name(n) == "IdentifierName":
                    try:
                        name = str(n).strip()
                    except (AttributeError, TypeError):
                        continue
                    if name and name.isidentifier():
                        out.add(name)
        except RecursionError:
            # Pathologically deep subtree — bail without partial data.
            return set()
        return out

    def _link_set(sigs: Set[str]) -> None:
        if len(sigs) < 2:
            return
        sigs_list = list(sigs)
        for i, a in enumerate(sigs_list):
            for b in sigs_list[i + 1:]:
                _add_pair(a, b)

    # Walk per-tree.  For every statement-bearing node we gather the
    # Both kinds of assignment expression contribute: blocking `=`
    # (AssignmentExpression) and nonblocking `<=` for sequential
    # registers (NonblockingAssignmentExpression).  Missing the
    # nonblocking form would lose every register's co-occurrences.
    _ASSIGN_KINDS = ("AssignmentExpression",
                     "NonblockingAssignmentExpression")

    for _path, root in _roots(trees):
        for node in _walk_syntax(root):
            kind = _kind_name(node)

            if kind == "ContinuousAssign":
                # `assign x = a + b;` → {x, a, b} all co-occur.
                for inner in _walk_syntax(node):
                    if _kind_name(inner) in _ASSIGN_KINDS:
                        _link_set(_idents_in(inner))
                continue

            if kind == "AlwaysBlock":
                # Walk down and collect each assignment's signal set
                # together with any enclosing conditional/case predicates.
                # We approximate "enclosing predicate" by walking the
                # always-block body once and tracking the most recent
                # predicate context — pyslang doesn't expose parent
                # pointers directly, so we re-walk with a context stack.
                _walk_always_for_cooccurrence(
                    node, _link_set, _idents_in, _ASSIGN_KINDS,
                )
                continue

    return cooc


def _walk_always_for_cooccurrence(
    block_node,
    link_set,
    idents_in,
    assign_kinds: Tuple[str, ...] = ("AssignmentExpression",
                                     "NonblockingAssignmentExpression"),
) -> None:
    """Walk an always-block body, tracking enclosing
    ``CaseStatement.expr`` / ``ConditionalStatement.predicate`` /
    ``IfStatement.cond`` so that gating signals get linked with the
    statements they gate.

    Calls ``link_set(signals)`` for each leaf assignment expression
    (blocking or nonblocking — controlled by ``assign_kinds``),
    where ``signals`` is the union of:
      * identifiers in the assignment's LHS and RHS, and
      * identifiers in every enclosing predicate / case-selector along
        the path from the always-block root down to the assignment.
    """
    # Iterative DFS with a per-path predicate-context stack.
    stack = [(block_node, set())]   # (node, accumulated predicate idents)
    while stack:
        node, ctx = stack.pop()
        kind = _kind_name(node)
        new_ctx = ctx
        if kind == "CaseStatement":
            try:
                new_ctx = ctx | idents_in(node.expr)
            except (AttributeError, TypeError):
                new_ctx = ctx
        elif kind in ("ConditionalStatement", "IfStatement"):
            # pyslang exposes the predicate on different attribute names
            # depending on syntax kind; try each cheaply.
            for attr in ("predicate", "cond", "condition"):
                pred = getattr(node, attr, None)
                if pred is not None:
                    try:
                        new_ctx = ctx | idents_in(pred)
                    except (AttributeError, TypeError):
                        pass
                    break
        elif kind in assign_kinds:
            sig_set = idents_in(node) | new_ctx
            link_set(sig_set)
            # Don't descend further into an assignment expression — its
            # children are already covered by `idents_in`.
            continue
        # Recurse with the updated context.
        try:
            for child in node:
                stack.append((child, new_ctx))
        except TypeError:
            pass


def _extract_signal_dataflow(
    trees: List[Tuple[str, Any]],
    valid_signals: Optional[Set[str]] = None,
    skip_signals: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Dict[str, Set[str]]], Dict[Tuple[str, str], int]]:
    """Extract a directional dataflow graph from RTL.

    For each ``LHS = RHS`` statement (continuous assign or always-block
    assignment), every identifier in RHS is recorded as **upstream** of
    every identifier in LHS, and conversely every LHS identifier is
    **downstream** of every RHS identifier.  The minimum cycle delay
    between a driver/driven pair is also tracked: ``0`` for purely
    combinational paths (continuous assigns and combinational always
    blocks) and ``1`` for assignments inside a clocked always block.

    The returned ``dataflow`` dict has shape::

        {
          signal: {
            "upstream":   {signals that drive this one},
            "downstream": {signals this one drives},
          },
          ...
        }

    The ``delay`` dict is keyed by ``(driver, driven)`` tuples and
    stores the minimum cycle distance observed across all statements
    that link them.

    Parameters
    ----------
    trees
        Parsed pyslang syntax trees from ``_parse_rtl_files``.
    valid_signals
        If provided, restrict the graph to these identifiers (typically
        ``facts.all_signals`` to filter spurious tokens).
    skip_signals
        Signals to exclude entirely (clocks and resets — they appear in
        sensitivity lists and reset branches without representing a real
        functional dataflow relationship).

    Notes
    -----
    Both LHS and RHS identifier extraction uses the existing
    ``IdentifierName``-walk pattern.  Bit-selects and hierarchical refs
    are resolved to their base identifier.  The delay tier is purely
    structural ("did this assignment appear inside a posedge/negedge
    always block?"); we do not attempt deep flop-chain analysis here.
    """
    valid = valid_signals
    skip = skip_signals or set()

    flow: Dict[str, Dict[str, Set[str]]] = {}
    delay: Dict[Tuple[str, str], int] = {}

    def _accept(s: str) -> bool:
        if not s or not s.isidentifier():
            return False
        if s in skip:
            return False
        if valid is not None and s not in valid:
            return False
        return True

    def _add(driver: str, driven: str, d: int) -> None:
        if not _accept(driver) or not _accept(driven) or driver == driven:
            return
        flow.setdefault(driver, {"upstream": set(), "downstream": set()})
        flow.setdefault(driven, {"upstream": set(), "downstream": set()})
        flow[driver]["downstream"].add(driven)
        flow[driven]["upstream"].add(driver)
        key = (driver, driven)
        # Keep the *minimum* observed delay (combinational beats sequential
        # if the same pair appears in both).
        if key not in delay or d < delay[key]:
            delay[key] = d

    def _idents_in(node) -> Set[str]:
        out: Set[str] = set()
        try:
            for n in _walk_syntax(node):
                if _kind_name(n) == "IdentifierName":
                    try:
                        name = str(n).strip()
                    except (AttributeError, TypeError):
                        continue
                    if name and name.isidentifier():
                        out.add(name)
        except RecursionError:
            return set()
        return out

    def _link(lhs_sigs: Set[str], rhs_sigs: Set[str], d: int) -> None:
        for r in rhs_sigs:
            for l in lhs_sigs:
                _add(r, l, d)

    # Pyslang distinguishes two assignment expression kinds:
    #   • AssignmentExpression           — blocking assigns (`=`).
    #   • NonblockingAssignmentExpression — sequential `<=` in clocked
    #                                       always blocks.
    # We accept both; a sequential always block typically uses the
    # nonblocking form, and missing it would lose every register's
    # dataflow relationships.
    _ASSIGN_KINDS = ("AssignmentExpression",
                     "NonblockingAssignmentExpression")

    for _path, root in _roots(trees):
        for node in _walk_syntax(root):
            kind = _kind_name(node)

            # Continuous assigns are always combinational (delay=0).
            if kind == "ContinuousAssign":
                for inner in _walk_syntax(node):
                    if _kind_name(inner) in _ASSIGN_KINDS:
                        try:
                            lhs_sigs = _idents_in(inner.left)
                            rhs_sigs = _idents_in(inner.right)
                        except (AttributeError, TypeError):
                            continue
                        _link(lhs_sigs, rhs_sigs, 0)
                continue

            # Always blocks: classify the block, then walk its body.
            if kind == "AlwaysBlock":
                try:
                    block_text = str(node)
                except Exception:
                    block_text = ""
                is_seq = ("posedge" in block_text or "negedge" in block_text)
                d = 1 if is_seq else 0
                for inner in _walk_syntax(node):
                    if _kind_name(inner) in _ASSIGN_KINDS:
                        try:
                            lhs_sigs = _idents_in(inner.left)
                            rhs_sigs = _idents_in(inner.right)
                        except (AttributeError, TypeError):
                            continue
                        _link(lhs_sigs, rhs_sigs, d)
                continue

    return flow, delay


def _direction_token_to_str(tok) -> Optional[str]:
    """Map a pyslang direction Token to a canonical string."""
    if tok is None:
        return None
    try:
        s = str(tok).lower()
    except Exception:
        return None
    if "input" in s:
        return "input"
    if "output" in s:
        return "output"
    if "inout" in s:
        return "inout"
    return None


def _identifier_token_text(tok) -> str:
    """Pull the cleaned identifier text from a pyslang Token.

    Robust to two artefacts of pyslang's text rendering:
      • Verbose repr ``Token(TokenKind.Identifier, "name")`` — peeled
        with a quote split.
      • Leading **trivia** such as line comments and whitespace (the
        Chisel emitter places ``// src/...:line:col`` annotations
        directly before each port name, so ``str(tok)`` returns
        ``"// src/...\\n               reset"`` instead of ``"reset"``).
        We strip everything up to the last whitespace boundary and
        validate the remainder as an identifier.
    """
    if tok is None:
        return ""
    try:
        s = str(tok)
    except Exception:
        return ""
    s = s.strip()
    if not s:
        return ""
    # Verbose Token repr: ``Token(TokenKind.Identifier, "name")``.
    if s.startswith("Token(") and '"' in s:
        try:
            s = s.split('"')[1]
        except IndexError:
            pass
    # If trivia (line comments / whitespace) is fused into the text,
    # the actual identifier is the trailing token.  Fall back to a
    # regex extraction of the last identifier-shaped substring.
    if not s.isidentifier():
        # Strip line-comments first so we never accidentally pull the
        # last word of a ``// ... name`` style trivia comment.
        s_no_comments = re.sub(r"//[^\n]*", "", s).strip()
        m = re.search(r"([A-Za-z_]\w*)\s*$", s_no_comments)
        if m:
            s = m.group(1)
    return s


def _extract_signal_port_info(
    trees: List[Tuple[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Extract per-signal port direction and instance-connection info.

    Returns a dict keyed by signal name with values::

        {
          "direction":   "input" | "output" | "inout" | "internal",
          "port_module": str | None,
          "connections": [
              (parent_module, instance_name, instance_type, port_name),
              ...
          ]
        }

    Walks each parsed tree, tracking the enclosing ``ModuleDeclaration``
    via a context stack so that ``PortDeclaration`` and
    ``HierarchyInstantiation`` nodes can be attributed to the correct
    module without needing parent pointers.

    Notes
    -----
    Pyslang surfaces port declarations in two distinct shapes:
      • **NonAnsi** (NVDLA style): separate ``PortDeclaration`` nodes
        inside the module body declare the direction of each port that
        was named in the (direction-less) port list.
      • **Ansi** (Chisel-emitted FifoX, modern SV): each port lives
        as an ``ImplicitAnsiPort`` (or ``ExplicitAnsiPort``) inside the
        ``AnsiPortList``.  The first port carries an explicit
        ``header.direction`` token; subsequent ports inherit until a
        new direction keyword is introduced.
    Both forms are handled here; for the Ansi case we walk the port
    list in order and propagate the most recent direction.
    """
    info: Dict[str, Dict[str, Any]] = {}

    def _ensure(sig: str) -> Dict[str, Any]:
        if sig not in info:
            info[sig] = {
                "direction":    "internal",
                "port_module":  None,
                # Set of all modules where this signal name appears
                # as a port.  Necessary because the same name can be
                # used as a port in multiple modules (e.g., a
                # ``req`` port that exists in both a parent and its
                # submodule with matching local-scope semantics).  The
                # singular ``port_module`` field stays for backwards
                # compatibility (last-writer-wins).
                "port_modules": set(),
                "connections":  [],
            }
        return info[sig]

    def _record_ansi_port_list(port_list_node, mod: str) -> None:
        """Walk an AnsiPortList in order, propagating direction across
        ports that omit it (the ``input a, b, c`` shorthand)."""
        try:
            ports = list(port_list_node.ports)
        except (AttributeError, TypeError):
            return
        current_direction: Optional[str] = None
        for p in ports:
            pkind = _kind_name(p)
            if pkind not in ("ImplicitAnsiPort", "ExplicitAnsiPort"):
                continue
            # Try the header's direction first.  When absent (inherited
            # from an earlier port), fall back to current_direction.
            d = None
            try:
                d = _direction_token_to_str(p.header.direction)
            except (AttributeError, TypeError):
                d = None
            if d is not None:
                current_direction = d
            direction = current_direction or "internal"
            # Extract the port name from the declarator.
            name = ""
            try:
                name = _identifier_token_text(p.declarator.name)
            except (AttributeError, TypeError):
                pass
            if not name or not name.isidentifier():
                continue
            rec = _ensure(name)
            # Don't overwrite a stronger direction from PortDeclaration.
            if rec["direction"] == "internal" or rec["port_module"] is None:
                rec["direction"] = direction
                rec["port_module"] = mod
            rec["port_modules"].add(mod)

    for _path, root in _roots(trees):
        # Walk with an explicit stack tracking enclosing module name.
        # Iterative DFS preserves order so we always know which
        # ModuleDeclaration we're inside when we hit a PortDeclaration
        # or a HierarchyInstantiation.
        stack: List[Tuple[Any, Optional[str]]] = [(root, None)]
        while stack:
            node, mod = stack.pop()
            kind = _kind_name(node)
            new_mod = mod
            if kind == "ModuleDeclaration":
                try:
                    new_mod = _identifier_token_text(node.header.name)
                except (AttributeError, TypeError):
                    new_mod = None
                # If the header carries an Ansi port list, record it
                # immediately — Ansi ports rely on ordered traversal of
                # the port list itself, not on later PortDeclaration
                # statements.
                if new_mod:
                    try:
                        plist = node.header.ports
                    except (AttributeError, TypeError):
                        plist = None
                    if plist is not None and _kind_name(plist) == "AnsiPortList":
                        _record_ansi_port_list(plist, new_mod)
            elif kind == "PortDeclaration" and mod is not None:
                # NonAnsi style: explicit ``input/output X;`` statements
                # inside the module body, possibly listing multiple
                # signals per declaration.
                try:
                    direction = _direction_token_to_str(node.header.direction)
                except (AttributeError, TypeError):
                    direction = None
                if direction:
                    try:
                        decls = list(node.declarators)
                    except (AttributeError, TypeError):
                        decls = []
                    for d in decls:
                        try:
                            name = _identifier_token_text(d.name)
                        except (AttributeError, TypeError):
                            continue
                        if not name or not name.isidentifier():
                            continue
                        rec = _ensure(name)
                        rec["direction"] = direction
                        rec["port_module"] = mod
                        rec["port_modules"].add(mod)
            elif kind == "HierarchyInstantiation" and mod is not None:
                # Extract the instance type and walk port connections.
                try:
                    inst_type = _identifier_token_text(node.type)
                except (AttributeError, TypeError):
                    inst_type = ""
                try:
                    instances = list(node.instances)
                except (AttributeError, TypeError):
                    instances = []
                for inst in instances:
                    try:
                        inst_name = _identifier_token_text(inst.decl.name)
                    except (AttributeError, TypeError):
                        # Fall back to walking for InstanceName token.
                        inst_name = ""
                        for sub in _walk_syntax(inst):
                            if _kind_name(sub) == "InstanceName":
                                try:
                                    inst_name = _identifier_token_text(sub.name)
                                except AttributeError:
                                    pass
                                break
                    # Record every NamedPortConnection inside this
                    # instance.  Treat the connected expression as a
                    # string and pull its identifier names — a
                    # connection like ``.req(top_req)`` records
                    # ``top_req`` as connected to port ``req`` on
                    # ``inst_type``.
                    for sub in _walk_syntax(inst):
                        if _kind_name(sub) != "NamedPortConnection":
                            continue
                        try:
                            port_name = _identifier_token_text(sub.name)
                        except (AttributeError, TypeError):
                            port_name = ""
                        if not port_name:
                            continue
                        # Identifier(s) in the connected expression.
                        try:
                            expr_node = sub.expr
                        except (AttributeError, TypeError):
                            expr_node = None
                        if expr_node is None:
                            continue
                        for inner in _walk_syntax(expr_node):
                            if _kind_name(inner) != "IdentifierName":
                                continue
                            try:
                                sig = str(inner).strip()
                            except (AttributeError, TypeError):
                                continue
                            if not sig or not sig.isidentifier():
                                continue
                            rec = _ensure(sig)
                            rec["connections"].append(
                                (mod, inst_name, inst_type, port_name)
                            )
            # Recurse with updated module context.
            try:
                for child in node:
                    stack.append((child, new_mod))
            except TypeError:
                pass

    return info


# Handshake suffix pairs.  Order matters within the longer-first rule
# in the extractor: ``pvld``/``prdy`` is matched before ``vld``/``rdy``
# to avoid ``foo_pvld`` being misclassified as ``foo_p`` + ``vld``.
_HANDSHAKE_SUFFIX_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    # NVDLA-flavoured short forms (must come before generic vld/rdy).
    ("pvld",     "prdy",     "valid_ready"),
    # Generic SV/AXI valid/ready.
    ("valid",    "ready",    "valid_ready"),
    # Short req/ack — match the most-specific tokens first.
    ("req_valid", "req_ready", "valid_ready"),
    ("req",      "ack",      "req_ack"),
    # FIFO push/pop control.
    ("push",     "pop",      "push_pop"),
    ("wr_en",    "rd_en",    "push_pop"),
    # Start/done style (typical AXI-Lite / config-pulse).
    ("start",    "done",     "start_done"),
    ("en",       "done",     "start_done"),
    # Common DMA pattern.
    ("rd_req",   "rd_rsp",   "valid_ready"),
    ("wr_req",   "wr_rsp",   "valid_ready"),
)


def _signals_share_link(
    a: str, b: str, facts: RTLFacts,
) -> bool:
    """Return True when two signals are confirmed-related via dataflow
    (direct or 1-hop) or via shared module (port-list or signal-to-
    module)."""
    flow = facts.signal_dataflow or {}
    sig2mod = facts.signal_to_module or {}
    port_info = facts.signal_port_info or {}
    # Direct: a drives b OR b drives a (either polarity counts as "linked").
    a_down = flow.get(a, {}).get("downstream", set())
    b_down = flow.get(b, {}).get("downstream", set())
    if b in a_down or a in b_down:
        return True
    # 1-hop transitive.
    a_up = flow.get(a, {}).get("upstream", set())
    b_up = flow.get(b, {}).get("upstream", set())
    if (a_down & b_down) or (a_up & b_up) or (a_down & b_up) or (a_up & b_down):
        return True
    # Shared-module residency.
    pi_a = port_info.get(a, {})
    pi_b = port_info.get(b, {})
    mods_a = set(pi_a.get("port_modules") or set())
    mods_b = set(pi_b.get("port_modules") or set())
    if pi_a.get("port_module"): mods_a.add(pi_a["port_module"])
    if pi_b.get("port_module"): mods_b.add(pi_b["port_module"])
    if sig2mod.get(a): mods_a.add(sig2mod[a])
    if sig2mod.get(b): mods_b.add(sig2mod[b])
    if mods_a & mods_b:
        return True
    return False


def _extract_localparam_encodings(
    trees: List[Tuple[str, Any]],
) -> Dict[str, str]:
    """Walk every ``ParameterDeclaration`` / ``LocalParameterDeclaration``
    and pull ``name -> literal_value_text`` for parameters whose
    initialiser is a single literal (the common state-encoding shape).

    We accept any width-prefixed literal (``2'b00``, ``8'h0a``, ``3'd5``)
    or a bare integer, ignoring more complex initialisers.  Returned
    map is the union across the whole compilation; same-name
    collisions across modules keep the first observation.
    """
    encoding: Dict[str, str] = {}
    literal_re = re.compile(
        r"^\s*(\d+'[bdhoBDHO][0-9a-fA-FxXzZ_]+|0x[0-9a-fA-F]+|\d+)\s*$"
    )
    for _path, root in _roots(trees):
        for node in _walk_syntax(root):
            kind = _kind_name(node)
            if kind not in ("ParameterDeclaration",
                            "LocalParameterDeclaration"):
                continue
            try:
                decls = list(node.declarators)
            except (AttributeError, TypeError):
                continue
            for d in decls:
                try:
                    name = _identifier_token_text(d.name)
                except (AttributeError, TypeError):
                    continue
                if not name or not name.isidentifier():
                    continue
                if name in encoding:
                    continue
                # Initialiser may be on `d.initializer` or as a child node.
                init_text = ""
                init = getattr(d, "initializer", None)
                if init is not None:
                    try:
                        init_text = str(init).strip()
                    except Exception:
                        init_text = ""
                # Strip leading `=` if present (initialiser includes it
                # in some pyslang versions).
                if init_text.startswith("="):
                    init_text = init_text[1:].strip()
                if literal_re.match(init_text):
                    encoding[name] = init_text
    return encoding


def _extract_state_machines(
    trees: List[Tuple[str, Any]],
    facts: RTLFacts,
) -> List[Dict[str, Any]]:
    """Detect candidate state machines from ``case (X)`` statements
    where ``X`` is a sequentially-driven signal.

    Strategy:
      1. Walk every ``CaseStatement``.  Skip cases whose selector text
         is anything other than a single bare identifier (filters out
         RAM-style ``case (rd_addr)`` and concatenated selectors).
      2. The selector identifier must be registered: present in
         ``facts.signal_drive_kind`` with value ``"seq"``.  Exception:
         when drive-kind info is missing (e.g., HLS-emitted code), we
         additionally accept selectors whose name contains ``state``
         or ``fsm`` (substring) so common idiomatic names are still
         picked up.
      3. From each ``CaseItem``, extract the case-pattern expressions:
         ``IdentifierName`` tokens become "state names",
         ``IntegerVectorExpression`` / ``IntegerLiteralExpression``
         tokens become "literal values".
      4. Collapse duplicates across multiple ``case`` blocks on the
         same selector signal (parent block can have multiple
         ``case`` constructs that all enumerate the same state set).

    The encoding map is filled by cross-referencing state-name
    identifiers against the localparam encodings extracted by
    ``_extract_localparam_encodings``.
    """
    drive = facts.signal_drive_kind or {}
    sig2mod = facts.signal_to_module or {}
    encodings = _extract_localparam_encodings(trees)

    # Collected per state-signal so we can collapse duplicates.
    by_signal: Dict[str, Dict[str, Any]] = {}
    selector_id_re = re.compile(r"^[A-Za-z_]\w*$")
    state_name_hint = re.compile(r"state|fsm|cstate|nstate", re.IGNORECASE)

    for _path, root in _roots(trees):
        for node in _walk_syntax(root):
            if _kind_name(node) != "CaseStatement":
                continue
            try:
                sel_text = str(node.expr).strip()
            except (AttributeError, TypeError):
                continue
            # Strip a single layer of parentheses from the selector.
            if sel_text.startswith("(") and sel_text.endswith(")"):
                sel_text = sel_text[1:-1].strip()
            if not selector_id_re.match(sel_text):
                continue
            # Selector must be registered (or naming-suggestive when
            # drive-kind info is absent).
            kind_val = drive.get(sel_text)
            if kind_val != "seq":
                if not (kind_val is None and state_name_hint.search(sel_text)):
                    continue

            entry = by_signal.setdefault(sel_text, {
                "state_signal": sel_text,
                "module":       sig2mod.get(sel_text),
                "states":       [],
                "literal_values": [],
                "encoding":     {},
                "_seen_states": set(),
                "_seen_lits":   set(),
            })

            # Walk case items and collect pattern tokens.  Use
            # StandardCaseItem (the typical ``LABEL: stmt;`` form);
            # skip default items (DefaultCaseItem) which add no
            # state-name info.
            for inner in _walk_syntax(node):
                ikind = _kind_name(inner)
                if ikind != "StandardCaseItem":
                    continue
                # Walk only the EXPRESSIONS (case label patterns), not
                # the clause body, to avoid pulling in identifiers that
                # appear inside the branch's statements.
                try:
                    exprs = inner.expressions
                except (AttributeError, TypeError):
                    continue
                for sub in _walk_syntax(exprs):
                    skind = _kind_name(sub)
                    if skind == "IdentifierName":
                        try:
                            nm = str(sub).strip()
                        except (AttributeError, TypeError):
                            continue
                        if not nm or not nm.isidentifier():
                            continue
                        # Skip the selector itself appearing in the
                        # pattern list (rare but possible).
                        if nm == sel_text:
                            continue
                        if nm not in entry["_seen_states"]:
                            entry["states"].append(nm)
                            entry["_seen_states"].add(nm)
                            if nm in encodings:
                                entry["encoding"][nm] = encodings[nm]
                    elif skind in ("IntegerVectorExpression",
                                   "IntegerLiteralExpression"):
                        try:
                            lit = str(sub).strip()
                        except (AttributeError, TypeError):
                            continue
                        if not lit:
                            continue
                        if lit not in entry["_seen_lits"]:
                            entry["literal_values"].append(lit)
                            entry["_seen_lits"].add(lit)

    # Finalise: drop the internal _seen_* sets, compute n_states,
    # and apply post-filters that distinguish FSMs from counters and
    # boilerplate.
    out: List[Dict[str, Any]] = []
    for sig, entry in by_signal.items():
        entry.pop("_seen_states", None)
        entry.pop("_seen_lits", None)
        entry["n_states"] = len(entry["states"]) + len(entry["literal_values"])

        # Filter 1 — degenerate.  A real FSM has at least two reachable
        # states; a single-branch case is usually HLS boilerplate (e.g.
        # ``state_var`` with one ``main_C_0`` case in a stub module).
        if entry["n_states"] < 2:
            continue
        # Filter 2 — counter / decode table.  A case with NO named
        # state identifiers and many literal patterns (>6) looks like
        # a counter or an address-decode table, not a state machine.
        # Most real FSMs use named ``localparam`` constants rather
        # than bare literals.
        if (not entry["states"]
                and len(entry["literal_values"]) > 6):
            continue
        # Filter 3 — counter-style naming.  Selectors named ``*_cnt``,
        # ``*_count``, ``*_idx``, ``*_index`` are loops or counters,
        # not FSMs.  This catches pdp's ``pooling_out_cnt`` / similar.
        low = sig.lower()
        if any(low.endswith(suf) for suf in
               ("_cnt", "_count", "_idx", "_index", "_addr",
                "_ptr", "_pointer", "_num", "_size")):
            continue
        out.append(entry)
    return out


# Naming-pattern hints for role classification.  Order matters: the
# more-specific patterns are checked first so ``foo_addr_valid`` is
# tagged as a handshake initiator (via the structured extractor) and
# never falls through to the address-name heuristic.
_NAME_HINT_DATA   = re.compile(r"_data\b|_pd\b|_payload\b|_dout\b|_din\b",
                               re.IGNORECASE)
_NAME_HINT_ADDR   = re.compile(r"_addr\b|_address\b|_baseaddr\b",
                               re.IGNORECASE)
_NAME_HINT_STATUS = re.compile(r"_status\b|_done\b|_busy\b|_idle\b|_err(or)?\b",
                               re.IGNORECASE)


# Modal-verb / requirement-phrase patterns we consider as evidence
# that a sentence states a verifiable property.  Order is informational
# only — the matcher returns the first that fires.
_DOC_MODAL_RE = re.compile(
    r"\b(must\s+(?:be\s+|not\s+|always\s+|remain\s+|hold\s+|equal\s+|maintain\s+)?"
    r"|shall\s+(?:be\s+|not\s+|remain\s+|always\s+)?"
    r"|should\s+(?:be\s+|not\s+|remain\s+)?"
    r"|required\s+to\s+(?:be\s+|remain\s+)?"
    r"|expected\s+to\s+(?:be\s+|remain\s+)?"
    r"|cannot\s+(?:be\s+|exceed\s+|drop\s+)?"
    r"|never\s+(?:be\s+|exceed\s+|drop\s+|equal\s+)?"
    r"|always\s+(?:remain\s+|hold\s+|be\s+)?)",
    re.IGNORECASE,
)

# Tokens we never consider as signal references even if they appear
# in the doc text (English words that overlap with valid SV identifiers).
_DOC_STOPWORDS = {
    "the", "a", "an", "is", "are", "be", "must", "shall", "should",
    "will", "may", "can", "and", "or", "not", "of", "in", "on", "to",
    "from", "with", "by", "for", "this", "that", "these", "those",
    "it", "its", "as", "if", "then", "else", "when", "while", "always",
    "never", "remain", "hold", "equal", "maintain", "cycle", "cycles",
    "valid", "ready", "data", "signal", "value", "reset", "clock",
    "after", "before", "during", "until", "next", "previous", "high",
    "low", "true", "false", "active", "inactive",
}


def _split_sentences(text: str) -> List[Tuple[int, str]]:
    """Split free-form text into (line_no, sentence) pairs.

    Cheap heuristic: split on `.`, `!`, `?` followed by whitespace, then
    on bare newlines (markdown bullets / tables).  Returns the 1-based
    line number where each sentence STARTS so the source pointer in
    ``documented_properties`` is accurate enough for the LLM and human
    review.
    """
    out: List[Tuple[int, str]] = []
    line_no = 1
    buf: List[str] = []
    sentence_start = 1
    for ln in text.splitlines(keepends=True):
        stripped = ln.strip()
        # Skip blockquotes and code fences entirely; they're not prose.
        if stripped.startswith("```") or stripped.startswith("~~~"):
            line_no += 1
            continue
        # Split this line into mini-sentences on `.`, `!`, `?`.
        parts = re.split(r"(?<=[.!?])\s+", stripped)
        for i, part in enumerate(parts):
            if not part:
                continue
            buf.append(part)
            # If the part ended with a sentence terminator OR this is
            # the last segment of a non-empty paragraph line, flush.
            if part.endswith((".", "!", "?")):
                joined = " ".join(buf).strip()
                if joined:
                    out.append((sentence_start, joined))
                buf = []
                sentence_start = line_no
        # End of line: also flush a partial buffer at line breaks so
        # bullet items don't all merge into one sentence.
        if buf and stripped == "":
            joined = " ".join(buf).strip()
            if joined:
                out.append((sentence_start, joined))
            buf = []
            sentence_start = line_no + 1
        line_no += 1
    if buf:
        out.append((sentence_start, " ".join(buf).strip()))
    return out


def _extract_documented_properties(
    docs_dir: Optional[str],
    facts: RTLFacts,
    *,
    max_properties: int = 60,
) -> List[Dict[str, Any]]:
    """Walk ``docs_dir`` for .md / .txt / .rst files, extract every
    sentence containing a modal verb AND at least one signal name
    from ``facts.all_signals``, and return a structured list.

    We do NOT try to parse the property into formal SVA — that would
    require real NLP.  Instead we surface the raw sentence as
    "evidence of intent" so the LLM can encode it.

    Returns an empty list when ``docs_dir`` is missing or empty.
    """
    if not docs_dir:
        return []
    try:
        docs_root = Path(docs_dir)
    except Exception:
        return []
    if not docs_root.exists():
        return []
    if not facts.all_signals:
        return []

    # Build a quick lookup; case-sensitive (Verilog identifiers are).
    signal_set = facts.all_signals
    out: List[Dict[str, Any]] = []
    seen_keys: Set[Tuple[str, str]] = set()

    for path in sorted(docs_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".md", ".txt", ".rst", ".markdown"):
            continue
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        for line_no, sent in _split_sentences(text):
            # Cheap modal filter first.
            mm = _DOC_MODAL_RE.search(sent)
            if not mm:
                continue
            modal = mm.group(0).strip().split()[0].lower()
            # Find signal references — case-sensitive identifier match.
            tokens = re.findall(r"\b([A-Za-z_]\w*)\b", sent)
            mentioned = {t for t in tokens
                         if t in signal_set
                         and t.lower() not in _DOC_STOPWORDS
                         and len(t) > 2}
            if not mentioned:
                continue
            key = (modal, sent[:80])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out.append({
                "signals":      sorted(mentioned),
                "modal":        modal,
                "raw_sentence": sent[:240],
                "source_file":  str(path.relative_to(docs_root.parent)
                                    if docs_root.parent
                                    else path),
                "source_line":  line_no,
            })
            if len(out) >= max_properties:
                return out
    return out


def _classify_signal_roles(facts: RTLFacts) -> Dict[str, str]:
    """Categorise every signal in ``facts.all_signals`` by role.

    Layered classification (first match wins):
      1. Clock / reset / out-of-scope: from existing structural facts.
      2. Handshake initiator / responder: from ``handshake_pairs``.
      3. State signal: from ``state_machines``.
      4. Naming-pattern: data / address / status suffix matches.
      5. Width + drive-kind: 1-bit control vs multi-bit data fallback.
      6. Unknown: nothing else matched.

    The result is a flat dict ``{signal_name: role}`` for every signal
    we can confidently classify; signals that fall through to
    "unknown" are still recorded so the prompt formatter can show a
    coverage stat.
    """
    out: Dict[str, str] = {}
    if not facts.all_signals:
        return out

    handshake_init: Set[str] = set()
    handshake_resp: Set[str] = set()
    for p in facts.handshake_pairs:
        if p.get("confidence") == "high":
            handshake_init.add(p["initiator"])
            handshake_resp.add(p["responder"])
    state_signals = {f["state_signal"] for f in facts.state_machines}

    widths = facts.signal_widths or {}
    drive  = facts.signal_drive_kind or {}
    out_of_scope = facts.out_of_scope_signals or set()

    for sig in facts.all_signals:
        # 1. Clock / reset / out-of-scope.
        if sig in facts.clock_signals:
            out[sig] = "clock"; continue
        if sig in facts.reset_signals:
            out[sig] = "reset"; continue
        if sig in out_of_scope:
            out[sig] = "out_of_scope"; continue
        # 2. Handshake roles.
        if sig in handshake_init:
            out[sig] = "handshake_init"; continue
        if sig in handshake_resp:
            out[sig] = "handshake_resp"; continue
        # 3. State.
        if sig in state_signals:
            out[sig] = "state"; continue
        # 4. Naming patterns.
        if _NAME_HINT_ADDR.search(sig):
            out[sig] = "address"; continue
        if _NAME_HINT_DATA.search(sig):
            out[sig] = "data"; continue
        if _NAME_HINT_STATUS.search(sig):
            out[sig] = "status"; continue
        # 5. Width + drive-kind fallback.
        w = widths.get(sig)
        if w is not None:
            if w == 1:
                out[sig] = "control"; continue
            if w > 1:
                # Multi-bit register without a more-specific name: data.
                out[sig] = "data"; continue
        # 6. Unknown.
        out[sig] = "unknown"
    return out


def _extract_handshake_pairs(facts: RTLFacts) -> List[Dict[str, Any]]:
    """Detect handshake pairs (initiator, responder) using name-pattern
    matching plus dataflow / module confirmation.

    For every signal whose name ends in a known initiator suffix
    (``valid``, ``req``, ``pvld``, ``push``, ``start``, ``en``, ...),
    look for a sibling signal sharing the same prefix and ending in
    the matching responder suffix.  When a candidate pair is found:

      • If a dataflow link or shared-module residency exists between
        them, classify as ``confidence="high"``.
      • Otherwise, ``confidence="medium"`` — surface to the LLM but
        with weaker prior.

    Operates on already-extracted facts (``all_signals``,
    ``signal_dataflow``, ``signal_port_info``, ``signal_to_module``)
    rather than re-walking the AST, so it costs O(N) in the signal
    count and finishes in milliseconds.

    Skips clock and reset signals — they appear with similar suffixes
    on some designs (e.g., ``rst_valid``) but never represent a
    protocol.
    """
    sigs = facts.all_signals
    if not sigs:
        return []
    skip = (facts.clock_signals or set()) | (facts.reset_signals or set())
    sig2mod = facts.signal_to_module or {}
    port_info = facts.signal_port_info or {}

    # Order suffixes longest-first so that the more specific pattern
    # wins when both could match (e.g., a signal ending in
    # ``req_valid`` matches the ``req_valid``/``req_ready`` pair, not
    # the bare ``valid``/``ready`` pair).
    ordered_pairs = sorted(
        _HANDSHAKE_SUFFIX_PAIRS,
        key=lambda p: -max(len(p[0]), len(p[1])),
    )

    seen_pairs: Set[Tuple[str, str]] = set()
    out: List[Dict[str, Any]] = []
    for sig in sorted(sigs):
        if sig in skip:
            continue
        for suf_init, suf_resp, protocol in ordered_pairs:
            init_marker = f"_{suf_init}"
            if not sig.endswith(init_marker):
                continue
            prefix = sig[: -len(init_marker)]
            partner = f"{prefix}_{suf_resp}"
            if partner not in sigs or partner in skip:
                continue
            key = tuple(sorted((sig, partner)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            # Module residency — collect the union for both signals.
            modules: Set[str] = set()
            for s in (sig, partner):
                pi = port_info.get(s, {})
                modules.update(pi.get("port_modules") or set())
                if pi.get("port_module"):
                    modules.add(pi["port_module"])
                if sig2mod.get(s):
                    modules.add(sig2mod[s])
            confidence = "high" if _signals_share_link(sig, partner, facts) \
                else "medium"
            out.append({
                "initiator":  sig,
                "responder":  partner,
                "protocol":   protocol,
                "modules":    modules,
                "confidence": confidence,
            })
            break  # don't double-classify on multiple matching suffixes
    return out


def pipeline_depth(
    facts: RTLFacts,
    src: str,
    dst: str,
    max_depth: int = 8,
) -> Optional[int]:
    """Return the minimum cycle delay (number of flop stages) along the
    shortest path from ``src`` to ``dst`` in the dataflow graph.

    Returns:
      • ``0`` if ``src`` reaches ``dst`` purely through combinational
        logic (no flops on the path).
      • ``N ≥ 1`` if the shortest path traverses ``N`` flops.
      • ``None`` if no path is found within ``max_depth`` cycles, or
        if either signal is absent from the graph.

    Uses BFS-with-priority over cumulative cycle delay (Dijkstra-lite,
    since edge weights are 0 or 1).  Memoised on
    ``facts.pipeline_depth_cache`` so repeated queries are O(1).

    Bounded at ``max_depth`` cycles to keep traversal cost manageable
    on large designs (NVDLA SDP has ~50K signals; without the bound,
    a single query could fan out to thousands of nodes).  ``8`` covers
    the typical SoC pipeline depths we observe in practice; the
    validator can pass a larger bound when checking ``$past(d, N)``
    with explicit ``N``.
    """
    flow = facts.signal_dataflow
    delay = facts.signal_dataflow_delay
    if not flow or src not in flow:
        return None
    cache_key = (src, dst)
    if cache_key in facts.pipeline_depth_cache:
        cached = facts.pipeline_depth_cache[cache_key]
        # Return the cached value as long as it didn't time out at a
        # lower max_depth than the current request (None at lower
        # depth might be reachable at higher depth).
        # Conservative: only reuse positive results unconditionally.
        if cached is not None or max_depth <= 8:
            return cached

    # 0-1 BFS using a deque so that combinational hops (delay 0) are
    # explored before sequential hops (delay 1).  This guarantees the
    # first time we visit ``dst`` we hold the minimum cumulative delay.
    from collections import deque
    visited: Dict[str, int] = {src: 0}
    queue = deque([(src, 0)])
    result: Optional[int] = None
    while queue:
        cur, d = queue.popleft()
        if cur == dst:
            result = d
            break
        if d >= max_depth:
            continue
        for nxt in flow.get(cur, {}).get("downstream", set()):
            edge_d = delay.get((cur, nxt), 0)
            new_d = d + edge_d
            if new_d > max_depth:
                continue
            if nxt not in visited or visited[nxt] > new_d:
                visited[nxt] = new_d
                # 0-weight edges go to the front (BFS-by-depth); 1-weight
                # to the back.
                if edge_d == 0:
                    queue.appendleft((nxt, new_d))
                else:
                    queue.append((nxt, new_d))

    facts.pipeline_depth_cache[cache_key] = result
    return result


def _extract_all_signals(trees: List[Tuple[str, Any]]) -> Set[str]:
    """
    Every signal name declared anywhere in the RTL files.

    Sources:
    - ``Declarator`` nodes inside ``NetDeclaration``, ``DataDeclaration``,
      and port declarations — covers ``wire``, ``reg``, ``logic``, ``input``,
      ``output``, ``inout``.
    - Implicit signals appearing as LHS of ``AssignmentExpression`` (some
      designs assign to undeclared wires which Verilog allows by default).

    Used by ``validate_signals`` to drop assertions referencing hallucinated
    signal names.
    """
    sigs: Set[str] = set()

    for _path, root in _roots(trees):
        for node in _walk_syntax(root):
            # Direct identifier extraction from Declarator nodes.
            if _kind_name(node) == 'Declarator':
                name = str(node).strip()
                # Declarator text may include init: "sig = expr"
                # take just the bare identifier prefix.
                base = re.split(r'[\s=\[]', name, maxsplit=1)[0]
                if base.isidentifier():
                    sigs.add(base)

            # Implicit wires appearing as LHS of any assignment.
            elif _kind_name(node) == 'AssignmentExpression':
                try:
                    lhs_text = str(node.left).strip()
                except AttributeError:
                    continue
                base = _lhs_base(lhs_text)
                if base:
                    sigs.add(base)

    return sigs


# ---------------------------------------------------------------------------
# Single entry point
# ---------------------------------------------------------------------------

def extract_rtl_facts(
    rtl_dir: str,
    signal_map: Optional[Dict[str, Any]] = None,
    top_module: str = "",
    module_facts_mode: str = "off",
    detect_out_of_scope: bool = True,
    sink_module_patterns: Optional[List[str]] = None,
    out_of_scope_max_iterations: int = 10,
    docs_dir: Optional[str] = None,
) -> RTLFacts:
    """
    Parse RTL files once and extract all structured facts via pyslang.

    NO CACHING — runs every call. Pyslang parsing is fast (<1s for typical
    designs) and avoids cache invalidation bugs.

    Parameters
    ----------
    rtl_dir : str
        Directory containing the RTL source files.
    signal_map : dict, optional
        Pre-extracted signal map from ``design_info.signal_map``. If
        provided, its keys are added to ``all_signals`` (so port-level
        signals are guaranteed to be present even if pyslang misses them).
    top_module : str, optional
        Name of the top-level module. Used by Stage 3 module scoping.
    module_facts_mode : str
        "off" (default), "lazy" (Option G), or "full" (Option F).

    Returns
    -------
    RTLFacts
        Populated facts dataclass. Check ``is_complete`` for parse status.
    """
    facts = RTLFacts()
    if not rtl_dir:
        facts.is_complete = False
        facts.parse_warnings.append("rtl_dir is empty")
        return facts

    trees = _parse_rtl_files(rtl_dir, facts)
    if not trees:
        return facts

    # ---- Flat extraction (unchanged, backward compatible) ---------------
    facts.signal_definitions = _extract_signal_definitions(trees)
    facts.case_selectors = _extract_case_selectors(trees)
    facts.combinational_signals, facts.signal_drive_kind = _extract_drive_kinds(trees)
    facts.constant_signal_pairs = _extract_constant_signal_pairs(trees)
    facts.all_signals = _extract_all_signals(trees)
    facts.signal_widths = _extract_signal_widths(trees)
    facts.signal_frequencies = _extract_signal_frequencies(trees)

    # Multi-clock-domain aware clock/reset detection.
    (
        facts.clock_signals,
        facts.reset_signals,
        facts.reset_polarity,
        facts.clock_reset_pairs,
        facts.reset_values,
    ) = _detect_clock_reset_pairs_and_resets(trees)

    # Signal co-occurrence graph for semantic-affinity validation in
    # the post-processor.  Excludes clock/reset signals (universal
    # presence would drown out meaningful pairings) and restricts the
    # graph to identifiers that pyslang has confirmed as real signals.
    facts.signal_cooccurrence = _extract_signal_cooccurrence(
        trees,
        skip_signals=facts.clock_signals | facts.reset_signals,
        valid_signals=facts.all_signals,
    )

    # Directional dataflow graph (upstream / downstream) — dual-use:
    # consumed by the prompt formatter (causality hints in the facts
    # block) and by ``validate_implication_direction`` in the
    # post-processor.  Same skip/valid filters as cooccurrence.
    facts.signal_dataflow, facts.signal_dataflow_delay = (
        _extract_signal_dataflow(
            trees,
            valid_signals=facts.all_signals,
            skip_signals=facts.clock_signals | facts.reset_signals,
        )
    )

    # Per-signal port direction and instance-connection info.  Powers
    # the cross-module-aware Tier 4 in the affinity validator and
    # adds direction tags to the prompt's signal lists.
    facts.signal_port_info = _extract_signal_port_info(trees)

    # Detected handshake pairs.  Runs AFTER the dataflow / port_info
    # extraction because the confidence classifier consults both.
    facts.handshake_pairs = _extract_handshake_pairs(facts)

    # Detected state machines.  Runs AFTER signal_drive_kind is
    # populated since the extractor uses ``"seq"`` membership as the
    # primary FSM filter.
    facts.state_machines = _extract_state_machines(trees, facts)

    # Signal role classification.  Runs LAST among the structured
    # extractors because it consumes outputs from every other one
    # (clocks, resets, handshakes, FSMs, widths, drive-kind).
    facts.signal_roles = _classify_signal_roles(facts)

    # Documented properties from design docs.  Optional — only runs
    # when ``docs_dir`` is supplied AND the directory exists.  Uses
    # ``facts.all_signals`` to filter sentences down to those that
    # reference real RTL signals.
    facts.documented_properties = _extract_documented_properties(
        docs_dir, facts,
    )

    # Augment all_signals with the design's port-level signal map.
    if signal_map:
        for name in signal_map.keys():
            facts.all_signals.add(name)
            if "." in name:
                facts.all_signals.add(name.split(".")[-1])

    # ---- Stage 3: module scoping ----------------------------------------
    if module_facts_mode in ("lazy", "full") and top_module:
        # Build module hierarchy for depth-limited scoping (Fix 4).
        facts.module_hierarchy = _build_module_hierarchy(trees, top_module)
        if facts.module_hierarchy:
            logger.info(
                "Module hierarchy: %d module(s), max depth %d",
                len(facts.module_hierarchy),
                max(facts.module_hierarchy.values()),
            )

    if module_facts_mode == "lazy":
        # Option G: build signal→module mapping, filter at format time.
        facts.signal_to_module = _build_signal_to_module(trees)
        if facts.signal_to_module:
            modules = set(facts.signal_to_module.values())
            logger.info(
                "Module scoping (lazy): %d signals mapped to %d module(s)%s",
                len(facts.signal_to_module),
                len(modules),
                f" (top: {top_module})" if top_module else "",
            )

    elif module_facts_mode == "full":
        # Option F: per-module extraction via ModuleDeclaration boundaries.
        module_map = _split_by_module(trees)
        if module_map:
            facts.per_module = _extract_per_module_facts(trees, module_map)
            # Backfill widths from flat extraction (compilation-level).
            for mod_name, mod_facts in facts.per_module.items():
                mod_facts.signal_widths = {
                    k: v for k, v in facts.signal_widths.items()
                    if k in mod_facts.all_signals
                }
            logger.info(
                "Module scoping (full): %d module(s) extracted: %s",
                len(facts.per_module),
                ", ".join(
                    f"{m}({len(f.all_signals)} sigs)"
                    for m, f in sorted(facts.per_module.items())
                ),
            )
        # Also build signal_to_module for fallback.
        facts.signal_to_module = _build_signal_to_module(trees)

    # ---- Out-of-scope signal detection (structural) ---------------------
    if detect_out_of_scope:
        patterns = sink_module_patterns or DEFAULT_SINK_MODULE_PATTERNS
        facts.out_of_scope_signals = _detect_out_of_scope_signals(
            trees,
            sink_name_patterns=patterns,
            max_iterations=out_of_scope_max_iterations,
        )
        if facts.out_of_scope_signals:
            sample = sorted(facts.out_of_scope_signals)[:5]
            logger.info(
                "Out-of-scope signals detected: %d (e.g., %s)",
                len(facts.out_of_scope_signals),
                ", ".join(sample),
            )

    # ---- Logging --------------------------------------------------------
    n_mixed = sum(1 for k in facts.signal_drive_kind.values() if k == "mixed")
    n_seq = sum(1 for k in facts.signal_drive_kind.values() if k == "seq")
    n_cooc_signals = len(facts.signal_cooccurrence)
    n_cooc_edges = sum(len(v) for v in facts.signal_cooccurrence.values()) // 2
    n_flow_signals = len(facts.signal_dataflow)
    n_flow_edges = sum(
        len(v["downstream"]) for v in facts.signal_dataflow.values()
    )
    logger.info(
        "RTL facts extracted: %d signal defs, %d case-driven, "
        "%d comb / %d seq / %d mixed drivers, "
        "%d const literals, %d total signals, %d widths, %d reset values, "
        "%d clock(s), %d reset(s), %d clock-reset pair(s), "
        "%d cooc signals / %d edges, "
        "%d flow signals / %d directed edges (complete=%s)",
        len(facts.signal_definitions),
        len(facts.case_selectors),
        len(facts.combinational_signals),
        n_seq,
        n_mixed,
        len(facts.constant_signal_pairs),
        len(facts.all_signals),
        len(facts.signal_widths),
        len(facts.reset_values),
        len(facts.clock_signals),
        len(facts.reset_signals),
        len(facts.clock_reset_pairs),
        n_cooc_signals,
        n_cooc_edges,
        n_flow_signals,
        n_flow_edges,
        facts.is_complete,
    )
    if facts.parse_warnings:
        for w in facts.parse_warnings[:5]:
            logger.warning("  parse warning: %s", w)

    return facts


# ---------------------------------------------------------------------------
# Stage 2: prompt formatter
# ---------------------------------------------------------------------------
#
# format_facts_for_prompt() turns an RTLFacts into a Markdown block ready
# to inject into the LLM system prompt. The formatter is design-agnostic
# and respects a token budget via two-tier loading + greedy fill.
#
# Two-tier strategy:
#   - CORE  (always emitted): clock/reset pairs, unusual reset values only,
#           signal allowlist hint, generic Bad Patterns, denylist top-N
#   - EXT   (greedy fill in priority order until hard cap):
#           Tier-1 widths -> full reset table -> case selectors ->
#           constant pairs -> Tier-2 widths
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """
    Cheap token estimator: ``len(text) // 4``.

    We leave 20% headroom in the hard cap so this approximation is safe
    even though Claude/Qwen tokenizers vary slightly.
    """
    return len(text) // 4


_ZERO_VALUE_RE = re.compile(
    r"^\s*(?:"
    r"\d+'[bhdo]0+|"           # 1'b0, 4'h0, 32'h00000000, etc.
    r"\{[^{}]*\{1'b0\}\s*\}|"  # {N{1'b0}}
    r"\{[^{}]*\{1'h0\}\s*\}|"  # {N{1'h0}}
    r"0+"                      # plain 0
    r")\s*$"
)


def _is_zero_reset_value(text: str) -> bool:
    """Return True if the reset value is "all zeros" (the common, boring case)."""
    return bool(_ZERO_VALUE_RE.match(text.strip()))


def _format_clock_reset_pairs(facts: RTLFacts) -> str:
    """
    Format the clock/reset pair table.

    Always small (typically 1-3 lines). Critical for MCD designs where the
    LLM might mix up which reset clears which clock domain.
    """
    if not facts.clock_reset_pairs:
        return ""

    # Collapse: if a clock has both a sync-only entry AND a reset-paired
    # entry, drop the sync-only one (it's noise once we know the reset).
    clocks_with_reset = {
        clk for clk, rst in facts.clock_reset_pairs if rst is not None
    }
    visible_pairs = {
        (clk, rst)
        for clk, rst in facts.clock_reset_pairs
        if rst is not None or clk not in clocks_with_reset
    }

    lines = ["## Clock and reset domains"]
    pairs = sorted(visible_pairs, key=lambda p: (p[0], p[1] or ""))
    for clock, reset in pairs:
        if reset is None:
            lines.append(f"- `{clock}` (synchronous-only, no async reset)")
        else:
            polarity = facts.reset_polarity.get(reset, "unknown")
            lines.append(
                f"- clock `{clock}` paired with reset `{reset}` "
                f"(active-{polarity})"
            )

    if len(pairs) > 1:
        lines.append(
            "- WARNING: this design has MULTIPLE clock domains. "
            "Each register's reset is fixed — do not assert a register "
            "against the wrong reset signal."
        )

    return "\n".join(lines)


def _format_reset_values(facts: RTLFacts, *, full: bool) -> str:
    """
    Format the reset values table.

    full=False: only registers whose reset value is NOT all-zero (the
                "interesting" cases the LLM is most likely to get wrong).
    full=True:  every register with a detected reset value, ranked by
                signal frequency.
    """
    if not facts.reset_values:
        return ""

    items = list(facts.reset_values.items())
    # Sort by frequency desc (most-used registers first), then name asc.
    items.sort(
        key=lambda kv: (-facts.signal_frequencies.get(kv[0], 0), kv[0])
    )

    rows = []
    for reg, (rst, val) in items:
        is_unusual = not _is_zero_reset_value(val)
        if not full and not is_unusual:
            continue
        marker = "  <-- unusual value" if is_unusual else ""
        rst_label = rst if rst else "(no async reset)"
        rows.append(f"- `{reg}` resets via `{rst_label}` to `{val}`{marker}")

    if not rows:
        return ""

    header = "## Reset values" if full else "## Reset values (unusual)"
    if not full:
        header += " — flagged because they are NOT all-zero"
    return "\n".join([header] + rows)


def _priority_signal_set(
    facts: RTLFacts,
    signal_map: Optional[Dict[str, Any]],
) -> Set[str]:
    """
    Build the "Tier-1 important signals" set: signals likely to appear
    in assertions, used to filter the widths table down to a useful core.

    Includes:
    - Every key in ``signal_map`` (the design's documented signals)
    - Every signal that appears in clocks, resets, reset_values,
      case_selectors, or constant_signal_pairs (these are referenced
      by other facts sections)
    """
    important: Set[str] = set()
    if signal_map:
        for k in signal_map.keys():
            important.add(k)
            if "." in k:
                important.add(k.split(".")[-1])
    important.update(facts.clock_signals)
    important.update(facts.reset_signals)
    important.update(facts.reset_values.keys())
    important.update(facts.case_selectors.keys())
    for sigs in facts.constant_signal_pairs.values():
        important.update(sigs)
    return important


def _format_signal_widths(
    facts: RTLFacts,
    *,
    only: Optional[Set[str]] = None,
) -> str:
    """
    Format the signal widths table, width-grouped, drive-kind annotated.

    Layout::

        ## Signal widths and timing
        Markers: (c)=combinational use |->, (s)=sequential use |=>,
                 (m)=mixed driver, no marker=no detected driver
        [1 bit]   sig_a(s) sig_b(c) sig_c(c) ...
        [3 bits]  in_code(c) op_code(c)
        [17 bits] out_data(c) acc_out(s)

    Parameters
    ----------
    only : set of str, optional
        If provided, restrict the table to these signal names (Tier-1
        filter). Otherwise include every signal in ``facts.signal_widths``.
    """
    if not facts.signal_widths:
        return ""

    by_width: Dict[int, List[str]] = {}
    for name, w in facts.signal_widths.items():
        if only is not None and name not in only:
            continue
        by_width.setdefault(w, []).append(name)

    if not by_width:
        return ""

    # Within each width, sort by frequency desc, name asc.
    def _rank(name: str) -> Tuple[int, str]:
        return (-facts.signal_frequencies.get(name, 0), name)

    lines = [
        "## Signal widths and timing",
        "Markers: (c)=combinational, use `|->`. (s)=sequential, use `|=>`. "
        "(m)=mixed driver. No marker = no detected driver.",
    ]
    for w in sorted(by_width.keys()):
        names = sorted(by_width[w], key=_rank)
        annotated = []
        for n in names:
            kind = facts.signal_drive_kind.get(n)
            suffix = f"({kind[0]})" if kind else ""
            annotated.append(f"{n}{suffix}")
        bit_label = "1 bit " if w == 1 else f"{w} bits"
        lines.append(f"[{bit_label}] " + " ".join(annotated))

    return "\n".join(lines)


def _format_case_selectors(facts: RTLFacts) -> str:
    """
    Format the case-driven signals section.

    Tells the LLM which selector expression actually drives each signal,
    preventing assertions like ``code == 3'b011`` when the RTL uses
    ``{is_8bit, in_code}`` as the selector.
    """
    if not facts.case_selectors:
        return ""

    items = sorted(
        facts.case_selectors.items(),
        key=lambda kv: (-facts.signal_frequencies.get(kv[0], 0), kv[0]),
    )

    lines = ["## Case-driven signals"]
    lines.append(
        "These signals are assigned in `case (...)` statements. "
        "When asserting on them, use the EXACT selector expression shown."
    )
    for sig, selectors in items:
        sel_text = ", ".join(sorted(selectors))
        lines.append(f"- `{sig}` is driven by `case ({sel_text})`")

    return "\n".join(lines)


def _format_constant_pairs(facts: RTLFacts) -> str:
    """
    Format the constant ownership section, in negative phrasing.
    """
    if not facts.constant_signal_pairs:
        return ""

    # Sort literals by the frequency of their owning signal(s) desc.
    def _lit_rank(item):
        lit, sigs = item
        max_freq = max(
            (facts.signal_frequencies.get(s, 0) for s in sigs), default=0
        )
        return (-max_freq, lit)

    items = sorted(facts.constant_signal_pairs.items(), key=_lit_rank)

    lines = [
        "## Constant ownership",
        "Each constant below belongs to specific signals only. "
        "Do NOT pair these constants with any other signal:",
    ]
    for lit, sigs in items:
        owners = ", ".join(f"`{s}`" for s in sorted(sigs))
        lines.append(f"- `{lit}` belongs to {owners}")

    return "\n".join(lines)


def _format_documented_properties(
    facts: RTLFacts,
    *,
    only: Optional[Set[str]] = None,
    max_properties: int = 12,
) -> str:
    """Format documented properties as a focused prompt section.

    Selects up to ``max_properties`` entries, ranked by:
      • Whether ANY of the entry's signals are in ``only`` (priority
        set) — entries that mention important signals come first.
      • Otherwise, by the number of signals mentioned (more signals
        = more specific property).

    Each row shows the modal phrase + the signals + the raw sentence
    (truncated).  Source file/line is included so the LLM (or a human
    reviewer) can trace back to the spec.
    """
    props = facts.documented_properties
    if not props:
        return ""

    def _rank(p: Dict[str, Any]) -> Tuple[int, int, str]:
        in_priority = 0
        if only is not None:
            in_priority = sum(1 for s in p["signals"] if s in only)
        return (-in_priority, -len(p["signals"]), p["raw_sentence"])

    ranked = sorted(props, key=_rank)[:max_properties]
    if not ranked:
        return ""

    lines = [
        "## Documented properties (from spec docs)",
        "Each row below was lifted from a `must`/`shall`/`should` "
        "sentence in the design documentation.  The signals named in "
        "each sentence are listed first.  When you write properties, "
        "**prefer to encode at least one assertion per documented "
        "requirement** — the LLM that wrote the docs already chose "
        "what matters.",
        "",
    ]
    for p in ranked:
        sigs = ", ".join(f"`{s}`" for s in p["signals"][:5])
        more = (f" +{len(p['signals']) - 5}"
                if len(p["signals"]) > 5 else "")
        sent = p["raw_sentence"]
        # Trim the sentence to keep prompt budget reasonable.
        if len(sent) > 140:
            sent = sent[:137] + "..."
        src = p.get("source_file") or "?"
        line_no = p.get("source_line") or 0
        lines.append(
            f"- [{p['modal']}] signals=[{sigs}{more}] @ {src}:{line_no}\n"
            f"  > \"{sent}\""
        )
    return "\n".join(lines)


def _format_state_machines(
    facts: RTLFacts,
    *,
    only: Optional[Set[str]] = None,
    max_machines: int = 6,
) -> str:
    """Format detected state machines as a prompt section.

    For each FSM (state signal + state name list + optional encoding):
      • State signal name and module (when scoped).
      • Up to 8 state-name patterns and 6 literal patterns, with the
        encoding map inlined when known.
      • One property template that the LLM can specialise per FSM:
        validity-range and one-hot guards.

    Returns an empty string when no FSMs were detected (typical of
    pure-datapath designs like CMAC).
    """
    fsms = facts.state_machines
    if not fsms:
        return ""

    # Restrict to FSMs whose selector is in ``only`` (if provided).
    candidates = fsms
    if only is not None:
        candidates = [f for f in fsms if f["state_signal"] in only]
        if not candidates:
            # Fall back to all — still informative even if outside the
            # priority set, since FSM signals tend to BE important.
            candidates = fsms

    candidates = candidates[:max_machines]
    lines = [
        "## State machines (detected)",
        "Each row below is a registered selector signal whose case "
        "branches form a state-machine state set.  When you write a "
        "property on one of these, **always** compare against one of "
        "the listed state values — comparing against an arbitrary "
        "literal (e.g., `state == 3'b111` when only states 0-2 exist) "
        "is a hallucination.",
        "",
    ]
    for fsm in candidates:
        sig = fsm["state_signal"]
        mod = fsm.get("module") or "(unscoped)"
        states = fsm.get("states", [])[:8]
        lits = fsm.get("literal_values", [])[:6]
        encoding = fsm.get("encoding") or {}
        more_states = max(0, len(fsm.get("states", [])) - len(states))
        more_lits = max(0, len(fsm.get("literal_values", [])) - len(lits))
        lines.append(f"- State signal `{sig}` (in {mod}, {fsm['n_states']} states):")
        if states:
            named = ", ".join(
                f"`{s}`{'='+encoding[s] if s in encoding else ''}"
                for s in states
            )
            extra = f", +{more_states} more" if more_states else ""
            lines.append(f"    states: {named}{extra}")
        if lits:
            lit_str = ", ".join(f"`{l}`" for l in lits)
            extra = f", +{more_lits} more" if more_lits else ""
            lines.append(f"    literal values: {lit_str}{extra}")
    lines.append("")
    lines.append("Property templates for an FSM with state signal `S` "
                 "and known states `{S0, S1, ...}`:")
    lines.append("- Valid-state membership: "
                 "`S inside {S0, S1, ...}`")
    lines.append("- One-hot encoding (substitute the state set): "
                 "`$onehot(S)`")
    lines.append("- Reset-state initialisation: "
                 "`$past(reset, 1) |-> S == <RESET_STATE>`")
    return "\n".join(lines)


def _format_handshake_pairs(
    facts: RTLFacts,
    *,
    only: Optional[Set[str]] = None,
    max_pairs: int = 12,
) -> str:
    """Format detected handshake pairs as a prompt section.

    Surfaces:
      • Up to ``max_pairs`` high-confidence (initiator, responder)
        pairs, ranked by combined signal importance (sum of
        ``signal_frequencies`` for initiator and responder, biased
        toward signals in the ``only`` priority set).
      • One property template per protocol type that appears in the
        listed pairs (persistence, stable-payload, no-double-grant) —
        compact reference for the LLM to mimic without re-deriving
        the standard shapes from scratch.

    Pairs of confidence ``"medium"`` (matched only by name pattern,
    no dataflow / module confirmation) are omitted; they're noisy.

    Returns an empty string if no high-confidence pairs exist (e.g.,
    pure datapath designs like CMAC have no handshakes).
    """
    pairs = facts.handshake_pairs
    if not pairs:
        return ""
    high = [p for p in pairs if p.get("confidence") == "high"]
    if not high:
        return ""

    freqs = facts.signal_frequencies or {}

    def _rank(p: Dict[str, Any]) -> Tuple[int, str]:
        importance_bonus = 0
        if only is not None:
            if p["initiator"] in only: importance_bonus += 100
            if p["responder"] in only: importance_bonus += 100
        score = (importance_bonus
                 + freqs.get(p["initiator"], 0)
                 + freqs.get(p["responder"], 0))
        return (-score, p["initiator"])

    ranked = sorted(high, key=_rank)[:max_pairs]
    if not ranked:
        return ""

    # Group by protocol so we can emit one set of templates per family.
    protocols_present = sorted({p["protocol"] for p in ranked})
    template_by_proto: Dict[str, List[str]] = {
        "valid_ready": [
            "Persistence (initiator must hold until acknowledged):",
            "  `(I && !R) |=> I`",
            "Stable payload during stall (substitute the data signal):",
            "  `(I && !R) |=> $stable(<payload>)`",
        ],
        "req_ack": [
            "Persistence (request held until acknowledged):",
            "  `(I && !R) |=> I`",
            "Acknowledge follows request within bounded time:",
            "  `I |-> ##[1:$] R`",
        ],
        "push_pop": [
            "No simultaneous push/pop on a single-port FIFO (substitute "
            "the FIFO's status signal):",
            "  `!(I && R)` (only when single-ported)",
            "Occupancy tracks pushes minus pops:",
            "  `(I && !R) |=> count == $past(count) + 1'b1`",
        ],
        "start_done": [
            "Done eventually follows start (use bounded interval if known):",
            "  `I |-> ##[1:$] R`",
            "Start de-asserts after acceptance:",
            "  `(I && R) |=> !I`",
        ],
    }

    lines = [
        "## Handshake protocols (detected)",
        "Each row below is a (initiator I, responder R) pair extracted "
        "from the RTL.  When you write a property on one of these pairs, "
        "use the templates for its protocol — same-cycle persistence "
        "(`I && !R |-> I`) is **always** a tautology; use `|=>` for the "
        "next-cycle form.",
        "",
        "Detected pairs (top by importance):",
    ]
    for p in ranked:
        mods = sorted(p.get("modules") or [])
        mod_str = (", ".join(mods[:2])
                   + (f" +{len(mods) - 2}" if len(mods) > 2 else "")
                   if mods else "(unscoped)")
        lines.append(
            f"- I=`{p['initiator']}` ↔ R=`{p['responder']}`  "
            f"({p['protocol']}, in {mod_str})"
        )
    lines.append("")
    lines.append("Property templates by protocol:")
    for proto in protocols_present:
        tmpl = template_by_proto.get(proto)
        if not tmpl:
            continue
        lines.append(f"- **{proto}**:")
        for t in tmpl:
            lines.append(f"  {t}")
    return "\n".join(lines)


def _format_port_directions(
    facts: RTLFacts,
    *,
    only: Optional[Set[str]] = None,
    max_lines: int = 30,
) -> str:
    """Format a compact port-direction summary for the prompt.

    Tells the LLM which signals are inputs vs outputs at the top
    module's boundary.  This matters for assertion direction: an
    *output*'s value should be checked relative to *input* state
    (e.g., ``input_valid |-> output_data == ...``), not the other way
    round.  Also lists internal signals declared as ports of a
    submodule — assertions on internal ports are usually scope errors.

    Selection:
      • Restrict to signals in ``only`` if provided.
      • Group by direction (input / output / inout / internal-port).
      • Cap total emitted rows at ``max_lines``.
    """
    info = facts.signal_port_info
    if not info:
        return ""

    by_dir: Dict[str, List[str]] = {
        "input": [], "output": [], "inout": [], "internal": [],
    }
    for sig, rec in info.items():
        if only is not None and sig not in only:
            continue
        d = rec.get("direction", "internal")
        if d not in by_dir:
            continue
        by_dir[d].append(sig)

    # Skip the section if nothing got selected.
    if not any(by_dir[d] for d in ("input", "output", "inout")):
        return ""

    lines = [
        "## Port directions (top-level boundary)",
        "Use these to write assertions that flow input → output: an "
        "output's value should be a function of input state, not the "
        "reverse.  An assertion on a submodule-internal port is usually "
        "a scoping mistake.",
    ]
    rows = 0
    for d, label in (
        ("input",  "Inputs"),
        ("output", "Outputs"),
        ("inout",  "Bidirectional"),
    ):
        if not by_dir[d] or rows >= max_lines:
            continue
        # Sort by frequency so important signals come first.
        freqs = facts.signal_frequencies
        names = sorted(
            by_dir[d],
            key=lambda n: (-freqs.get(n, 0), n),
        )
        # Keep the line readable.
        head = ", ".join(f"`{n}`" for n in names[:12])
        extra = (f", ... +{len(names) - 12} more"
                 if len(names) > 12 else "")
        lines.append(f"- **{label}**: {head}{extra}")
        rows += 1
    return "\n".join(lines)


def _format_signal_dataflow(
    facts: RTLFacts,
    *,
    only: Optional[Set[str]] = None,
    max_lines: int = 25,
) -> str:
    """Format the directional dataflow graph as causality hints.

    Each line summarises one signal's outgoing relationships:

        - `req` → `grant` (1-cycle), `state` (comb)

    The block tells the LLM which direction implications should run
    in:  if X drives Y, then `X |-> Y` (or `X |=> Y` for sequential)
    is the right shape, not the reverse.

    Selection strategy:
      • Restrict to signals in ``only`` if provided (typically the
        priority set returned by ``_priority_signal_set``).
      • Sort by combined fan-out (downstream count) so the most
        connected signals appear first.
      • Cap at ``max_lines`` rows to keep the section compact.
      • For each row, list up to 4 downstream targets to keep lines
        readable; if more exist, append ``+N more``.
    """
    flow = facts.signal_dataflow
    if not flow:
        return ""

    delay = facts.signal_dataflow_delay or {}

    # Filter to the priority set when available.
    candidates = list(flow.items())
    if only is not None:
        candidates = [(s, v) for s, v in candidates if s in only]

    # Rank by downstream fan-out (most connected first).
    candidates.sort(key=lambda kv: (-len(kv[1].get("downstream", set())), kv[0]))

    lines = [
        "## Causality (driver → driven)",
        "These dataflow relationships were extracted from the RTL "
        "source. **Implications should follow the arrow**: when X "
        "drives Y, write `X |-> Y` (combinational, marked `comb`) or "
        "`X |=> Y` (next-cycle, marked `1-cycle`). Reversing the "
        "direction (`Y |-> X`) is almost always a bug.",
    ]
    rows_emitted = 0
    for sig, edges in candidates:
        downstream = edges.get("downstream", set())
        if not downstream:
            continue
        # Group downstream by minimum delay between (sig, dst).
        # For sequential targets, we additionally bucket by total
        # pipeline depth (computed on demand) so a 3-stage pipeline is
        # surfaced as such instead of generically "1-cycle".
        comb_targets: List[str] = []
        seq_buckets: Dict[int, List[str]] = {}
        for dst in sorted(downstream):
            d = delay.get((sig, dst), 0)
            if d == 0:
                comb_targets.append(dst)
                continue
            depth = pipeline_depth(facts, sig, dst, max_depth=8)
            # `depth` may be lower than the recorded direct edge delay
            # if a shorter path exists through other intermediates.
            # Treat None as "1+" (at least the direct edge's delay).
            n = depth if depth and depth >= 1 else d
            seq_buckets.setdefault(n, []).append(dst)
        parts: List[str] = []
        if comb_targets:
            head = ", ".join(f"`{t}`" for t in comb_targets[:4])
            extra = (f" +{len(comb_targets) - 4} more"
                     if len(comb_targets) > 4 else "")
            parts.append(f"{head}{extra} (comb)")
        for n in sorted(seq_buckets):
            tgts = seq_buckets[n]
            head = ", ".join(f"`{t}`" for t in tgts[:4])
            extra = (f" +{len(tgts) - 4} more"
                     if len(tgts) > 4 else "")
            label = f"{n}-cycle" if n == 1 else f"{n}-cycle pipeline"
            parts.append(f"{head}{extra} ({label})")
        if not parts:
            continue
        lines.append(f"- `{sig}` → " + "; ".join(parts))
        rows_emitted += 1
        if rows_emitted >= max_lines:
            break

    if rows_emitted == 0:
        return ""
    return "\n".join(lines)


def _format_negative_constraints(
    facts: RTLFacts,
    *,
    denylist: Optional[List[str]] = None,
) -> str:
    """
    Format the signal allowlist hint and (if available) the hallucination
    denylist.
    """
    lines = [
        "## Signal name constraints",
        "Use ONLY signals listed in the widths table above (or the "
        "documented signal map). Do NOT invent signal names.",
    ]
    if denylist:
        # Top-5 most frequent
        top = denylist[:5]
        lines.append(
            "Previously hallucinated names (do NOT use these): "
            + ", ".join(f"`{n}`" for n in top)
        )
    return "\n".join(lines)


def _format_bad_patterns() -> str:
    """
    Generic SVA mistakes the LLM commonly makes, regardless of design.
    Static text — same for every design.
    """
    return "\n".join([
        "## Common SVA mistakes to avoid",
        "- Width mismatch: `assert (sig == 4'b0011)` when `sig` is 3 bits "
        "-> use `3'b011`. Always match the literal width to the signal width.",
        "- Wrong implication operator: `assert property (... |=> result)` "
        "when `result` is combinational -> use `|->`. `|=>` is only for "
        "sequential signals that update on the next clock.",
        "- Conjunction instead of implication: "
        "`assert (cond_a && cond_b && result)` asserts all three are "
        "simultaneously true; for `cond_a -> result` use "
        "`assert (!cond_a || result)`.",
        "- Out-of-range bit select: `assert (sig[N])` when `sig` has fewer "
        "than `N+1` bits. Always check the declared width.",
        "- Missing property wrapper: a bare `(cond) |-> (result)` is not a "
        "valid statement. Use "
        "`assert property (@(posedge clk) disable iff (!rst) (cond) |-> (result));`.",
        "- Incomplete assertion: every assertion must be a COMPLETE, SINGLE-LINE "
        "statement ending with a semicolon. WRONG: `assert (sig ==` (truncated). "
        "WRONG: splitting across multiple lines without proper continuation. "
        "RIGHT: `assert (condition) else $error(\"msg\");` all on one line.",
    ])


def _format_good_protocol_patterns() -> str:
    """
    Canonical SVA templates for common protocol patterns.

    These are static examples the LLM can follow verbatim for:
    - Valid/ready handshake (stability + persistence)
    - Credit-based flow control
    - Reset-value checks

    Included because small/quantized LLMs frequently invert the temporal
    operators in handshake assertions (e.g., emitting
    ``(valid && !ready) |-> (valid && ready)`` which is a logical
    contradiction instead of ``|=> valid`` which is the correct
    persistence property).
    """
    return "\n".join([
        "## Reference SVA patterns (copy these templates)",
        "Valid/ready handshake — valid must PERSIST until ready acknowledges:",
        "  `assert property (@(posedge clk) disable iff (!rst) "
        "(valid && !ready) |=> valid) else $error(\"valid dropped before ready\");`",
        "Valid/ready handshake — data must be STABLE while valid pending:",
        "  `assert property (@(posedge clk) disable iff (!rst) "
        "(valid && !ready) |=> $stable(data)) else $error(\"data changed before ack\");`",
        "Credit pulse — one credit per completed transaction:",
        "  `assert property (@(posedge clk) disable iff (!rst) "
        "credit_vld |-> (credit_size != 0)) else $error(\"zero-sized credit\");`",
        "Reset value — register must clear on async reset:",
        "  `assert property (@(posedge clk) !rst_n |-> (reg_sig == '0)) "
        "else $error(\"reg did not reset\");`",
        "AVOID: `(valid && !ready) |-> (valid && ready)` — this is a "
        "logical contradiction, not a handshake property.",
    ])


def format_facts_for_prompt(
    facts: RTLFacts,
    *,
    soft_token_budget: int = 1500,
    hard_token_budget: int = 2400,
    denylist: Optional[List[str]] = None,
    signal_map: Optional[Dict[str, Any]] = None,
    full_facts: Optional["RTLFacts"] = None,
) -> str:
    """
    Format an RTL-facts block for injection into the LLM system prompt.

    Two-tier loading:
    - CORE sections always emit (clock/reset pairs, unusual reset values,
      negative constraints, bad patterns).
    - EXTENDED sections fill the remaining budget greedily, in priority
      order: Tier-1 widths -> full reset table -> case selectors ->
      constant pairs -> Tier-2 widths.

    The function never exceeds ``hard_token_budget`` and tries to stay
    under ``soft_token_budget``. Token counts are estimated as
    ``len(text) // 4`` (cheap; we leave 20% headroom in the cap).

    Parameters
    ----------
    facts : RTLFacts
        Extracted RTL facts (from ``extract_rtl_facts``).
    soft_token_budget : int
        Target budget. Greedy fill stops adding extended sections once
        this is exceeded.
    hard_token_budget : int
        Absolute ceiling. No section is added if it would push past this.
    denylist : list of str, optional
        Hallucinated signal names from previous runs, sorted by count
        descending. The top-5 are emitted in the negative constraints
        section. Pass None or an empty list to skip.
    signal_map : dict, optional
        Design's signal_map (from ``design_info.signal_map``). Used to
        compute the Tier-1 widths filter (signals likely to appear in
        assertions).

    Returns
    -------
    str
        A Markdown-formatted block ready to inject into the system prompt.
        Empty string if facts contain nothing useful.
    """
    if not facts.is_complete and not facts.signal_widths:
        # Nothing reliable to say.
        return ""

    # ---- CORE sections (always included if non-empty) -------------------
    core: List[str] = []
    pairs_block = _format_clock_reset_pairs(facts)
    if pairs_block:
        core.append(pairs_block)

    unusual_resets = _format_reset_values(facts, full=False)
    if unusual_resets:
        core.append(unusual_resets)

    constraints = _format_negative_constraints(facts, denylist=denylist)
    if constraints:
        core.append(constraints)

    bad_patterns = _format_bad_patterns()
    good_patterns = _format_good_protocol_patterns()
    core.append(bad_patterns)

    core_text = "\n\n".join(core)
    core_tokens = _estimate_tokens(core_text)

    # ---- EXTENDED sections (greedy fill in priority order) --------------
    # Build candidates lazily so we don't pay formatting cost for skipped
    # ones.
    important = _priority_signal_set(facts, signal_map)

    candidates: List[Tuple[str, str]] = []  # (name, formatted_text)

    # Priority 1: Tier-1 widths (only signals in `important`).
    tier1 = _format_signal_widths(facts, only=important)
    if tier1:
        candidates.append(("widths_tier1", tier1))

    # Priority 2: directional dataflow (causality).  Comes right after
    # widths because direction-of-implication is a high-leverage hint
    # — it prevents wrong-direction implications, which are the #1
    # semantic bug we observe in LLM-emitted assertions.
    dataflow_block = _format_signal_dataflow(facts, only=important)
    if dataflow_block:
        candidates.append(("dataflow", dataflow_block))

    # Port directions — short, high-leverage section that tells the
    # LLM which side of a property each signal lives on.
    port_dir_block = _format_port_directions(facts, only=important)
    if port_dir_block:
        candidates.append(("port_directions", port_dir_block))

    # Handshake pairs — protocol-aware templates so the LLM emits
    # correct same-cycle vs next-cycle shapes for valid/ready, req/ack,
    # push/pop, start/done.  Higher priority than case selectors /
    # constants because handshake bugs are the most-frequent semantic
    # mistake we observe.
    handshake_block = _format_handshake_pairs(facts, only=important)
    if handshake_block:
        candidates.append(("handshakes", handshake_block))

    # State machines — listing the state set lets the LLM write
    # membership and one-hot properties without inventing state values.
    # Sit alongside handshakes; both encode protocol structure.
    fsm_block = _format_state_machines(facts, only=important)
    if fsm_block:
        candidates.append(("state_machines", fsm_block))

    # Documented properties — sentences from the spec that mention
    # design signals + a modal verb.  High priority because it directly
    # reflects designer intent ("the LLM that wrote the docs already
    # chose what matters").
    doc_block = _format_documented_properties(facts, only=important)
    if doc_block:
        candidates.append(("documented_properties", doc_block))

    # Priority 3: full reset table — replaces unusual-only if we can fit it.
    full_resets = _format_reset_values(facts, full=True)
    if full_resets and full_resets != unusual_resets:
        candidates.append(("resets_full", full_resets))

    # Priority 4: case selectors.
    case_block = _format_case_selectors(facts)
    if case_block:
        candidates.append(("case_selectors", case_block))

    # Priority 5: constant pairs.
    const_block = _format_constant_pairs(facts)
    if const_block:
        candidates.append(("constant_pairs", const_block))

    # Priority 5: Tier-2 widths (everything not in Tier-1).
    rest = (
        set(facts.signal_widths.keys()) - important
        if facts.signal_widths
        else set()
    )
    if rest:
        tier2 = _format_signal_widths(facts, only=rest)
        if tier2:
            # Drop the redundant header by replacing it with a compact one.
            tier2 = tier2.replace(
                "## Signal widths and timing\n"
                "Markers: (c)=combinational, use `|->`. "
                "(s)=sequential, use `|=>`. "
                "(m)=mixed driver. No marker = no detected driver.\n",
                "## Additional signal widths\n",
            )
            candidates.append(("widths_tier2", tier2))

    # Greedy fill: add candidates in order until we hit the soft budget.
    # Once over soft, only add a candidate if it still fits under hard.
    selected: List[str] = []
    selected_names: List[str] = []
    running_tokens = core_tokens
    sep_tokens = _estimate_tokens("\n\n")  # ~0-1 token, but be honest

    # Collect names of "swap-out" sections that should replace a core entry.
    swap_out_unusual = False

    for name, block in candidates:
        block_tokens = _estimate_tokens(block) + sep_tokens

        # Special case: full resets replace the unusual-only resets in core.
        if name == "resets_full" and unusual_resets:
            saved = _estimate_tokens(unusual_resets) + sep_tokens
            net = block_tokens - saved
            if running_tokens + net > hard_token_budget:
                continue
            running_tokens += net
            selected.append(block)
            selected_names.append(name)
            swap_out_unusual = True
            continue

        if running_tokens + block_tokens > hard_token_budget:
            continue
        running_tokens += block_tokens
        selected.append(block)
        selected_names.append(name)

    # Assemble final output. Order in the prompt:
    #   1. clock/reset pairs (core)
    #   2. reset values (full if selected, else unusual)
    #   3. signal widths (tier-1, then tier-2 if selected)
    #   4. case selectors
    #   5. constant pairs
    #   6. negative constraints
    #   7. bad patterns

    final: List[str] = []
    if pairs_block:
        final.append(pairs_block)

    if "resets_full" in selected_names:
        # use full
        idx = selected_names.index("resets_full")
        final.append(selected[idx])
    elif unusual_resets:
        final.append(unusual_resets)

    for tier_name in ("widths_tier1", "widths_tier2"):
        if tier_name in selected_names:
            idx = selected_names.index(tier_name)
            final.append(selected[idx])

    # Port directions — emit BEFORE causality so the LLM first knows
    # the boundary, then reads cross-signal causality.
    if "port_directions" in selected_names:
        idx = selected_names.index("port_directions")
        final.append(selected[idx])

    # Causality (direction-of-implication hints) — right after widths so
    # the LLM has signal-shape and signal-direction info adjacent.
    if "dataflow" in selected_names:
        idx = selected_names.index("dataflow")
        final.append(selected[idx])

    # Handshakes — sit alongside causality, before generic case/constant
    # sections, because protocol shape is more decision-relevant.
    if "handshakes" in selected_names:
        idx = selected_names.index("handshakes")
        final.append(selected[idx])

    # State machines — directly after handshakes; both are protocol
    # structure that informs what properties to emit.
    if "state_machines" in selected_names:
        idx = selected_names.index("state_machines")
        final.append(selected[idx])

    # Documented properties — placed last among the structured
    # sections so the LLM reads it after structural context (widths,
    # causality, port directions, handshakes, FSMs) is already in mind.
    if "documented_properties" in selected_names:
        idx = selected_names.index("documented_properties")
        final.append(selected[idx])

    for opt_name in ("case_selectors", "constant_pairs"):
        if opt_name in selected_names:
            idx = selected_names.index(opt_name)
            final.append(selected[idx])

    # Fix 6: when module-scoped, add a compact list of combinational
    # signals from submodules so the LLM still knows to use |-> for them.
    # This directly addresses the drive-kind regression observed on nvdla_mul
    # where scoping lost submodule annotations (fix_next_cycle 9→16).
    if full_facts is not None and full_facts is not facts:
        all_comb = full_facts.combinational_signals
        scoped_sigs = facts.all_signals
        submodule_comb = sorted(
            all_comb - scoped_sigs,
            key=lambda n: (-full_facts.signal_frequencies.get(n, 0), n),
        )
        if submodule_comb:
            # Also include seq signals from submodules for completeness.
            all_seq = {
                n for n, k in full_facts.signal_drive_kind.items()
                if k == "seq"
            }
            submodule_seq = sorted(
                all_seq - scoped_sigs,
                key=lambda n: (-full_facts.signal_frequencies.get(n, 0), n),
            )
            lines = ["## Submodule signal timing (outside top module)"]
            lines.append(
                "These signals are in submodules. Use the correct "
                "implication operator when referencing them:"
            )
            # Compact: comb names on one line, seq on another.
            comb_names = ", ".join(f"{n}" for n in submodule_comb[:50])
            if len(submodule_comb) > 50:
                comb_names += f", ... +{len(submodule_comb) - 50} more"
            lines.append(f"- Combinational (use `|->`): {comb_names}")
            if submodule_seq:
                seq_names = ", ".join(f"{n}" for n in submodule_seq[:30])
                if len(submodule_seq) > 30:
                    seq_names += f", ... +{len(submodule_seq) - 30} more"
                lines.append(f"- Sequential (use `|=>`): {seq_names}")
            submod_block = "\n".join(lines)
            # Only add if within budget.
            if (running_tokens + _estimate_tokens(submod_block)
                    <= hard_token_budget):
                final.append(submod_block)
                running_tokens += _estimate_tokens(submod_block)

    if constraints:
        final.append(constraints)

    final.append(bad_patterns)
    final.append(good_patterns)

    header = (
        "# RTL Facts (this design)\n"
        "Use the facts below as authoritative ground truth about this "
        "specific design. They were extracted directly from the RTL "
        "source via a parser, not inferred."
    )
    return header + "\n\n" + "\n\n".join(final)


def format_batch_facts(
    facts: RTLFacts,
    signal_names: Set[str],
) -> str:
    """
    Format a compact per-batch facts section for signals in this batch.

    Unlike the full ``format_facts_for_prompt()`` which covers the entire
    design, this produces a small (~50-150 token) block focused on the
    specific signals appearing in the current batch's skeletons. It
    repeats info from the system-prompt facts block intentionally (helps
    attention on smaller models).

    Each signal gets one line with width, drive kind, and any special
    attributes (reset value, case selector, constant ownership).

    Parameters
    ----------
    facts : RTLFacts
        Pre-extracted RTL facts (same instance used for the system prompt).
    signal_names : set of str
        Signal identifiers extracted from the batch's skeletons.

    Returns
    -------
    str
        Compact Markdown block, or empty string if no matching signals.
    """
    if not signal_names or not facts.signal_widths:
        return ""

    # Only include signals we have facts for.
    known = signal_names & set(facts.signal_widths.keys())
    if not known:
        # Fall back to signals in all_signals even without width info.
        known = signal_names & facts.all_signals
        if not known:
            return ""

    # Sort by frequency (most-used first) for consistent ordering.
    ordered = sorted(
        known,
        key=lambda n: (-facts.signal_frequencies.get(n, 0), n),
    )

    lines = ["## Signals in this batch"]

    for name in ordered:
        parts = [f"`{name}`"]

        # Width.
        w = facts.signal_widths.get(name)
        if w is not None:
            parts.append(f"{w} bit{'s' if w != 1 else ''}")

        # Drive kind.
        dk = facts.signal_drive_kind.get(name)
        if dk:
            label = {"comb": "(c) use |->", "seq": "(s) use |=>",
                     "mixed": "(m) check carefully"}.get(dk, "")
            if label:
                parts.append(label)

        # Case selector.
        if name in facts.case_selectors:
            sels = ", ".join(sorted(facts.case_selectors[name]))
            parts.append(f"case-driven by `{sels}`")

        # Reset value.
        if name in facts.reset_values:
            rst, val = facts.reset_values[name]
            unusual = " UNUSUAL" if not _is_zero_reset_value(val) else ""
            rst_label = rst if rst else "no async reset"
            parts.append(f"resets via {rst_label} to `{val}`{unusual}")

        # Constant ownership.
        for lit, owners in facts.constant_signal_pairs.items():
            if name in owners:
                parts.append(f"owns constant `{lit}`")
                break  # one is enough for context

        lines.append("- " + ", ".join(parts))

    return "\n".join(lines)


def format_already_covered(prior_assertions: str) -> str:
    """
    Format a compact "already covered" section from prior batch output.

    Extracts signal+condition pairs from assertion text to produce a
    compact list (~30 tokens per assertion) that tells the LLM what's
    already been generated. This prevents duplication across batches.

    Parameters
    ----------
    prior_assertions : str
        Raw SVA text from prior batch(es).

    Returns
    -------
    str
        Compact list, or empty string if no prior assertions.
    """
    if not prior_assertions or not prior_assertions.strip():
        return ""

    # Extract compact signatures: "signal == value" or "condition |-> consequence"
    signatures: List[str] = []
    for line in prior_assertions.splitlines():
        stripped = line.strip()
        if not stripped.startswith("assert"):
            continue
        # Strip the error message for compactness.
        clean = re.split(r'\belse\s+\$\w+\s*\(', stripped, maxsplit=1)[0]
        # Strip assert/property keywords and outer parens.
        clean = re.sub(r'^assert\s+(property\s+)?', '', clean).strip()
        clean = re.sub(r';\s*$', '', clean).strip()
        # Strip outer parens if balanced.
        if clean.startswith('(') and clean.endswith(')'):
            inner = clean[1:-1]
            depth = 0
            balanced = True
            for ch in inner:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth < 0:
                        balanced = False
                        break
            if balanced and depth == 0:
                clean = inner.strip()
        # Truncate long signatures.
        if len(clean) > 100:
            clean = clean[:97] + "..."
        if clean:
            signatures.append(clean)

    if not signatures:
        return ""

    lines = [
        f"## Already generated ({len(signatures)} assertions) — do NOT repeat these",
    ]
    for sig in signatures:
        lines.append(f"- `{sig}`")

    return "\n".join(lines)


def format_facts_reminders(
    facts: RTLFacts,
    *,
    denylist: Optional[List[str]] = None,
) -> str:
    """
    Build a compact "Critical reminders" block for the END of the system
    prompt (instruction sandwich pattern).

    This repeats the highest-leverage rules from the facts block so they
    benefit from recency bias. Kept intentionally short (~100-150 tokens)
    because it's NOT cached by prefix-caching engines.

    Parameters
    ----------
    facts : RTLFacts
        Extracted RTL facts (same instance used for the main block).
    denylist : list of str, optional
        Hallucinated signal names (top-N, pre-filtered).

    Returns
    -------
    str
        Markdown block, or empty string if facts are empty.
    """
    if not facts.signal_widths and not facts.combinational_signals:
        return ""

    lines = [
        "═══════════════════════════════════════════════════════════",
        "CRITICAL REMINDERS (from RTL Facts above)",
        "═══════════════════════════════════════════════════════════",
    ]

    # 1. Drive-kind reminder (biggest Stage 2 win — reinforce it).
    n_comb = len(facts.combinational_signals)
    if n_comb:
        lines.append(
            f"- This design has {n_comb} combinational signal(s) marked (c) "
            f"in the widths table. Use `|->` for them, NEVER `|=>`."
        )

    # 2. MCD reminder (if applicable).
    if len(facts.clock_reset_pairs) > 1:
        lines.append(
            "- MULTIPLE clock domains detected. Each register has a "
            "specific reset — do NOT mix reset signals across domains."
        )

    # 3. Unusual reset values.
    unusual = [
        (reg, val)
        for reg, (rst, val) in facts.reset_values.items()
        if not _is_zero_reset_value(val)
    ]
    if unusual:
        examples = ", ".join(
            f"`{reg}` resets to `{val}`" for reg, val in unusual[:3]
        )
        lines.append(f"- Unusual reset values: {examples}. Check the facts table.")

    # 4. Denylist reminder.
    if denylist:
        top = denylist[:3]
        lines.append(
            "- Do NOT use these hallucinated signal names: "
            + ", ".join(f"`{n}`" for n in top)
        )

    # 5. Formatting reminder.
    lines.append(
        "- Every assertion MUST be a complete, single-line statement "
        "ending with `;`. No truncated or multi-line splits."
    )

    # 6. Signal names.
    lines.append(
        "- Use ONLY signals from the RTL Facts widths table. "
        "Do NOT invent signal names."
    )

    return "\n".join(lines)
