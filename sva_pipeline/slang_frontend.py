"""
slang_frontend.py
-----------------
Unified RTL analysis and SVA linting using pyslang.

Replaces both ``design_graph.py`` (Yosys-based structural extraction) and
``verible_lint`` (Verible-based syntax checking) with a single tool:
the Slang SystemVerilog compiler, accessed via its Python bindings.

Key advantages over the previous Yosys + Verible approach:
  - Single dependency (``pip install pyslang``) instead of two system tools.
  - In-process calls via Python API — no subprocess overhead or temp files.
  - Full IEEE 1800-2017 compliance with parameter resolution.
  - Access to internal signals, comments, and always-block analysis.
  - ~100x faster SVA linting (in-memory parse vs subprocess per assertion).

Fallback: if ``pyslang`` is not installed, the module exposes helper
functions that delegate to the existing Yosys (``design_graph.py``) and
Verible (``tools.py:verible_lint``) implementations.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Check if pyslang is available
# ---------------------------------------------------------------------------

try:
    import pyslang
    SLANG_AVAILABLE = True
    logger.debug("pyslang is available.")
except ImportError:
    SLANG_AVAILABLE = False
    logger.debug("pyslang not installed — will fall back to Yosys/Verible.")


# ---------------------------------------------------------------------------
# Dataclasses — reuse PortInfo/ModuleInfo from design_graph.py
# ---------------------------------------------------------------------------

from .design_graph import PortInfo, ModuleInfo


@dataclass
class DesignInfo:
    """
    Complete design analysis result produced by ``build_design_info()``.

    Carries everything the pipeline needs — hierarchy, signal map, clock
    detection — so that ``main.py`` never has to load external files.
    """
    top_module: str
    modules: Dict[str, ModuleInfo] = field(default_factory=dict)
    hierarchy_tree: Dict[str, List[str]] = field(default_factory=dict)

    # Auto-generated text outputs (replace hierarchy.txt and signal_map.json).
    hierarchy_text: str = ""
    signal_map: Dict[str, Any] = field(default_factory=dict)
    graph_summary_text: str = ""

    # Auto-detected design characteristics.
    has_clock: bool = False
    clock_signal: Optional[str] = None
    reset_signal: Optional[str] = None


# ---------------------------------------------------------------------------
# Naming-convention heuristics for signal type classification
# ---------------------------------------------------------------------------

# Patterns are tried in order; first match wins.
_TYPE_PATTERNS: List[Tuple[str, str]] = [
    (r"(?:^|_)clk(?:$|_)|^clock$|^pclk$|^aclk$|^sclk$", "clock"),
    (r"(?:^|_)rst(?:$|_)|^reset|^prstn$|^rst_n$|^arst", "reset"),
    (r"(?:^|_)(?:valid|vld|pvld)(?:$|_)", "control"),
    (r"(?:^|_)(?:ready|rdy|prdy)(?:$|_)", "control"),
    (r"(?:^|_)(?:en|enable)(?:$|_)", "control"),
    (r"(?:^|_)(?:sel|cs|psel)(?:$|_)", "control"),
    (r"(?:^|_)(?:wr|write|pwrite|we)(?:$|_)", "control"),
    (r"(?:^|_)(?:rd|read)(?:$|_)", "control"),
    (r"(?:^|_)(?:addr|adr|paddr)(?:$|_)", "address"),
    (r"(?:^|_)(?:data|dat|wdat|rdat|pwdata|prdata)(?:$|_)", "data"),
    (r"(?:^|_)(?:nposted|posted)(?:$|_)", "control"),
]


def _classify_signal_type(name: str, direction: str, width: int) -> str:
    """
    Infer a signal type from its name using hardware naming conventions.

    Falls back to width-based heuristics: 1-bit outputs → status,
    1-bit inputs → control, multi-bit → data.
    """
    lower = name.lower()
    for pattern, sig_type in _TYPE_PATTERNS:
        if re.search(pattern, lower):
            return sig_type

    # Width-based fallback.
    if width == 1:
        return "status" if direction == "output" else "control"
    return "data"


# ---------------------------------------------------------------------------
# Build design info from pyslang
# ---------------------------------------------------------------------------

def build_design_info(
    rtl_dir: str,
    top_module: str,
    top_file: str = "",
    allow_pyslang_warnings: bool = False,
) -> DesignInfo:
    """
    Analyse the RTL design and produce a complete :class:`DesignInfo`.

    Uses pyslang if available, otherwise falls back to the existing
    Yosys-based ``design_graph.py`` module.

    Parameters
    ----------
    rtl_dir : str
        Directory containing Verilog/SystemVerilog source files.
    top_module : str
        Top-level module name.
    allow_pyslang_warnings : bool
        When False (default), pipeline stops on pyslang warnings (with a
        Y/n prompt in a TTY, hard fail otherwise). Fatal diagnostics
        always stop regardless of this flag.

    Returns
    -------
    DesignInfo
    """
    if SLANG_AVAILABLE:
        return _build_with_slang(
            rtl_dir, top_module, top_file,
            allow_warnings=allow_pyslang_warnings,
        )
    else:
        return _build_with_yosys_fallback(rtl_dir, top_module)


# pyslang diagnostic codes that make structure extraction unreliable.
# These always stop the pipeline regardless of allow_pyslang_warnings.
_FATAL_PYSLANG_CODES = frozenset({
    "UnknownModule",          # module not defined (missing stubs)
    "UnknownSystemName",      # undefined system function ($foo)
    "UndeclaredIdentifier",   # reference to undeclared signal
    "UndeclaredButFoundNested",
    "UnknownMember",          # struct/interface member missing
    "UsedBeforeDeclared",
})

# Codes demoted to warnings — lint-style issues that don't break extraction.
# Behaviour depends on allow_pyslang_warnings: default False stops, True continues.
# Add codes here as we observe them in real designs.
_WARNING_PYSLANG_CODES = frozenset({
    "SignConversion",
    "WidthExpand",
    "PortWidthExpand",
    "ArithOpMismatch",
    "CaseTypeMismatch",
    "MissingTimeScale",
    "UnsignedArithShift",
    "EventExpressionConstant",
    "MultiBitEdge",
    "PortDoesNotExist",       # demoted per team decision (review later)
    "UnconnectedNamedPort",
})


def _extract_code_name(code) -> str:
    """Extract the bare diagnostic code name (e.g. 'UnknownModule')."""
    # pyslang returns DiagCode(CodeName) when stringified.
    s = str(code)
    if s.startswith("DiagCode(") and s.endswith(")"):
        return s[len("DiagCode("):-1]
    return s


def _classify_and_gate_diagnostics(errors, source_manager, allow_warnings: bool):
    """
    Classify pyslang diagnostics into fatal vs warning and decide whether
    to stop the pipeline.

    Fatal diagnostics (e.g., UnknownModule) always raise SystemExit.
    Warnings stop the pipeline unless ``allow_warnings`` is True.
    In a TTY session the user is prompted Y/n before the hard stop on
    warnings; non-interactive sessions fail immediately.
    """
    import sys

    fatal_errors = []
    warning_errors = []
    unknown_errors = []

    for d in errors:
        code_name = _extract_code_name(d.code)
        if code_name in _FATAL_PYSLANG_CODES:
            fatal_errors.append((code_name, d))
        elif code_name in _WARNING_PYSLANG_CODES:
            warning_errors.append((code_name, d))
        else:
            unknown_errors.append((code_name, d))

    def _summarise(bucket, label):
        if not bucket:
            return
        counts: Dict[str, int] = {}
        for code_name, _d in bucket:
            counts[code_name] = counts.get(code_name, 0) + 1
        logger.warning("  %s:", label)
        for code_name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            # Show up to 3 sample locations for this code.
            samples = [d for c, d in bucket if c == code_name][:3]
            logger.warning("    %s: %d occurrence(s)", code_name, n)
            for d in samples:
                try:
                    fn = source_manager.getFileName(d.location)
                    line = source_manager.getLineNumber(d.location)
                    logger.warning("      → %s:%d", fn, line)
                except Exception:
                    pass

    if fatal_errors:
        logger.error(
            "pyslang reported %d FATAL error(s) that prevent reliable "
            "design extraction:", len(fatal_errors),
        )
        _summarise(fatal_errors, "fatal")
        if warning_errors or unknown_errors:
            logger.error(
                "Also %d warning(s):",
                len(warning_errors) + len(unknown_errors),
            )
            _summarise(warning_errors, "warnings (classified)")
            _summarise(unknown_errors, "warnings (unclassified codes)")
        logger.error(
            "Fatal errors must be resolved before the pipeline can "
            "proceed. Common causes: missing module stubs, missing "
            "include files, or unsupported SystemVerilog constructs."
        )
        raise SystemExit(2)

    # No fatals — only warnings.
    all_warnings = warning_errors + unknown_errors
    if not all_warnings:
        return

    logger.warning(
        "pyslang reported %d warning(s) during compilation:",
        len(all_warnings),
    )
    _summarise(warning_errors, "warnings (classified)")
    _summarise(unknown_errors, "warnings (unclassified codes)")

    if allow_warnings:
        logger.warning(
            "allow_pyslang_warnings=true — continuing with extraction."
        )
        return

    # Warnings present and not allowed — gate.
    if sys.stdin.isatty():
        logger.warning(
            "Structure extraction may still succeed, but results can be "
            "unreliable. Continue anyway? [y/N]"
        )
        try:
            answer = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer == "y" or answer == "yes":
            logger.warning("User chose to continue despite warnings.")
            return
        logger.error("User aborted due to pyslang warnings.")
        raise SystemExit(3)
    else:
        logger.error(
            "pyslang emitted warnings and allow_pyslang_warnings=False "
            "(default). Set 'agent.allow_pyslang_warnings: true' in your "
            "YAML to proceed, or fix the reported issues. Aborting."
        )
        raise SystemExit(3)


def _build_with_slang(
    rtl_dir: str, top_module: str, top_file: str = "",
    allow_warnings: bool = False,
) -> DesignInfo:
    """
    Build DesignInfo using pyslang's in-process compilation.
    """
    # Collect all .v / .sv files from the RTL directory.
    rtl_files: List[str] = []
    for root, _, files in os.walk(rtl_dir):
        for fname in sorted(files):
            if Path(fname).suffix in {".v", ".sv"}:
                rtl_files.append(os.path.join(root, fname))

    # If a top_file is specified, ensure it's in the list and at the front.
    if top_file:
        # Resolve relative to rtl_dir if needed.
        if not os.path.isabs(top_file) and not os.path.exists(top_file):
            top_file = os.path.join(rtl_dir, top_file)
        top_file = os.path.normpath(top_file)
        # Remove from list if already present, then prepend.
        rtl_files = [f for f in rtl_files if os.path.normpath(f) != top_file]
        if os.path.exists(top_file):
            rtl_files.insert(0, top_file)
            logger.info("Top file: %s", top_file)

    if not rtl_files:
        logger.warning("No RTL files found in '%s'.", rtl_dir)
        return DesignInfo(top_module=top_module)

    logger.info("Parsing %d RTL file(s) with pyslang …", len(rtl_files))

    # Parse and compile.
    trees = []
    for f in rtl_files:
        try:
            trees.append(pyslang.SyntaxTree.fromFile(f))
        except Exception as exc:
            logger.warning("pyslang could not parse %s: %s", f, exc)

    comp = pyslang.Compilation()
    for t in trees:
        comp.addSyntaxTree(t)

    # Classify diagnostics and decide whether to proceed.
    diags = comp.getAllDiagnostics()
    errors = [d for d in diags if d.isError()]
    if errors:
        _classify_and_gate_diagnostics(
            errors, comp.sourceManager, allow_warnings=allow_warnings,
        )

    # Find the top module instance.
    top_instances = list(comp.getRoot().topInstances)
    target_inst = None
    for inst in top_instances:
        if inst.name == top_module:
            target_inst = inst
            break

    if target_inst is None:
        if top_instances:
            # Use the first top instance if exact name not found.
            target_inst = top_instances[0]
            logger.warning(
                "Module '%s' not found — using '%s' as top.",
                top_module, target_inst.name,
            )
            top_module = target_inst.name
        else:
            logger.error("No top instances found in compilation.")
            return DesignInfo(top_module=top_module)

    # Extract all modules from the compilation.
    modules: Dict[str, ModuleInfo] = {}
    hierarchy_tree: Dict[str, List[str]] = {}

    # Walk all top instances and their children.
    _extract_module_recursive(target_inst, modules, hierarchy_tree)

    # Detect clock and reset from the top module source.
    has_clock, clock_signal = _detect_clock(target_inst)
    reset_signal = _detect_reset(target_inst)

    # Build signal map.
    signal_map = _build_signal_map(modules, top_module)

    # Build hierarchy text.
    hierarchy_text = _build_hierarchy_text(top_module, modules, hierarchy_tree)

    # Build graph summary.
    graph_summary_text = _build_graph_summary(top_module, modules, hierarchy_tree)

    info = DesignInfo(
        top_module=top_module,
        modules=modules,
        hierarchy_tree=hierarchy_tree,
        hierarchy_text=hierarchy_text,
        signal_map=signal_map,
        graph_summary_text=graph_summary_text,
        has_clock=has_clock,
        clock_signal=clock_signal,
        reset_signal=reset_signal,
    )

    logger.info(
        "Design info built: %d module(s), has_clock=%s, clock=%s, reset=%s",
        len(modules), has_clock, clock_signal, reset_signal,
    )
    return info


def _extract_module_recursive(
    inst,
    modules: Dict[str, ModuleInfo],
    hierarchy_tree: Dict[str, List[str]],
) -> None:
    """
    Recursively extract module information from a pyslang instance.
    """
    body = inst.body
    mod_name = inst.name

    # Skip if we've already processed this module name (parameterised variants
    # may appear multiple times with different names).
    if mod_name in modules:
        return

    # Extract ports.
    ports: Dict[str, PortInfo] = {}
    for port in body.portList:
        direction = str(port.direction).replace("ArgumentDirection.", "").lower()
        # Map pyslang direction names to our convention.
        dir_map = {"in": "input", "out": "output", "inout": "inout"}
        direction = dir_map.get(direction, direction)

        width = port.type.bitWidth if hasattr(port.type, "bitWidth") else 1

        ports[port.name] = PortInfo(
            name=port.name,
            direction=direction,
            width=width,
        )

    # Extract cell instantiations (submodules) — look for child instances
    # by checking portConnections or definition references.
    cells: Dict[str, str] = {}
    children: List[str] = []

    # Use the syntax to find module instantiations.
    src = str(body.syntax) if body.syntax else ""
    # Simple heuristic: find instance patterns in source.
    # More robust: iterate body members if available.
    inst_pattern = re.compile(r"^\s*(\w+)\s+(?:#\s*\(.*?\)\s+)?(\w+)\s*\(", re.MULTILINE)
    for match in inst_pattern.finditer(src):
        cell_type = match.group(1)
        cell_name = match.group(2)
        # Skip Verilog keywords that look like instantiations.
        keywords = {
            "module", "endmodule", "input", "output", "inout", "wire", "reg",
            "logic", "assign", "always", "initial", "if", "else", "begin",
            "end", "case", "default", "for", "generate", "parameter",
            "localparam", "function", "endfunction", "task", "endtask",
            "always_comb", "always_ff", "always_latch",
        }
        if cell_type not in keywords and cell_name not in keywords:
            cells[cell_name] = cell_type
            if cell_type not in children:
                children.append(cell_type)

    modules[mod_name] = ModuleInfo(
        name=mod_name,
        ports=ports,
        cells=cells,
    )

    if children:
        hierarchy_tree[mod_name] = sorted(children)


def _detect_clock(inst) -> Tuple[bool, Optional[str]]:
    """
    Detect if the design has a clock signal by looking for ``posedge``
    patterns in the top module source and checking port names.

    Returns (has_clock, clock_signal_name).
    """
    src = str(inst.body.syntax) if inst.body.syntax else ""
    posedge_signals = set(re.findall(r"posedge\s+(\w+)", src))

    # Filter to likely clock names.
    clock_patterns = re.compile(
        r"(?:^|_)clk(?:$|_)|^clock$|^pclk$|^aclk$|^sclk$|^hclk$",
        re.IGNORECASE,
    )
    for sig in posedge_signals:
        if clock_patterns.search(sig):
            return True, sig

    # If any posedge signal exists, it's probably a clock.
    if posedge_signals:
        return True, sorted(posedge_signals)[0]

    # Fallback: when the top module has no posedge (pure wiring modules
    # like NVDLA_cacc that only instantiate submodules), scan input ports
    # for clock-named signals. Prefer canonical clock names (ending in
    # "clk") over overrides/gates (containing "ovr", "gate", "sync").
    port_clock = _find_port_by_pattern(
        inst, clock_patterns, direction="input",
        prefer_suffix="clk",
        demote_substrings=["ovr", "gate", "sync", "slcg"],
    )
    if port_clock:
        return True, port_clock

    return False, None


def _find_port_by_pattern(
    inst, pattern, direction: Optional[str] = None,
    prefer_suffix: Optional[str] = None,
    demote_substrings: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Scan the top-module ports for an input whose name matches ``pattern``.

    Used as a fallback when the top is pure submodule wiring and has no
    procedural blocks of its own. Ranks matches:
        1. Names ending with ``prefer_suffix`` (e.g., "clk"/"rstn")
        2. Names not containing any ``demote_substrings`` (e.g., "ovr",
           "gate" — signals that are overrides / gated versions rather
           than the canonical clock/reset)
        3. Everything else

    Returns the best-ranked matching port, or None.
    """
    try:
        body = inst.body
    except AttributeError:
        return None

    candidates: List[str] = []
    for member in body:
        tname = type(member).__name__
        if tname != 'PortSymbol':
            continue
        name = getattr(member, 'name', '')
        if not name:
            continue
        if direction is not None:
            member_dir = str(getattr(member, 'direction', '')).lower()
            want = direction.lower().strip()
            if want == "input":
                if not member_dir.endswith(".in"):
                    continue
            elif want == "output":
                if not member_dir.endswith(".out"):
                    continue
            elif want not in member_dir:
                continue
        if pattern.search(name):
            candidates.append(name)

    if not candidates:
        return None

    def _rank(name: str) -> Tuple[int, int, str]:
        # Lower is better.
        demote = 0
        if demote_substrings:
            for sub in demote_substrings:
                if sub.lower() in name.lower():
                    demote += 1
        suffix_score = 0 if (prefer_suffix and
                              name.lower().endswith(prefer_suffix.lower())) else 1
        return (demote, suffix_score, name)

    candidates.sort(key=_rank)
    return candidates[0]


def _detect_reset(inst) -> Optional[str]:
    """
    Detect the reset signal by looking for ``negedge`` patterns and
    reset naming conventions.
    """
    src = str(inst.body.syntax) if inst.body.syntax else ""
    negedge_signals = set(re.findall(r"negedge\s+(\w+)", src))

    reset_patterns = re.compile(
        r"(?:^|_)rstn?(?:$|_)|(?:^|_)rst_n(?:$|_)|^reset|^arst|^nrst",
        re.IGNORECASE,
    )
    for sig in negedge_signals:
        if reset_patterns.search(sig):
            return sig

    # Also check for !signal patterns in always blocks (active-low reset).
    active_low = re.findall(r"if\s*\(\s*!(\w+)\s*\)", src)
    for sig in active_low:
        if reset_patterns.search(sig):
            return sig

    # Fallback: scan input ports for reset-named signals (same rationale
    # as _detect_clock's port-name fallback). Prefer names ending with
    # "rstn"/"rst_n"/"rst" over other matches.
    for suffix in ("rstn", "rst_n", "rst"):
        port_reset = _find_port_by_pattern(
            inst, reset_patterns, direction="input",
            prefer_suffix=suffix,
            demote_substrings=["ovr", "gate", "sync"],
        )
        if port_reset and port_reset.lower().endswith(suffix):
            return port_reset

    # Fall back to any match if no suffix-ranked winner.
    port_reset = _find_port_by_pattern(
        inst, reset_patterns, direction="input",
        demote_substrings=["ovr", "gate", "sync"],
    )
    return port_reset


def _build_signal_map(
    modules: Dict[str, ModuleInfo],
    top_module: str,
) -> Dict[str, Any]:
    """
    Generate a signal map dict from extracted module information.

    Uses naming-convention heuristics for type classification.
    """
    signal_map: Dict[str, Any] = {}

    for mod_name, mod_info in modules.items():
        for port_name, port in mod_info.ports.items():
            # Use unqualified names for top-module ports, qualified for others.
            if mod_name == top_module:
                key = port_name
            else:
                key = f"{mod_name}.{port_name}"

            sig_type = _classify_signal_type(
                port_name, port.direction, port.width
            )

            # Build a description from the available information.
            if port.width > 1:
                desc = (
                    f"{port.direction} [{port.width - 1}:0] "
                    f"of {mod_name}"
                )
            else:
                desc = f"{port.direction} of {mod_name}"

            signal_map[key] = {
                "module": mod_name,
                "direction": port.direction,
                "width": port.width,
                "type": sig_type,
                "description": desc,
            }

    logger.info("Signal map: %d entries auto-generated.", len(signal_map))
    return signal_map


def _build_hierarchy_text(
    top_module: str,
    modules: Dict[str, ModuleInfo],
    hierarchy_tree: Dict[str, List[str]],
) -> str:
    """
    Generate a human-readable hierarchy text (replaces hierarchy.txt).
    """
    lines = [f"Design hierarchy for {top_module}:", ""]

    def _walk(mod_name: str, indent: int = 0) -> None:
        prefix = "  " * indent
        mod = modules.get(mod_name)
        port_count = len(mod.ports) if mod else 0
        lines.append(f"{prefix}{mod_name} ({port_count} ports)")
        for child in hierarchy_tree.get(mod_name, []):
            _walk(child, indent + 1)

    _walk(top_module)
    return "\n".join(lines)


def _build_graph_summary(
    top_module: str,
    modules: Dict[str, ModuleInfo],
    hierarchy_tree: Dict[str, List[str]],
) -> str:
    """
    Generate a compact design summary for the system prompt.
    """
    lines: List[str] = []

    # Show modules in order: top first, then alphabetical.
    ordered = [top_module] + sorted(
        m for m in modules if m != top_module
    )

    for mod_name in ordered:
        mod = modules.get(mod_name)
        if not mod:
            continue

        is_top = "(top)" if mod_name == top_module else ""
        lines.append(f"Module: {mod_name} {is_top}".strip())

        # Port summary.
        if mod.ports:
            port_strs = []
            for p in sorted(mod.ports.values(), key=lambda x: x.name):
                d = p.direction[:3]
                port_strs.append(f"{p.name}({d},{p.width})")
            lines.append(f"  Ports: {', '.join(port_strs)}")

        # Submodules.
        children = hierarchy_tree.get(mod_name, [])
        if children:
            lines.append(f"  Submodules: {', '.join(children)}")

        lines.append("")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Yosys fallback for build_design_info
# ---------------------------------------------------------------------------

def _build_with_yosys_fallback(
    rtl_dir: str,
    top_module: str,
) -> DesignInfo:
    """
    Fall back to the existing Yosys-based design_graph module.
    """
    logger.info("pyslang not available — falling back to Yosys.")

    from .design_graph import (
        build_design_graph,
        generate_signal_map,
        graph_summary,
    )

    graph = build_design_graph(rtl_dir, top_module=top_module)
    if graph is None:
        logger.error("Yosys fallback also failed.")
        return DesignInfo(top_module=top_module)

    sig_map = generate_signal_map(graph)
    summary = graph_summary(graph)

    # Detect clock/reset from hierarchy text (less reliable than Slang).
    has_clock = False
    clock_signal = None
    reset_signal = None

    return DesignInfo(
        top_module=top_module,
        modules=graph.modules,
        hierarchy_tree=graph.hierarchy_tree,
        hierarchy_text=str(graph.raw_netlist.get("creator", "")),
        signal_map=sig_map,
        graph_summary_text=summary,
        has_clock=has_clock,
        clock_signal=clock_signal,
        reset_signal=reset_signal,
    )


# ---------------------------------------------------------------------------
# SVA linting via pyslang (replaces verible_lint)
# ---------------------------------------------------------------------------

def slang_lint(
    sva_code: str,
    reject_assert_property: bool = False,
) -> str:
    """
    Validate SystemVerilog Assertion syntax using pyslang.

    Same return format as the original ``verible_lint()``:
    ``"PASS: ..."`` on success, ``"FAIL: ..."`` on failure.

    Falls back to Verible if pyslang is not available.

    Parameters
    ----------
    sva_code : str
        SVA code snippet (without outer module wrapper).
    reject_assert_property : bool
        When True, reject concurrent assertions (for combinational designs).

    Returns
    -------
    str
    """
    if not SLANG_AVAILABLE:
        # Fall back to Verible.
        from .tools import verible_lint as _verible_lint
        return _verible_lint(
            sva_code, reject_assert_property=reject_assert_property,
        )

    # Semantic check: reject "assert property" for combinational designs.
    if reject_assert_property:
        if re.search(r"\bassert\s+property\s*\(", sva_code, re.IGNORECASE):
            return (
                "FAIL: Code contains 'assert property (...)' which is a "
                "CONCURRENT assertion and is wrong for a combinational design.\n"
                "This design has no clock — use IMMEDIATE assertions.\n\n"
                "  WRONG (concurrent): assert property (...) else $error(...);\n"
                "  RIGHT (immediate) : assert (...) else $error(...);\n\n"
                "Remove every 'property' keyword and resubmit."
            )

    # Semantic check: reject .width attribute.
    if re.search(r"\w+\.width\b", sva_code):
        return (
            "FAIL: Code uses '<signal>.width' which is NOT valid SystemVerilog.\n"
            "Use $bits(<signal>) instead.\n\n"
            "  WRONG : assert (fullkeys.width == 1408) else $error(...);\n"
            "  RIGHT : assert ($bits(fullkeys) == 1408) else $error(...);"
        )

    # Detect whether the code contains concurrent assertions (assert property)
    # to decide the wrapping strategy:
    # - Concurrent assertions go at module scope.
    # - Immediate assertions go inside always_comb.
    is_concurrent = bool(
        re.search(r"\bassert\s+property\b", sva_code, re.IGNORECASE)
    )

    if is_concurrent:
        # Concurrent assertions live at module scope.
        sv_wrapper = (
            "module _sva_check_wrapper(\n"
            "  input logic clk,\n"
            "  input logic rst_n\n"
            ");\n\n"
            f"{sva_code}\n\n"
            "endmodule\n"
        )
    else:
        # Immediate assertions live inside a procedural block.
        sv_wrapper = (
            "module _sva_check_wrapper(\n"
            "  input logic clk,\n"
            "  input logic rst_n\n"
            ");\n\n"
            "always_comb begin\n"
            f"{sva_code}\n"
            "end\n\n"
            "endmodule\n"
        )

    try:
        tree = pyslang.SyntaxTree.fromText(sv_wrapper)
        comp = pyslang.Compilation()
        comp.addSyntaxTree(tree)

        diags = comp.getAllDiagnostics()

        # Filter out identifier-related errors — the SVA snippet references
        # design signals that aren't declared in our minimal wrapper module.
        # These are expected and should not be treated as failures.
        # We only care about syntax and structural errors.
        _IGNORE_CODES = {
            "DiagCode(UndeclaredIdentifier)",
            "DiagCode(TypoIdentifier)",
        }
        errors = [
            d for d in diags
            if d.isError()
            and str(d.code) not in _IGNORE_CODES
        ]

        if not errors:
            return (
                "PASS: SVA syntax is valid.\n"
                "(pyslang found no parse or semantic errors)"
            )

        # Format error messages.
        engine = pyslang.DiagnosticEngine(comp.sourceManager)
        client = pyslang.TextDiagnosticClient()
        engine.addClient(client)
        for d in errors:
            engine.issue(d)

        formatted = client.getString().strip()
        # Replace internal source references with <assertion>.
        formatted = re.sub(r"source:\d+:\d+:", "<assertion>:", formatted)

        return f"FAIL: Syntax/semantic errors detected:\n{formatted[:1200]}"

    except Exception as exc:
        return f"FAIL: pyslang error: {exc}"
