"""
design_graph.py
---------------
Pre-built design graph extracted from Yosys at pipeline startup.

Instead of spawning a Yosys subprocess on every ``yosys_extract`` tool call,
this module runs Yosys **once**, parses the JSON netlist into a set of Python
dataclasses, and provides instant lookup functions the agent can use
throughout the ReAct loop.

Additionally, the graph can auto-generate a ``signal_map`` dict compatible
with the existing ``signal_map.json`` schema, eliminating the need for a
manually authored file.
"""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses representing the elaborated design
# ---------------------------------------------------------------------------

@dataclass
class PortInfo:
    """One port of a Verilog module."""
    name: str
    direction: str   # "input", "output", "inout"
    width: int
    bits: List[int] = field(default_factory=list)  # raw bit indices from Yosys


@dataclass
class ModuleInfo:
    """One module in the elaborated design."""
    name: str
    ports: Dict[str, PortInfo] = field(default_factory=dict)
    # cell_name -> cell_type (i.e. submodule instance name -> module name)
    cells: Dict[str, str] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DesignGraph:
    """
    Complete design graph extracted from a Yosys JSON netlist.

    Attributes
    ----------
    top_module : str
        Name of the top-level module.
    modules : dict
        Module name -> ModuleInfo for every module in the design.
    hierarchy_tree : dict
        Parent module name -> list of child module names (instantiation tree).
    raw_netlist : dict
        The full Yosys JSON netlist for advanced queries.
    """
    top_module: str
    modules: Dict[str, ModuleInfo] = field(default_factory=dict)
    hierarchy_tree: Dict[str, List[str]] = field(default_factory=dict)
    raw_netlist: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Build the graph from Yosys
# ---------------------------------------------------------------------------

def build_design_graph(
    rtl_dir: str,
    yosys_bin: str = "yosys",
    top_module: str = "AES",
) -> Optional[DesignGraph]:
    """
    Run Yosys once to elaborate the design and extract a full JSON netlist.

    Parses the netlist into a :class:`DesignGraph` with module, port, and
    hierarchy information.  Returns ``None`` if Yosys is not installed or
    elaboration fails (the caller should fall back to per-call extraction).

    Parameters
    ----------
    rtl_dir : str
        Directory containing Verilog/SystemVerilog source files.
    yosys_bin : str
        Path to the Yosys binary.
    top_module : str
        Top-level module name for hierarchy elaboration.

    Returns
    -------
    DesignGraph or None
    """
    # Collect all .v / .sv files.
    rtl_files: List[str] = []
    for root, _, files in os.walk(rtl_dir):
        for fname in sorted(files):
            if Path(fname).suffix in {".v", ".sv"}:
                rtl_files.append(os.path.join(root, fname))

    if not rtl_files:
        logger.warning("No RTL files found in '%s' — cannot build design graph.", rtl_dir)
        return None

    # Temporary file for the JSON netlist output.
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        json_out = tmp.name

    try:
        # Build a Yosys script that reads everything, elaborates, and dumps JSON.
        # Paths are quoted to handle directory names with spaces.
        read_cmds = "\n".join(f'read_verilog -sv "{f}"' for f in rtl_files)
        yosys_script = (
            f"{read_cmds}\n"
            f"hierarchy -check -top {top_module}\n"
            f"proc\n"
            f"write_json {json_out}\n"
        )

        logger.info(
            "Running Yosys to build design graph (top=%s, %d files) …",
            top_module, len(rtl_files),
        )

        result = subprocess.run(
            [yosys_bin, "-p", yosys_script],
            capture_output=True,
            text=True,
            timeout=180,
        )

        if result.returncode != 0:
            err = (result.stderr or result.stdout)[:500]
            logger.error("Yosys failed — falling back to per-call extraction.\n%s", err)
            return None

        # Parse the JSON netlist.
        with open(json_out, "r", encoding="utf-8") as fh:
            raw_netlist = json.load(fh)

        return _parse_netlist(raw_netlist, top_module)

    except FileNotFoundError:
        logger.warning("Yosys binary '%s' not found — design graph disabled.", yosys_bin)
        return None
    except subprocess.TimeoutExpired:
        logger.warning("Yosys timed out (180 s) — design graph disabled.")
        return None
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.error("Failed to parse Yosys JSON: %s", exc)
        return None
    finally:
        if os.path.exists(json_out):
            os.unlink(json_out)


def _parse_netlist(raw: Dict[str, Any], top_module: str) -> DesignGraph:
    """
    Convert a raw Yosys JSON netlist dict into a DesignGraph.

    Walks the ``modules`` section, extracting ports and cell (submodule)
    instantiations for each module.
    """
    graph = DesignGraph(
        top_module=top_module,
        raw_netlist=raw,
    )

    raw_modules = raw.get("modules", {})

    for mod_name, mod_data in raw_modules.items():
        # Parse ports.
        ports: Dict[str, PortInfo] = {}
        for port_name, port_data in mod_data.get("ports", {}).items():
            bits = port_data.get("bits", [])
            ports[port_name] = PortInfo(
                name=port_name,
                direction=port_data.get("direction", "unknown"),
                width=len(bits),
                bits=bits,
            )

        # Parse cells (submodule instantiations).
        cells: Dict[str, str] = {}
        for cell_name, cell_data in mod_data.get("cells", {}).items():
            cell_type = cell_data.get("type", "unknown")
            # Skip Yosys internal cells (start with $).
            if not cell_type.startswith("$"):
                cells[cell_name] = cell_type

        module_info = ModuleInfo(
            name=mod_name,
            ports=ports,
            cells=cells,
            attributes=mod_data.get("attributes", {}),
        )
        graph.modules[mod_name] = module_info

        # Build hierarchy tree from cells.
        child_types = sorted(set(cells.values()))
        if child_types:
            graph.hierarchy_tree[mod_name] = child_types

    logger.info(
        "Design graph built: %d module(s), top=%s",
        len(graph.modules), top_module,
    )
    return graph


# ---------------------------------------------------------------------------
# Signal map auto-generation
# ---------------------------------------------------------------------------

def generate_signal_map(
    graph: DesignGraph,
    manual_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Auto-generate a signal_map dict from the design graph.

    Walks every module's ports and creates an entry for each signal.  If a
    ``manual_map`` is provided, its entries take precedence (manual
    annotations override auto-generated ones).

    Parameters
    ----------
    graph : DesignGraph
        Pre-built design graph.
    manual_map : dict, optional
        Existing manually-authored signal_map entries.

    Returns
    -------
    dict
        Signal map compatible with signal_map.json schema.
    """
    auto_map: Dict[str, Any] = {}

    for mod_name, mod_info in graph.modules.items():
        for port_name, port_info in mod_info.ports.items():
            # Use module-qualified names for non-top-level signals to
            # avoid collisions (e.g. multiple modules have 'in'/'out').
            if mod_name == graph.top_module:
                key = port_name
            else:
                key = f"{mod_name}.{port_name}"

            # Infer a basic type from direction and width.
            if port_info.direction == "input" and port_info.width == 1:
                sig_type = "control"
            elif port_info.direction == "output" and port_info.width == 1:
                sig_type = "status"
            else:
                sig_type = "data"

            auto_map[key] = {
                "module": mod_name,
                "direction": port_info.direction,
                "width": port_info.width,
                "type": sig_type,
                "description": (
                    f"Auto-generated: {port_info.direction} port "
                    f"[{port_info.width - 1}:0] of module {mod_name}"
                    if port_info.width > 1
                    else f"Auto-generated: {port_info.direction} port of module {mod_name}"
                ),
            }

    # Merge: manual entries override auto-generated.
    if manual_map:
        for key, val in manual_map.items():
            auto_map[key] = val

    logger.info(
        "Signal map: %d auto-generated + %d manual = %d total entries.",
        len(auto_map) - (len(manual_map) if manual_map else 0),
        len(manual_map) if manual_map else 0,
        len(auto_map),
    )
    return auto_map


# ---------------------------------------------------------------------------
# Graph summary for system prompt
# ---------------------------------------------------------------------------

def graph_summary(graph: DesignGraph) -> str:
    """
    Produce a compact multi-line text summary of the design for the system prompt.

    Format:
      Module: AES (top)
        Ports: enable(in,1), e128(out,1), d128(out,1), ...
        Submodules: AES_Encrypt, AES_Decrypt

    Parameters
    ----------
    graph : DesignGraph

    Returns
    -------
    str
    """
    if not graph or not graph.modules:
        return "Design graph not available."

    lines: List[str] = []

    # Show modules in a deterministic order: top module first, then alphabetical.
    ordered_modules: List[str] = []
    if graph.top_module in graph.modules:
        ordered_modules.append(graph.top_module)
    for name in sorted(graph.modules.keys()):
        if name != graph.top_module:
            ordered_modules.append(name)

    for mod_name in ordered_modules:
        mod = graph.modules[mod_name]
        is_top = "(top)" if mod_name == graph.top_module else ""
        lines.append(f"Module: {mod_name} {is_top}".strip())

        # Port summary — compact single line.
        if mod.ports:
            port_strs = []
            for p in sorted(mod.ports.values(), key=lambda x: x.name):
                dir_short = p.direction[:3]  # "inp" -> "in", "out" -> "out"
                port_strs.append(f"{p.name}({dir_short},{p.width})")
            lines.append(f"  Ports: {', '.join(port_strs)}")

        # Submodule list.
        children = graph.hierarchy_tree.get(mod_name, [])
        if children:
            lines.append(f"  Submodules: {', '.join(children)}")

        lines.append("")  # blank line between modules

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Graph-backed lookup functions (replace subprocess calls)
# ---------------------------------------------------------------------------

def graph_lookup_module(graph: DesignGraph, module_name: str) -> str:
    """
    Look up a module in the pre-built graph and return a formatted port table.

    Same output format as the original ``yosys_extract()`` function so the
    agent sees a consistent interface.

    Parameters
    ----------
    graph : DesignGraph
    module_name : str
        Exact module name (case-sensitive).

    Returns
    -------
    str
        Formatted port listing, or an error message.
    """
    if module_name not in graph.modules:
        available = ", ".join(sorted(graph.modules.keys()))
        return (
            f"[yosys_extract] Module '{module_name}' not found in design graph.\n"
            f"Available modules: {available}"
        )

    mod = graph.modules[module_name]
    if not mod.ports:
        return f"[yosys_extract] Module '{module_name}' has no ports."

    lines = [f"Module: {module_name}", "Ports:"]
    for port in sorted(mod.ports.values(), key=lambda p: p.name):
        if port.width == 1:
            lines.append(f"  {port.direction:6s}  {port.name}")
        else:
            lines.append(f"  {port.direction:6s}  [{port.width - 1}:0]  {port.name}")

    # Also show submodule instantiations.
    if mod.cells:
        lines.append("\nSubmodule instances:")
        for inst_name, cell_type in sorted(mod.cells.items()):
            lines.append(f"  {inst_name} -> {cell_type}")

    return "\n".join(lines)


def graph_lookup_signal(graph: DesignGraph, signal_name: str) -> str:
    """
    Search all modules' ports for a signal name (partial, case-insensitive).

    Returns a formatted string in the same style as ``signal_map_lookup()``.

    Parameters
    ----------
    graph : DesignGraph
    signal_name : str
        Exact or partial signal name to search for.

    Returns
    -------
    str
    """
    query = signal_name.lower().strip()
    matches: List[Tuple[str, str, PortInfo]] = []  # (module, port_name, port)

    for mod_name, mod in graph.modules.items():
        for port_name, port in mod.ports.items():
            if query in port_name.lower():
                matches.append((mod_name, port_name, port))

    if not matches:
        return (
            f"[graph_lookup_signal] No signals matching '{signal_name}' "
            f"found in the design graph."
        )

    lines = [f"Design graph results for '{signal_name}':"]
    for mod_name, port_name, port in sorted(matches):
        lines.append(f"\n  Signal : {port_name}")
        lines.append(f"    module         : {mod_name}")
        lines.append(f"    direction      : {port.direction}")
        lines.append(f"    width          : {port.width}")

    return "\n".join(lines)
