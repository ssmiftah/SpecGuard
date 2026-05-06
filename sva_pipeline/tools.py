"""
tools.py
--------
Concrete implementations of every tool the ReAct agent can call, plus the
OpenAI-compatible tool-definition schema that is passed to the Qwen3 chat
template so the model knows what tools are available.

Each tool function:
  - Takes only plain Python types (str, int) so the agent's JSON args can be
    directly unpacked into the call.
  - Returns a plain string — the "observation" that the agent reasons over.
  - Handles errors gracefully (missing binaries, bad input, etc.) and returns
    a descriptive error string rather than raising, because the agent should
    be able to read the error and adapt its next action.

Tools provided
--------------
1. rtl_retrieve      – delegated to FAISSRetriever in rag.py (wired in agent)
2. doc_retrieve      – delegated to FAISSRetriever in rag.py (wired in agent)
3. yosys_extract     – runs Yosys to extract module port/structure info
4. signal_map_lookup – queries the pre-loaded signal_map.json
5. verible_lint      – runs verible-verilog-syntax to validate SVA syntax
"""

import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool 3 — Yosys structural extraction
# ---------------------------------------------------------------------------

def yosys_extract(module_name: str, rtl_dir: str, yosys_bin: str = "yosys") -> str:
    """
    Run Yosys to extract port and structural information for `module_name`.

    Yosys reads all .v/.sv files in `rtl_dir`, elaborates the requested
    module, and emits a JSON netlist.  We then parse that JSON to produce a
    human-readable port summary that the agent can use to verify signal
    names, directions, and bit widths before writing assertions.

    Parameters
    ----------
    module_name : str
        The exact Verilog module identifier (e.g. "AES_Encrypt").
    rtl_dir : str
        Directory containing the RTL source files.

    Returns
    -------
    str
        A formatted string listing module ports, or an error description.
    """
    # Collect every Verilog/SV file in the RTL directory.
    rtl_files: List[str] = []
    for root, _, files in os.walk(rtl_dir):
        for fname in files:
            if Path(fname).suffix in {".v", ".sv"}:
                rtl_files.append(os.path.join(root, fname))

    if not rtl_files:
        return f"[yosys_extract] No RTL files found in '{rtl_dir}'."

    # Write the JSON netlist to a temporary file so Yosys has a target.
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        json_out = tmp.name

    try:
        # Build a compact Yosys script:
        #   read_verilog -sv  : parse each source file (SV extensions enabled)
        #   hierarchy         : resolve the module hierarchy from the given top
        #   proc              : convert procedural blocks to netlists
        #   write_json        : serialise the elaborated design to JSON
        # Quote each path so Yosys handles directory names with spaces correctly.
        read_cmds = "\n".join(f'read_verilog -sv "{f}"' for f in rtl_files)
        yosys_script = (
            f"{read_cmds}\n"
            f"hierarchy -check -top {module_name}\n"
            f"proc\n"
            f"write_json {json_out}\n"
        )

        result = subprocess.run(
            [yosys_bin, "-p", yosys_script],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            # Return the first 800 chars of stderr so the agent can diagnose.
            err = (result.stderr or result.stdout)[:800]
            return f"[yosys_extract] Yosys returned an error:\n{err}"

        # Parse the JSON netlist.
        with open(json_out, "r") as fh:
            netlist = json.load(fh)

        modules = netlist.get("modules", {})
        if module_name not in modules:
            available = ", ".join(modules.keys())
            return (
                f"[yosys_extract] Module '{module_name}' not found in netlist.\n"
                f"Available modules: {available}"
            )

        mod = modules[module_name]
        ports = mod.get("ports", {})
        if not ports:
            return f"[yosys_extract] Module '{module_name}' has no ports."

        # Format the port table for the agent.
        lines = [f"Module: {module_name}", "Ports:"]
        for port_name, port_info in sorted(ports.items()):
            direction = port_info.get("direction", "unknown")
            bits = port_info.get("bits", [])
            width = len(bits)
            if width == 1:
                lines.append(f"  {direction:6s}  {port_name}")
            else:
                lines.append(f"  {direction:6s}  [{width - 1}:0]  {port_name}")

        return "\n".join(lines)

    except FileNotFoundError:
        return (
            "[yosys_extract] 'yosys' binary not found. "
            "Please install Yosys and ensure it is on PATH."
        )
    except subprocess.TimeoutExpired:
        return "[yosys_extract] Yosys timed out after 120 s."
    except json.JSONDecodeError as exc:
        return f"[yosys_extract] Failed to parse Yosys JSON output: {exc}"
    except Exception as exc:  # pylint: disable=broad-except
        return f"[yosys_extract] Unexpected error: {exc}"
    finally:
        # Always clean up the temporary file.
        if os.path.exists(json_out):
            os.unlink(json_out)


# ---------------------------------------------------------------------------
# Tool 4 — Signal map lookup
# ---------------------------------------------------------------------------

def signal_map_lookup(signal_name: str, signal_map: Dict[str, Any]) -> str:
    """
    Look up a signal (or substring) in the pre-loaded signal map.

    The signal map is a JSON dict where each key is a signal name and the
    value is a dict of attributes (module, direction, width, description, …).
    Lookup is case-insensitive and also matches partial names so the agent
    can search by keyword (e.g. "key" returns all key-related signals).

    Parameters
    ----------
    signal_name : str
        Exact or partial signal name to search for.
    signal_map : dict
        The parsed content of signal_map.json, injected at agent init time.

    Returns
    -------
    str
        Formatted signal info, or a not-found message.
    """
    if not signal_map:
        return "[signal_map_lookup] Signal map is empty or not loaded."

    query = signal_name.lower().strip()

    # Collect all signals whose name contains the query string.
    matches: List[str] = [k for k in signal_map if query in k.lower()]

    if not matches:
        return (
            f"[signal_map_lookup] No signals found matching '{signal_name}'.\n"
            f"Available signals: {', '.join(sorted(signal_map.keys()))}"
        )

    lines = [f"Signal map results for '{signal_name}':"]
    for name in sorted(matches):
        info = signal_map[name]
        lines.append(f"\n  Signal : {name}")
        for attr, val in info.items():
            lines.append(f"    {attr:15s}: {val}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 5 — Verible SVA syntax checker
# ---------------------------------------------------------------------------

def verible_lint(
    sva_code: str,
    verible_bin: str = "verible-verilog-syntax",
    reject_assert_property: bool = False,
) -> str:
    """
    Validate SystemVerilog Assertion syntax using verible-verilog-syntax.

    The snippet is wrapped in a minimal module shell so Verible has a valid
    compilation unit, written to a temp file, and checked.  The raw output
    (errors / warnings) is returned so the agent can diagnose failures and
    rewrite the assertion.

    Parameters
    ----------
    sva_code : str
        One or more SVA property/assert statements (without outer module).

    Returns
    -------
    str
        "PASS: …" if syntax is valid, "FAIL: …" with error details otherwise.
    """
    # Wrap the snippet in a trivial module so Verible has a complete SV unit.
    # Immediate assertions (assert (...) else ...) must live inside a
    # procedural block, so we place the code inside always_comb.
    # A clock and reset wire are also declared for concurrent assertions.
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

    # Write to a named temp file — Verible needs a real path, not stdin.
    with tempfile.NamedTemporaryFile(
        suffix=".sv", mode="w", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(sv_wrapper)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [verible_bin, tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            # Secondary semantic checks for patterns that are syntactically
            # valid to Verible but semantically wrong for this design.

            # (a) "assert property (...)" is a CONCURRENT assertion and
            # requires a clock edge.  For purely combinational designs (no
            # clock), this is semantically wrong.  For clocked designs,
            # concurrent assertions are valid and expected.
            if reject_assert_property and re.search(r"\bassert\s+property\s*\(", sva_code, re.IGNORECASE):
                return (
                    "FAIL: Code contains 'assert property (...)' which is a "
                    "CONCURRENT assertion and is WRONG for a combinational design.\n"
                    "The AES design has no clock — you MUST use IMMEDIATE assertions.\n\n"
                    "  WRONG (concurrent): assert property (out == expected) else $error(...);\n"
                    "  RIGHT (immediate) : assert (out == expected) else $error(...);\n\n"
                    "Remove every 'property' keyword and resubmit."
                )

            # (b) "signal.width" is not a valid SystemVerilog runtime attribute.
            # Bit-width is a static elaboration property; use $bits(signal) instead.
            if re.search(r"\w+\.width\b", sva_code):
                return (
                    "FAIL: Code uses '<signal>.width' which is NOT valid SystemVerilog.\n"
                    "Signal bit-width is a static elaboration property, not a runtime value.\n\n"
                    "  WRONG : assert (fullkeys.width == 1408) else $error(...);\n"
                    "  RIGHT : assert ($bits(fullkeys) == 1408) else $error(...);\n\n"
                    "Replace every '<signal>.width' with '$bits(<signal>)' and resubmit.\n"
                    "Note: $bits() only works if the signal is directly visible in scope.\n"
                    "For internal submodule signals that are inaccessible, document the\n"
                    "width constraint as a comment instead of an assertion."
                )
            return (
                "PASS: SVA syntax is valid.\n"
                "(verible-verilog-syntax found no parse errors)"
            )
        else:
            # Combine stdout and stderr; strip the temp-file path from
            # the output so the agent sees signal/line info, not system paths.
            raw = (result.stderr + result.stdout).strip()
            cleaned = raw.replace(tmp_path, "<assertion>")
            return f"FAIL: Syntax errors detected:\n{cleaned[:1200]}"

    except FileNotFoundError:
        return (
            "[verible_lint] 'verible-verilog-syntax' not found. "
            "Please install Verible and ensure it is on PATH."
        )
    except subprocess.TimeoutExpired:
        return "[verible_lint] Verible timed out after 30 s."
    except Exception as exc:  # pylint: disable=broad-except
        return f"[verible_lint] Unexpected error: {exc}"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def dispatch_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    rtl_retriever,      # FAISSRetriever or HybridRetriever
    doc_retriever,      # FAISSRetriever or HybridRetriever
    signal_map: Dict[str, Any],
    rtl_dir: str,
    rtl_top_k: int = 5,
    doc_top_k: int = 5,
    yosys_bin: str = "yosys",
    verible_bin: str = "verible-verilog-syntax",
    design_graph: Optional[Any] = None,       # Optional[DesignGraph]
    full_rtl_injected: bool = False,          # True when full RTL is in prompt
    reject_assert_property: bool = True,      # False for clocked designs
) -> str:
    """
    Route a parsed tool call to the correct implementation.

    Called by the agent after parsing a `<tool_call>` block from the model
    output.  All five tools are handled here so the agent loop stays clean.

    Parameters
    ----------
    tool_name : str
        Name of the tool, exactly matching one of the TOOL_DEFINITIONS names.
    tool_args : dict
        Keyword arguments extracted from the model's JSON tool call.
    rtl_retriever : FAISSRetriever or HybridRetriever
        Pre-built index for RTL source files.
    doc_retriever : FAISSRetriever or HybridRetriever
        Pre-built index for documentation files.
    signal_map : dict
        Parsed signal_map.json dict.
    rtl_dir : str
        Path to the RTL source directory (for Yosys).
    rtl_top_k / doc_top_k : int
        Default k for RAG retrieval (can be overridden by tool_args).
    design_graph : DesignGraph, optional
        Pre-built design graph.  When provided, yosys_extract and
        signal_map_lookup use instant graph lookups instead of subprocesses.
    full_rtl_injected : bool
        When True, the full RTL source is already in the system prompt,
        so rtl_retrieve returns a redirect message instead of searching.

    Returns
    -------
    str
        Observation string to feed back into the ReAct conversation.
    """
    try:
        if tool_name == "rtl_retrieve":
            # When full RTL is in the system prompt, retrieval is unnecessary.
            if full_rtl_injected:
                return (
                    "[rtl_retrieve] The full RTL source code is already in "
                    "your system context.  No retrieval needed — refer to the "
                    "COMPLETE RTL SOURCE CODE section in the system prompt."
                )
            if rtl_retriever is None:
                return (
                    "[rtl_retrieve] RTL source is already in your system context. "
                    "Refer to the COMPLETE RTL SOURCE CODE section."
                )
            query = tool_args.get("query", "")
            k = int(tool_args.get("k", rtl_top_k))
            results = rtl_retriever.retrieve(query, k=k)
            if not results:
                return "[rtl_retrieve] No relevant RTL chunks found."
            # Format results with source metadata so the agent knows which
            # module / file each snippet came from.
            parts = []
            for i, r in enumerate(results, 1):
                src = r["metadata"].get("source", "?")
                mod = r["metadata"].get("module", "?")
                score = r["score"]
                parts.append(
                    f"[Result {i} | file={Path(src).name} | module={mod} | score={score:.3f}]\n"
                    f"{r['text']}"
                )
            return "\n\n---\n\n".join(parts)

        elif tool_name == "doc_retrieve":
            if doc_retriever is None:
                return (
                    "[doc_retrieve] Documentation is already in your system "
                    "context (injected alongside the RTL source code).  "
                    "Refer to the DESIGN DOCUMENTATION section in the prompt."
                )
            query = tool_args.get("query", "")
            k = int(tool_args.get("k", doc_top_k))
            results = doc_retriever.retrieve(query, k=k)
            if not results:
                return "[doc_retrieve] No relevant documentation chunks found."
            parts = []
            for i, r in enumerate(results, 1):
                src = r["metadata"].get("source", "?")
                score = r["score"]
                parts.append(
                    f"[Result {i} | file={Path(src).name} | score={score:.3f}]\n"
                    f"{r['text']}"
                )
            return "\n\n---\n\n".join(parts)

        elif tool_name == "yosys_extract":
            module = tool_args.get("module_name", "")
            if not module:
                return "[yosys_extract] 'module_name' argument is required."
            # Use pre-built graph for instant lookup when available.
            if design_graph is not None:
                from .design_graph import graph_lookup_module
                return graph_lookup_module(design_graph, module)
            return yosys_extract(module, rtl_dir, yosys_bin=yosys_bin)

        elif tool_name == "signal_map_lookup":
            signal = tool_args.get("signal_name", "")
            if not signal:
                return "[signal_map_lookup] 'signal_name' argument is required."
            # Start with the signal_map dict lookup.
            result = signal_map_lookup(signal, signal_map)
            # Supplement with graph data when available.
            if design_graph is not None:
                from .design_graph import graph_lookup_signal
                graph_result = graph_lookup_signal(design_graph, signal)
                if "No signals matching" not in graph_result:
                    result += "\n\n--- Design graph data ---\n" + graph_result
            return result

        elif tool_name in ("verible_lint", "slang_lint"):
            code = tool_args.get("sva_code", "")
            if not code:
                return "[verible_lint] 'sva_code' argument is required."
            # Use Slang-based linting (falls back to Verible if unavailable).
            from .slang_frontend import slang_lint
            return slang_lint(
                code,
                reject_assert_property=reject_assert_property,
            )

        else:
            return f"[dispatch_tool] Unknown tool: '{tool_name}'."

    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Tool '%s' raised an exception", tool_name)
        return f"[dispatch_tool] Tool '{tool_name}' raised an error: {exc}"


# ---------------------------------------------------------------------------
# OpenAI-compatible tool definitions (passed to Qwen3 chat template)
# ---------------------------------------------------------------------------
# Qwen3's tokenizer.apply_chat_template() accepts a `tools` list in the
# OpenAI function-calling schema format.  The model uses these descriptions
# to decide when and how to call each tool.

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "rtl_retrieve",
            "description": (
                "Search the RTL (Verilog/SystemVerilog) source index for code "
                "snippets relevant to the query.  Use this to find how specific "
                "logic, state machines, or interfaces are implemented in the design."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language or code-level search query, "
                            "e.g. 'key expansion always block' or 'AES_Encrypt output port'."
                        ),
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "doc_retrieve",
            "description": (
                "Search the design documentation index for specification passages "
                "relevant to the query.  Use this to find what a module is supposed "
                "to do, timing requirements, protocol rules, or design intent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language query about the design specification, "
                            "e.g. 'AES encryption output must equal expected value'."
                        ),
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "yosys_extract",
            "description": (
                "Use Yosys to extract the port list and structural information for "
                "a given module.  Returns port names, directions, and bit widths. "
                "Use this before writing assertions to confirm exact signal names."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "module_name": {
                        "type": "string",
                        "description": (
                            "Exact Verilog module identifier, e.g. 'AES_Encrypt'."
                        ),
                    },
                },
                "required": ["module_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "signal_map_lookup",
            "description": (
                "Look up one or more signals in the design signal map JSON. "
                "Returns the signal's module membership, direction, width, type, "
                "and description.  Supports partial/substring matching."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "signal_name": {
                        "type": "string",
                        "description": (
                            "Exact or partial signal name to search for, "
                            "e.g. 'key' to find all key-related signals."
                        ),
                    },
                },
                "required": ["signal_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "slang_lint",
            "description": (
                "Validate SystemVerilog Assertion (SVA) syntax using Slang "
                "(IEEE 1800-2017 compiler).  Returns PASS or FAIL with details. "
                "Always call this before finalising any assertion."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sva_code": {
                        "type": "string",
                        "description": (
                            "The SVA code snippet to check, without an outer module "
                            "wrapper.  May include multiple assert/property statements."
                        ),
                    },
                },
                "required": ["sva_code"],
            },
        },
    },
]
