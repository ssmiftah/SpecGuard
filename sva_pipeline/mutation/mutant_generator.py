"""
Mutant generator — parses RTL source and produces mutated file copies.

Walks the DUT source file line-by-line, identifies mutable lines (skipping
declarations, comments, and submodule instantiation blocks), applies
mutation operators, and writes each mutant to a separate file.
"""

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .config import MutationConfig
from .operators import Mutation, apply_operators

logger = logging.getLogger(__name__)

# Lines matching these patterns are never mutated.
_SKIP_PATTERNS = [
    re.compile(r"^\s*//"),                           # comment lines
    re.compile(r"^\s*$"),                             # blank lines
    re.compile(r"^\s*(module|endmodule)\b"),          # module boundaries
    re.compile(r"^\s*(input|output|inout)\b"),        # port declarations
    re.compile(r"^\s*(wire|reg|logic|integer)\b"),    # signal declarations
    re.compile(r"^\s*(parameter|localparam)\b"),      # parameters
    re.compile(r"^\s*(begin|end)\s*$"),               # block delimiters only
    re.compile(r"^\s*\);?\s*$"),                      # closing paren/semicolon
    re.compile(r"^\s*\.\w+\s*\("),                    # port map connections
]


def _should_skip_line(line: str) -> bool:
    """Return True if the line should not be mutated."""
    return any(p.match(line) for p in _SKIP_PATTERNS)


def _find_instantiation_spans(lines: List[str]) -> Set[int]:
    """
    Find line ranges that are submodule instantiation blocks.

    Returns a set of 0-based line indices that should be excluded
    from mutation (port maps of instantiated submodules).
    """
    skip_lines: Set[int] = set()
    in_inst = False
    paren_depth = 0

    # Heuristic: a line that looks like "ModuleName instance_name (" or
    # "ModuleName #(...) instance_name (" starts an instantiation.
    inst_start_re = re.compile(
        r"^\s*(\w+)\s+(?:#\s*\(.*?\)\s+)?(\w+)\s*\("
    )
    # Keywords that are NOT module names.
    keywords = {
        "module", "endmodule", "input", "output", "inout", "wire", "reg",
        "logic", "assign", "always", "initial", "if", "else", "begin",
        "end", "case", "default", "for", "generate", "parameter",
        "localparam", "function", "endfunction", "task", "endtask",
        "always_comb", "always_ff", "always_latch", "integer",
    }

    for i, line in enumerate(lines):
        if in_inst:
            skip_lines.add(i)
            paren_depth += line.count("(") - line.count(")")
            if paren_depth <= 0:
                in_inst = False
                paren_depth = 0
        else:
            m = inst_start_re.match(line)
            if m and m.group(1) not in keywords and m.group(2) not in keywords:
                in_inst = True
                skip_lines.add(i)
                paren_depth = line.count("(") - line.count(")")
                if paren_depth <= 0:
                    in_inst = False

    return skip_lines


def _build_width_groups(signal_map: Dict[str, Any]) -> Dict[int, List[str]]:
    """
    Group signal names by their bit width for SIGNAL_SWAP operator.

    Only includes signals that appear in behavioral code (not clocks/resets).
    """
    groups: Dict[int, List[str]] = {}
    for name, info in signal_map.items():
        # Skip clocks, resets, and hierarchical names (submodule ports).
        if info.get("type") in ("clock", "reset"):
            continue
        if "." in name:
            continue
        width = info.get("width", 1)
        groups.setdefault(width, []).append(name)
    return groups


def generate_mutants(
    dut_source: str,
    dut_filename: str,
    config: MutationConfig,
    signal_map: Dict[str, Any] = None,
    top_module: str = "",
) -> List[Dict[str, Any]]:
    """
    Generate all single-point mutants from a DUT source file.

    Parameters
    ----------
    dut_source : str
        Full text of the DUT Verilog source file.
    dut_filename : str
        Original filename (for metadata).
    config : MutationConfig
        Mutation testing configuration.
    signal_map : dict, optional
        Signal map from DesignInfo (used for SIGNAL_SWAP width matching).

    Returns
    -------
    list of dict
        Each dict has: id, mutation (Mutation object), source (full mutated text).
    """
    lines = dut_source.splitlines(keepends=True)
    inst_spans = _find_instantiation_spans(
        [l.rstrip("\n") for l in lines]
    )
    width_groups = _build_width_groups(signal_map) if signal_map else {}

    all_mutations: List[Mutation] = []

    for i, line in enumerate(lines):
        stripped = line.rstrip("\n")

        # Skip non-mutable lines.
        if i in inst_spans:
            continue
        if _should_skip_line(stripped):
            continue

        # Apply all enabled operators to this line.
        line_mutations = apply_operators(
            stripped,
            line_no=i + 1,  # 1-based
            enabled_operators=config.operators,
            width_groups=width_groups,
        )
        all_mutations.extend(line_mutations)

    # Deduplicate by hashing the mutated line content.
    seen_hashes: Set[str] = set()
    unique_mutations: List[Mutation] = []
    for m in all_mutations:
        h = hashlib.md5(
            f"{m.line_no}:{m.mutated}".encode()
        ).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_mutations.append(m)

    # Cap at max_mutants.
    if len(unique_mutations) > config.max_mutants:
        logger.info(
            "Capping mutants from %d to %d.",
            len(unique_mutations), config.max_mutants,
        )
        unique_mutations = unique_mutations[:config.max_mutants]

    # Build the full mutated source for each mutant.
    mutants = []
    for idx, mutation in enumerate(unique_mutations):
        # Replace the original line with the mutated line.
        mutated_lines = list(lines)
        mutated_lines[mutation.line_no - 1] = mutation.mutated + "\n"
        mutated_source = "".join(mutated_lines)

        mutants.append({
            "id": idx,
            "mutation": mutation,
            "source": mutated_source,
            "filename": dut_filename,
            "top_module": top_module,
        })

    logger.info(
        "Generated %d mutant(s) from %s (%d lines, %d mutable).",
        len(mutants), dut_filename, len(lines),
        len(lines) - len(inst_spans) - sum(
            1 for l in lines if _should_skip_line(l.rstrip("\n"))
        ),
    )
    return mutants
