"""
Mutation operators for Verilog/SystemVerilog RTL.

Each operator identifies mutable sites in a line of RTL code and produces
one or more single-point mutations.  A mutation is a ``(original, replacement)``
string pair that can be applied via ``str.replace(original, replacement, 1)``.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass
class Mutation:
    """One single-point mutation."""
    operator: str       # operator name, e.g. "OP_REPLACE"
    line_no: int        # 1-based line number in the source file
    original: str       # the full original line
    mutated: str        # the full mutated line
    description: str    # human-readable description


# ---------------------------------------------------------------------------
# Operator 1: Operator replacement
# ---------------------------------------------------------------------------

# Map each operator to its replacement set.
_OP_MAP = {
    "&&": ["||"],
    "||": ["&&"],
    "&":  ["|"],
    "|":  ["&"],
    "^":  ["&", "|"],
    "==": ["!="],
    "!=": ["=="],
    ">=": [">", "=="],
    "<=": ["<", "=="],
    ">":  [">=", "<"],
    "<":  ["<=", ">"],
    "+":  ["-"],
    "-":  ["+"],
}

# Sort by length descending so we match longer operators first (e.g. "==" before "=").
_OP_PATTERNS = sorted(_OP_MAP.keys(), key=len, reverse=True)
# Build a regex that matches any of the operators as whole tokens.
# Negative lookbehind/ahead prevents matching inside <=, >=, ==, !=, etc.
_OP_RE = re.compile(
    r"(?<!=)(?<!!)"
    + "("
    + "|".join(re.escape(op) for op in _OP_PATTERNS)
    + ")"
    + r"(?!=)"
)


def op_replace(line: str, line_no: int) -> List[Mutation]:
    """Replace logical/arithmetic/comparison operators."""
    mutations = []
    # Find all operator occurrences in the line.
    for match in _OP_RE.finditer(line):
        op = match.group(0)
        if op not in _OP_MAP:
            continue
        for replacement in _OP_MAP[op]:
            # Replace only this specific occurrence.
            new_line = line[:match.start()] + replacement + line[match.end():]
            mutations.append(Mutation(
                operator="OP_REPLACE",
                line_no=line_no,
                original=line,
                mutated=new_line,
                description=f"Replace '{op}' with '{replacement}'",
            ))
    return mutations


# ---------------------------------------------------------------------------
# Operator 2: Constant replacement
# ---------------------------------------------------------------------------

# Match Verilog-style constants: N'bXXX, N'hXXX, N'dXXX, or bare integers.
_CONST_RE = re.compile(r"\d+'[bhd][0-9a-fA-F_]+|\b\d+\b")


def const_replace(line: str, line_no: int) -> List[Mutation]:
    """Mutate constant values (flip 0↔1, ±1)."""
    mutations = []
    for match in _CONST_RE.finditer(line):
        const = match.group(0)
        replacements = []

        if const == "1'b0":
            replacements.append("1'b1")
        elif const == "1'b1":
            replacements.append("1'b0")
        elif "'" in const:
            # For wider constants, flip the last hex/binary digit.
            prefix, value = const.rsplit("'", 1)
            base_char = value[0]  # b, h, or d
            digits = value[1:]
            if digits and digits != "0":
                # Flip last digit: for hex, toggle LSB.
                last = digits[-1]
                if base_char == "b":
                    flipped = "0" if last == "1" else "1"
                elif base_char == "h":
                    flipped = hex(int(last, 16) ^ 1)[2:]
                else:  # d
                    try:
                        flipped = str(int(last) ^ 1)
                    except ValueError:
                        continue
                new_const = prefix + "'" + base_char + digits[:-1] + flipped
                replacements.append(new_const)
        else:
            # Bare integer literal.
            try:
                val = int(const)
                if val == 0:
                    replacements.append("1")
                elif val == 1:
                    replacements.append("0")
                else:
                    replacements.append(str(val - 1))
                    replacements.append(str(val + 1))
            except ValueError:
                continue

        for repl in replacements:
            new_line = line[:match.start()] + repl + line[match.end():]
            mutations.append(Mutation(
                operator="CONST_REPLACE",
                line_no=line_no,
                original=line,
                mutated=new_line,
                description=f"Replace constant '{const}' with '{repl}'",
            ))
    return mutations


# ---------------------------------------------------------------------------
# Operator 3: Signal swap
# ---------------------------------------------------------------------------

def signal_swap(
    line: str,
    line_no: int,
    width_groups: Dict[int, List[str]],
) -> List[Mutation]:
    """
    Swap two signals of the same width in an expression.

    Parameters
    ----------
    width_groups : dict
        Maps bit-width → list of signal names with that width.
        Built from DesignInfo.signal_map.
    """
    mutations = []
    for width, signals in width_groups.items():
        if len(signals) < 2:
            continue
        for i, sig_a in enumerate(signals):
            if sig_a not in line:
                continue
            for sig_b in signals[i + 1:]:
                if sig_b in line and sig_a != sig_b:
                    # Swap A with B.
                    new_line = line.replace(sig_a, "___SWAP___").replace(
                        sig_b, sig_a
                    ).replace("___SWAP___", sig_b)
                    if new_line != line:
                        mutations.append(Mutation(
                            operator="SIGNAL_SWAP",
                            line_no=line_no,
                            original=line,
                            mutated=new_line,
                            description=f"Swap '{sig_a}' with '{sig_b}'",
                        ))
    return mutations


# ---------------------------------------------------------------------------
# Operator 4: Bit-slice mutation
# ---------------------------------------------------------------------------

_BITSLICE_RE = re.compile(r"\[(\d+):(\d+)\]")


def bitslice_mut(line: str, line_no: int) -> List[Mutation]:
    """Shift bit ranges by ±1."""
    mutations = []
    for match in _BITSLICE_RE.finditer(line):
        hi = int(match.group(1))
        lo = int(match.group(2))
        # Shift both bounds down by 1 (if lo > 0).
        if lo > 0:
            new_slice = f"[{hi - 1}:{lo - 1}]"
            new_line = line[:match.start()] + new_slice + line[match.end():]
            mutations.append(Mutation(
                operator="BITSLICE_MUT",
                line_no=line_no,
                original=line,
                mutated=new_line,
                description=f"Shift [{hi}:{lo}] down to [{hi-1}:{lo-1}]",
            ))
        # Shift both bounds up by 1.
        new_slice_up = f"[{hi + 1}:{lo + 1}]"
        new_line_up = line[:match.start()] + new_slice_up + line[match.end():]
        mutations.append(Mutation(
            operator="BITSLICE_MUT",
            line_no=line_no,
            original=line,
            mutated=new_line_up,
            description=f"Shift [{hi}:{lo}] up to [{hi+1}:{lo+1}]",
        ))
    return mutations


# ---------------------------------------------------------------------------
# Operator 5: Condition negation
# ---------------------------------------------------------------------------

_COND_NEG_RE = re.compile(r"(!)\s*(\w+)|(\w+)\s*(==)\s*(\w+)")


def cond_negate(line: str, line_no: int) -> List[Mutation]:
    """Toggle negation in conditions: !sig → sig, sig → !sig."""
    mutations = []

    # Pattern 1: !signal → signal
    for match in re.finditer(r"!(\w+)", line):
        sig = match.group(1)
        new_line = line[:match.start()] + sig + line[match.end():]
        mutations.append(Mutation(
            operator="COND_NEGATE",
            line_no=line_no,
            original=line,
            mutated=new_line,
            description=f"Remove negation: !{sig} → {sig}",
        ))

    # Pattern 2: if (signal) — add negation (only in if/ternary contexts).
    for match in re.finditer(r"if\s*\(\s*(\w+)\s*\)", line):
        sig = match.group(1)
        if not sig.startswith("!"):
            new_cond = f"if (!{sig})"
            new_line = line[:match.start()] + new_cond + line[match.end():]
            mutations.append(Mutation(
                operator="COND_NEGATE",
                line_no=line_no,
                original=line,
                mutated=new_line,
                description=f"Negate condition: if ({sig}) → if (!{sig})",
            ))

    return mutations


# ---------------------------------------------------------------------------
# Operator 6: Assignment deletion
# ---------------------------------------------------------------------------

_ASSIGN_RE = re.compile(r"^\s*(assign\s+\w+\s*=|[\w\[\]:]+\s*<=)")


def assign_delete(line: str, line_no: int) -> List[Mutation]:
    """Comment out an assignment statement."""
    if _ASSIGN_RE.search(line):
        new_line = "// MUTANT_DELETED: " + line.lstrip()
        return [Mutation(
            operator="ASSIGN_DELETE",
            line_no=line_no,
            original=line,
            mutated=new_line,
            description="Delete assignment",
        )]
    return []


# ---------------------------------------------------------------------------
# Operator 7: Sensitivity list mutation
# ---------------------------------------------------------------------------

def sensitivity_mut(line: str, line_no: int) -> List[Mutation]:
    """Swap posedge ↔ negedge."""
    mutations = []
    if "posedge" in line:
        new_line = line.replace("posedge", "negedge", 1)
        mutations.append(Mutation(
            operator="SENSITIVITY_MUT",
            line_no=line_no,
            original=line,
            mutated=new_line,
            description="Swap posedge → negedge",
        ))
    if "negedge" in line:
        new_line = line.replace("negedge", "posedge", 1)
        mutations.append(Mutation(
            operator="SENSITIVITY_MUT",
            line_no=line_no,
            original=line,
            mutated=new_line,
            description="Swap negedge → posedge",
        ))
    return mutations


# ---------------------------------------------------------------------------
# Dispatcher: apply all enabled operators to a line
# ---------------------------------------------------------------------------

# Map operator names to functions.
_OPERATOR_FUNCS = {
    "OP_REPLACE": op_replace,
    "CONST_REPLACE": const_replace,
    "BITSLICE_MUT": bitslice_mut,
    "COND_NEGATE": cond_negate,
    "ASSIGN_DELETE": assign_delete,
    "SENSITIVITY_MUT": sensitivity_mut,
    # SIGNAL_SWAP is handled separately (needs width_groups).
}


def apply_operators(
    line: str,
    line_no: int,
    enabled_operators: List[str],
    width_groups: Dict[int, List[str]] = None,
) -> List[Mutation]:
    """
    Apply all enabled mutation operators to a single line.

    Returns a list of Mutation objects (one per single-point mutation).
    """
    all_mutations: List[Mutation] = []

    for op_name in enabled_operators:
        if op_name == "SIGNAL_SWAP":
            if width_groups:
                all_mutations.extend(signal_swap(line, line_no, width_groups))
        elif op_name in _OPERATOR_FUNCS:
            all_mutations.extend(_OPERATOR_FUNCS[op_name](line, line_no))

    return all_mutations
