"""
ast_assertions.py
-----------------
AST-guided assertion generation — extracts RTL patterns via regex and
generates SVA assertion skeletons deterministically (no LLM needed).

Extracts these RTL patterns from Verilog source text:
  1. Case branches:     case(sel) val: out = expr; endcase
  2. Direct assignments: assign out = expr;
  3. Ternary muxes:     assign out = sel ? a : b;
  4. Sequential resets:  if (!rst) reg <= 0;
  5. Sequential func:    if (cond) reg <= expr;
  6. Comb comparisons:   out_w = (in == const);

Each pattern is converted to a syntactically correct SVA assertion
using templates.  The assertions can be used directly (ast_only=True)
or passed to the LLM for semantic enrichment (descriptions, protocol
assertions, edge cases).

Principle: AST provides STRUCTURE (100% accurate, 0 cost).
           LLM provides SEMANTICS (descriptions, protocol logic).
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RTLPattern:
    """One extracted RTL pattern from source code."""
    pattern_type: str       # "case_branch", "direct_assign", "ternary_mux",
                            # "wire_passthrough", "seq_reset", "seq_func",
                            # "comb_comparison"
    source_line: int        # approximate line number
    source_text: str        # the raw RTL text that matched
    selector: Optional[str] # for case: the case selector expression
    condition: Optional[str] # case value or if-condition
    lhs: str                # output signal being assigned
    rhs: str                # expression being assigned
    is_sequential: bool     # True if inside always @(posedge ...)
    clock: Optional[str]    # clock signal if sequential
    reset: Optional[str]    # reset signal if sequential
    reset_is_async: bool = False   # True if reset is in sensitivity list (async)
    reset_polarity: str = "low"    # "low" = active-low (!rst), "high" = active-high (rst)


@dataclass
class AssertionSkeleton:
    """One generated assertion skeleton."""
    assertion_text: str     # ready-to-use SVA
    pattern_type: str       # which pattern generated this
    source_line: int        # which RTL line this came from
    source_text: str        # the RTL text this was derived from
    description: str        # auto-generated human-readable description


# ---------------------------------------------------------------------------
# Pattern extraction — regex-based
# ---------------------------------------------------------------------------

def extract_patterns(
    source: str,
    clock: Optional[str] = None,
    reset: Optional[str] = None,
) -> List[RTLPattern]:
    """
    Extract all RTL behavioral patterns from module source text.

    Calls each pattern extractor and returns a merged, deduplicated list.
    """
    patterns: List[RTLPattern] = []

    # Strip comment lines and preprocessor directives to avoid false matches.
    clean_source = _strip_comments(source)

    # Wrap any always-without-begin/end in synthetic begin/end so the
    # rest of the extractor (which assumes the block form) can process
    # them uniformly.
    clean_source = _normalise_always_blocks(clean_source)

    # Extract in order of specificity.
    patterns.extend(_extract_case_patterns(clean_source))
    patterns.extend(_extract_assign_patterns(clean_source))
    patterns.extend(_extract_always_patterns(clean_source, clock, reset))

    # Deduplicate by (lhs, rhs) — keep the first match.
    # This catches the same assignment extracted from different contexts
    # (e.g., continuous assign + combinational always block).
    seen = set()
    unique = []
    for p in patterns:
        key = (p.lhs, p.rhs, p.condition or "")
        if key not in seen:
            seen.add(key)
            unique.append(p)

    logger.info(
        "AST extraction: %d patterns found (%d unique).",
        len(patterns), len(unique),
    )
    return unique


def _strip_comments(source: str) -> str:
    """Remove single-line comments and preprocessor directives."""
    lines = []
    for line in source.splitlines():
        stripped = line.strip()
        # Skip preprocessor directives.
        if stripped.startswith("`"):
            continue
        # Skip pure comment lines.
        if stripped.startswith("//"):
            continue
        # Remove inline comments.
        comment_pos = line.find("//")
        if comment_pos >= 0:
            line = line[:comment_pos]
        lines.append(line)
    return "\n".join(lines)


def _line_number(source: str, pos: int) -> int:
    """Approximate line number for a character position."""
    return source[:pos].count("\n") + 1


# ---------------------------------------------------------------------------
# Pattern 1: Case branches
# ---------------------------------------------------------------------------

# Match: case(selector_expr) ... endcase   (also casez / casex)
_CASE_BLOCK_RE = re.compile(
    r"\bcase[zx]?\s*\((.+?)\)(.*?)\bendcase\b",
    re.DOTALL,
)

# Match a case branch: value(s): begin ... end
# The value list accepts:
#   - typed literals: ``2'b00``, ``8'h0f``, ``3'd5``  (also wildcards
#     ``2'b0?``, ``2'b0x`` for casez/casex)
#   - bare identifiers (parameter names): ``IDLE``, ``STATE_RUN``
#   - parenthesized expressions: ``(32'h7004 & 32'hfff)`` (NVDLA REG
#     decoders use this form for masked-address case selectors).
# Multiple comma-separated values per branch are captured as one
# string and split downstream by _CASE_VALUE_TOKEN_RE.
#
# IMPORTANT: the value list is structured as ``VAL (, VAL)*`` (comma
# REQUIRED between successive values) — the older form
# ``(VAL\s*,?\s*)+`` made the comma optional, which let each `+`
# iteration consume as little as one bare identifier and produced
# catastrophic backtracking on files with many identifiers but few
# ``:\s*begin`` suffixes (NV_NVDLA_CMAC_REG_single.v hung >60s).
_VAL_ALT = (
    r"\d+'[bhdoBHDO][\w?xz]+"   # typed literal
    r"|[A-Za-z_]\w*"             # bare identifier
    r"|\([^()]*\)"               # parenthesized expression (one level)
)
_CASE_BRANCH_BEGIN_RE = re.compile(
    r"((?:" + _VAL_ALT + r")(?:\s*,\s*(?:" + _VAL_ALT + r"))*)\s*:\s*"
    r"begin(.*?)end",
    re.DOTALL,
)

# Match a case branch single-statement form: ``value(s): lhs = rhs;``
# The negative lookahead ``(?!begin)`` keeps this disjoint from the
# block form above.  Single-statement branches are common in compact
# RTL and were silently dropped before this fix wired the regex into
# the extractor loop.
_CASE_BRANCH_SINGLE_RE = re.compile(
    r"((?:" + _VAL_ALT + r")(?:\s*,\s*(?:" + _VAL_ALT + r"))*)\s*:\s*"
    r"(?!begin)(\w+(?:\[[^\]]+\])?\s*<?=\s*[^;]+;)",
)

# Tokenise the value list captured by the branch regexes.  Accepts
# typed literals AND parameter-name identifiers.
_CASE_VALUE_TOKEN_RE = re.compile(
    r"\d+'[bhdoBHDO][\w?xz]+|[A-Za-z_]\w*"
)

# Match: default: begin ... end  OR  default: lhs = rhs;
_DEFAULT_BRANCH_RE = re.compile(
    r"default\s*:\s*begin(.*?)end",
    re.DOTALL,
)
_DEFAULT_BRANCH_SINGLE_RE = re.compile(
    r"default\s*:\s*(?!begin)(\w+(?:\[[^\]]+\])?\s*<?=\s*[^;]+;)",
)

# Match an assignment inside a case branch body.
_ASSIGNMENT_RE = re.compile(
    r"(\w+)\s*=\s*(.+?)\s*;",
)

# Match Verilog literals: N'bXXX, N'hXXX, N'dXXX (also octal + casez/x wildcards)
_LITERAL_RE = re.compile(r"\d+'[bhdoBHDO][\w_?xz]+")


def _extract_case_patterns(source: str) -> List[RTLPattern]:
    """Extract case-statement branch patterns.

    Recognised forms (after this round of fixes):
      * ``case`` / ``casez`` / ``casex``
      * Block-form branches: ``2'b00: begin ... end``
      * Single-statement branches: ``2'b00: out = expr;``
      * Multi-value branches: ``2'b00, 2'b01: ...`` (one pattern per value)
      * Bare-identifier values (parameter names): ``IDLE: ...``
      * ``default`` branches in either block or single-stmt form
    """
    patterns = []

    def _emit(case_start, branch_start, values_str, body_or_stmt,
              is_block):
        values = _CASE_VALUE_TOKEN_RE.findall(values_str)
        if is_block:
            assignments = _ASSIGNMENT_RE.findall(body_or_stmt)
        else:
            am = _ASSIGNMENT_RE.search(body_or_stmt)
            assignments = [am.groups()] if am else []
        for value in values:
            for lhs, rhs in assignments:
                lhs = lhs.strip()
                rhs = rhs.strip()
                if lhs.startswith("'") or "bx" in rhs:
                    continue
                patterns.append(RTLPattern(
                    pattern_type="case_branch",
                    source_line=_line_number(
                        source, case_start + branch_start
                    ),
                    source_text=f"{selector} == {value}: {lhs} = {rhs}",
                    selector=selector,
                    condition=value,
                    lhs=lhs, rhs=rhs,
                    is_sequential=False,
                    clock=None, reset=None,
                ))

    for case_match in _CASE_BLOCK_RE.finditer(source):
        selector = case_match.group(1).strip()
        case_body = case_match.group(2)
        case_start = case_match.start()

        # Block-form branches.  Track their spans so we can blank
        # them out before scanning for single-statement branches —
        # otherwise the single-stmt regex could re-match the first
        # statement inside a block body.
        spans: List[Tuple[int, int]] = []
        for branch_match in _CASE_BRANCH_BEGIN_RE.finditer(case_body):
            spans.append((branch_match.start(), branch_match.end()))
            _emit(case_start, branch_match.start(),
                  branch_match.group(1).strip(),
                  branch_match.group(2),
                  is_block=True)

        # Default block.
        default_match = _DEFAULT_BRANCH_RE.search(case_body)
        if default_match:
            spans.append((default_match.start(), default_match.end()))
            for lhs, rhs in _ASSIGNMENT_RE.findall(default_match.group(1)):
                lhs = lhs.strip()
                rhs = rhs.strip()
                if lhs.startswith("'") or "bx" in rhs:
                    continue
                patterns.append(RTLPattern(
                    pattern_type="case_branch",
                    source_line=_line_number(
                        source, case_start + default_match.start()
                    ),
                    source_text=f"{selector} == default: {lhs} = {rhs}",
                    selector=selector,
                    condition="default",
                    lhs=lhs, rhs=rhs,
                    is_sequential=False,
                    clock=None, reset=None,
                ))

        # Mask out block spans before searching for single-statement
        # branches — otherwise an inner ``lhs = rhs;`` would be
        # matched as a single-stmt branch.
        case_body_no_blocks = case_body
        for start, end in sorted(spans, reverse=True):
            case_body_no_blocks = (
                case_body_no_blocks[:start]
                + " " * (end - start)
                + case_body_no_blocks[end:]
            )

        # Single-statement branches.
        for branch_match in _CASE_BRANCH_SINGLE_RE.finditer(
                case_body_no_blocks):
            _emit(case_start, branch_match.start(),
                  branch_match.group(1).strip(),
                  branch_match.group(2),
                  is_block=False)

        # Default single-statement form.
        for dm in _DEFAULT_BRANCH_SINGLE_RE.finditer(case_body_no_blocks):
            am = _ASSIGNMENT_RE.search(dm.group(1))
            if not am:
                continue
            lhs, rhs = am.group(1).strip(), am.group(2).strip()
            if lhs.startswith("'") or "bx" in rhs:
                continue
            patterns.append(RTLPattern(
                pattern_type="case_branch",
                source_line=_line_number(source, case_start + dm.start()),
                source_text=f"{selector} == default: {lhs} = {rhs}",
                selector=selector, condition="default",
                lhs=lhs, rhs=rhs,
                is_sequential=False, clock=None, reset=None,
            ))

    return patterns


# ---------------------------------------------------------------------------
# Pattern 2: Continuous assignments
# ---------------------------------------------------------------------------

_CONT_ASSIGN_RE = re.compile(
    r"assign\s+(\w+(?:\[.*?\])?)\s*=\s*(.+?)\s*;",
)


def _extract_assign_patterns(source: str) -> List[RTLPattern]:
    """Extract continuous assignment patterns (assign out = expr;)."""
    patterns = []

    for match in _CONT_ASSIGN_RE.finditer(source):
        lhs = match.group(1).strip()
        rhs = match.group(2).strip()

        # Classify: ternary, passthrough, or general.
        if "?" in rhs and ":" in rhs:
            ptype = "ternary_mux"
        elif re.match(r"^\w+(\[.*\])?$", rhs):
            ptype = "wire_passthrough"
        else:
            ptype = "direct_assign"

        patterns.append(RTLPattern(
            pattern_type=ptype,
            source_line=_line_number(source, match.start()),
            source_text=f"assign {lhs} = {rhs}",
            selector=None,
            condition=None,
            lhs=lhs,
            rhs=rhs,
            is_sequential=False,
            clock=None,
            reset=None,
        ))

    return patterns


# ---------------------------------------------------------------------------
# Pattern 3: Always blocks (sequential + combinational)
# ---------------------------------------------------------------------------

# Match: always @(posedge clk or negedge rst) begin ... end
_ALWAYS_BLOCK_RE = re.compile(
    r"always\s*@\s*\((.+?)\)\s*begin(.*?)(?=\balways\b|\bendmodule\b)",
    re.DOTALL,
)


# Detect ``always @(...) <statement>`` without a surrounding
# ``begin/end``.  Used by ``_normalise_always_blocks`` to wrap such
# blocks in synthetic ``begin/end`` so the rest of the extractor
# (which assumes block form) sees them.
_ALWAYS_NO_BEGIN_RE = re.compile(
    # Inner content of the @(...) sensitivity list — strictly anchored
    # to avoid catastrophic backtracking.  Allows one level of nested
    # parens (e.g. ``@(rising_edge(clk))``) without overlapping
    # alternatives.
    #   [^()]*               — leading non-paren run
    #   (?:\([^()]*\)[^()]*)* — pairs of (balanced parens) + non-paren
    # This pattern has exactly one way to match each character.
    #
    # The lookahead ``(?!\s*begin\b)`` guards against accidentally
    # wrapping an already-blocked always.
    r"\balways\s*@\s*\([^()]*(?:\([^()]*\)[^()]*)*\)(?!\s*begin\b)",
)


def _normalise_always_blocks(source: str) -> str:
    """Wrap each ``always @(...) <statement>`` in synthetic
    ``begin/end`` so the rest of the extractor (which presumes block
    form) can process it uniformly.  No-op for already-blocked
    always statements.

    The body of a no-begin always is the single statement that
    follows: typically ``lhs = expr;``, ``lhs <= expr;``, or an
    ``if-else`` statement.  We find the end of that statement by
    walking forward until we either:
      - close a balanced ``if-else`` chain (matching every ``if`` to
        its corresponding else+statement or hitting end-of-file
        without an ``else``),
      - hit a ``;`` outside any pending ``if`` chain.
    The wrapped form is then ``always @(...) begin <body> end``.
    """
    out: List[str] = []
    pos = 0
    n = len(source)
    while pos < n:
        m = _ALWAYS_NO_BEGIN_RE.search(source, pos)
        if not m:
            out.append(source[pos:])
            break
        out.append(source[pos:m.end()])
        # Find the end of the single statement.
        cursor = m.end()
        depth_paren = 0
        # Skip whitespace.
        while cursor < n and source[cursor].isspace():
            cursor += 1
        # We need to recognise an ``if (...) body [else body]`` or a
        # bare assignment.  Simple heuristic: consume tokens until we
        # see a top-level ``;`` AND we are not at the start of an
        # ``else`` keyword (which would extend the statement).
        body_start = cursor
        while cursor < n:
            ch = source[cursor]
            if ch == "(":
                depth_paren += 1
            elif ch == ")":
                depth_paren -= 1
            elif ch == ";" and depth_paren == 0:
                # Check whether the next non-space token is ``else``.
                look = cursor + 1
                while look < n and source[look].isspace():
                    look += 1
                if source[look:look + 4] == "else" and \
                        (look + 4 == n or not source[look + 4].isalnum()
                         and source[look + 4] != "_"):
                    cursor = look + 4
                    continue
                # Statement ends here.
                cursor += 1
                break
            cursor += 1
        body = source[body_start:cursor]
        out.append(f"begin {body} end")
        pos = cursor
    return "".join(out)

# Match reset: if (!rst) begin ... end (active-low — common in NVDLA)
_RESET_IF_RE = re.compile(
    r"if\s*\(\s*!(\w+)\s*\)\s*begin(.*?)end",
    re.DOTALL,
)

# Match reset: if (rst) begin ... end (active-high — common in industry)
# Caller must verify the captured signal is the design's reset signal
# (via reset_polarity=="high" on the surrounding always-block) before
# treating these as seq_reset patterns.  Without that guard this regex
# would also match generic conditional assigns like ``if (en) ...``.
_RESET_IF_HIGH_RE = re.compile(
    r"if\s*\(\s*(\w+)\s*\)\s*begin(.*?)end",
    re.DOTALL,
)

# Match non-blocking assignment: signal <= expr;
_NB_ASSIGN_RE = re.compile(
    r"(\w+(?:\[.*?\])?)\s*<=\s*(.+?)\s*;",
)

# Match if-condition before assignment: if(cond) signal <= expr;
_IF_NB_ASSIGN_RE = re.compile(
    r"if\s*\((.+?)\)\s*(?:begin)?\s*(\w+)\s*<=\s*(.+?)\s*;",
)


# Match `if (cond) begin ... end` block — companion to _IF_NB_ASSIGN_RE
# for the multi-LHS case (one block can contain several non-blocking
# assignments, all sharing the same condition).
_IF_NB_BLOCK_RE = re.compile(
    r"if\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*begin(.*?)end",
    re.DOTALL,
)

# Match combinational always block: always @(signal or signal ...)
_COMB_ALWAYS_RE = re.compile(
    r"always\s*@\s*\((?!.*(?:posedge|negedge))(.+?)\)\s*begin(.*?)(?=\balways\b|\bendmodule\b)",
    re.DOTALL,
)

# Match blocking assignment in combinational block: signal = expr;
# Two refinements over a naive `\w+\s*=\s*(.+?)\s*;`:
#   1. Allow bit-select on LHS (``out[3:0] = ...``) — without this,
#      the regex skips bit-select assignments and may instead match a
#      later identifier appearing on the RHS as if it were the LHS.
#   2. ``=(?!=)`` — reject the match when the ``=`` is the first half
#      of ``==`` so we don't split a comparison operator and produce
#      garbage RHS like ``=0 ? a : b``.
_BLOCKING_ASSIGN_RE = re.compile(
    r"(\w+(?:\[[^\]]+\])?)\s*=(?!=)\s*(.+?)\s*;",
)


# Match `if (cond) begin ... end else begin ... end` with up to one
# level of parenthesised sub-expressions in the condition.  The
# condition group uses a strictly-anchored form
# (``[^()]*(\([^()]*\)[^()]*)*``) so each character belongs to
# exactly one alternative — this is critical to avoid catastrophic
# regex backtracking on large RTL.  Captures:
#   group 1 — the if condition
#   group 2 — the then-branch body
#   group 3 — the else-branch body
_IF_ELSE_BLOCK_RE = re.compile(
    r"if\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*"
    r"begin(.*?)end\s*else\s*begin(.*?)end",
    re.DOTALL,
)


# Match `if (cond) statement; else statement;` (single-statement form).
# Same anchored cond pattern.
_IF_ELSE_SINGLE_RE = re.compile(
    r"if\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*"
    r"([^{};]+;)\s*else\s+([^{};]+;)",
    re.DOTALL,
)


# Match mixed-form: ``if (cond) begin ... end else single_stmt;``
# Captures: 1=cond, 2=then-block-body, 3=else-single-statement
_IF_BLOCK_ELSE_SINGLE_RE = re.compile(
    r"if\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*"
    r"begin(.*?)end\s*else\s+([^{};]+;)",
    re.DOTALL,
)


# Match mixed-form: ``if (cond) single_stmt; else begin ... end``
_IF_SINGLE_ELSE_BLOCK_RE = re.compile(
    r"if\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*"
    r"([^{};]+;)\s*else\s*begin(.*?)end",
    re.DOTALL,
)


# Components of an `if-else if-else if-...-else` cascade.  Used by the
# cascade parser, NOT by the simpler if-else extractor — cascades
# have to be parsed top-down with proper begin/end nesting because
# each branch can itself contain blocks.
_IF_HEAD_RE = re.compile(
    r"\bif\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*begin\b",
    re.DOTALL,
)
_ELSE_IF_HEAD_RE = re.compile(
    r"\s*\belse\s+if\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*begin\b",
    re.DOTALL,
)
_ELSE_HEAD_RE = re.compile(
    r"\s*\belse\s*begin\b",
    re.DOTALL,
)


def _find_matching_end(text: str, start_pos: int) -> int:
    """Find the index of the ``end`` keyword that matches a ``begin``
    whose CLOSING delimiter sits at ``start_pos`` (i.e. start_pos is
    the first char *after* the begin token).  Returns -1 on no match.

    Counts begin/end pairs so nested blocks are skipped.  Treats
    ``case``/``endcase`` and ``module``/``endmodule`` as orthogonal
    (those use different delimiters and shouldn't affect the count).
    """
    depth = 1
    cursor = start_pos
    n = len(text)
    while cursor < n:
        m = re.search(r"\b(begin|end)\b", text[cursor:])
        if not m:
            return -1
        kw = m.group(1)
        kw_start = cursor + m.start()
        kw_end = cursor + m.end()
        if kw == "begin":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return kw_start
        cursor = kw_end
    return -1


def _parse_if_cascade(
    body: str, start_pos: int,
) -> Tuple[Optional[List[Tuple[str, str]]], Optional[str], int]:
    """Parse an ``if (...) begin ... end [else if ...]* [else ...]``
    cascade starting at ``body[start_pos]``.

    Returns ``(chain, final_else_body, end_pos)`` where:
      * ``chain`` is a list of ``(condition, branch_body)`` pairs,
        one per ``if`` / ``else if`` clause (in priority order).
      * ``final_else_body`` is the body of the trailing ``else``
        clause, or ``None`` if there isn't one.
      * ``end_pos`` is the index just after the cascade's last
        ``end`` token.

    Returns ``(None, None, start_pos)`` when no cascade starts at
    ``start_pos`` or the structure is malformed.
    """
    head = _IF_HEAD_RE.match(body, start_pos)
    if not head:
        return None, None, start_pos
    chain: List[Tuple[str, str]] = []
    cond = head.group(1).strip()
    body_open = head.end()  # just after `begin`
    body_close = _find_matching_end(body, body_open)
    if body_close < 0:
        return None, None, start_pos
    chain.append((cond, body[body_open:body_close]))
    cursor = body_close + len("end")

    final_else: Optional[str] = None
    while True:
        eim = _ELSE_IF_HEAD_RE.match(body, cursor)
        if eim:
            ec = eim.group(1).strip()
            eb_open = eim.end()
            eb_close = _find_matching_end(body, eb_open)
            if eb_close < 0:
                return None, None, start_pos
            chain.append((ec, body[eb_open:eb_close]))
            cursor = eb_close + len("end")
            continue
        em = _ELSE_HEAD_RE.match(body, cursor)
        if em:
            eb_open = em.end()
            eb_close = _find_matching_end(body, eb_open)
            if eb_close < 0:
                return None, None, start_pos
            final_else = body[eb_open:eb_close]
            cursor = eb_close + len("end")
        break
    return chain, final_else, cursor


def _extract_always_patterns(
    source: str,
    clock: Optional[str],
    reset: Optional[str],
) -> List[RTLPattern]:
    """Extract patterns from always blocks."""
    patterns = []

    # Sequential always blocks (posedge/negedge in sensitivity).
    for match in _ALWAYS_BLOCK_RE.finditer(source):
        sensitivity = match.group(1)
        body = match.group(2)

        # Detect clock and reset from sensitivity list.
        posedge = re.findall(r"posedge\s+(\w+)", sensitivity)
        negedge = re.findall(r"negedge\s+(\w+)", sensitivity)
        block_clock = posedge[0] if posedge else clock
        block_reset = negedge[0] if negedge else reset

        # Determine if reset is asynchronous (appears in sensitivity list)
        # and its polarity (active-low = negedge, active-high = posedge).
        reset_is_async = False
        reset_polarity = "low"  # default: active-low

        if block_reset and block_reset in [n for n in negedge]:
            # negedge rst in sensitivity → async active-low reset
            reset_is_async = True
            reset_polarity = "low"
        elif block_reset and block_reset in [p for p in posedge if p != block_clock]:
            # posedge rst in sensitivity (not the clock) → async active-high reset
            reset_is_async = True
            reset_polarity = "high"

        # Also detect polarity from the if-condition in the body.
        rst_if_match = re.search(r"if\s*\(\s*(!?)(\w+)\s*\)", body)
        if rst_if_match and rst_if_match.group(2) == block_reset:
            if rst_if_match.group(1) == "!":
                reset_polarity = "low"   # if (!rst) → active-low
            else:
                reset_polarity = "high"  # if (rst)  → active-high

        if not posedge and not negedge:
            # Combinational always block — extract blocking assignments.
            # Strip out case blocks first (already handled by case extractor).
            body_no_case = re.sub(
                r"case\s*\(.+?\).*?endcase", "", body, flags=re.DOTALL
            )

            def _is_simulation_only(cond_text: str) -> bool:
                """Skip if-conditions that are simulation-only PLI
                probes (``$value$plusargs``, ``$test$plusargs``,
                ``$random``-driven coverage hooks).  These don't
                describe RTL behaviour and produce non-synthesisable
                assertions that pyslang rejects."""
                lower = cond_text.lower()
                if "$value$plusargs" in lower:
                    return True
                if "$test$plusargs" in lower:
                    return True
                if "coverage" in lower:
                    return True
                return False

            # Run the cascade extractor BEFORE the simple if-else
            # extractor.  Cascades (``if (a) ... else if (b) ... else
            # ...``) with ≥2 conditions get folded into a single
            # nested-ternary assertion per LHS — both more compact and
            # more correct than the simple if-else extractor's
            # behaviour (which would mis-attribute inner-cascade
            # assignments to the outer else-branch).
            body_no_cascade = body_no_case
            cursor = 0
            while cursor < len(body_no_cascade):
                m = _IF_HEAD_RE.search(body_no_cascade, cursor)
                if not m:
                    break
                if_start = m.start()
                chain, final_else, end_pos = _parse_if_cascade(
                    body_no_cascade, if_start,
                )
                # Cascade = ≥2 conditions (true if-else if priority
                # chain).  1-condition if-else (with or without final
                # else) belongs to the if_branch row-clustering path,
                # not here — that code already produces a single
                # packed assertion for multi-LHS branches.
                is_cascade = (chain is not None and len(chain) >= 2)
                if not is_cascade:
                    cursor = if_start + len("if")
                    continue
                if _is_simulation_only(chain[0][0]):
                    body_no_cascade = (
                        body_no_cascade[:if_start]
                        + " " * (end_pos - if_start)
                        + body_no_cascade[end_pos:]
                    )
                    cursor = end_pos
                    continue

                # Per-branch LHS → RHS map.
                branch_assigns: List[Dict[str, str]] = []
                for cond_str, branch_body in chain:
                    m_lhs: Dict[str, str] = {}
                    for am in _BLOCKING_ASSIGN_RE.finditer(branch_body):
                        ilhs = am.group(1).strip()
                        irhs = am.group(2).strip()
                        if ilhs in ("begin", "end", "if", "else",
                                    "case", "default"):
                            continue
                        m_lhs[ilhs] = irhs
                    branch_assigns.append(m_lhs)
                final_assigns: Dict[str, str] = {}
                if final_else is not None:
                    for am in _BLOCKING_ASSIGN_RE.finditer(final_else):
                        ilhs = am.group(1).strip()
                        irhs = am.group(2).strip()
                        if ilhs in ("begin", "end", "if", "else",
                                    "case", "default"):
                            continue
                        final_assigns[ilhs] = irhs

                # LHS that appear in EVERY branch (including the
                # final else, when present).  These are the signals
                # whose value at any cycle is determined by the
                # cascade — fair game for nested-ternary compaction.
                lhs_universe = set(branch_assigns[0].keys())
                for m_lhs in branch_assigns[1:]:
                    lhs_universe &= set(m_lhs.keys())
                if final_else is not None:
                    lhs_universe &= set(final_assigns.keys())

                for clhs in sorted(lhs_universe):
                    parts = []
                    for (ccond, _), m_lhs in zip(chain, branch_assigns):
                        parts.append(f"({ccond}) ? ({m_lhs[clhs]})")
                    if final_else is not None:
                        tail = f"({final_assigns[clhs]})"
                    else:
                        # No final else — the cascade leaves clhs at
                        # its previous value (latch).  Express as
                        # ``: clhs`` so the assertion still holds.
                        tail = clhs
                    nested_ternary = " : ".join(parts + [tail])
                    cond_summary = " | ".join(c for c, _ in chain)
                    patterns.append(RTLPattern(
                        pattern_type="cascade_assignment",
                        source_line=_line_number(
                            source, match.start() + if_start
                        ),
                        source_text=(
                            f"if-cascade ({len(chain)} conds"
                            f"{', + final else' if final_else else ''}) "
                            f"-> {clhs}"
                        ),
                        selector=cond_summary[:100],
                        condition=None,
                        lhs=clhs, rhs=nested_ternary,
                        is_sequential=False,
                        clock=None, reset=None,
                    ))

                # Blank out the cascade so subsequent extractors
                # don't double-emit.
                body_no_cascade = (
                    body_no_cascade[:if_start]
                    + " " * (end_pos - if_start)
                    + body_no_cascade[end_pos:]
                )
                cursor = end_pos

            # Strip-and-extract if-else blocks.  Their inner blocking
            # assignments emit ``if_branch`` patterns instead of bare
            # ``direct_assign`` so the if-condition + which-branch info
            # isn't lost.  The block source is replaced with whitespace
            # to preserve absolute char offsets for line-number lookup.
            body_no_ifelse = body_no_cascade

            # Iterate over cascade-blanked body so cascades aren't
            # re-extracted here as 2-branch if-else patterns.
            for ie_match in list(_IF_ELSE_BLOCK_RE.finditer(body_no_cascade)):
                cond = ie_match.group(1).strip()
                if _is_simulation_only(cond):
                    # Still strip from body so the generic blocking-
                    # assign pass doesn't re-extract them as
                    # direct_assigns.
                    body_no_ifelse = (
                        body_no_ifelse[:ie_match.start()]
                        + " " * (ie_match.end() - ie_match.start())
                        + body_no_ifelse[ie_match.end():]
                    )
                    continue
                then_body = ie_match.group(2)
                else_body = ie_match.group(3)
                ie_start = ie_match.start()

                for inner_re, inner_text, branch_label in (
                    (_BLOCKING_ASSIGN_RE, then_body, "then"),
                    (_BLOCKING_ASSIGN_RE, else_body, "else"),
                ):
                    for am in inner_re.finditer(inner_text):
                        ilhs = am.group(1).strip()
                        irhs = am.group(2).strip()
                        if ilhs in ("begin", "end", "if", "else",
                                    "case", "default"):
                            continue
                        patterns.append(RTLPattern(
                            pattern_type="if_branch",
                            source_line=_line_number(
                                source, match.start() + ie_start
                            ),
                            source_text=f"if ({cond}) {ilhs} = {irhs}"
                                        if branch_label == "then"
                                        else f"else {ilhs} = {irhs}",
                            selector=cond,
                            condition=branch_label,
                            lhs=ilhs, rhs=irhs,
                            is_sequential=False,
                            clock=None, reset=None,
                        ))
                # Blank out the matched range to prevent double-extraction.
                body_no_ifelse = (
                    body_no_ifelse[:ie_match.start()]
                    + " " * (ie_match.end() - ie_match.start())
                    + body_no_ifelse[ie_match.end():]
                )

            # Mixed-form if-else: ``if (c) begin ... end else single;``
            # Run BEFORE the single-statement variant so the trailing
            # else-statement isn't first picked up as a stand-alone
            # blocking assignment.
            for ie_match in list(
                    _IF_BLOCK_ELSE_SINGLE_RE.finditer(body_no_ifelse)):
                cond = ie_match.group(1).strip()
                if _is_simulation_only(cond):
                    body_no_ifelse = (
                        body_no_ifelse[:ie_match.start()]
                        + " " * (ie_match.end() - ie_match.start())
                        + body_no_ifelse[ie_match.end():]
                    )
                    continue
                then_body = ie_match.group(2)
                else_stmt = ie_match.group(3)
                ie_start = ie_match.start()
                # Then-branch block body: iterate _BLOCKING_ASSIGN_RE.
                for am in _BLOCKING_ASSIGN_RE.finditer(then_body):
                    ilhs = am.group(1).strip()
                    irhs = am.group(2).strip()
                    if ilhs in ("begin", "end", "if", "else", "case",
                                "default"):
                        continue
                    patterns.append(RTLPattern(
                        pattern_type="if_branch",
                        source_line=_line_number(
                            source, match.start() + ie_start
                        ),
                        source_text=f"if ({cond}) {ilhs} = {irhs}",
                        selector=cond, condition="then",
                        lhs=ilhs, rhs=irhs,
                        is_sequential=False, clock=None, reset=None,
                    ))
                # Else-branch single statement: parse one assignment.
                am = _BLOCKING_ASSIGN_RE.search(else_stmt)
                if am:
                    ilhs = am.group(1).strip()
                    irhs = am.group(2).strip()
                    if ilhs not in ("begin", "end", "if", "else", "case",
                                     "default"):
                        patterns.append(RTLPattern(
                            pattern_type="if_branch",
                            source_line=_line_number(
                                source, match.start() + ie_start
                            ),
                            source_text=f"else {ilhs} = {irhs}",
                            selector=cond, condition="else",
                            lhs=ilhs, rhs=irhs,
                            is_sequential=False, clock=None, reset=None,
                        ))
                body_no_ifelse = (
                    body_no_ifelse[:ie_match.start()]
                    + " " * (ie_match.end() - ie_match.start())
                    + body_no_ifelse[ie_match.end():]
                )

            # Mirror form: ``if (c) single; else begin ... end``
            for ie_match in list(
                    _IF_SINGLE_ELSE_BLOCK_RE.finditer(body_no_ifelse)):
                cond = ie_match.group(1).strip()
                if _is_simulation_only(cond):
                    body_no_ifelse = (
                        body_no_ifelse[:ie_match.start()]
                        + " " * (ie_match.end() - ie_match.start())
                        + body_no_ifelse[ie_match.end():]
                    )
                    continue
                then_stmt = ie_match.group(2)
                else_body = ie_match.group(3)
                ie_start = ie_match.start()
                # Then-branch single statement.
                am = _BLOCKING_ASSIGN_RE.search(then_stmt)
                if am:
                    ilhs = am.group(1).strip()
                    irhs = am.group(2).strip()
                    if ilhs not in ("begin", "end", "if", "else", "case",
                                     "default"):
                        patterns.append(RTLPattern(
                            pattern_type="if_branch",
                            source_line=_line_number(
                                source, match.start() + ie_start
                            ),
                            source_text=f"if ({cond}) {ilhs} = {irhs}",
                            selector=cond, condition="then",
                            lhs=ilhs, rhs=irhs,
                            is_sequential=False, clock=None, reset=None,
                        ))
                # Else-branch block body.
                for am in _BLOCKING_ASSIGN_RE.finditer(else_body):
                    ilhs = am.group(1).strip()
                    irhs = am.group(2).strip()
                    if ilhs in ("begin", "end", "if", "else", "case",
                                "default"):
                        continue
                    patterns.append(RTLPattern(
                        pattern_type="if_branch",
                        source_line=_line_number(
                            source, match.start() + ie_start
                        ),
                        source_text=f"else {ilhs} = {irhs}",
                        selector=cond, condition="else",
                        lhs=ilhs, rhs=irhs,
                        is_sequential=False, clock=None, reset=None,
                    ))
                body_no_ifelse = (
                    body_no_ifelse[:ie_match.start()]
                    + " " * (ie_match.end() - ie_match.start())
                    + body_no_ifelse[ie_match.end():]
                )

            # Single-statement if-else form: `if (c) lhs = a; else lhs = b;`
            for ie_match in list(_IF_ELSE_SINGLE_RE.finditer(body_no_ifelse)):
                cond = ie_match.group(1).strip()
                if _is_simulation_only(cond):
                    body_no_ifelse = (
                        body_no_ifelse[:ie_match.start()]
                        + " " * (ie_match.end() - ie_match.start())
                        + body_no_ifelse[ie_match.end():]
                    )
                    continue
                then_stmt = ie_match.group(2)
                else_stmt = ie_match.group(3)
                ie_start = ie_match.start()
                for inner_text, branch_label in (
                    (then_stmt, "then"),
                    (else_stmt, "else"),
                ):
                    am = _BLOCKING_ASSIGN_RE.search(inner_text)
                    if not am:
                        continue
                    ilhs = am.group(1).strip()
                    irhs = am.group(2).strip()
                    if ilhs in ("begin", "end", "if", "else",
                                "case", "default"):
                        continue
                    patterns.append(RTLPattern(
                        pattern_type="if_branch",
                        source_line=_line_number(
                            source, match.start() + ie_start
                        ),
                        source_text=f"if ({cond}) {ilhs} = {irhs}"
                                    if branch_label == "then"
                                    else f"else {ilhs} = {irhs}",
                        selector=cond,
                        condition=branch_label,
                        lhs=ilhs, rhs=irhs,
                        is_sequential=False,
                        clock=None, reset=None,
                    ))
                body_no_ifelse = (
                    body_no_ifelse[:ie_match.start()]
                    + " " * (ie_match.end() - ie_match.start())
                    + body_no_ifelse[ie_match.end():]
                )

            for assign_match in _BLOCKING_ASSIGN_RE.finditer(body_no_ifelse):
                lhs = assign_match.group(1).strip()
                rhs = assign_match.group(2).strip()
                # Skip keywords and pragmas.
                if lhs in ("begin", "end", "if", "else", "case", "default"):
                    continue
                # Check if it's a comparison.
                if re.match(r"\(.+?\s*==\s*.+?\)", rhs):
                    ptype = "comb_comparison"
                else:
                    ptype = "direct_assign"
                patterns.append(RTLPattern(
                    pattern_type=ptype,
                    source_line=_line_number(source, match.start() + assign_match.start()),
                    source_text=f"{lhs} = {rhs}",
                    selector=None,
                    condition=None,
                    lhs=lhs,
                    rhs=rhs,
                    is_sequential=False,
                    clock=None,
                    reset=None,
                ))
            continue

        # Sequential block — find reset assignments.  Choose the
        # reset-detector by polarity: active-low searches for
        # ``if (!rst)``; active-high searches for ``if (rst)`` and
        # additionally verifies the captured signal matches
        # ``block_reset`` (so we don't misclassify non-reset
        # conditionals as resets).
        reset_match = None
        if reset_polarity == "low":
            reset_match = _RESET_IF_RE.search(body)
        elif reset_polarity == "high" and block_reset:
            for cand in _RESET_IF_HIGH_RE.finditer(body):
                if cand.group(1) == block_reset:
                    reset_match = cand
                    break
        if reset_match:
            rst_signal = reset_match.group(1)
            rst_body = reset_match.group(2)
            rst_cond_text = (
                f"!{rst_signal}" if reset_polarity == "low" else rst_signal
            )

            for nb_match in _NB_ASSIGN_RE.finditer(rst_body):
                lhs = nb_match.group(1).strip()
                rhs = nb_match.group(2).strip()
                patterns.append(RTLPattern(
                    pattern_type="seq_reset",
                    source_line=_line_number(source, match.start()),
                    source_text=f"if ({rst_cond_text}) {lhs} <= {rhs}",
                    selector=None,
                    condition=rst_cond_text,
                    lhs=lhs,
                    rhs=rhs,
                    is_sequential=True,
                    clock=block_clock,
                    reset=rst_signal,
                    reset_is_async=reset_is_async,
                    reset_polarity=reset_polarity,
                ))

        # Find functional assignments in the else-branch.
        # Look for if(cond) signal <= expr; patterns after the reset block.
        else_body = body
        if reset_match:
            # Get everything after the reset if-else.
            else_pos = body.find("end else begin", reset_match.start())
            if else_pos >= 0:
                else_body = body[else_pos + len("end else begin"):]

        # Multi-LHS block form first: ``if (cond) begin <multiple
        # assignments> end``.  Without this, _IF_NB_ASSIGN_RE below
        # only captures the FIRST assignment in the block (because
        # its regex is a single ``if (cond) ... lhs <= rhs;`` shape),
        # and the remaining assignments fall through to the
        # unconditional _NB_ASSIGN_RE pass — losing their condition
        # and breaking condition-clustering for the family.
        else_body_no_blocks = else_body
        for blk in list(_IF_NB_BLOCK_RE.finditer(else_body)):
            condition = blk.group(1).strip()
            blk_body = blk.group(2)
            if "coverage" in condition.lower() or "$value$plusargs" in condition.lower():
                # Strip simulation-only pragma bodies so they don't
                # leak through to the unconditional pass.
                else_body_no_blocks = (
                    else_body_no_blocks[:blk.start()]
                    + " " * (blk.end() - blk.start())
                    + else_body_no_blocks[blk.end():]
                )
                continue
            for nb in _NB_ASSIGN_RE.finditer(blk_body):
                lhs = nb.group(1).strip()
                rhs = nb.group(2).strip()
                if lhs.startswith("'"):
                    continue
                patterns.append(RTLPattern(
                    pattern_type="seq_func",
                    source_line=_line_number(source, match.start()),
                    source_text=f"if ({condition}) {lhs} <= {rhs}",
                    selector=None,
                    condition=condition,
                    lhs=lhs,
                    rhs=rhs,
                    is_sequential=True,
                    clock=block_clock,
                    reset=block_reset,
                    reset_is_async=reset_is_async,
                    reset_polarity=reset_polarity,
                ))
            # Blank out the block in the leftover-body so neither
            # `_IF_NB_ASSIGN_RE` nor `_NB_ASSIGN_RE` re-process it.
            else_body_no_blocks = (
                else_body_no_blocks[:blk.start()]
                + " " * (blk.end() - blk.start())
                + else_body_no_blocks[blk.end():]
            )

        for if_match in _IF_NB_ASSIGN_RE.finditer(else_body_no_blocks):
            condition = if_match.group(1).strip()
            lhs = if_match.group(2).strip()
            rhs = if_match.group(3).strip()
            # Skip VCS coverage pragmas.
            if "coverage" in condition.lower() or lhs.startswith("'"):
                continue
            patterns.append(RTLPattern(
                pattern_type="seq_func",
                source_line=_line_number(source, match.start()),
                source_text=f"if ({condition}) {lhs} <= {rhs}",
                selector=None,
                condition=condition,
                lhs=lhs,
                rhs=rhs,
                is_sequential=True,
                clock=block_clock,
                reset=block_reset,
                reset_is_async=reset_is_async,
                reset_polarity=reset_polarity,
            ))

        # Simple unconditional assignments in else-branch.
        for nb_match in _NB_ASSIGN_RE.finditer(else_body):
            lhs = nb_match.group(1).strip()
            rhs = nb_match.group(2).strip()
            # Skip if already captured by if-condition extraction.
            if any(p.lhs == lhs and p.pattern_type == "seq_func" for p in patterns):
                continue
            # Skip reset values (already captured).
            if any(p.lhs == lhs and p.pattern_type == "seq_reset" for p in patterns):
                continue
            patterns.append(RTLPattern(
                pattern_type="seq_func",
                source_line=_line_number(source, match.start()),
                source_text=f"{lhs} <= {rhs}",
                selector=None,
                condition=None,
                lhs=lhs,
                rhs=rhs,
                is_sequential=True,
                clock=block_clock,
                reset=block_reset,
            ))

    return patterns


# ---------------------------------------------------------------------------
# Skeleton generation — template-based
# ---------------------------------------------------------------------------

def generate_skeletons(
    patterns: List[RTLPattern],
    is_combinational: bool,
) -> List[AssertionSkeleton]:
    """
    Convert extracted patterns into assertion skeletons using templates.
    """
    skeletons: List[AssertionSkeleton] = []

    for p in patterns:
        skel = _pattern_to_skeleton(p, is_combinational)
        if skel:
            skeletons.append(skel)

    # Add invariant assertions derived from pattern analysis.
    invariants = _generate_invariant_skeletons(patterns, is_combinational)
    skeletons.extend(invariants)

    logger.info("Generated %d assertion skeleton(s) (%d invariants).",
                len(skeletons), len(invariants))
    return skeletons


def _generate_invariant_skeletons(
    patterns: List[RTLPattern],
    is_combinational: bool,
) -> List[AssertionSkeleton]:
    """
    Generate invariant assertions from pattern analysis.

    Detects common hardware patterns and generates correct assertions:

    * **Decoder one-hot**: when ≥2 ``comb_comparison`` patterns
      decode the same source signal into multiple flags
      (``cfg_is_int8 = (mode == 0)``, ``cfg_is_fp16 = (mode == 1)``,
      ``cfg_is_int16 = (mode == 2)``), emit:

      - ``$onehot0({...})`` — at most one flag is high (always
        emitted, replaces the legacy ``(a+b+c) <= 1`` form which
        is mathematically equivalent but less idiomatic).
      - ``$onehot({...})`` — exactly one flag is high (emitted only
        when the decoded value set covers ``{0, 1, ..., 2^N - 1}``
        for some N, which proves the decoder is complete and rules
        out the all-zero case the prior form missed).
    """
    skeletons: List[AssertionSkeleton] = []

    # --- Decoder family detection -----------------------------------------
    comparison_groups: Dict[str, List[RTLPattern]] = {}
    for p in patterns:
        if p.pattern_type == "comb_comparison" and "==" in p.rhs:
            match = re.match(r"\((\w+(?:\[.*?\])?)\s*==", p.rhs)
            if match:
                source_signal = match.group(1)
                comparison_groups.setdefault(source_signal, []).append(p)

    def _decoded_value(rhs: str) -> Optional[int]:
        """Extract the integer the comparison is checking against.

        Recognises decimal (``5``, ``5'd5``) and hex/binary literals
        (``5'h1f``, ``2'b10``).  Returns None when the value isn't a
        plain integer literal (which prevents accidental
        completeness inference for symbolic comparisons).
        """
        m = re.search(r"==\s*(?:\d+'[bdhoBDHO]([0-9a-fA-FxXzZ_]+)|"
                      r"(\d+))", rhs)
        if not m:
            return None
        if m.group(2) is not None:
            return int(m.group(2))
        # Sized literal — base-detection from the prefix.
        prefix_match = re.search(r"==\s*\d+'([bdhoBDHO])", rhs)
        if not prefix_match:
            return None
        base_char = prefix_match.group(1).lower()
        digits = m.group(1).replace("_", "")
        # Drop x/z so partial-X literals don't crash int parsing.
        digits = re.sub(r"[xXzZ]", "0", digits)
        try:
            return int(digits, {"b": 2, "d": 10, "h": 16, "o": 8}[base_char])
        except (ValueError, KeyError):
            return None

    def _is_complete_decoder(values: List[int]) -> bool:
        """True when the decoded value set is exactly
        ``{0, 1, ..., 2^N - 1}`` for some N ≥ 1 — proves the source
        signal must take exactly one of these values, which makes
        ``$onehot`` (rather than ``$onehot0``) sound."""
        if not values:
            return False
        n = len(values)
        # n must be a power of two; values must be 0..n-1.
        if n & (n - 1) != 0:
            return False
        return sorted(values) == list(range(n))

    for source_signal, group in comparison_groups.items():
        if len(group) < 2:
            continue
        flag_names = [p.lhs for p in group]
        flag_concat = "{" + ", ".join(flag_names) + "}"

        # Always emit $onehot0 (replaces legacy <=1 sum form).
        skeletons.append(AssertionSkeleton(
            assertion_text=(
                f"assert ($onehot0({flag_concat})) "
                f'else $error("At most one of {", ".join(flag_names)} '
                f'may be active (decoded from {source_signal})");'
            ),
            pattern_type="invariant_onehot0",
            source_line=group[0].source_line,
            source_text=f"Decoded from {source_signal}",
            description=(
                f"At most one of {', '.join(flag_names)} may be high "
                f"(decoded from {source_signal})"
            ),
        ))

        # Try to detect completeness — if the decoder covers
        # every value of the source, exactly-one is provable.
        decoded_values = [_decoded_value(p.rhs) for p in group]
        if all(v is not None for v in decoded_values) and \
                _is_complete_decoder(decoded_values):
            skeletons.append(AssertionSkeleton(
                assertion_text=(
                    f"assert ($onehot({flag_concat})) "
                    f'else $error("Exactly one of '
                    f'{", ".join(flag_names)} must be active '
                    f'(complete decoder of {source_signal})");'
                ),
                pattern_type="invariant_onehot",
                source_line=group[0].source_line,
                source_text=f"Complete decoder of {source_signal}",
                description=(
                    f"Exactly one of {', '.join(flag_names)} must be "
                    f"high (complete decoder of {source_signal})"
                ),
            ))

    return skeletons


def _pattern_to_skeleton(
    p: RTLPattern,
    is_combinational: bool,
) -> Optional[AssertionSkeleton]:
    """Convert one RTL pattern to an assertion skeleton."""

    if p.pattern_type == "case_branch":
        if p.condition == "default":
            # Default case — skip (hard to express as a guard).
            return None
        assertion = (
            f"assert (!({p.selector} == {p.condition}) || "
            f"({p.lhs} == {p.rhs})) "
            f'else $error("{p.lhs} mismatch when {p.selector}=={p.condition}");'
        )
        desc = f"When {p.selector} == {p.condition}, {p.lhs} must equal {p.rhs}"

    elif p.pattern_type in ("direct_assign", "comb_comparison"):
        assertion = (
            f"assert ({p.lhs} == ({p.rhs})) "
            f'else $error("{p.lhs} assignment mismatch");'
        )
        desc = f"{p.lhs} must always equal {p.rhs}"

    elif p.pattern_type == "wire_passthrough":
        assertion = (
            f"assert ({p.lhs} == {p.rhs}) "
            f'else $error("{p.lhs} passthrough mismatch");'
        )
        desc = f"{p.lhs} must equal {p.rhs}"

    elif p.pattern_type == "ternary_mux":
        assertion = (
            f"assert ({p.lhs} == ({p.rhs})) "
            f'else $error("{p.lhs} mux select mismatch");'
        )
        desc = f"{p.lhs} must equal {p.rhs}"

    elif p.pattern_type == "seq_reset":
        clk = p.clock or "clk"
        rst = p.reset or "rst_n"

        # Reset condition based on polarity.
        rst_cond = f"!{rst}" if p.reset_polarity == "low" else rst

        if p.reset_is_async:
            # Async reset: register clears immediately — use |-> (same cycle).
            assertion = (
                f"assert property (@(posedge {clk}) "
                f"{rst_cond} |-> {p.lhs} == {p.rhs}) "
                f'else $error("{p.lhs} async reset value mismatch");'
            )
            desc = f"On async reset, {p.lhs} must immediately be {p.rhs}"
        else:
            # Sync reset: register clears on next clock edge — use |=> (next cycle).
            assertion = (
                f"assert property (@(posedge {clk}) "
                f"{rst_cond} |=> {p.lhs} == {p.rhs}) "
                f'else $error("{p.lhs} sync reset value mismatch");'
            )
            desc = f"On sync reset, {p.lhs} must be {p.rhs} next cycle"

    elif p.pattern_type == "seq_func":
        clk = p.clock or "clk"
        rst = p.reset or "rst_n"

        # Disable condition based on polarity.
        disable_cond = f"!{rst}" if p.reset_polarity == "low" else rst

        if p.condition:
            assertion = (
                f"assert property (@(posedge {clk}) disable iff ({disable_cond}) "
                f"({p.condition}) |=> {p.lhs} == $past({p.rhs})) "
                f'else $error("{p.lhs} functional update mismatch");'
            )
            desc = f"When {p.condition}, {p.lhs} must update to {p.rhs}"
        else:
            assertion = (
                f"assert property (@(posedge {clk}) disable iff ({disable_cond}) "
                f"{p.lhs} == $past({p.rhs})) "
                f'else $error("{p.lhs} register delay mismatch");'
            )
            desc = f"{p.lhs} must follow {p.rhs} by one cycle"

    elif p.pattern_type == "if_branch":
        # selector = if-condition; condition = "then" or "else"
        cond = p.selector or ""
        if p.condition == "then":
            assertion = (
                f"assert (!({cond}) || ({p.lhs} == ({p.rhs}))) "
                f'else $error("{p.lhs} mismatch in then-branch of if({cond})");'
            )
            desc = f"When {cond}, {p.lhs} must equal {p.rhs}"
        else:  # "else"
            assertion = (
                f"assert (({cond}) || ({p.lhs} == ({p.rhs}))) "
                f'else $error("{p.lhs} mismatch in else-branch of if({cond})");'
            )
            desc = f"When !({cond}), {p.lhs} must equal {p.rhs}"

    elif p.pattern_type == "cascade_assignment":
        # rhs is already a nested-ternary expression covering all
        # branches in priority order.  Single assertion captures the
        # entire if-else if-else chain's behaviour for this LHS.
        sel_summary = (p.selector or "")[:60]
        assertion = (
            f"assert ({p.lhs} == ({p.rhs})) "
            f'else $error("{p.lhs} mismatch in if-cascade '
            f'({sel_summary})");'
        )
        desc = (
            f"{p.lhs} must equal the value selected by the "
            f"if-else if-else cascade"
        )

    else:
        return None

    return AssertionSkeleton(
        assertion_text=assertion,
        pattern_type=p.pattern_type,
        source_line=p.source_line,
        source_text=p.source_text,
        description=desc,
    )


# ---------------------------------------------------------------------------
# Top-level convenience functions
# ---------------------------------------------------------------------------

def generate_ast_assertions(
    source: str,
    clock: Optional[str] = None,
    reset: Optional[str] = None,
    is_combinational: bool = True,
    max_case_branches: int = 50,
    allowed_signals: Optional[Set[str]] = None,
    skip_trivial_internal: bool = False,
) -> List[AssertionSkeleton]:
    """
    Extract RTL patterns and generate assertion skeletons in one call.

    Parameters
    ----------
    source : str
        Full Verilog/SystemVerilog module source text.
    clock : str, optional
        Clock signal name (for sequential assertion templates).
    reset : str, optional
        Reset signal name (for reset assertion templates).
    is_combinational : bool
        True for combinational designs (immediate assertions).
    max_case_branches : int
        Skip per-branch case assertions if a case has more than this many branches.
    allowed_signals : set of str, optional
        When provided AND ``skip_trivial_internal`` is True, trivial
        pattern types (direct_assign, wire_passthrough, comb_comparison)
        are filtered to only those whose LHS signal name is in this set.
        Typically the signal map keys plus top-module ports.
    skip_trivial_internal : bool
        Gates the allowed_signals filter. Default False (preserve current
        behavior); True enables the filter.

    Returns
    -------
    list of AssertionSkeleton
    """
    patterns = extract_patterns(source, clock=clock, reset=reset)

    # Filter out case branches that exceed the limit.
    if max_case_branches > 0:
        # Count case branches per selector.
        case_counts: Dict[str, int] = {}
        for p in patterns:
            if p.pattern_type == "case_branch" and p.selector:
                case_counts[p.selector] = case_counts.get(p.selector, 0) + 1

        # Remove branches from oversized cases.
        filtered = []
        for p in patterns:
            if (p.pattern_type == "case_branch"
                    and p.selector
                    and case_counts.get(p.selector, 0) > max_case_branches):
                continue
            filtered.append(p)

        if len(filtered) < len(patterns):
            dropped = len(patterns) - len(filtered)
            logger.info(
                "Skipped %d case branches (case has > %d branches).",
                dropped, max_case_branches,
            )
        patterns = filtered

    # Skip trivial-restatement patterns for internal signals.
    # Only drops pure RTL-to-SVA restatements (direct_assign,
    # wire_passthrough, comb_comparison). Keeps high-value patterns:
    # case_branch, seq_reset, seq_func, ternary_mux.
    _TRIVIAL_TYPES = {"direct_assign", "wire_passthrough", "comb_comparison"}
    if skip_trivial_internal and allowed_signals:
        filtered = []
        dropped_by_type: Dict[str, int] = {}
        for p in patterns:
            if p.pattern_type in _TRIVIAL_TYPES:
                # Strip bit-selects and braces to get the base identifier.
                base = re.sub(r'\[.*?\]|\{.*?\}', '', p.lhs).strip()
                base = base.split(',')[0].strip()  # handle concat LHS
                if base and base not in allowed_signals:
                    dropped_by_type[p.pattern_type] = (
                        dropped_by_type.get(p.pattern_type, 0) + 1
                    )
                    continue
            filtered.append(p)
        if dropped_by_type:
            total_dropped = sum(dropped_by_type.values())
            logger.info(
                "Skipped %d trivial assertion(s) on internal signals "
                "(not in signal_map): %s",
                total_dropped,
                ", ".join(f"{k}={v}" for k, v in dropped_by_type.items()),
            )
        patterns = filtered

    return generate_skeletons(patterns, is_combinational)


def format_skeletons_as_sva(skeletons: List[AssertionSkeleton]) -> str:
    """Format skeletons into a ready-to-use SVA code string."""
    lines = []
    for skel in skeletons:
        lines.append(f"// {skel.description}")
        lines.append(skel.assertion_text)
        lines.append("")
    return "\n".join(lines)


def format_skeletons_for_llm(skeletons: List[AssertionSkeleton]) -> str:
    """
    Format skeletons into a structured table for LLM enrichment.

    The LLM receives this as context and can review, improve descriptions,
    and add protocol/edge-case assertions not captured by structural patterns.
    """
    lines = [
        f"The following {len(skeletons)} assertion skeletons were automatically "
        "extracted from the RTL source code. Each is syntactically correct.",
        "",
    ]

    for i, skel in enumerate(skeletons, 1):
        lines.append(f"Skeleton {i} [{skel.pattern_type}] (line {skel.source_line}):")
        lines.append(f"  RTL: {skel.source_text}")
        lines.append(f"  SVA: {skel.assertion_text}")
        lines.append(f"  Desc: {skel.description}")
        lines.append("")

    return "\n".join(lines)
