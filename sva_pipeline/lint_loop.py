"""
lint_loop.py
------------
Post-generation linting feedback loop.

After the ReAct agent produces a batch of SVA assertions, this module:
  1. Splits the raw SVA string into individual assertions (with comments).
  2. Lints each assertion independently via verible_lint().
  3. Collects failures into a JSON report persisted to disk.
  4. Feeds failures back to the agent for targeted repair.
  5. Repeats until all assertions pass or a refinement cap is reached.

The final output contains only assertions that passed Verible validation.
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from .rtl_facts import RTLFacts
from .slang_frontend import slang_lint

if TYPE_CHECKING:
    from .agent import SVAAgent
    from .config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Assertion splitter
# ---------------------------------------------------------------------------

def split_assertions(sva_code: str) -> List[Dict[str, str]]:
    """
    Split a raw SVA string into individual assertion entries.

    Each entry is a dict with two keys:
      "comment"   : preceding // comment lines (may be empty string)
      "assertion" : the assert statement itself, including multi-line bodies

    The splitter walks line-by-line and:
      - Accumulates comment lines (starting with //) and blank lines into a
        comment buffer.
      - When a line containing 'assert' is found, starts an assertion buffer.
      - Continues accumulating into the assertion buffer until a line ends
        with ';' (tracking parenthesis depth and string literals to avoid
        false positives from ';' inside $error messages).
      - Emits the (comment, assertion) pair and resets both buffers.

    Parameters
    ----------
    sva_code : str
        Raw SVA code string, potentially with multiple assertions and comments.

    Returns
    -------
    list of dict
        Each dict has "comment" (str) and "assertion" (str) keys.
    """
    if not sva_code or not sva_code.strip():
        return []

    lines = sva_code.splitlines()
    results: List[Dict[str, str]] = []

    comment_buf: List[str] = []
    assertion_buf: List[str] = []
    in_assertion = False

    # Track parenthesis depth and whether we are inside a string literal
    # so that a ';' inside $error("some; text") is not treated as the
    # statement terminator.
    paren_depth = 0
    in_string = False

    for line in lines:
        stripped = line.strip()

        if not in_assertion:
            # Outside an assertion — accumulate comments or detect a new assert.
            if not stripped:
                # Blank line: belongs to the comment block above.
                comment_buf.append("")
            elif stripped.startswith("//"):
                comment_buf.append(stripped)
            elif "assert" in stripped.lower():
                # Start of a new assertion.
                in_assertion = True
                assertion_buf = [stripped]
                paren_depth = 0
                in_string = False
                # Count parens and strings in this first line.
                paren_depth, in_string = _update_paren_state(
                    stripped, paren_depth, in_string
                )
                # Check if the assertion is complete on this single line.
                if _assertion_complete(stripped, paren_depth, in_string):
                    results.append({
                        "comment": "\n".join(comment_buf).strip(),
                        "assertion": "\n".join(assertion_buf).strip(),
                    })
                    comment_buf = []
                    assertion_buf = []
                    in_assertion = False
                    paren_depth = 0
                    in_string = False
            else:
                # Non-comment, non-assert line — treat as part of the comment
                # context (e.g. a stray descriptive line).
                comment_buf.append(stripped)
        else:
            # A new `assert` line while still accumulating means the previous
            # assertion never triggered _assertion_complete. Two cases:
            #   a) the previous buffer IS complete (ends with `;`, parens
            #      balanced) but the per-line check missed it — commit it.
            #   b) the previous buffer is truncated (no `;`, or unbalanced
            #      parens) — discard to stop it swallowing all following
            #      assertions into one blob.
            if not in_string and re.match(r'^\s*assert\b', stripped, re.IGNORECASE):
                prev_joined = " ".join(assertion_buf).rstrip()
                prev_balanced = (paren_depth == 0 and not in_string
                                 and prev_joined.endswith(";"))
                if prev_balanced:
                    results.append({
                        "comment": "\n".join(comment_buf).strip(),
                        "assertion": "\n".join(assertion_buf).strip(),
                    })
                    comment_buf = []
                else:
                    logger.warning(
                        "SVA splitter: discarding truncated assertion "
                        "fragment: %s", prev_joined[:120],
                    )
                assertion_buf = [stripped]
                paren_depth = 0
                in_string = False
                paren_depth, in_string = _update_paren_state(
                    stripped, paren_depth, in_string
                )
                if _assertion_complete(stripped, paren_depth, in_string):
                    results.append({
                        "comment": "\n".join(comment_buf).strip(),
                        "assertion": "\n".join(assertion_buf).strip(),
                    })
                    comment_buf = []
                    assertion_buf = []
                    in_assertion = False
                    paren_depth = 0
                    in_string = False
                continue

            # Inside a multi-line assertion — keep accumulating.
            assertion_buf.append(stripped)
            paren_depth, in_string = _update_paren_state(
                stripped, paren_depth, in_string
            )
            if _assertion_complete(stripped, paren_depth, in_string):
                results.append({
                    "comment": "\n".join(comment_buf).strip(),
                    "assertion": "\n".join(assertion_buf).strip(),
                })
                comment_buf = []
                assertion_buf = []
                in_assertion = False
                paren_depth = 0
                in_string = False

    # If we ended mid-assertion (truncated output), still emit what we have.
    if assertion_buf:
        logger.warning(
            "SVA splitter: assertion buffer not empty at end of input — "
            "possible truncation. Emitting partial assertion."
        )
        results.append({
            "comment": "\n".join(comment_buf).strip(),
            "assertion": "\n".join(assertion_buf).strip(),
        })

    return results


def _update_paren_state(
    line: str, paren_depth: int, in_string: bool
) -> Tuple[int, bool]:
    """
    Walk characters in `line` to update parenthesis depth and string state.

    Tracks double-quoted string literals so parentheses and semicolons inside
    strings are not counted.  Handles escaped quotes (\\") within strings.

    Returns
    -------
    (paren_depth, in_string)
    """
    i = 0
    while i < len(line):
        ch = line[i]
        if in_string:
            if ch == "\\" and i + 1 < len(line):
                i += 2  # skip escaped character
                continue
            if ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)
        i += 1
    return paren_depth, in_string


def _assertion_complete(
    line: str, paren_depth: int, in_string: bool
) -> bool:
    """
    Return True if the current line ends a complete assertion statement.

    An assertion is complete when:
      - The line ends with ';'  (after stripping whitespace)
      - All parentheses are balanced (depth == 0)
      - We are not inside a string literal
    """
    stripped = line.rstrip()
    return stripped.endswith(";") and paren_depth == 0 and not in_string


# ---------------------------------------------------------------------------
# Per-assertion linting
# ---------------------------------------------------------------------------

def lint_single_assertion(
    assertion: str,
    reject_assert_property: bool = True,
) -> Dict[str, str]:
    """
    Lint one assertion through slang_lint().

    Parameters
    ----------
    assertion : str
        A single assert statement (without module wrapper).
    reject_assert_property : bool
        When True, reject concurrent assertions (for combinational designs).

    Returns
    -------
    dict with keys "status" ("PASS" or "FAIL") and "error" (str).
    """
    result = slang_lint(
        assertion,
        reject_assert_property=reject_assert_property,
    )
    if result.startswith("PASS"):
        return {"status": "PASS", "error": ""}
    else:
        return {"status": "FAIL", "error": result}


def lint_all_assertions(
    assertions: List[Dict[str, str]],
    reject_assert_property: bool = True,
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """
    Lint every assertion individually through pyslang.

    Parameters
    ----------
    assertions : list of dict
        Each dict has "comment" and "assertion" keys (from split_assertions).
    reject_assert_property : bool
        When True, reject concurrent assertions (for combinational designs).

    Returns
    -------
    passed : list of dict
        Assertions that passed linting (same format as input).
    failures : list of dict
        Assertions that failed, enriched with "error" and "index" keys.
    """
    passed: List[Dict[str, str]] = []
    failures: List[Dict[str, Any]] = []

    for i, entry in enumerate(assertions):
        lint_result = lint_single_assertion(
            entry["assertion"],
            reject_assert_property=reject_assert_property,
        )

        if lint_result["status"] == "PASS":
            passed.append(entry)
            logger.info(
                "  Assertion %d/%d: PASS", i + 1, len(assertions)
            )
        else:
            failures.append({
                "index": i,
                "comment": entry["comment"],
                "assertion": entry["assertion"],
                "error": lint_result["error"],
            })
            logger.warning(
                "  Assertion %d/%d: FAIL — %s",
                i + 1, len(assertions), lint_result["error"][:100],
            )

    logger.info(
        "Lint summary: %d passed, %d failed (out of %d)",
        len(passed), len(failures), len(assertions),
    )
    return passed, failures


# ---------------------------------------------------------------------------
# JSON report persistence
# ---------------------------------------------------------------------------

def save_lint_report(report: Dict[str, Any], path: str) -> None:
    """
    Write the lint failure report to disk as pretty-printed JSON.

    The report is overwritten on each call so it always reflects the latest
    state.  Individual iteration history is preserved inside the report's
    "iterations" list.

    Parameters
    ----------
    report : dict
        The full lint report (see run_lint_loop for schema).
    path : str
        Output file path (e.g. "./lint_failures.json").
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    logger.info("Lint report written to %s", path)


# ---------------------------------------------------------------------------
# Fix bare property fragments (missing assert wrapper)
# ---------------------------------------------------------------------------

def fix_bare_property_fragments(
    sva_code: str,
    clock_signal: Optional[str] = None,
    reset_signal: Optional[str] = None,
) -> str:
    """
    Wrap bare property expressions that are missing the assert wrapper.

    The LLM sometimes emits property expressions without the
    ``assert property (...)`` wrapper, e.g.::

        (cond) |-> (result) else $error("...");

    This is syntactically invalid SV. If a clock/reset is available, wrap
    in ``assert property (@(posedge clk) disable iff (!rst) ...)``.
    For bare expressions without implication operators, wrap in
    ``assert (...)``.
    """
    # Work on the full text to handle multi-line bare fragments.
    # Pattern: a comment line, then one or more non-assert lines containing
    # a property expression, ending with `) else $error("...");`
    #
    # We use the assertion splitter's approach: walk lines, accumulate
    # non-assert/non-comment lines until we see `else $error`.

    lines = sva_code.splitlines()
    result_lines: List[str] = []
    fixed = 0
    i = 0

    # Format detector: a line is part of a fully-formed concurrent
    # assertion (NOT a bare fragment) if it contains any of the keywords
    # `property NAME;` (property declaration), `endproperty`, or the
    # `assert property (...)` invocation. Such lines must pass through
    # unchanged — the bare-fragment heuristic was designed for naked
    # `(cond) |-> (result) else $error(...)` blobs that are missing the
    # outer wrapper, NOT for already-wrapped concurrent assertions.
    _CONCURRENT_FORM_RE = re.compile(
        r"\bproperty\s+\w+\s*;|\bendproperty\b|\bassert\s+property\b",
        re.IGNORECASE,
    )

    while i < len(lines):
        stripped = lines[i].strip()

        # Pass through blanks, comments, assert lines, and any line that
        # already participates in a concurrent property block.
        if (not stripped
                or stripped.startswith("//")
                or stripped.lower().startswith("assert")
                or _CONCURRENT_FORM_RE.search(stripped)):
            result_lines.append(lines[i])
            i += 1
            continue

        # Potential bare fragment — accumulate lines until we find
        # `else $error(...)` or hit a comment/assert/blank/concurrent line.
        frag_lines = [stripped]
        j = i + 1
        while j < len(lines):
            nxt = lines[j].strip()
            if (not nxt
                    or nxt.startswith("//")
                    or nxt.lower().startswith("assert")
                    or _CONCURRENT_FORM_RE.search(nxt)):
                break
            frag_lines.append(nxt)
            if re.search(r'else\s+\$error\s*\(', nxt):
                j += 1
                break
            j += 1

        fragment = " ".join(frag_lines)

        # Check if this is a bare property/expression.
        has_implication = bool(re.search(r'\|[-=]>', fragment))
        has_else_error = bool(re.search(r'\)\s*else\s+\$error\s*\(', fragment))

        if not has_else_error:
            # Not a recognizable fragment — keep original lines.
            result_lines.append(lines[i])
            i += 1
            continue

        # Extract body and else clause from the joined fragment.
        else_match = re.search(
            r'\)\s*(else\s+\$error\s*\(".*?"\)\s*;?)\s*$', fragment
        )
        if not else_match:
            result_lines.append(lines[i])
            i += 1
            continue

        else_clause = else_match.group(1).rstrip(';') + ";"
        body = fragment[:else_match.start()].strip()

        # Remove wrapping parens if balanced.
        while body.startswith("(") and body.endswith(")"):
            inner = body[1:-1]
            depth = 0
            balanced = True
            for ch in inner:
                if ch == "(": depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth < 0:
                        balanced = False
                        break
            if balanced and depth == 0:
                body = inner.strip()
            else:
                break

        if has_implication and clock_signal:
            reset_clause = f" disable iff (!{reset_signal})" if reset_signal else ""
            # Single line so deduplicate_assertions can match it.
            new_line = (
                f"assert property (@(posedge {clock_signal}){reset_clause} "
                f"{body}) {else_clause}"
            )
        elif has_implication and not clock_signal:
            logger.debug("Removed bare fragment (no clock): %s", fragment[:80])
            if result_lines and result_lines[-1].strip().startswith("//"):
                result_lines.pop()
            fixed += 1
            i = j
            continue
        else:
            new_line = f"assert ({body}) {else_clause}"

        result_lines.append(new_line)
        fixed += 1
        logger.debug("Wrapped bare fragment: %s", fragment[:80])
        i = j  # skip all consumed lines

    if fixed:
        logger.info("Wrapped %d bare property fragment(s).", fixed)
    return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# Fix immediate AND-form assertions
# ---------------------------------------------------------------------------

def _split_top_level_and(expr: str) -> List[str]:
    """
    Split an expression on top-level ``&&`` operators, respecting
    parenthesis depth so that ``(a && b) && c`` splits into
    ``["(a && b)", "c"]``, not ``["(a", "b)", "c"]``.
    """
    clauses: List[str] = []
    depth = 0
    current: List[str] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '&' and i + 1 < len(expr) and expr[i + 1] == '&' and depth == 0:
            clauses.append("".join(current).strip())
            current = []
            i += 2  # skip both &
            continue
        current.append(ch)
        i += 1
    if current:
        clauses.append("".join(current).strip())
    return [c for c in clauses if c]


def _extract_checked_signal(error_msg: str) -> Optional[str]:
    """
    Extract the signal name being checked from an assertion error message.

    Matches patterns like:
    - "out_data mismatch"
    - "X incorrect"
    - "X output incorrect"
    - "expected X"
    - "X must be Y"
    - "X should be Y"
    """
    patterns = [
        # "out_data mismatch", "out_inv incorrect"
        r'(\w+)\s+(?:incorrect|mismatch|wrong|error|failure)',
        # "output incorrect" → look for signal before "output"
        r'([a-zA-Z_]\w*)\s+output\s+(?:incorrect|mismatch|wrong)',
        # "expected out_data", "check out_data"
        r'(?:expected|check(?:ing)?|verify(?:ing)?)\s+(\w+)',
        # "out_data must be", "out_data should be"
        r'(\w+)\s+(?:should|must)\s+be',
        # "X pattern" (as in "zero pattern incorrect")
        r'(\w+)\s+pattern\s+(?:incorrect|mismatch|wrong)',
    ]
    noise_words = {
        'the', 'a', 'an', 'output', 'input', 'signal', 'value',
        'result', 'data', 'when', 'if', 'not', 'is', 'are',
        'code', 'mode', 'pattern', 'format', 'sign', 'flag',
        'zero', 'one', 'bit', 'byte', 'word', 'width', 'size',
    }
    for pattern in patterns:
        m = re.search(pattern, error_msg, re.IGNORECASE)
        if m:
            sig = next((g for g in m.groups() if g), None)
            if sig and sig.lower() not in noise_words:
                return sig
    return None


def fix_immediate_and_form(sva_code: str) -> str:
    """
    Fix immediate assertions that use AND instead of implication.

    Detects::

        assert (cond1 && cond2 && result_check) else $error("result mismatch");

    This asserts ALL conditions true simultaneously — always fails when
    cond1 is false. Rewrites to implication form::

        assert (!(cond1 && cond2) || (result_check)) else $error("...");

    Uses the error message to identify which clause(s) are the
    "consequence" (the thing being checked) vs the "condition" (the
    input state). Design-agnostic — works by parsing the error message,
    not by knowing signal names.
    """
    lines = sva_code.splitlines()
    fixed_lines: List[str] = []
    fixed = 0

    for line in lines:
        stripped = line.strip()

        # Only target immediate assertions: assert (...) else $error("...");
        match = re.match(
            r'assert\s*\((.+)\)\s*else\s*\$error\s*\("(.+?)"\)\s*;\s*$',
            stripped,
        )
        if not match:
            fixed_lines.append(line)
            continue

        body = match.group(1).strip()
        error_msg = match.group(2)

        # Skip if already has implication form.
        if '||' in body or '|->' in body or '|=>' in body:
            fixed_lines.append(line)
            continue

        # Skip assert property (concurrent).
        if 'property' in stripped.lower():
            fixed_lines.append(line)
            continue

        # Split on top-level &&.
        clauses = _split_top_level_and(body)
        if len(clauses) < 3:
            fixed_lines.append(line)
            continue

        # Extract checked signal from error message.
        checked = _extract_checked_signal(error_msg)
        if not checked:
            fixed_lines.append(line)
            continue

        # Split into conditions vs consequences (case-insensitive match).
        conditions = []
        consequences = []
        for clause in clauses:
            if re.search(r'\b' + re.escape(checked) + r'\b', clause, re.IGNORECASE):
                consequences.append(clause)
            else:
                conditions.append(clause)

        if not conditions or not consequences:
            fixed_lines.append(line)
            continue

        cond_str = " && ".join(conditions)
        conseq_str = " && ".join(consequences)
        new_line = (
            f'assert (!({cond_str}) || ({conseq_str})) '
            f'else $error("{error_msg}");'
        )
        fixed_lines.append(new_line)
        fixed += 1
        logger.debug("Fixed AND-form: %s", stripped[:80])

    if fixed:
        logger.info("Fixed %d immediate AND-form assertion(s).", fixed)
    return "\n".join(fixed_lines)


# ---------------------------------------------------------------------------
# Post-processing: fix common LLM assertion mistakes
# ---------------------------------------------------------------------------

def fix_immediate_implication(sva_code: str) -> str:
    """
    Fix immediate assertions that incorrectly use implication operators.

    Both ``|->`` and ``->`` are only valid inside ``assert property``.
    In immediate ``assert (...)``, the equivalent is ``!(cond) || (consequence)``.

    Rewrites:
      assert (COND |-> CONSEQUENCE) else $error("...");
      assert (COND -> CONSEQUENCE) else $error("...");
    to:
      assert (!(COND) || (CONSEQUENCE)) else $error("...");
    """
    import re

    lines = sva_code.splitlines()
    fixed_lines = []

    for line in lines:
        stripped = line.strip()

        # Only fix immediate assertions (not assert property).
        # Check for both |-> and -> (but not <= or >= or !=).
        if (stripped.startswith("assert")
                and "assert property" not in stripped
                and ("->" in stripped)):

            match = re.match(
                r'assert\s*\((.+?)\)\s*else\s*(.*)',
                stripped,
                re.DOTALL,
            )
            if match:
                body = match.group(1).strip()
                else_part = match.group(2).strip()

                # Try |-> first (more specific), then -> .
                # Use regex to split on |-> or standalone ->
                # but NOT on >= or <=  or !=.
                impl_match = re.search(r'\|?->', body)
                if impl_match:
                    operator = impl_match.group(0)  # "|->", "->", or "|->"
                    parts = body.split(operator, 1)
                    cond = parts[0].strip()
                    conseq = parts[1].strip()

                    # Remove outer parens from condition if present.
                    if cond.startswith("(") and cond.endswith(")"):
                        cond_inner = cond[1:-1].strip()
                    else:
                        cond_inner = cond

                    new_body = f"!({cond_inner}) || ({conseq})"
                    new_line = f"assert ({new_body}) else {else_part}"
                    fixed_lines.append(new_line)
                    logger.debug(
                        "Fixed '%s' in immediate assert: %s",
                        operator, stripped[:60],
                    )
                    continue

        fixed_lines.append(line)

    return "\n".join(fixed_lines)




def fix_double_negation(sva_code: str) -> str:
    """
    Simplify double-negated assertion patterns.

    The LLM sometimes wraps an AST skeleton's ``!(cond) || (result)``
    in another negation, producing::

        assert (!( !(cond) || (result) ) || (result))

    This is logically equivalent to::

        assert (!(cond) || (result))

    *only* when the inner and outer ``result`` clauses are the same
    expression (the De Morgan / absorption identity).  If they differ,
    the rewrite is unsound — the boolean simplification becomes::

        assert ((cond && !inner_result) || outer_result)

    which is NOT what the original assertion meant.  This function
    therefore verifies (after whitespace/paren normalisation) that
    ``inner_result`` and ``outer_result`` are structurally equal
    before applying the simplification.

    The body extraction is paren-balanced (it does not rely on lazy
    regex captures), so it tolerates nested parentheses and arbitrary
    whitespace.  The detector also rejects bodies whose inner clause
    isn't of the form ``!(X) || Y`` (i.e., the LLM error pattern
    being targeted).
    """
    lines = sva_code.splitlines()
    fixed_lines: List[str] = []
    fixed = 0

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("assert") or "property" in stripped.split("(", 1)[0]:
            fixed_lines.append(line)
            continue

        # Locate `else` clause (paren-balanced, since `else $error("...")`
        # may appear inside an inner string we don't care about).  Use the
        # first occurrence of `) else` at depth 1 — that is the close of
        # the outer assert paren immediately followed by the else clause.
        depth = 0
        else_pos = -1
        i = 0
        while i < len(stripped) - 5:
            ch = stripped[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and stripped[i+1:i+6].lstrip().startswith("else"):
                    else_pos = i
                    break
            i += 1
        if else_pos < 0:
            fixed_lines.append(line)
            continue

        prefix = stripped[:else_pos+1]   # `assert (...)` inclusive
        suffix = stripped[else_pos+1:].strip()  # `else $error(...);`
        # Strip `assert` + opening paren and the matching close.
        m = re.match(r"assert\s*\(", prefix)
        if not m:
            fixed_lines.append(line)
            continue
        body = prefix[m.end():-1].strip()
        body = _strip_outer_parens(body)

        # Body must be `!(<inner>) || <outer_result>` at top level.
        if not (body.startswith("!") or body.startswith("~")):
            fixed_lines.append(line)
            continue
        # Find the close paren of the outer negation argument.
        if not body[1:].lstrip().startswith("("):
            fixed_lines.append(line)
            continue
        # Locate the parenthesised argument of the leading `!`.
        ko = body.index("(", 1)
        depth = 0
        end_neg = -1
        for j in range(ko, len(body)):
            ch = body[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end_neg = j
                    break
        if end_neg < 0:
            fixed_lines.append(line)
            continue
        inner_body = body[ko+1:end_neg].strip()
        rest = body[end_neg+1:].strip()
        if not rest.startswith("||"):
            fixed_lines.append(line)
            continue
        outer_result = rest[2:].strip()
        if not outer_result:
            fixed_lines.append(line)
            continue

        # Inner must itself be an implication shape: `!(<cond>) || <inner_result>`.
        if not (inner_body.startswith("!") or inner_body.startswith("~")):
            fixed_lines.append(line)
            continue
        if not inner_body[1:].lstrip().startswith("("):
            fixed_lines.append(line)
            continue
        ki = inner_body.index("(", 1)
        depth = 0
        end_inner_neg = -1
        for j in range(ki, len(inner_body)):
            ch = inner_body[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end_inner_neg = j
                    break
        if end_inner_neg < 0:
            fixed_lines.append(line)
            continue
        inner_rest = inner_body[end_inner_neg+1:].strip()
        if not inner_rest.startswith("||"):
            fixed_lines.append(line)
            continue
        inner_result = inner_rest[2:].strip()

        # Soundness guard: only apply the De Morgan / absorption rewrite
        # when the two `result` clauses are structurally equal.
        if _normalize_expr(inner_result) != _normalize_expr(outer_result):
            fixed_lines.append(line)
            continue

        new_line = f"assert ({inner_body}) {suffix}"
        fixed_lines.append(new_line)
        fixed += 1
        logger.debug("Fixed double negation: %s", stripped[:80])

    if fixed:
        logger.info("Fixed %d double-negation assertion(s).", fixed)
    return "\n".join(fixed_lines)


def deduplicate_assertions(sva_code: str) -> str:
    """
    Remove duplicate assertions that check the same signal and condition.

    The LLM often regenerates assertions that the AST already produced.
    This keeps the first occurrence (which is the AST version, since AST
    assertions come before LLM assertions in the output).
    """
    import re

    lines = sva_code.splitlines()
    seen_signatures: set = set()
    kept_lines: List[str] = []
    removed = 0

    for line in lines:
        stripped = line.strip()

        # Extract assertion signature: the core condition being checked.
        if stripped.startswith("assert"):
            # Normalise whitespace for comparison.
            normalised = re.sub(r'\s+', ' ', stripped)
            # Remove error message (varies between AST and LLM versions).
            sig = re.sub(r'else\s+\$error\(.*$', '', normalised).strip()

            if sig in seen_signatures:
                removed += 1
                logger.debug("Deduplicated: %s", stripped[:60])
                continue
            seen_signatures.add(sig)

        kept_lines.append(line)

    if removed:
        logger.info("Deduplicated %d assertion(s).", removed)
    return "\n".join(kept_lines)


def remove_wrong_style_assertions(sva_code: str, config: Any) -> str:
    """
    Remove assertions that use the wrong style for the design.

    - On combinational designs (reject_assert_property=True):
      remove any ``assert property`` that crept in from the LLM.
    - On clocked designs: keep everything.

    Also removes unconditional output value assertions that aren't
    guarded by a case/condition — these are a common LLM mistake
    where it over-generalises case-specific checks.
    """
    import re

    reject = getattr(config, 'reject_assert_property', None)
    lines = sva_code.splitlines()
    kept_lines: List[str] = []
    removed = 0

    for line in lines:
        stripped = line.strip()

        # Remove concurrent assertions on combinational designs.
        if reject and "assert property" in stripped:
            removed += 1
            logger.debug("Removed concurrent assertion on combinational design: %s", stripped[:60])
            # Also skip the comment line above if it's the previous kept line.
            if kept_lines and kept_lines[-1].strip().startswith("//"):
                kept_lines.pop()
            continue

        kept_lines.append(line)

    if removed:
        logger.info("Removed %d wrong-style assertion(s).", removed)
    return "\n".join(kept_lines)


def fix_condition_only_assertions(
    sva_code: str,
    facts: Optional[RTLFacts] = None,
) -> str:
    """
    Repair (or, if repair is unsound, drop) condition-only assertions
    whose error message implies a missing consequent.

    Pattern (the LLM's mistake):
        assert (COND) else $error("OUTPUT must be VALUE");

    Here ``COND`` is a complete predicate and the *error message*
    encodes the LLM's stated intent — the assertion was meant to be::

        assert (!(COND) || (OUTPUT == VALUE)) else $error("...");

    We do not blindly perform that rewrite (the message could be
    creative).  Instead, the path is **try-repair → validate →
    keep-or-drop**:

      1. Detect the shape (condition-only body + expectation phrasing
         like ``must be``, ``should be``, ``must equal``).
      2. Extract candidate ``OUTPUT == VALUE`` pairs from the message.
      3. Validate each candidate against design facts:
           • ``OUTPUT`` must exist in ``facts.all_signals`` (i.e., be a
             real port or internal signal in the RTL).
           • ``VALUE`` must be a Verilog literal whose declared width
             does not exceed the signal's width (when both are known).
           • ``OUTPUT`` must not already appear in ``COND`` (otherwise
             the repair would be redundant or self-referential).
      4. If at least one candidate passes validation, emit the rewrite
         using all validated candidates ANDed together.
      5. If no candidate validates, drop the assertion AND leave a
         tagged comment ``// REPAIR_FAILED: ...`` so that downstream
         lint-feedback iterations can show the LLM what was missing.

    The ``facts`` argument is optional.  Without facts the validator
    cannot verify signal existence, so we conservatively skip the
    rewrite (we neither repair nor drop) — the assertion passes
    through unchanged.  This preserves backward compatibility for any
    caller that still invokes this function without facts.
    """
    expectation_re = re.compile(
        r"\b(?:must\s+be|should\s+be|must\s+equal|must\s+produce|"
        r"shall\s+be|expected\s+to\s+be)\b",
        re.IGNORECASE,
    )
    # Loose extraction of `IDENT = VALUE`, `IDENT == VALUE`,
    # `IDENT must be VALUE`, etc., from the message.  We extract
    # liberally and let the validator decide what is real.  The value
    # pattern accepts:
    #   • Sized literals:   8'h00, 32'b101, 16'd255
    #   • Plain integers:   0, 1, 42, 0xff
    #   • Brace replication / concatenation:   {N{1'b0}}
    candidate_re = re.compile(
        r"([A-Za-z_]\w*(?:\.\w+)*)\s*"
        r"(?:=|==|must\s+be|should\s+be|must\s+equal|shall\s+be)\s+"
        r"("
        r"\d+'[bdhoBDHO][0-9a-fA-FxXzZ_]+"   # sized literal: 8'h00
        r"|0x[0-9a-fA-F]+"                    # 0xff style
        r"|\d+"                                # bare integer: 0, 1, 42
        r"|\{[^}]+\}"                         # brace expression
        r")\b",
        re.IGNORECASE,
    )

    all_signals = getattr(facts, "all_signals", None) if facts else None
    signal_widths = getattr(facts, "signal_widths", {}) if facts else {}
    if all_signals is None and not signal_widths:
        # Without facts we cannot validate — pass through unchanged.
        return sva_code

    lines = sva_code.splitlines()
    out_lines: List[str] = []
    repaired = 0
    dropped  = 0

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("assert") \
                or "property" in stripped.split("(", 1)[0]:
            out_lines.append(line); continue

        match = re.match(
            r'assert\s*\((.+)\)\s*else\s*\$error\s*\("(.+?)"\)\s*;\s*$',
            stripped,
        )
        if not match:
            out_lines.append(line); continue
        condition = match.group(1).strip()
        error_msg = match.group(2).strip()

        # Genuine condition-only: no implication / disjunction / temporal.
        if ("||" in condition or "|->" in condition or "|=>" in condition
                or "##" in condition):
            out_lines.append(line); continue

        # Message must indicate a missing expected output.
        if not expectation_re.search(error_msg):
            out_lines.append(line); continue

        # Try to extract candidate consequents from the message.
        cands = candidate_re.findall(error_msg)
        validated: List[Tuple[str, str]] = []
        for sig, val in cands:
            sig = sig.strip()
            val = val.strip().rstrip(".")
            # Skip if already in the condition (redundant repair).
            if re.search(r"\b" + re.escape(sig) + r"\b", condition):
                continue
            # Validator 1: signal must exist in the design.
            if all_signals is not None:
                # accept hierarchical leaf as well as full name
                leaf = sig.rsplit(".", 1)[-1]
                if sig not in all_signals and leaf not in all_signals:
                    continue
            # Validator 2: literal width must not exceed signal width
            # when both are known.
            sig_width = signal_widths.get(sig) \
                or signal_widths.get(sig.rsplit(".", 1)[-1])
            lit_width = _literal_width(val) if "'" in val else None
            if sig_width is not None and lit_width is not None \
                    and lit_width > sig_width:
                continue
            validated.append((sig, val))

        if validated:
            consequence = " && ".join(f"{s} == {v}" for s, v in validated)
            new_line = (
                f'assert (!({condition}) || ({consequence})) '
                f'else $error("{error_msg}");'
            )
            out_lines.append(new_line)
            repaired += 1
            logger.info("Repaired condition-only assertion: added "
                        "consequent (%s): %s", consequence, stripped[:80])
            continue

        # Repair failed — drop the assertion but leave a breadcrumb so
        # later lint-feedback iterations can prompt the LLM with the
        # specific shortfall.  The comment is harmless to compilers and
        # informative to the agent on the next pass.
        dropped += 1
        out_lines.append(
            f"// REPAIR_FAILED: condition-only assertion (message says "
            f"'{error_msg[:80]}'); LLM omitted the consequent and the "
            f"message did not yield a validated repair candidate."
        )
        if out_lines and len(out_lines) >= 2 \
                and out_lines[-2].strip().startswith("//") \
                and not out_lines[-2].strip().startswith("// REPAIR_FAILED"):
            # If the pre-existing comment was the LLM's own description
            # of the assertion, leave it — context for the next pass.
            pass
        logger.info("Dropped condition-only assertion (no validated "
                    "repair candidate): %s", stripped[:90])

    if repaired:
        logger.info("Repaired %d condition-only assertion(s).", repaired)
    if dropped:
        logger.info("Dropped %d unrepairable condition-only assertion(s) "
                    "(left REPAIR_FAILED breadcrumbs).", dropped)
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Fix |=> on combinational logic → |->
# ---------------------------------------------------------------------------

def fix_next_cycle_on_combinational(
    sva_code: str,
    facts: RTLFacts,
) -> str:
    """
    Convert |=> (next-cycle) to |-> (same-cycle) in assertions that
    reference purely combinational signals.

    The LLM often wraps combinational checks in concurrent assertions with
    |=> (next-cycle implication), but combinational signals change in the
    same cycle. Using |=> checks the NEXT cycle's value, which is incorrect.

    Detection: uses ``facts.combinational_signals`` (extracted via pyslang
    from continuous assigns and combinational always blocks). Any assertion
    using |=> whose consequence references one of those signals is
    converted to |-> (same-cycle).

    Parameters
    ----------
    sva_code : str
        Raw SVA code string.
    facts : RTLFacts
        Pre-extracted RTL facts (must include ``combinational_signals``).

    Returns
    -------
    str
        SVA code with |=> fixed to |-> for combinational signals.
    """
    comb_signals = facts.combinational_signals
    if not comb_signals:
        return sva_code

    logger.info(
        "Combinational signal detection: %d signals from RTL.",
        len(comb_signals),
    )

    lines = sva_code.splitlines()
    fixed_lines = []
    fixed = 0

    for line in lines:
        stripped = line.strip()
        if "|=>" in stripped:
            # Find the FIRST top-level |=> (paren-aware) — that is the
            # property's primary implication.  Nested |=> inside
            # sub-properties stay untouched.
            idx = _find_top_level(stripped, "|=>")
            if idx < 0:
                # Top-level operator not found (only nested |=>) → skip.
                fixed_lines.append(line)
                continue
            consequence = stripped[idx + 3:]
            consequence_sigs = _extract_signal_names(consequence)
            # If ANY signal in the consequence is combinational, fix it
            # — but rewrite only this one operator, preserving any nested
            # |=> in subexpressions.
            if consequence_sigs & comb_signals:
                # Replace the first |=> in the original `line` (not just
                # in `stripped`, so leading whitespace is preserved).
                rel_idx = line.find("|=>")
                if rel_idx >= 0:
                    new_line = line[:rel_idx] + "|->" + line[rel_idx + 3:]
                    fixed_lines.append(new_line)
                    fixed += 1
                    logger.debug("Fixed |=> to |-> (combinational): %s", stripped[:80])
                    continue
        fixed_lines.append(line)

    if fixed:
        logger.info("Fixed %d assertion(s): |=> to |-> on combinational signals.", fixed)
    return "\n".join(fixed_lines)


def fix_same_cycle_past_on_sequential(
    sva_code: str,
    facts: RTLFacts,
) -> str:
    """
    Convert |-> to |=> when the consequence uses ``$past(...)`` and the
    LHS of the equality is a sequential (flop-driven) signal.

    For a flop ``always @(posedge clk) if (en) q <= d;``, the correct
    temporal relation is ``(en) |=> q == $past(d)`` (next-cycle q equals
    d sampled when en was high). The form ``(en) |-> q == $past(d)`` is
    an off-by-one: it asserts q *now* equals d from the *previous* cycle,
    which is only true if d was stable across the enable window.

    Detection: any assertion containing both ``|->`` and ``$past(`` whose
    LHS (signal immediately after ``|->`` and before the first ``==``) is
    marked ``"seq"`` in ``facts.signal_drive_kind``.
    """
    drive = facts.signal_drive_kind
    if not drive:
        return sva_code

    lines = sva_code.splitlines()
    fixed_lines = []
    fixed = 0

    # Capture the LHS of the equality immediately following `|->`.
    # Allows hierarchical references (`top.sub.q`) and bit/part-selects.
    # The drive-kind lookup uses the leaf identifier (last segment) since
    # `signal_drive_kind` is keyed by simple signal names.
    lhs_re = re.compile(
        r"\|->\s*\(?\s*"
        r"([A-Za-z_]\w*(?:\.\w+)*)"     # hierarchical name
        r"(?:\s*\[[^\]]+\])?"            # optional bit/part-select
        r"\s*=="
    )

    for line in lines:
        stripped = line.strip()
        if "|->" in stripped and "$past(" in stripped:
            m = lhs_re.search(stripped)
            if m:
                full_lhs = m.group(1)
                leaf = full_lhs.rsplit(".", 1)[-1]  # use leaf for facts lookup
                if drive.get(leaf) == "seq" or drive.get(full_lhs) == "seq":
                    # Replace only the first |-> at top level (the
                    # property's primary implication), leaving any nested
                    # |-> in sub-properties untouched.
                    rel_idx = _find_top_level(line, "|->")
                    if rel_idx < 0:
                        fixed_lines.append(line); continue
                    new_line = line[:rel_idx] + "|=>" + line[rel_idx + 3:]
                    fixed_lines.append(new_line)
                    fixed += 1
                    logger.debug(
                        "Fixed |-> to |=> ($past on sequential LHS %s): %s",
                        full_lhs, stripped[:80],
                    )
                    continue
        fixed_lines.append(line)

    if fixed:
        logger.info(
            "Fixed %d assertion(s): |-> to |=> for $past-on-sequential.",
            fixed,
        )
    return "\n".join(fixed_lines)


# ---------------------------------------------------------------------------
# Remove semantically vacuous assertions (4 patterns from manual review of
# SpecGuard outputs on cmacfull / rubik / sdp — each pattern parses cleanly
# and uses real signals, so other validators don't catch them).
# ---------------------------------------------------------------------------

# Disable-iff prefix capture — used by the dead-code-under-disable
# detector. We extract the disable expression with a paren-balanced
# scan rather than a greedy regex so nested parens (e.g.
# `disable iff (!rst && (mode == X))`) parse cleanly.
_DISABLE_IFF_PREFIX_RE = re.compile(r"disable\s+iff\s*\(", re.IGNORECASE)


def _extract_disable_iff_expr(s: str) -> Optional[Tuple[str, int]]:
    """Return (disable_expression, end_index) for the first
    `disable iff (...)` clause in `s`, with the inside of the parens
    extracted via a balanced scan.  Returns None if not present."""
    m = _DISABLE_IFF_PREFIX_RE.search(s)
    if not m:
        return None
    start = m.end()
    depth = 1
    i = start
    while i < len(s):
        ch = s[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return s[start:i].strip(), i + 1
        i += 1
    return None


def _analyse_dead_under_disable(s: str) -> Optional[Dict[str, Any]]:
    """If `s` is an `assert` whose antecedent contains a top-level
    conjunct equal to its `disable iff` expression, return a dict
    describing the situation:

        {
          "disable_expr":  raw disable expression text,
          "match_idx":     index into top-level conjunct list,
          "antecedent":    antecedent text (parens stripped),
          "ante_start":    char offset of the antecedent within `s`,
          "ante_end":      char offset (exclusive) of the antecedent,
          "impl_op":       "|->" or "|=>",
        }

    Return None when the assertion is not the dead-code shape.

    Generalised across:
      • Any depth of paren nesting in the disable expression.
      • Hierarchical signal references (`top.sub.sig`).
      • Bit-selects on the disable signal.
      • Word-boundary-correct identifier matching (so `rst` does
        not match the suffix of `sig_rst`)."""
    res = _extract_disable_iff_expr(s)
    if res is None:
        return None
    disable_expr, end_idx = res
    rest = s[end_idx:].strip()
    rest_offset = len(s) - len(s[end_idx:].lstrip())
    # First top-level implication.
    impl_idx = -1
    op_used = ""
    for op in ("|->", "|=>"):
        idx = _find_top_level(rest, op)
        if idx >= 0 and (impl_idx < 0 or idx < impl_idx):
            impl_idx = idx
            op_used = op
    if impl_idx < 0:
        return None
    antecedent_raw = rest[:impl_idx]
    antecedent_str = _strip_outer_parens(antecedent_raw.strip())
    if not antecedent_str:
        return None
    disable_norm = _normalize_expr(disable_expr)
    if not disable_norm:
        return None
    conjuncts = _split_top_level(antecedent_str, "&&")
    match_idx = -1
    for i, c in enumerate(conjuncts):
        if _normalize_expr(c) == disable_norm:
            match_idx = i; break
    # Also single-conjunct case (no `&&` in antecedent).
    if match_idx < 0 and _normalize_expr(antecedent_str) == disable_norm:
        match_idx = 0  # whole antecedent IS the dead conjunct
        conjuncts = [antecedent_str]
    if match_idx < 0:
        return None
    return {
        "disable_expr": disable_expr,
        "match_idx":    match_idx,
        "antecedent":   antecedent_str,
        "conjuncts":    conjuncts,
        "ante_start":   rest_offset,
        "ante_end":     rest_offset + impl_idx,
        "impl_op":      op_used,
    }


def remove_dead_code_under_disable(sva_code: str) -> str:
    """**Repair** assertions where `disable iff (D)` and a top-level
    conjunct of the antecedent equals `D`; only **drop** when repair
    is impossible.

    Two repair paths:

      1. **Strip-conjunct**: if the antecedent is `D && X1 && X2 ...`
         (multi-conjunct, with the dead one alongside other meaningful
         conjuncts), remove only the dead conjunct and keep the rest.
         The repaired assertion checks the SAME relationship the LLM
         intended, just without the redundant gate that made it
         vacuous.

      2. **Drop with breadcrumb**: if the antecedent is *only* `D`
         (single-conjunct, nothing left after stripping), drop the
         assertion and emit a `// REPAIR_FAILED` comment so that
         downstream lint-feedback iterations can show the LLM the
         shortfall.  Without other conjuncts there is no
         consequence-bearing structure to preserve.

    Observed in SpecGuard cmacfull/rubik outputs as the pattern
    `disable iff (!rstn) (... && !rstn && ...) |-> ...` where the LLM
    accidentally repeated the reset signal in the antecedent — a
    typical case where the rest of the antecedent is meaningful and
    the strip-conjunct repair preserves real verification value.
    """
    lines = sva_code.splitlines()
    out: List[str] = []
    repaired = 0
    dropped  = 0
    for ln in lines:
        s = ln.strip()
        if not s.startswith("assert"):
            out.append(ln); continue
        info = _analyse_dead_under_disable(s)
        if info is None:
            out.append(ln); continue
        # Determine whether repair is possible.
        survivors = [c for i, c in enumerate(info["conjuncts"])
                     if i != info["match_idx"]]
        if survivors:
            # Strip-conjunct repair: rebuild the antecedent without
            # the dead conjunct, preserving everything else verbatim.
            new_antecedent = " && ".join(c.strip() for c in survivors)
            # Splice into the original line at the antecedent's slot.
            # We compute the substring positions once via the analysis
            # offsets relative to `s`, then map back into `ln` (which
            # may have leading whitespace).
            indent = ln[:len(ln) - len(ln.lstrip())]
            head   = s[:info["ante_start"]]
            tail   = s[info["ante_end"]:]
            # The antecedent slot is where we splice the new string;
            # we wrap it in parens for safety regardless of original.
            new_line = f"{indent}{head}({new_antecedent}){tail}"
            out.append(new_line)
            repaired += 1
            logger.info("Repaired dead-code-under-disable (stripped "
                        "conjunct '%s'): %s",
                        info["conjuncts"][info["match_idx"]].strip(),
                        s[:90])
            continue
        # No surviving conjuncts → drop with breadcrumb.
        dropped += 1
        out.append(
            f"// REPAIR_FAILED: dead-code-under-disable — antecedent "
            f"matched the disable clause '{info['disable_expr'][:60]}' "
            f"with no other conjuncts to preserve."
        )
        if out and len(out) >= 2 \
                and out[-2].strip().startswith("//") \
                and not out[-2].strip().startswith("// REPAIR_FAILED"):
            pass  # keep the LLM's own header for context on next pass
        logger.info("Dropped dead-code-under-disable (no repair "
                    "possible): %s", s[:90])
    if repaired:
        logger.info("Repaired %d dead-code-under-disable assertion(s).",
                    repaired)
    if dropped:
        logger.info("Dropped %d dead-code-under-disable assertion(s) "
                    "(left REPAIR_FAILED breadcrumbs).", dropped)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Shared expression utilities for missing-implication and tautology
# detectors.  These operate on raw assertion bodies after the
# `assert ...` / `else ...` shells have been stripped, and they reason
# about parens explicitly so they don't trip on nested grouping.
# ---------------------------------------------------------------------------

def _is_balanced(expr: str) -> bool:
    """Return True if `expr` has matched parentheses (standard scan)."""
    depth = 0
    for ch in expr:
        if ch == "(": depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0: return False
    return depth == 0


def _strip_outer_parens(expr: str) -> str:
    """Repeatedly strip a single layer of fully-wrapping outer parens."""
    s = expr.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        wraps_all = False
        for i, ch in enumerate(s):
            if ch == "(": depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    wraps_all = (i == len(s) - 1)
                    break
        if not wraps_all:
            break
        s = s[1:-1].strip()
    return s


def _split_top_level(expr: str, sep: str) -> List[str]:
    """Split `expr` on a multi-character separator at depth 0 only."""
    parts: List[str] = []
    depth = 0
    last = 0
    n, m = len(expr), len(sep)
    i = 0
    while i <= n - m:
        ch = expr[i]
        if ch == "(": depth += 1; i += 1; continue
        if ch == ")": depth -= 1; i += 1; continue
        if depth == 0 and expr[i:i+m] == sep:
            parts.append(expr[last:i].strip())
            i += m
            last = i
            continue
        i += 1
    parts.append(expr[last:].strip())
    return [p for p in parts if p]


def _find_top_level(expr: str, needle: str) -> int:
    """Return index of first depth-0 occurrence of `needle`, or -1."""
    depth = 0
    n, m = len(expr), len(needle)
    i = 0
    while i <= n - m:
        ch = expr[i]
        if ch == "(": depth += 1
        elif ch == ")": depth -= 1
        elif depth == 0 and expr[i:i+m] == needle:
            return i
        i += 1
    return -1


# Atom shape: optional `!`/`~`, optional parens, identifier with
# hierarchical reference and bit-select / part-select.  An atom is
# something that has no operators beyond its own negation/grouping.
_ATOM_RE = re.compile(
    r"^\s*[!~]?\s*\(?\s*"
    r"[A-Za-z_]\w*(?:\.\w+)*"
    r"(?:\[[^\]]+\])?"
    r"\s*\)?\s*$"
)


def _is_atomic_bool(expr: str) -> bool:
    """A boolean atom: `sig`, `!sig`, `(sig)`, `!sig.field[7:0]`, etc.
    No operators, no comparisons, no arithmetic."""
    return bool(_ATOM_RE.match(expr))


def _normalize_expr(expr: str) -> str:
    """Aggressively normalize an expression for structural equality:
    strip whitespace, redundant outer parens, and a leading `+`."""
    return _strip_outer_parens(expr).replace(" ", "")


_ASSERT_SHELL_RE   = re.compile(r"^\s*assert\s+(?:property\s*)?", re.IGNORECASE)
_ELSE_TAIL_RE      = re.compile(r"\)\s*else\b.*$|;\s*$", re.IGNORECASE)
_CLOCK_SPEC_RE     = re.compile(r"^\s*@\s*\([^)]*\)\s*", re.IGNORECASE)
_DISABLE_IFF_RE    = re.compile(r"^\s*disable\s+iff\s*\(", re.IGNORECASE)


def _strip_clock_and_disable(ante: str) -> str:
    """Strip `@(posedge clk)` and `disable iff (X)` prefixes from an
    antecedent.  Uses a paren-balanced scan for `disable iff` so that
    nested parens in the disable expression don't break the strip."""
    s = ante.strip()
    s = _CLOCK_SPEC_RE.sub("", s)
    m = _DISABLE_IFF_RE.match(s)
    if m:
        # find the matching ')' for the opening paren of disable iff
        depth = 0
        end = -1
        for i in range(m.end() - 1, len(s)):
            ch = s[i]
            if ch == "(": depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = i; break
        if end != -1:
            s = s[end+1:].lstrip()
    return s


def _extract_assertion_body(s: str) -> str:
    """Pull the top-level body out of an `assert (...) else ...;` line.

    Returns the inside of the outermost parens, with the assert/else
    shell removed.  If the line doesn't have a parseable shape, returns
    the original line unchanged so the caller can pass it through."""
    txt = s.strip()
    txt = _ASSERT_SHELL_RE.sub("", txt, count=1)
    # Drop `else $error(...);` tail and trailing semicolons.
    txt = re.sub(r"\)\s*else\s+\$\w+\s*\([^;]*\)\s*;?\s*$", ")", txt)
    txt = re.sub(r";\s*$", "", txt)
    txt = txt.strip()
    # Strip outer parens once (the body sits inside `assert (...)`).
    if txt.startswith("(") and txt.endswith(")") and _is_balanced(txt):
        # Verify the outer parens wrap the whole expression.
        depth = 0
        wraps = False
        for i, ch in enumerate(txt):
            if ch == "(": depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    wraps = (i == len(txt) - 1); break
        if wraps:
            txt = txt[1:-1].strip()
    return txt


# ---------------------------------------------------------------------------
# Detector: missing implication (violation antecedent without consequent)
# ---------------------------------------------------------------------------

_CMP_RE   = re.compile(r"==|!=|<=|>=|\s<\s|\s>\s")
_ARITH_RE = re.compile(r"[+*/%]|(?<![!~\-])\s-\s")


def _is_violation_antecedent(body: str) -> bool:
    """A body is a 'violation antecedent missing implication' if and
    only if:
      • Its top-level structure is a `&&` chain (≥2 conjuncts).
      • Every conjunct is an atomic boolean (`sig` or `!sig`, with
        optional grouping/slicing/dotted refs).
      • At least one conjunct is negated.

    The function returns False on:
      • Bodies with any `||` at top level (mode-encoding alternations).
      • Bodies wrapped fully in `!( ... )` (body-level negation, e.g.
        mutual exclusion).
      • Bodies containing any equality, comparison, or arithmetic
        operator (real predicates: conservation laws, reset value
        checks, range checks).
      • Bodies whose conjuncts include subexpressions with operators
        (those represent richer invariants, not bare violations).

    Doing the work positively rather than via blocklist ensures the
    detector is conservative: anything outside the known bug shape is
    kept.
    """
    if "&&" not in body:
        return False
    if "||" in body:
        return False
    if _CMP_RE.search(body) or _ARITH_RE.search(body):
        return False
    # If the entire body is `!( ... )` wrapping, treat as body-level
    # negation (mutual exclusion / De Morgan invariant) — keep.
    s = body.strip()
    if s.startswith("!(") or s.startswith("~("):
        # paren-balanced check for full wrap
        depth = 0
        wraps_all = False
        for i, ch in enumerate(s[1:], start=1):
            if ch == "(": depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    wraps_all = (i == len(s) - 1); break
        if wraps_all:
            return False
    conjuncts = _split_top_level(body, "&&")
    if len(conjuncts) < 2:
        return False
    if not all(_is_atomic_bool(c) for c in conjuncts):
        return False
    if not any(c.lstrip().startswith(("!", "~")) for c in conjuncts):
        return False
    return True


def remove_missing_implication_assertions(sva_code: str) -> str:
    """Drop `assert (sig && !sig && ...)` shaped bodies — pure-boolean
    `&&` chains over plain signals (with at least one negation). These
    are violation antecedents that should have appeared on the LEFT
    side of `|->`; emitting them as a bare conjunction asserts that
    the violation always holds (almost certainly the LLM's mistake).

    Whitelisted (kept untouched):
      • Bodies with any equality/comparison: conservation laws
        (`a == b && c == d`), reset checks (`x == 0 && y == 0`),
        range checks (`a > 0 && b < N`).
      • Bodies with arithmetic: invariant arithmetic relations.
      • Bodies wrapped in `!( ... )`: mutual exclusion / De Morgan
        invariants.
      • Bodies with any `||` at top level: mode-encoding, OR-chains.
      • Bodies with implication / cycle delay (`|->`, `|=>`, `##`):
        conditional already.
      • Concurrent `assert property (...)` form: leave to other
        detectors.

    Observed in earlier SpecGuard runs:
      • sdp: `assert (valid && !ready && !pop)` — shape match.
      • cmacfull: similar `req && !grant && !done` patterns.
    """
    lines = sva_code.splitlines()
    kept: List[str] = []
    removed = 0
    for ln in lines:
        s = ln.strip()
        if not s.startswith("assert"):
            kept.append(ln); continue
        if "property" in s.split("(", 1)[0]:
            kept.append(ln); continue
        if _find_top_level(s, "|->") >= 0 or _find_top_level(s, "|=>") >= 0:
            kept.append(ln); continue
        if "##" in s:
            kept.append(ln); continue
        body = _extract_assertion_body(s)
        if not _is_violation_antecedent(body):
            kept.append(ln); continue
        removed += 1
        logger.info("Removed missing-implication conjunction: %s", s[:90])
        if kept and kept[-1].strip().startswith("//"):
            kept.pop()
    if removed:
        logger.info("Removed %d missing-implication conjunction(s).", removed)
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Detector: tautological implications — antecedent structurally implies
# the consequent.  Generic across any antecedent / consequent shape.
# ---------------------------------------------------------------------------

def _conjunct_implies(antecedent: str, consequent: str) -> bool:
    """Return True when the implication `antecedent -> consequent` is
    structurally a tautology — no SAT, just normalised string equality
    over expression structure.

    Two cases captured:

    1. **Conjunct-membership**: the (whole) consequent matches some
       top-level conjunct of the antecedent.  Examples:
         `(A && X) -> A`,   `(!A && X) -> !A`,   `(X) -> X`.

    2. **Disjunct-membership in the consequent**: at least one
       top-level disjunct of the consequent matches a top-level conjunct
       of the antecedent (or the antecedent itself).  Examples:
         `(A) -> (A || B)`,        `(A && X) -> (B || A)`,
         `(!A) -> (C || !A)`.
       This captures `X |-> (X || ...)` style tautologies that would
       otherwise slip past a pure conjunct check.
    """
    if not consequent:
        return False
    cn = _normalize_expr(consequent)
    ante_conjuncts = [_normalize_expr(c) for c in _split_top_level(antecedent, "&&")]
    ante_conjuncts = [c for c in ante_conjuncts if c]

    # Case 1: whole consequent equals one of the antecedent's conjuncts.
    if cn in ante_conjuncts:
        return True

    # Case 2: any top-level disjunct of the consequent equals an
    # antecedent conjunct (or the antecedent as a whole).
    disjuncts = _split_top_level(consequent, "||")
    if len(disjuncts) > 1:
        ante_norm = _normalize_expr(antecedent)
        for d in disjuncts:
            dn = _normalize_expr(d)
            if not dn:
                continue
            if dn in ante_conjuncts or dn == ante_norm:
                return True
    return False


def _split_same_cycle_implication(body: str) -> Optional[Tuple[str, str]]:
    """If `body` is a *same-cycle* implication whose consequent has no
    temporal delay, return (antecedent, consequent). Otherwise None.

    Handled forms (the only ones where structural conjunct-overlap
    is a real tautology):
      • `X |-> Y` (overlapping implication, same cycle).
      • `!(X) || Y` (De Morgan immediate equivalent of `X -> Y`).
      • `(!X) || Y`.

    NOT considered (returns None even on conjunct overlap, since the
    consequent observes a different cycle than the antecedent):
      • `X |=> Y`         — non-overlapping (next cycle).
      • `X |-> ##N Y`     — delayed consequent.
      • `X |-> $past(...)` / similar temporal references."""
    s = body.strip()
    idx = _find_top_level(s, "|->")
    if idx >= 0:
        ante = _strip_outer_parens(s[:idx].strip())
        ante = _strip_clock_and_disable(ante)
        ante = _strip_outer_parens(ante)
        conseq_raw = s[idx+3:].strip()
        # Reject any cycle-delay or temporal operator in the consequent
        # since the conjunct-implies check no longer applies once the
        # consequent is shifted in time.
        if re.search(r"##|\$past|\$stable|\$rose|\$fell|\bs_eventually\b",
                     conseq_raw):
            return None
        conseq = _strip_outer_parens(conseq_raw)
        return ante, conseq
    # De Morgan: top-level `!X || Y` (immediate, same cycle).
    or_parts = _split_top_level(s, "||")
    if len(or_parts) == 2:
        left, right = or_parts
        l = left.strip()
        if l.startswith("!") or l.startswith("~"):
            inner = _strip_outer_parens(l[1:].strip())
            return inner, _strip_outer_parens(right)
    return None


def remove_tautological_implications(sva_code: str) -> str:
    """Drop assertions whose consequent is structurally implied by
    their antecedent — true regardless of design state.

    Captured shapes (general, not pattern-specific) — only when the
    consequent observes the SAME cycle as the antecedent:
      • `assert (X |-> Y)` where Y matches a top-level conjunct of X.
      • `assert property (... X |-> Y ...)` with the same condition.
      • `assert (!(X) || Y)` (De Morgan immediate form of `X -> Y`).

    Deliberately NOT flagged (different cycle → meaningful invariants):
      • `X |=> Y` — next-cycle persistence: `(valid && !ready) |=> valid`
        is a real protocol assertion, not a tautology.
      • `X |-> ##N Y` — delayed consequent.
      • `X |-> $past(Y)` / `$stable(Y)` etc.

    Examples flagged as tautologies:
      • `assert (!(A && !B) || A)`        → A is a conjunct of `A && !B`
      • `assert ((A && X) |-> A)`         → A is a conjunct of `A && X`
      • `assert ((!ready && X) |-> !ready)` → !ready is a conjunct
      • `assert (X |-> X)`                → trivial self-implication

    The check is purely structural (whitespace-normalized string
    equality on conjuncts) — no SAT solving, so it errs on the side
    of keeping when the consequent isn't a literal conjunct.
    """
    lines = sva_code.splitlines()
    kept: List[str] = []
    removed = 0
    for ln in lines:
        s = ln.strip()
        if not s.startswith("assert"):
            kept.append(ln); continue
        body = _extract_assertion_body(s)
        impl = _split_same_cycle_implication(body)
        if impl is None:
            kept.append(ln); continue
        ante, conseq = impl
        if _conjunct_implies(ante, conseq):
            removed += 1
            logger.info("Removed tautological implication: %s", s[:90])
            if kept and kept[-1].strip().startswith("//"):
                kept.pop()
            continue
        kept.append(ln)
    if removed:
        logger.info("Removed %d tautological implication(s).", removed)
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Detector: vacuous antecedents — assertions whose antecedent is
# structurally constant-false, so the consequent never gets checked.
# These look real but verify nothing.
# ---------------------------------------------------------------------------

# Constant-zero / constant-false literal forms commonly emitted by LLMs.
_CONST_ZERO_RE = re.compile(
    r"^\s*(?:1'b0|1'h0|1'd0|0|'0|\d+'b0+|\d+'h0+|\d+'d0+)\s*$",
    re.IGNORECASE,
)
# Constant-one forms (would make an antecedent trivially true — not
# vacuous, but combined with a nonsense consequent might be a different
# bug; we don't flag this here).


def _normalised_atom_negated(atom: str) -> Tuple[bool, str]:
    """Return (is_negated, base_atom_normalised). Tracks a single
    leading `!` or `~`; everything past that is the base."""
    s = atom.strip()
    neg = False
    while s.startswith("!") or s.startswith("~"):
        neg = not neg
        s = s[1:].strip()
    return neg, _normalize_expr(s)


def _is_constant_false(expr: str) -> bool:
    """Return True if `expr` is structurally a contradiction.

    Captures three forms:
      • A constant-zero literal (`1'b0`, `0`, `'0`, ...).
      • A self-contradicting conjunction `X && !X` (or `!X && X`,
        and chains containing such a pair anywhere).
      • A self-comparison that is always false: `X != X`, `X < X`,
        `X > X` (handled here so that vacuous-antecedent checks
        also catch these patterns)."""
    s = _strip_outer_parens(expr.strip())
    if not s:
        return False
    if _CONST_ZERO_RE.match(s):
        return True
    # Self-comparison contradictions: `X != X`, `X < X`, `X > X`.
    for op in ("!=", "<", ">"):
        idx = _find_top_level(s, op)
        if idx >= 0:
            lhs = _normalize_expr(s[:idx])
            rhs = _normalize_expr(s[idx + len(op):])
            if lhs and lhs == rhs:
                return True
    # `X && !X` somewhere at top level → contradiction.
    conjuncts = _split_top_level(s, "&&")
    if len(conjuncts) >= 2:
        seen_pos: Set[str] = set()
        seen_neg: Set[str] = set()
        for c in conjuncts:
            neg, base = _normalised_atom_negated(c)
            if not base:
                continue
            if neg and base in seen_pos:
                return True
            if (not neg) and base in seen_neg:
                return True
            (seen_neg if neg else seen_pos).add(base)
    return False


def remove_vacuous_antecedent(sva_code: str) -> str:
    """Drop assertions whose antecedent is structurally constant-false.
    Such assertions never check anything — they look real (compile
    fine, reference real signals) but provide zero verification value.

    Captured antecedent shapes:
      • Constant-zero literal: `assert ((1'b0) |-> X)`.
      • Self-contradiction: `assert ((a && !a) |-> X)`.
      • Same patterns under De Morgan immediate form
        (`assert (!(constant-true) || X)` is symmetric — we leave
        constant-true antecedents alone since they collapse to the
        consequent being asserted unconditionally, which other
        detectors handle).

    Bare bodies that are themselves contradictions (`assert (1'b0)`,
    `assert (a && !a)`) are also dropped — these would always fire
    and are essentially placeholders.
    """
    lines = sva_code.splitlines()
    kept: List[str] = []
    removed = 0
    for ln in lines:
        s = ln.strip()
        if not s.startswith("assert"):
            kept.append(ln); continue
        body = _extract_assertion_body(s)
        if not body:
            kept.append(ln); continue
        impl = _split_same_cycle_implication(body)
        target = impl[0] if impl is not None else body
        # For implications, also check the |=> form (vacuous antecedent
        # is meaningless under any cycle relation).
        if impl is None:
            for op in ("|=>",):
                idx = _find_top_level(body, op)
                if idx >= 0:
                    target = _strip_outer_parens(
                        _strip_clock_and_disable(
                            _strip_outer_parens(body[:idx].strip())
                        )
                    )
                    break
        if _is_constant_false(target):
            removed += 1
            logger.info("Removed vacuous-antecedent assertion: %s", s[:90])
            if kept and kept[-1].strip().startswith("//"):
                kept.pop()
            continue
        kept.append(ln)
    if removed:
        logger.info("Removed %d vacuous-antecedent assertion(s).", removed)
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Detector: self-comparisons — `(X OP X)` where LHS and RHS are
# structurally identical.  Always-true / always-false depending on OP.
# ---------------------------------------------------------------------------

# Comparison operators we test for self-equality / self-inequality.
# Order matters for prefix matching: `==`, `!=` before `<`, `>`.
_SELF_CMP_OPS = ("==", "!=", "<=", ">=", "<", ">")


def _check_self_comparison(body: str) -> Optional[str]:
    """If `body` reduces to `X OP X` with X structurally identical on
    both sides, return a short reason string.  Otherwise None.

    The check operates on the OUTERMOST comparison only; nested
    comparisons inside a complex expression are intentionally ignored
    so we don't over-fire on legitimate predicates like
    `((a == b) && (b == c))` where each comparand differs."""
    s = _strip_outer_parens(body.strip())
    if not s or "&&" in s or "||" in s:
        # A logical combination — skip.  An inner self-comparison can
        # appear, but it is part of a richer predicate; let the
        # downstream lint loop or formal handle it.
        return None
    for op in _SELF_CMP_OPS:
        idx = _find_top_level(s, op)
        if idx < 0:
            continue
        lhs = _normalize_expr(s[:idx])
        rhs = _normalize_expr(s[idx + len(op):])
        if not lhs or not rhs:
            continue
        if lhs == rhs:
            if op in ("==", "<=", ">="):
                return f"always true ({lhs} {op} {rhs})"
            return f"always false ({lhs} {op} {rhs})"
        # First top-level comparison decides; don't keep scanning.
        return None
    return None


# ---------------------------------------------------------------------------
# Detector: vacuous BODY — assertions whose entire body is structurally
# constant-true (never fires) or whose body simplifies to such via
# ternary identities like ``(cond ? 1'b1 : 1'b1)`` and
# ``(cond ? const : const)``.
#
# Distinct from ``remove_vacuous_antecedent`` (which catches
# constant-false antecedents).  This one catches the OPPOSITE failure
# mode: bodies that are constant-TRUE, which silently never fire and
# create false confidence in the assertion harness.
# ---------------------------------------------------------------------------

# A constant-one literal — accepts any positive width, with optional
# explicit width prefix.  Includes the bare ``1`` integer that some
# LLMs emit in place of ``1'b1``.
_CONST_TRUE_RE = re.compile(
    r"^\s*(?:1'b1|1'h1|1'd1|1|'1)\s*$",
    re.IGNORECASE,
)


def _is_constant_true(expr: str) -> bool:
    """Return True if `expr` is structurally always-true.

    Captures (without SAT solving):
      • Bare constant-one literals: ``1'b1``, ``1``, ``'1``, ``32'hffff``
        is NOT considered constant-true (it's a multi-bit literal whose
        boolean value is true, but conservatively we only flag the
        literally-named cases to avoid edge cases).
      • Ternary with structurally-identical true and false branches
        whose value is constant-true: ``(cond ? 1'b1 : 1'b1)``.
      • Ternary whose two branches are textually identical AND each is
        itself constant-true: ``(X ? Y : Y)`` where Y is const-true.
      • Self-comparisons that always hold (X == X) — already covered
        by ``remove_self_comparison_assertions``; we re-check here so
        a body that simplifies to ``(X == X)`` after ternary collapse
        is still caught.
    """
    s = _strip_outer_parens(expr.strip())
    if not s:
        return False
    if _CONST_TRUE_RE.match(s):
        return True
    # Ternary with equal branches at top level.
    q_idx = _find_top_level(s, "?")
    if q_idx >= 0:
        # Find matching `:` at the same depth.
        depth = 0
        c_idx = -1
        for i in range(q_idx + 1, len(s)):
            ch = s[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == ":" and depth == 0:
                c_idx = i; break
        if c_idx > q_idx:
            t_branch = _strip_outer_parens(s[q_idx + 1:c_idx].strip())
            f_branch = _strip_outer_parens(s[c_idx + 1:].strip())
            if _normalize_expr(t_branch) == _normalize_expr(f_branch):
                # Both branches identical → body equals that branch.
                # Recurse to catch nested vacuity.
                return _is_constant_true(t_branch)
    # Self-comparison (X == X) etc.
    for op in ("==", "<=", ">="):
        idx = _find_top_level(s, op)
        if idx >= 0:
            lhs = _normalize_expr(s[:idx])
            rhs = _normalize_expr(s[idx + len(op):])
            if lhs and lhs == rhs:
                return True
    return False


def _simplify_constant_ternary(body: str) -> Optional[Tuple[str, str]]:
    """If `body` matches `(cond) ? CTRUE : CFALSE` where the branches
    are constant 0/1 literals, return ``(simplified_body, reason)`` for
    a sound rewrite:

      • ``(X) ? 1'b1 : 1'b0``  →  ``X``           (identity)
      • ``(X) ? 1'b0 : 1'b1``  →  ``!(X)``        (inversion)
      • Other constant combinations don't yield a meaningful predicate
        (both true → vacuous-true; both false → vacuous-false; mixed
        non-0/1 → not a Boolean simplification).

    Returns ``None`` when no sound simplification applies.
    """
    s = _strip_outer_parens(body.strip())
    q_idx = _find_top_level(s, "?")
    if q_idx < 0:
        return None
    depth = 0
    c_idx = -1
    for i in range(q_idx + 1, len(s)):
        ch = s[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == ":" and depth == 0:
            c_idx = i; break
    if c_idx < 0:
        return None
    cond = _strip_outer_parens(s[:q_idx].strip())
    t_branch = _strip_outer_parens(s[q_idx + 1:c_idx].strip())
    f_branch = _strip_outer_parens(s[c_idx + 1:].strip())
    is_const_true  = lambda b: bool(_CONST_TRUE_RE.match(b.strip()))
    is_const_false = lambda b: bool(_CONST_ZERO_RE.match(b.strip()))
    if not cond:
        return None
    if is_const_true(t_branch) and is_const_false(f_branch):
        return f"({cond})", "(X ? 1 : 0) → X"
    if is_const_false(t_branch) and is_const_true(f_branch):
        return f"!({cond})", "(X ? 0 : 1) → !X"
    return None


# Captures a `property NAME; ... endproperty` declaration and the
# property body between them.  Used to inline named-property bodies
# before the vacuous-body check so that
# ``assert property (p_xyz);`` is judged on what ``p_xyz`` actually
# asserts, not on the bare reference line.
_PROPERTY_DECL_RE = re.compile(
    r"\bproperty\s+([A-Za-z_]\w*)\s*[;]\s*(.*?)\s*\bendproperty\b",
    re.DOTALL | re.IGNORECASE,
)
_ASSERT_PROP_REF_RE = re.compile(
    r"\bassert\s+property\s*\(\s*([A-Za-z_]\w*)\s*\)",
    re.IGNORECASE,
)


def _extract_property_bodies(text: str) -> Dict[str, str]:
    """Return ``{property_name: body_text}`` for every named property
    declaration in `text`.  The body excludes the ``property NAME;``
    header and ``endproperty`` keyword.

    Trailing semicolon is stripped from the body so downstream
    constant-checks operating on the body don't see ``(1'b1);`` and
    miss the constant-true match.
    """
    out: Dict[str, str] = {}
    for m in _PROPERTY_DECL_RE.finditer(text):
        body = m.group(2).strip()
        # Strip a trailing `;` (the statement terminator that lives
        # inside the property block before `endproperty`).
        while body.endswith(";"):
            body = body[:-1].rstrip()
        out[m.group(1)] = body
    return out


def _split_property_assert_on_one_line(text: str) -> str:
    """Some pipelines (notably the security pass) emit the named
    property declaration AND the ``assert property (NAME)`` statement
    on a single line: ``property p; ...; endproperty assert property
    (p);``.  Per-line detectors that match ``startswith("assert")``
    miss the assert because the line starts with ``property``.

    This pre-pass splits every ``endproperty assert property`` boundary
    onto its own line so downstream line-iterating detectors see the
    assert at the start of a line.  Idempotent — calling it twice has
    no effect.
    """
    return re.sub(
        r"\bendproperty\s+(?=assert\s+property\b)",
        "endproperty\n",
        text,
        flags=re.IGNORECASE,
    )


def remove_vacuous_body_assertions(sva_code: str) -> str:
    """Drop or repair assertions whose body is structurally constant
    or trivially simplifiable.

    Three handled shapes:

      1. **Constant-true body** (drop): ``assert (1'b1)``,
         ``assert ((cond) ? 1'b1 : 1'b1)``, ``assert (X == X)``.
         These never fire and carry no information — the LLM omitted
         the real predicate.  Drop with a ``// REPAIR_FAILED``
         breadcrumb.

      2. **Boolean ternary identity** (repair):
         ``assert ((cond) ? 1'b1 : 1'b0)`` is structurally equivalent
         to ``assert (cond)`` — sound rewrite.  Likewise
         ``assert ((cond) ? 1'b0 : 1'b1)`` becomes ``assert (!(cond))``.
         Repair preserves the LLM's evident intent (encoded in the
         ternary's ``cond``) instead of dropping useful information.

      3. **Constant-false body** (drop): ``assert (1'b0)`` always
         fires.  Either the LLM emitted a placeholder or it intended
         a violation report — neither is useful as-is.  Drop with
         breadcrumb.

    The repair path is sound because ``(X ? 1 : 0)`` and ``X`` are
    Boolean equivalents — the rewrite cannot change assertion
    semantics.  We do NOT attempt to repair more elaborate ternaries
    (those would require knowing whether the branch values mean
    anything beyond Boolean lift).
    """
    # Pre-scan property declarations so that ``assert property (p_xyz)``
    # can be judged against the body of ``property p_xyz; ...
    # endproperty``, not against the bare reference line.  Without
    # this inline step, vacuous bodies that live inside named
    # properties (a common security-pass output shape) escape the
    # detector entirely.
    property_bodies = _extract_property_bodies(sva_code)
    # Map property name → set of line indices that comprise its
    # declaration (so we can drop them when the asserting line goes).
    decl_line_ranges: Dict[str, Tuple[int, int]] = {}
    if property_bodies:
        # Build a per-line annotation: which (if any) property name
        # is being declared on this line, and whether it ends here.
        in_prop: Optional[str] = None
        prop_start: int = -1
        for idx, ln in enumerate(sva_code.splitlines()):
            if in_prop is None:
                m = re.search(r"\bproperty\s+([A-Za-z_]\w*)\s*[;]", ln)
                if m and m.group(1) in property_bodies:
                    in_prop = m.group(1)
                    prop_start = idx
            if in_prop is not None and "endproperty" in ln:
                decl_line_ranges[in_prop] = (prop_start, idx)
                in_prop = None

    lines = sva_code.splitlines()
    # Two-pass model: first pass classifies every line as
    # repair / drop / keep; second pass emits the final list.  Avoids
    # the index-misalignment bug where in-loop deletion of decl lines
    # shifts everything downstream.
    line_action: Dict[int, Tuple[str, Optional[str]]] = {}
    drop_idx: Set[int] = set()
    drop_breadcrumbs: Dict[int, str] = {}
    dropped  = 0
    repaired = 0
    for line_idx, line in enumerate(lines):
        s = line.strip()
        if not s.startswith("assert"):
            continue
        body = _extract_assertion_body(s)
        body_clean = _strip_clock_and_disable(body)

        # If this is `assert property (NAME);`, swap the body for the
        # named property's actual body so the vacuity check applies
        # to real content rather than to the property identifier.
        ref_m = _ASSERT_PROP_REF_RE.search(s)
        ref_name: Optional[str] = None
        if ref_m and ref_m.group(1) in property_bodies:
            ref_name = ref_m.group(1)
            inlined = property_bodies[ref_name]
            body_clean = _strip_clock_and_disable(inlined)

        # Repair path — sound Boolean identity simplification.
        repair = _simplify_constant_ternary(body_clean)
        if repair is not None:
            new_body, reason = repair
            indent = line[:len(line) - len(line.lstrip())]
            new_line = None
            if "assert property" in s:
                m = re.match(
                    r"(.*?assert\s+property\s*\()(.+)(\)\s*(?:else.*)?;?\s*$)",
                    s, re.DOTALL,
                )
                if m:
                    head, _, tail = m.group(1), m.group(2), m.group(3)
                    new_line = f"{indent}{head}{new_body}{tail}"
            else:
                m = re.match(
                    r"(.*?assert\s*\()(.+)(\)\s*(?:else.*)?;?\s*$)",
                    s, re.DOTALL,
                )
                if m:
                    head, _, tail = m.group(1), m.group(2), m.group(3)
                    new_line = f"{indent}{head}{new_body}{tail}"
            if new_line is not None:
                line_action[line_idx] = ("repair", new_line)
                repaired += 1
                logger.info("Repaired vacuous-ternary (%s): %s",
                            reason, s[:90])
                continue

        # Constant-true body — drop with breadcrumb.
        if _is_constant_true(body_clean):
            drop_idx.add(line_idx)
            drop_breadcrumbs[line_idx] = (
                "// REPAIR_FAILED: vacuous body — assertion body "
                "simplifies to constant-true and never fires; LLM "
                "omitted a real predicate."
            )
            if ref_name and ref_name in decl_line_ranges:
                start, end = decl_line_ranges[ref_name]
                for j in range(start, end + 1):
                    drop_idx.add(j)
            dropped += 1
            logger.info("Dropped vacuous-body assertion: %s", s[:90])
            continue

        # Constant-false body — drop with breadcrumb.
        if _is_constant_false(body_clean):
            drop_idx.add(line_idx)
            drop_breadcrumbs[line_idx] = (
                "// REPAIR_FAILED: constant-false body — assertion "
                "always fires regardless of design state; LLM either "
                "emitted a placeholder or inverted the property."
            )
            if ref_name and ref_name in decl_line_ranges:
                start, end = decl_line_ranges[ref_name]
                for j in range(start, end + 1):
                    drop_idx.add(j)
            dropped += 1
            logger.info("Dropped constant-false-body assertion: %s",
                        s[:90])
            continue

    # Second pass: rebuild output respecting per-line actions.
    out_lines: List[str] = []
    for i, line in enumerate(lines):
        if i in drop_idx:
            if i in drop_breadcrumbs:
                out_lines.append(drop_breadcrumbs[i])
            continue
        if i in line_action:
            kind, payload = line_action[i]
            if kind == "repair" and payload is not None:
                out_lines.append(payload); continue
        out_lines.append(line)
    if repaired:
        logger.info("Vacuous body: repaired %d ternary(ies).", repaired)
    if dropped:
        logger.info("Vacuous body: dropped %d assertion(s).", dropped)
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Detector: `inside`-set width mismatch — flag (or drop) `LHS inside
# {SET}` constructs where the LHS width does not equal the literal
# width(s) in SET.
#
# An `inside` clause with a width-mismatched literal is a structural
# bug: pyslang's set-membership semantics extend the narrower side
# with zeros, but the LLM almost never intends this — it usually
# means the assertion was constructed from inconsistent fragments.
# ---------------------------------------------------------------------------

# Matches LHS `inside {literal-or-range, ...}` at depth 0.
# LHS may include a bit-select.  Captures LHS-base, LHS-select, set-text.
_INSIDE_RE = re.compile(
    r"([A-Za-z_]\w*(?:\.\w+)*)"
    r"(\[[^\]]+\])?"
    r"\s+inside\s*\{([^{}]+)\}",
    re.IGNORECASE,
)


def _bit_select_width(sel: Optional[str]) -> Optional[int]:
    """Return the width of a bit-select string like ``[7:0]`` or
    ``[5]`` or ``None`` if not parseable as a numeric range."""
    if not sel:
        return None
    s = sel.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    inner = s[1:-1].strip()
    if ":" in inner:
        parts = inner.split(":")
        try:
            a, b = int(parts[0].strip()), int(parts[1].strip())
            return abs(a - b) + 1
        except ValueError:
            return None
    # Single index → 1 bit.
    try:
        int(inner)
        return 1
    except ValueError:
        return None


def validate_inside_widths(
    sva_code: str,
    facts: Optional["RTLFacts"] = None,
    mode: str = "drop",
) -> str:
    """Validate `LHS inside {SET}` clauses for LHS / literal width
    consistency.

    For each match:
      1. Compute LHS width: bit-select width if explicit, otherwise
         the signal's declared width from ``facts.signal_widths``.
      2. For every Verilog literal in SET, parse its declared width.
      3. If the LHS width is known AND every set-literal width is
         known AND any set-literal width disagrees with the LHS width,
         flag the assertion as structurally malformed.

    Conservative — when a width can't be determined (parameter-shaped
    bit-select, expressions in the set, missing facts) we abstain
    rather than guess.

    No repair: padding the literal is opinionated and could change
    intent.  Drops with a ``// REPAIR_FAILED`` breadcrumb so the LLM
    can re-emit on the next iteration.
    """
    widths = getattr(facts, "signal_widths", {}) if facts else {}
    if mode not in ("flag", "drop"):
        raise ValueError(f"validate_inside_widths: unknown mode {mode!r}")

    lines = sva_code.splitlines()
    out_lines: List[str] = []
    flagged = 0
    dropped = 0
    for line in lines:
        s = line.strip()
        if not s.startswith("assert") or "inside" not in s:
            out_lines.append(line); continue
        bug: Optional[str] = None
        for m in _INSIDE_RE.finditer(s):
            lhs_sig, lhs_sel, set_text = m.group(1), m.group(2), m.group(3)
            # Determine LHS width.
            lhs_w = _bit_select_width(lhs_sel) if lhs_sel else None
            if lhs_w is None:
                lhs_w = widths.get(lhs_sig)
            if lhs_w is None:
                continue
            # Parse every literal in the set; ignore set entries that
            # aren't simple Verilog literals (ranges, identifiers, etc.).
            lit_widths: List[Tuple[str, int]] = []
            for lit in set_text.split(","):
                lit = lit.strip()
                lw = _literal_width(lit) if "'" in lit else None
                if lw is not None:
                    lit_widths.append((lit, lw))
            if not lit_widths:
                continue
            # Find any width mismatch.
            for lit, lw in lit_widths:
                if lw != lhs_w:
                    bug = (f"{lhs_sig}{lhs_sel or ''} ({lhs_w}-bit) "
                           f"inside {{{lit}}} ({lw}-bit literal)")
                    break
            if bug:
                break
        if bug is None:
            out_lines.append(line); continue
        if mode == "flag":
            out_lines.append(f"// INSIDE_WIDTH_MISMATCH: {bug}")
            out_lines.append(line)
            flagged += 1
            logger.info("Flagged inside-width mismatch (%s): %s",
                        bug, s[:90])
        else:
            out_lines.append(
                f"// REPAIR_FAILED: inside-set width mismatch — {bug}; "
                f"padding would be opinionated."
            )
            dropped += 1
            logger.info("Dropped inside-width mismatch: %s", s[:90])
    if flagged:
        logger.info("Inside-width: flagged %d.", flagged)
    if dropped:
        logger.info("Inside-width: dropped %d.", dropped)
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Detector: generate-loop variable used outside generate scope.
#
# Catches assertions referencing identifiers that look like
# generate-loop indices (e.g. ``i``, ``j``, ``k``, ``ii``, ``jj``)
# but do NOT exist as real RTL signals.  Such assertions won't
# elaborate — the loop variable has no scope outside its generate
# block.  The bug appears when the LLM lifts an AST pattern from
# inside a generate without realising the surrounding scope.
# ---------------------------------------------------------------------------

# Single- or double-letter lowercase identifier (typical loop indices).
_LOOP_VAR_NAME_RE = re.compile(r"^[a-z]{1,2}$")


def validate_loop_var_scope(
    sva_code: str,
    facts: Optional["RTLFacts"] = None,
    mode: str = "drop",
) -> str:
    """Drop (or flag) assertions that reference identifiers shaped
    like generate-loop indices but absent from the **declared signal
    set** (signals with a known width in ``facts.signal_widths``).

    Heuristic but conservative:
      1. Only single- or double-letter all-lowercase tokens are
         considered loop-var candidates (``i``, ``j``, ``k``, ``ii``).
      2. The token must appear in the assertion body (after stripping
         the clock + disable shells AND Verilog literals).
      3. The token must NOT have a declared width.  Real RTL signals
         always have a width; generate-loop variables and parameters
         do not.  Pyslang surfaces loop-loop indices in
         ``all_signals``, so a pure ``all_signals`` membership test
         lets them through; ``signal_widths`` is the cleaner truth
         set for "is this a real wire I can reference outside a
         generate block?".

    Any single positive (loop-shaped + width-less) is enough to
    flag the assertion: there's no design under which `i` resolves
    correctly outside its generate scope without a width somewhere.

    Drop is the default (the assertion cannot elaborate); breadcrumb
    captures which loop variable was orphaned so the next-iteration
    prompt can prompt for either ``generate`` wrapping or signal
    substitution.
    """
    widths = getattr(facts, "signal_widths", None) if facts else None
    if not widths:
        return sva_code
    if mode not in ("flag", "drop"):
        raise ValueError(f"validate_loop_var_scope: unknown mode {mode!r}")

    lines = sva_code.splitlines()
    out_lines: List[str] = []
    flagged = 0
    dropped = 0
    for line in lines:
        s = line.strip()
        if not s.startswith("assert"):
            out_lines.append(line); continue
        body = _extract_assertion_body(s)
        body = _strip_clock_and_disable(body)
        # Strip the trailing `else $error/$fatal(...)` clause
        # explicitly.  ``_extract_assertion_body``'s regex breaks when
        # the error message itself contains a `;` (a common shape
        # in security-pass-emitted CWE messages like
        # ``"... cfg_reg_en; requires ..."``), leaving the entire
        # message text in `body` — and the message contains short
        # English words like ``on``, ``be``, ``to`` that match the
        # loop-var name regex.
        body = re.sub(
            r"\)\s*else\s+\$\w+\s*\(.*$",
            "",
            body,
            flags=re.DOTALL,
        )
        # Strip every double-quoted string so any leftover message
        # text can't contribute to the token set.
        body = re.sub(r'"[^"]*"', " ", body)
        # Strip Verilog literals BEFORE tokenising — otherwise the
        # extractor pulls letter-suffixes out of bases like ``4'hf``
        # (the trailing ``hf`` looks like a 2-letter loop-var name).
        cleaned = re.sub(
            r"\b\d+'[bdhoBDHO][0-9a-fA-FxXzZ_?]+",
            " ",
            body,
        )
        cleaned = re.sub(r"\b\d+'\s*[bdhoBDHO]\b", " ", cleaned)
        # Find candidate loop-var tokens; case-sensitive.
        tokens = set(re.findall(r"\b([A-Za-z_]\w*)\b", cleaned))
        offenders = [t for t in tokens
                     if _LOOP_VAR_NAME_RE.match(t)
                     and t not in widths]
        if not offenders:
            out_lines.append(line); continue
        offenders.sort()
        msg = (f"loop-shaped identifier(s) {{{','.join(offenders)}}} "
               f"not in design signal map; assertion likely lifted "
               f"from a generate block without scope wrapping.")
        if mode == "flag":
            out_lines.append(f"// LOOP_VAR_OUT_OF_SCOPE: {msg}")
            out_lines.append(line)
            flagged += 1
            logger.info("Flagged loop-var-out-of-scope (%s): %s",
                        offenders, s[:90])
        else:
            out_lines.append(
                f"// REPAIR_FAILED: {msg}"
            )
            dropped += 1
            logger.info("Dropped loop-var-out-of-scope: %s", s[:90])
    if flagged:
        logger.info("Loop-var scope: flagged %d.", flagged)
    if dropped:
        logger.info("Loop-var scope: dropped %d.", dropped)
    return "\n".join(out_lines)


def remove_self_comparison_assertions(sva_code: str) -> str:
    """Drop `assert (X OP X)` where LHS and RHS are structurally
    identical (after whitespace and outer-paren normalisation).

    These assertions are tautologies (`==`, `<=`, `>=`) or always-fail
    placeholders (`!=`, `<`, `>`) — neither has verification value.

    Captured shapes:
      • `assert (sig == sig)`.
      • `assert (top.x.y >= top.x.y)`.
      • `assert (data[7:0] != data[7:0])`.
      • Inside an implication consequent: `assert (A |-> (X == X))`.

    NOT captured (kept untouched):
      • `assert (sig[7:0] == sig[15:8])` — different bit-selects.
      • `assert (sig == ~sig)`            — different operand.
      • `assert ((a == b) && (b == c))`   — compound predicate; let the
        lint loop handle inner false-equality patterns.
    """
    lines = sva_code.splitlines()
    kept: List[str] = []
    removed = 0
    for ln in lines:
        s = ln.strip()
        if not s.startswith("assert"):
            kept.append(ln); continue
        body = _extract_assertion_body(s)
        targets: List[str] = []
        impl = _split_same_cycle_implication(body)
        if impl is not None:
            # Check the consequent of a same-cycle implication.
            targets.append(impl[1])
        else:
            # No same-cycle implication — check the body directly.
            # Also unwrap `|=>` consequents for completeness.
            j = _find_top_level(body, "|=>")
            if j >= 0:
                targets.append(_strip_outer_parens(body[j+3:].strip()))
            else:
                targets.append(body)
        reason = None
        for t in targets:
            r = _check_self_comparison(t)
            if r:
                reason = r; break
        if reason:
            removed += 1
            logger.info("Removed self-comparison assertion (%s): %s",
                        reason, s[:90])
            if kept and kept[-1].strip().startswith("//"):
                kept.pop()
            continue
        kept.append(ln)
    if removed:
        logger.info("Removed %d self-comparison assertion(s).", removed)
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Detector: semantic affinity — assertions whose signals have no
# design-level relationship (according to the RTL co-occurrence graph).
#
# Catches "frankenstein" assertions: bodies that mix signals from
# unrelated subsystems.  A pure structural detector cannot reach this
# class of LLM hallucination because every individual signal is real,
# every width matches, and every operator is syntactically valid.
# ---------------------------------------------------------------------------

def _assertion_signals_for_affinity(
    body: str,
    facts: "RTLFacts",
) -> Set[str]:
    """Extract the set of *eligible* design signals from an assertion
    body for the affinity check.

    Filters applied (in order):
      • Drop clock and reset signals.  They appear in nearly every
        assertion and would mask real co-occurrence patterns.
      • Drop known parameter names from ``facts.parameters`` (and
        single-token UPPER_CASE identifiers as a heuristic backstop
        when the parameter list is incomplete).  Parameters are
        compile-time constants, not RTL signals, and don't belong
        in the co-occurrence graph.
      • Drop short generate-loop indices (one- or two-letter lower
        case names like ``i``, ``j``, ``ii``).  These are loop
        variables introduced inside generate blocks and are not
        signals in the design's signal map.
      • If ``facts.all_signals`` is populated, restrict to identifiers
        that pyslang has confirmed as real signals.

    Returns the filtered set; the caller decides whether the set is
    large enough to score.
    """
    raw = _extract_signal_names(body)

    skip_clk_rst = (getattr(facts, "clock_signals", set()) or set()) \
        | (getattr(facts, "reset_signals", set()) or set())

    # Build a parameter-name set from facts (best-effort).
    param_names: Set[str] = set()
    for p in (getattr(facts, "parameters", []) or []):
        if isinstance(p, dict):
            n = p.get("name")
            if n: param_names.add(n)
        elif isinstance(p, str):
            param_names.add(p)

    all_sigs = getattr(facts, "all_signals", set()) or set()

    out: Set[str] = set()
    for sig in raw:
        if sig in skip_clk_rst:
            continue
        # Explicit parameters.
        if sig in param_names:
            continue
        # Heuristic backstop for parameters: ALL_UPPER_CASE identifiers
        # of length ≥ 2, with no lowercase letter and no digits-only.
        if (len(sig) >= 2 and sig == sig.upper()
                and any(c.isalpha() for c in sig)
                and not sig.isdigit()):
            continue
        # Generate-loop indices: short (≤2 chars), all-lowercase,
        # purely alphabetic.  Catches `i`, `j`, `k`, `ii`, `jj`.
        if (len(sig) <= 2 and sig == sig.lower() and sig.isalpha()):
            continue
        # If we have a signal manifest, demand membership.
        if all_sigs and sig not in all_sigs:
            continue
        out.add(sig)
    return out


def _affinity_link_ratio(
    sigs: Set[str],
    cooc: Dict[str, Set[str]],
    signal_to_module: Optional[Dict[str, str]] = None,
    signal_port_info: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[float, int, int]:
    """Compute the fraction of signal pairs in `sigs` that are linked,
    using a four-tier definition of "linked".

    Two signals A and B are linked when ANY of the following holds:
      1. **Direct statement co-occurrence**: B is in cooc(A) — they
         appear together in the same continuous assign or single
         always-block assignment.
      2. **1-hop transitive co-occurrence**: cooc(A) and cooc(B)
         share at least one neighbour.  Captures pairs that connect
         through intermediate combinational logic.
      3. **Same module**: signal_to_module(A) == signal_to_module(B).
         Captures handshake pairs and other port-level invariants
         where the signals are sibling ports/wires of the same module
         but never appear in a single statement together.
      4. **Same instance port-connection**: A and B are both wired into
         the same submodule instance (perhaps as different ports).
         Captures cross-module relationships where two signals connect
         through a common submodule — a Slice's input and output, for
         instance — without sharing a single RTL statement.

    The tiers progress from *tight* (statement) to *loose* (port
    connection across modules).  A pair is considered linked the
    moment any tier matches.

    Returns ``(link_ratio, linked_pairs, total_pairs)``.  When the
    set has fewer than two scoreable signals, returns ``(1.0, 0, 0)``
    (the assertion is not flag-eligible).
    """
    sig_list = sorted(sigs)
    n = len(sig_list)
    if n < 2:
        return 1.0, 0, 0
    total = n * (n - 1) // 2
    linked = 0
    nbrs = {s: cooc.get(s, set()) for s in sig_list}
    sig2mod = signal_to_module or {}
    port_info = signal_port_info or {}
    # Pre-compute "instance keys" for each signal: the set of
    # (parent_module, instance_name) tuples this signal participates
    # in.  Two signals are Tier-4-linked if these sets intersect.
    inst_keys: Dict[str, Set[Tuple[str, str]]] = {}
    for s in sig_list:
        info = port_info.get(s)
        if not info:
            inst_keys[s] = set()
            continue
        keys: Set[Tuple[str, str]] = set()
        for (parent_mod, inst_name, _inst_type, _port_name) in info.get(
            "connections", []
        ):
            if parent_mod and inst_name:
                keys.add((parent_mod, inst_name))
        inst_keys[s] = keys
    for i in range(n):
        si = sig_list[i]
        ni = nbrs[si]
        mi = sig2mod.get(si)
        ki = inst_keys[si]
        for j in range(i + 1, n):
            sj = sig_list[j]
            # Tier 1: direct co-occurrence.
            if sj in ni:
                linked += 1; continue
            # Tier 2: 1-hop transitive.
            nj = nbrs[sj]
            if ni and nj and (ni & nj):
                linked += 1; continue
            # Tier 3: same module.
            if mi is not None:
                mj = sig2mod.get(sj)
                if mj is not None and mi == mj:
                    linked += 1; continue
            # Tier 4: same instance port-connection.
            if ki and inst_keys[sj] and (ki & inst_keys[sj]):
                linked += 1; continue
    return (linked / total) if total else 1.0, linked, total


def validate_semantic_affinity(
    sva_code: str,
    facts: Optional["RTLFacts"] = None,
    threshold: float = 0.0,
    mode: str = "flag",
) -> str:
    """Validate that each assertion's signals have a real design-level
    relationship by checking the RTL co-occurrence graph in
    ``facts.signal_cooccurrence``.

    For each assertion:

      1. Extract the eligible signal set (in ``all_signals``, excluding
         clock and reset).
      2. Compute the *link ratio*: the fraction of distinct signal
         pairs that co-occur in any single RTL statement.
      3. If the assertion has at least two scoreable signals AND the
         link ratio is at or below ``threshold``, the assertion is
         flagged or dropped (per ``mode``).

    Parameters
    ----------
    threshold : float, default 0.0
        Pairs are considered "structurally unrelated" when the link
        ratio does not exceed this value.  ``0.0`` flags only the
        strongest signal: assertions where *no* pair of signals
        co-occurs anywhere in the RTL.  Higher values are more
        aggressive but risk false positives on legitimate
        cross-cutting properties.
    mode : {"flag", "drop"}, default "flag"
        ``"flag"`` annotates suspicious assertions with a
        ``// LOW_AFFINITY`` comment but keeps them in the output —
        useful while calibrating the threshold.
        ``"drop"`` removes them and leaves a ``// REPAIR_FAILED``
        breadcrumb for the lint-feedback loop.

    Notes
    -----
    Single-signal assertions (`assert (valid)`) and assertions that
    only reference clock/reset are exempt by construction (the link
    ratio is undefined and we return early).

    Without ``facts.signal_cooccurrence`` populated, this function
    is a no-op: the validator silently passes everything through so
    callers without facts (older test harnesses, ablation flows)
    aren't affected.
    """
    cooc = getattr(facts, "signal_cooccurrence", None) if facts else None
    if not cooc:
        return sva_code
    if mode not in ("flag", "drop"):
        raise ValueError(f"validate_semantic_affinity: unknown mode "
                         f"{mode!r}; expected 'flag' or 'drop'")

    sig2mod = getattr(facts, "signal_to_module", {}) or {}
    port_info = getattr(facts, "signal_port_info", {}) or {}

    lines = sva_code.splitlines()
    out_lines: List[str] = []
    flagged = 0
    dropped = 0
    for line in lines:
        s = line.strip()
        if not s.startswith("assert"):
            out_lines.append(line); continue
        body = _extract_assertion_body(s)
        # Strip clock/disable shells first so we score only the
        # property body's signals.
        body = _strip_clock_and_disable(body)
        sigs = _assertion_signals_for_affinity(body, facts)
        ratio, linked, total = _affinity_link_ratio(
            sigs, cooc, sig2mod, port_info,
        )
        if total < 1:
            out_lines.append(line); continue
        if ratio > threshold:
            out_lines.append(line); continue
        # Below-or-equal threshold → suspicious.
        sig_list_short = ", ".join(sorted(sigs)[:6])
        more = "..." if len(sigs) > 6 else ""
        if mode == "flag":
            out_lines.append(
                f"// LOW_AFFINITY (link={linked}/{total}): "
                f"signals [{sig_list_short}{more}] never co-occur in any "
                f"RTL statement; assertion may mix unrelated subsystems."
            )
            out_lines.append(line)
            flagged += 1
            logger.info("Flagged low-affinity assertion (link=%d/%d): %s",
                        linked, total, s[:90])
        else:  # mode == "drop"
            out_lines.append(
                f"// REPAIR_FAILED: low semantic affinity "
                f"(link={linked}/{total}); signals [{sig_list_short}{more}] "
                f"never co-occur in any RTL statement."
            )
            dropped += 1
            if out_lines and len(out_lines) >= 2 \
                    and out_lines[-2].strip().startswith("//") \
                    and not out_lines[-2].strip().startswith("// REPAIR_FAILED"):
                pass  # keep prior comment for the next-pass agent
            logger.info("Dropped low-affinity assertion (link=%d/%d): %s",
                        linked, total, s[:90])
    if flagged:
        logger.info("Semantic-affinity: flagged %d assertion(s) "
                    "(threshold=%.2f).", flagged, threshold)
    if dropped:
        logger.info("Semantic-affinity: dropped %d assertion(s) "
                    "(threshold=%.2f).", dropped, threshold)
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Detector: implication direction — flag `A |-> B` (or `A |=> B`) when
# B is structurally upstream of A in the RTL dataflow graph (i.e., the
# LLM swapped antecedent and consequent).
# ---------------------------------------------------------------------------

def _antecedent_consequent_signals(
    body: str,
    facts: "RTLFacts",
) -> Optional[Tuple[Set[str], Set[str], str]]:
    """Split an assertion body into (antecedent_signals, consequent_signals,
    op) for any implication form, returning ``None`` if the body is not
    an implication.

    Filters apply the same exclusions as the affinity check
    (clocks, resets, parameters, generate-loop indices, signals not
    in ``facts.all_signals``)."""
    s = body.strip()
    op = None
    idx = -1
    for cand_op in ("|->", "|=>"):
        i = _find_top_level(s, cand_op)
        if i >= 0 and (idx < 0 or i < idx):
            idx = i
            op = cand_op
    if idx < 0 or op is None:
        return None
    ante_str = _strip_outer_parens(
        _strip_clock_and_disable(_strip_outer_parens(s[:idx].strip()))
    )
    conseq_str = _strip_outer_parens(s[idx + len(op):].strip())
    if not ante_str or not conseq_str:
        return None
    ante_sigs = _assertion_signals_for_affinity(ante_str, facts)
    conseq_sigs = _assertion_signals_for_affinity(conseq_str, facts)
    return ante_sigs, conseq_sigs, op


def _is_strictly_upstream(
    candidate_upstream: str,
    candidate_downstream: str,
    flow: Dict[str, Dict[str, Set[str]]],
    max_depth: int = 3,
) -> bool:
    """Return True if `candidate_upstream` reaches `candidate_downstream`
    via the dataflow graph within ``max_depth`` hops AND the reverse
    direction does NOT also hold within the same depth.

    The asymmetry matters: cyclic feedback paths (state register that
    drives next-state and is driven by next-state) would otherwise
    flag in both directions.  We only flag a one-way upstream
    relationship, which is the unambiguous "wrong-direction" signal.
    """
    if candidate_upstream == candidate_downstream:
        return False

    def _reaches(src: str, dst: str) -> bool:
        if src not in flow:
            return False
        seen = {src}
        frontier = [(src, 0)]
        while frontier:
            cur, d = frontier.pop()
            if d >= max_depth:
                continue
            for nxt in flow.get(cur, {}).get("downstream", set()):
                if nxt == dst:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append((nxt, d + 1))
        return False

    forward  = _reaches(candidate_upstream, candidate_downstream)
    backward = _reaches(candidate_downstream, candidate_upstream)
    return forward and not backward


def validate_implication_direction(
    sva_code: str,
    facts: Optional["RTLFacts"] = None,
    max_depth: int = 3,
    mode: str = "flag",
) -> str:
    """Flag (or drop) assertions whose implication direction is reversed
    relative to the RTL dataflow.

    For each `assert (A |-> B)` or `assert (A |=> B)`:

      1. Extract the antecedent's signal set and the consequent's
         signal set.
      2. For every (a, b) cross-product pair, check whether `b` is
         strictly upstream of `a` in the dataflow graph (within
         ``max_depth`` hops).
      3. If MORE than half of cross-product pairs are
         "wrong-direction", flag the assertion as having a
         reversed implication.

    The cross-product/majority approach handles assertions with
    multiple signals on either side gracefully — we don't flag
    on a single coincidental pair, only when the dominant direction
    runs the wrong way.

    Parameters
    ----------
    max_depth : int
        Number of dataflow hops to search before giving up.  Bounded
        so this remains O(N) across the whole pipeline; depth=3
        catches the common 1- to 3-stage relationships in practice.
    mode : {"flag", "drop"}
        ``"flag"`` annotates the assertion with a comment but keeps it.
        ``"drop"`` removes it and leaves a ``// REPAIR_FAILED`` breadcrumb.

    Notes
    -----
    Without a populated ``facts.signal_dataflow`` this function is a
    no-op.  Cyclic feedback paths (state register ↔ next-state) are
    handled by requiring strict one-way upstream — both-direction
    reachable pairs are NOT flagged.
    """
    flow = getattr(facts, "signal_dataflow", None) if facts else None
    if not flow:
        return sva_code
    if mode not in ("flag", "drop"):
        raise ValueError(f"validate_implication_direction: unknown mode "
                         f"{mode!r}; expected 'flag' or 'drop'")

    lines = sva_code.splitlines()
    out_lines: List[str] = []
    flagged = 0
    dropped = 0
    for line in lines:
        s = line.strip()
        if not s.startswith("assert"):
            out_lines.append(line); continue
        body = _extract_assertion_body(s)
        body = _strip_clock_and_disable(body)
        split = _antecedent_consequent_signals(body, facts)
        if split is None:
            out_lines.append(line); continue
        ante_sigs, conseq_sigs, op = split
        if not ante_sigs or not conseq_sigs:
            out_lines.append(line); continue
        # Skip pairs that overlap (signals on both sides — not a
        # direction question, possibly a tautology handled elsewhere).
        ante_only = ante_sigs - conseq_sigs
        conseq_only = conseq_sigs - ante_sigs
        if not ante_only or not conseq_only:
            out_lines.append(line); continue
        # Cross-product directional check.
        wrong = 0
        right = 0
        for a in ante_only:
            for b in conseq_only:
                if _is_strictly_upstream(b, a, flow, max_depth=max_depth):
                    wrong += 1
                elif _is_strictly_upstream(a, b, flow, max_depth=max_depth):
                    right += 1
        # Only flag when wrong-direction strictly dominates.  If neither
        # direction reaches (no path either way), we have no evidence
        # and abstain.
        if wrong == 0 or wrong <= right:
            out_lines.append(line); continue
        if mode == "flag":
            sample_pair = (
                f"{sorted(conseq_only)[0]} drives {sorted(ante_only)[0]}"
            )
            out_lines.append(
                f"// REVERSED_DIRECTION ({op}, evidence={wrong}>{right}): "
                f"in RTL dataflow, {sample_pair} — antecedent should likely "
                f"be on the consequent side."
            )
            out_lines.append(line)
            flagged += 1
            logger.info("Flagged reversed implication direction "
                        "(wrong=%d, right=%d): %s", wrong, right, s[:90])
        else:
            sample_pair = (
                f"{sorted(conseq_only)[0]} drives {sorted(ante_only)[0]}"
            )
            out_lines.append(
                f"// REPAIR_FAILED: reversed implication direction "
                f"({op}); RTL dataflow shows {sample_pair} "
                f"(wrong={wrong}, right={right})."
            )
            dropped += 1
            logger.info("Dropped reversed implication direction: %s", s[:90])
    if flagged:
        logger.info("Implication direction: flagged %d assertion(s).",
                    flagged)
    if dropped:
        logger.info("Implication direction: dropped %d assertion(s).",
                    dropped)
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Detector (informational): documented-property coverage — for each
# property lifted from spec docs, check whether the assertion file
# contains at least one assertion mentioning all of its signals.  Does
# NOT remove assertions; appends a single ``// DOC_COVERAGE:`` summary
# comment at the end of the file so the metrics block can pick it up
# and the next-iteration prompt can list the gaps.
# ---------------------------------------------------------------------------

def annotate_doc_coverage(
    sva_code: str,
    facts: Optional["RTLFacts"] = None,
) -> str:
    """Append a ``// DOC_COVERAGE: covered=N total=M missing=[...]``
    summary line to the SVA file, where ``covered`` is the number of
    documented properties whose every signal appears in at least one
    emitted assertion.

    Designed to be informational only — never modifies or removes any
    assertion.  When ``facts.documented_properties`` is empty (no docs
    were available, or none yielded usable sentences), the function is
    a no-op.
    """
    props = getattr(facts, "documented_properties", None) if facts else None
    if not props:
        return sva_code
    # Build the "all signals mentioned somewhere" set across the file.
    referenced: Set[str] = set()
    for line in sva_code.splitlines():
        s = line.strip()
        if not s.startswith("assert"):
            continue
        referenced |= _extract_signal_names(s)
    covered = 0
    missing: List[str] = []
    for p in props:
        signals = set(p.get("signals") or [])
        if signals and signals.issubset(referenced):
            covered += 1
        else:
            # Identify a short label for the missing property.
            label = (p.get("modal") or "?") + ":" + ",".join(
                sorted(signals)[:3]
            )
            missing.append(label)
    if covered == len(props):
        # Full coverage — still leave a positive breadcrumb.
        annotation = (f"// DOC_COVERAGE: covered={covered}/{len(props)} "
                      "(all documented properties have at least one "
                      "matching assertion)")
    else:
        miss_str = "; ".join(missing[:6])
        more = (f" +{len(missing) - 6}" if len(missing) > 6 else "")
        annotation = (
            f"// DOC_COVERAGE: covered={covered}/{len(props)} "
            f"missing=[{miss_str}{more}]"
        )
    logger.info("Doc coverage: %d/%d documented properties covered.",
                covered, len(props))
    if not sva_code.endswith("\n"):
        sva_code += "\n"
    return sva_code + annotation + "\n"


# ---------------------------------------------------------------------------
# Detector: state-machine literal hallucinations — flag assertions
# comparing an FSM's state signal against a literal that's not in the
# state set extracted from the RTL.  The LLM occasionally invents
# state values like ``state == 3'b111`` when only states 0-2 exist;
# without an FSM-aware check, those compile cleanly and never fire.
# ---------------------------------------------------------------------------

def _normalise_literal(lit: str) -> str:
    """Normalise a Verilog literal for equality comparison.

    Folds whitespace, lowercases the base character, and strips
    underscores so that ``2'b00`` matches ``2'B00`` and ``2'b0_0``.
    """
    s = lit.strip().replace(" ", "").replace("_", "")
    # Lowercase the base char (b/h/d/o) but keep the value digits.
    m = re.match(r"^(\d*'?)([bdhoBDHO]?)(.+)$", s)
    if m:
        return f"{m.group(1)}{m.group(2).lower()}{m.group(3).lower()}"
    return s.lower()


def validate_state_assertions(
    sva_code: str,
    facts: Optional["RTLFacts"] = None,
    mode: str = "flag",
) -> str:
    """Flag (or drop) assertions whose comparisons reference an FSM
    state literal that does NOT appear in the FSM's extracted state
    set.

    For each ``facts.state_machines`` entry, build the union of:
      • Literal patterns observed in case items (``literal_values``).
      • Encoded values for named states (``encoding`` map).

    Then for each assertion, find every comparison of the form
    ``state_signal {==,!=,<=,>=,<,>} <LITERAL>`` and check the
    literal against that union (whitespace and base-char normalised).
    Unknown literals are flagged — they suggest a fabricated state
    value rather than one extracted from the RTL.

    Notes
    -----
    Comparisons against an *identifier* (e.g., ``state == BUSY``) are
    not checked; they're either valid state-name references or will
    be caught by the existing signal-validation pass.

    Bare integer literals like ``0`` or ``1`` are tolerated because
    they're commonly used as catch-all reset values regardless of
    the formal encoding; only width-specified literals
    (``2'b00``, ``8'h0a``) are checked against the state set.
    """
    fsms = getattr(facts, "state_machines", None) if facts else None
    if not fsms:
        return sva_code
    if mode not in ("flag", "drop"):
        raise ValueError(f"validate_state_assertions: unknown mode "
                         f"{mode!r}; expected 'flag' or 'drop'")

    # Build per-state-signal known-value set, normalised.
    fsm_values: Dict[str, Set[str]] = {}
    for fsm in fsms:
        sig = fsm["state_signal"]
        vals: Set[str] = set()
        for v in fsm.get("literal_values", []):
            vals.add(_normalise_literal(v))
        for v in (fsm.get("encoding") or {}).values():
            vals.add(_normalise_literal(v))
        if vals:
            fsm_values[sig] = vals

    if not fsm_values:
        return sva_code

    # Match `<sig> OP <literal>` where literal is width-prefixed.
    # We anchor on the state signal name, so the regex is built per
    # signal to keep matches precise.
    width_lit = r"\d+'[bdhoBDHO][0-9a-fA-FxXzZ_]+"

    lines = sva_code.splitlines()
    out_lines: List[str] = []
    flagged = 0
    dropped = 0
    for line in lines:
        s = line.strip()
        if not s.startswith("assert"):
            out_lines.append(line); continue
        body = _extract_assertion_body(s)
        # Find the first FSM signal mentioned in this assertion;
        # short-circuit if none are present.
        match_info: Optional[Tuple[str, str, str]] = None
        for sig, known_vals in fsm_values.items():
            sig_re = re.compile(
                r"\b" + re.escape(sig) + r"\b\s*"
                r"(==|!=|<=|>=|<|>)\s*"
                r"(" + width_lit + r")",
            )
            m = sig_re.search(body)
            if not m:
                continue
            op, lit = m.group(1), m.group(2)
            if _normalise_literal(lit) in known_vals:
                continue  # legitimate state comparison
            match_info = (sig, op, lit)
            break
        if match_info is None:
            out_lines.append(line); continue
        sig, op, lit = match_info
        known_str = ", ".join(sorted(fsm_values[sig])[:8])
        if mode == "flag":
            out_lines.append(
                f"// STATE_HALLUCINATION: {sig} {op} {lit} — literal not "
                f"in extracted FSM state set {{{known_str}}}; possible "
                f"fabricated state value."
            )
            out_lines.append(line)
            flagged += 1
            logger.info("Flagged state-literal hallucination "
                        "(%s %s %s): %s", sig, op, lit, s[:90])
        else:
            out_lines.append(
                f"// REPAIR_FAILED: state-literal hallucination — "
                f"{sig} {op} {lit} not in extracted state set "
                f"{{{known_str}}}."
            )
            dropped += 1
            logger.info("Dropped state-literal hallucination: %s", s[:90])
    if flagged:
        logger.info("State-machine: flagged %d assertion(s).", flagged)
    if dropped:
        logger.info("State-machine: dropped %d assertion(s).", dropped)
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Detector: cross-module assertions — flag assertions whose signals come
# from modules with no port-connection link.  Catches the family of
# "frankenstein" assertions where the LLM mixes ports from unrelated
# subsystems.  Distinct from the affinity check because it works at
# module granularity rather than statement-level co-occurrence.
# ---------------------------------------------------------------------------

def _signal_module_set(
    sig: str,
    facts: "RTLFacts",
) -> Set[str]:
    """Return the set of modules a signal participates in.

    A signal participates in a module if ANY of:
      • It is declared as a port of that module.
      • It appears in that module's body (per ``signal_to_module``).
      • It is wired into an instance inside that module (parent side).
      • It is wired INTO that module via an instance port-connection
        in some parent module (i.e., the module is the instance type
        that uses this signal — counted because the signal flows into
        that submodule's logic).

    Handles signals that exist in multiple modules (clock/reset) by
    returning the full set rather than a single module.
    """
    out: Set[str] = set()
    sig2mod = getattr(facts, "signal_to_module", {}) or {}
    port_info = getattr(facts, "signal_port_info", {}) or {}
    info = port_info.get(sig, {})
    # Every module where this signal name appears as a port (a single
    # name like ``req`` may be a port of several modules in a hierarchy
    # — see _extract_signal_port_info docstring).
    pms = info.get("port_modules") or set()
    out.update(pms)
    pm = info.get("port_module")
    if pm:
        out.add(pm)
    fallback = sig2mod.get(sig)
    if fallback:
        out.add(fallback)
    for (parent_mod, _inst_name, inst_type, _port_name) in info.get(
        "connections", []
    ):
        # Both ends of a wire are functionally related: the parent
        # module that owns the wire AND the submodule whose port the
        # wire drives.  Including both stops the validator from
        # treating signals connected through an instance as if they
        # came from unrelated subsystems.
        if parent_mod:
            out.add(parent_mod)
        if inst_type:
            out.add(inst_type)
    return out


def validate_cross_module_assertions(
    sva_code: str,
    facts: Optional["RTLFacts"] = None,
    mode: str = "flag",
) -> str:
    """Flag (or drop) assertions whose signals split across modules
    with NO port-connection or shared-instance path between them.

    For each assertion:
      1. Extract eligible signals (same filters as affinity check).
      2. Collect the module-participation set of each signal via
         ``_signal_module_set``.
      3. If the union of all signal modules is non-empty AND the
         pairwise intersection is empty (no two signals share a
         module), AND there is no Tier-4 instance link between them,
         the assertion mixes genuinely unrelated subsystems — flag.

    The detector deliberately abstains when:
      • Any signal lacks module info (we cannot prove unrelatedness).
      • All signals share at least one module (uncontroversial).
      • A Tier-4 instance link bridges the modules (legitimate
        cross-module port-level invariant).

    Notes
    -----
    Lower-recall sibling of the affinity check.  Where affinity
    operates at signal-pair granularity (and may flag a single odd
    pair in an otherwise good assertion), this detector requires
    *every* signal pair to be cross-module-disjoint, so its precision
    is higher but it catches fewer cases.  Both run in flag-only mode
    by default; the union of their flags is what surfaces to the
    operator.
    """
    if not facts:
        return sva_code
    port_info = getattr(facts, "signal_port_info", None)
    sig2mod = getattr(facts, "signal_to_module", None)
    if not port_info and not sig2mod:
        return sva_code
    if mode not in ("flag", "drop"):
        raise ValueError(f"validate_cross_module_assertions: unknown "
                         f"mode {mode!r}; expected 'flag' or 'drop'")

    lines = sva_code.splitlines()
    out_lines: List[str] = []
    flagged = 0
    dropped = 0
    for line in lines:
        s = line.strip()
        if not s.startswith("assert"):
            out_lines.append(line); continue
        body = _extract_assertion_body(s)
        body = _strip_clock_and_disable(body)
        sigs = _assertion_signals_for_affinity(body, facts)
        if len(sigs) < 2:
            out_lines.append(line); continue
        # Per-signal module participation.
        per_sig_mods: Dict[str, Set[str]] = {
            sig: _signal_module_set(sig, facts) for sig in sigs
        }
        # Abstain if any signal is module-info-less.
        if any(not m for m in per_sig_mods.values()):
            out_lines.append(line); continue
        # If there is a single module shared by every signal, we're fine.
        common = set.intersection(*per_sig_mods.values())
        if common:
            out_lines.append(line); continue
        # Otherwise, see whether a Tier-4 instance link bridges any pair
        # of signals.  Build the inst_key sets.
        inst_keys: Dict[str, Set[Tuple[str, str]]] = {}
        for sig in sigs:
            info = port_info.get(sig, {}) if port_info else {}
            keys: Set[Tuple[str, str]] = set()
            for (pm, iname, _t, _p) in info.get("connections", []):
                if pm and iname:
                    keys.add((pm, iname))
            inst_keys[sig] = keys
        sig_list = sorted(sigs)
        any_inst_link = False
        for i in range(len(sig_list)):
            for j in range(i + 1, len(sig_list)):
                if inst_keys[sig_list[i]] & inst_keys[sig_list[j]]:
                    any_inst_link = True
                    break
            if any_inst_link:
                break
        if any_inst_link:
            out_lines.append(line); continue
        # All signals are real, scoped to known modules, with no shared
        # module and no shared instance — strongly suspicious.
        mods_summary = ", ".join(
            f"{sig}∈{{{','.join(sorted(per_sig_mods[sig]))}}}"
            for sig in sig_list[:4]
        )
        if mode == "flag":
            out_lines.append(
                f"// CROSS_MODULE: signals split across modules with no "
                f"port-connection link; {mods_summary}"
            )
            out_lines.append(line)
            flagged += 1
            logger.info("Flagged cross-module assertion: %s", s[:90])
        else:
            out_lines.append(
                f"// REPAIR_FAILED: cross-module assertion — signals split "
                f"across modules with no port-connection link; {mods_summary}"
            )
            dropped += 1
            logger.info("Dropped cross-module assertion: %s", s[:90])
    if flagged:
        logger.info("Cross-module: flagged %d assertion(s).", flagged)
    if dropped:
        logger.info("Cross-module: dropped %d assertion(s).", dropped)
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Detector: $past depth — flag `q == $past(d, N)` where N does not match
# the actual flop-chain depth from `d` to `q` in the dataflow graph.
# Off-by-N latency errors are a common LLM bug class that nothing else
# in the pipeline reaches.
# ---------------------------------------------------------------------------

# Match `LHS == $past(RHS_SIGNAL, N)` (with optional N — defaults to 1
# if omitted in SVA).  We accept hierarchical refs and bit-selects on
# both LHS and the $past argument.
_PAST_EQ_RE = re.compile(
    r"\(?\s*"
    r"([A-Za-z_]\w*(?:\.\w+)*)(?:\[[^\]]+\])?"     # LHS signal (capture base)
    r"\s*==\s*"
    r"\$past\s*\(\s*"
    r"([A-Za-z_]\w*(?:\.\w+)*)(?:\[[^\]]+\])?"     # past arg (capture base)
    r"(?:\s*,\s*(\d+))?"                            # optional depth N
    r"\s*\)",
)


def validate_past_depth(
    sva_code: str,
    facts: Optional["RTLFacts"] = None,
    mode: str = "flag",
    max_lookup_depth: int = 16,
) -> str:
    """Validate that ``q == $past(d, N)`` patterns use the correct
    depth N relative to the actual flop-chain depth from ``d`` to
    ``q`` in the RTL dataflow graph.

    The check is purely structural:
      1. Extract every ``LHS == $past(RHS, N)`` occurrence (N=1 if
         omitted) from the assertion body.
      2. Look up the minimum cycle delay from RHS to LHS via
         ``rtl_facts.pipeline_depth`` (BFS bounded at
         ``max_lookup_depth``).
      3. If the dataflow shows a definite cycle delay D and N != D,
         flag the assertion as a likely off-by-N error.

    The validator deliberately *abstains* (doesn't flag) when:
      • The RHS or LHS isn't in the dataflow graph (we can't measure).
      • No path is found within the search bound (could be deeper
        than expected, or could be cross-module — neither is a clear
        off-by-N signal).

    Parameters
    ----------
    mode : {"flag", "drop"}
        Same convention as the other validators.

    Notes
    -----
    Pyslang-extracted dataflow doesn't always trace through generate
    blocks or chisel-emitted intermediate wires cleanly, so the
    abstain-on-no-path rule keeps the false-positive rate low at the
    cost of some recall.
    """
    flow = getattr(facts, "signal_dataflow", None) if facts else None
    if not flow:
        return sva_code
    if mode not in ("flag", "drop"):
        raise ValueError(f"validate_past_depth: unknown mode "
                         f"{mode!r}; expected 'flag' or 'drop'")

    try:
        from .rtl_facts import pipeline_depth as _pdepth
    except ImportError:
        from sva_pipeline.rtl_facts import pipeline_depth as _pdepth

    lines = sva_code.splitlines()
    out_lines: List[str] = []
    flagged = 0
    dropped = 0
    for line in lines:
        s = line.strip()
        if not s.startswith("assert"):
            out_lines.append(line); continue
        if "$past" not in s:
            out_lines.append(line); continue
        body = _extract_assertion_body(s)
        body = _strip_clock_and_disable(body)
        # Each `$past` mismatch is independent; flag the first we find
        # to keep the message clean (one flag per assertion).
        mismatch: Optional[Tuple[str, str, int, int]] = None
        for m in _PAST_EQ_RE.finditer(body):
            lhs, rhs_arg = m.group(1), m.group(2)
            n_str = m.group(3)
            n = int(n_str) if n_str else 1
            actual = _pdepth(facts, rhs_arg, lhs,
                             max_depth=max(max_lookup_depth, n + 4))
            # Abstain if dataflow has no measurement.
            if actual is None or actual == 0:
                continue
            if actual != n:
                mismatch = (lhs, rhs_arg, n, actual)
                break
        if mismatch is None:
            out_lines.append(line); continue
        lhs, rhs_arg, n, actual = mismatch
        if mode == "flag":
            out_lines.append(
                f"// PAST_DEPTH_MISMATCH: $past({rhs_arg}, {n}) used to "
                f"compare against {lhs}, but RTL dataflow shows "
                f"{actual}-cycle delay from {rhs_arg} to {lhs}; "
                f"likely off-by-{abs(actual - n)} latency error."
            )
            out_lines.append(line)
            flagged += 1
            logger.info("Flagged $past depth mismatch (got %d, expected "
                        "%d): %s", n, actual, s[:90])
        else:
            out_lines.append(
                f"// REPAIR_FAILED: $past depth mismatch — "
                f"$past({rhs_arg}, {n}) but actual flop depth is {actual}."
            )
            dropped += 1
            logger.info("Dropped $past depth mismatch: %s", s[:90])
    if flagged:
        logger.info("$past depth: flagged %d assertion(s).", flagged)
    if dropped:
        logger.info("$past depth: dropped %d assertion(s).", dropped)
    return "\n".join(out_lines)


# Pattern: bare `(sig == LIT)` body with no antecedent / implication.
# Matches `assert (sig == LIT)` and `assert property (... (sig == LIT))`.
# Captures the BODY between the outermost parens AFTER stripping the
# clock spec and disable-iff prefix.
_BARE_EQ_BODY_RE = re.compile(
    r"\(\s*([A-Za-z_]\w*(?:\[[^\]]+\])?)\s*==\s*"
    r"(\d+'?[bdhoBDHOxXzZ?]?[\w?]*|\{[^}]+\})\s*\)"
)


def remove_unconditional_equality_assertions(sva_code: str, facts: Optional[RTLFacts] = None) -> str:
    """Drop assertions whose body — after stripping clock spec and
    `disable iff` shells — is just `(signal == LITERAL)` with no
    antecedent, no implication, no temporal operator.  These claim the
    signal is always equal to the literal on every clock cycle, which
    is almost always wrong (most signals change).

    Now handles BOTH shapes:
      • Immediate: ``assert (sig == LIT) else ...;``
      • Concurrent: ``assert property (@(posedge clk) [disable iff (...)]
        sig == LIT) [else ...];``

    The concurrent form was previously missed because the early-return
    on ``|->`` / ``|=>`` / ``##`` / ``$past`` is preceded by clock-spec
    inspection that doesn't actually contain those operators in the
    bare-equality concurrent shape — so we explicitly extract the
    body and check the residue.

    Whitelist parameter-style identifiers (UPPER_CASE) to avoid
    false-positives on legitimate static-config checks.  Observed in
    SpecGuard cmacfull outputs: 7 always-zero datapath bugs (immediate
    form) AND in cdp/pdp run_01 outputs: 2 always-zero security
    assertions in concurrent form.
    """
    # Pre-extract named property bodies so that
    # ``assert property (p_xyz)`` is judged on what ``p_xyz`` actually
    # asserts, not on the property identifier alone.  Without this
    # the unconditional-equality bug pattern wrapped in a named
    # property (a common security-pass shape, e.g. CWE-1295 "debug
    # message" props in cdp / pdp) silently escapes.
    property_bodies = _extract_property_bodies(sva_code)
    decl_line_ranges: Dict[str, Tuple[int, int]] = {}
    if property_bodies:
        in_prop: Optional[str] = None
        prop_start: int = -1
        for idx, ln in enumerate(sva_code.splitlines()):
            if in_prop is None:
                m_ = re.search(r"\bproperty\s+([A-Za-z_]\w*)\s*[;]", ln)
                if m_ and m_.group(1) in property_bodies:
                    in_prop = m_.group(1)
                    prop_start = idx
            if in_prop is not None and "endproperty" in ln:
                decl_line_ranges[in_prop] = (prop_start, idx)
                in_prop = None

    lines = sva_code.splitlines()
    # Two-pass model: first pass marks which line indices to drop
    # (assert refs that fail the check + their associated property
    # declarations); second pass rebuilds `kept` by skipping marked
    # indices.  Avoids the index-misalignment bug where in-loop
    # slicing of `kept` shifts everything that comes after.
    drop_idx: Set[int] = set()
    removed = 0
    param_names: Set[str] = set()
    if facts is not None and hasattr(facts, "parameters"):
        for p in (getattr(facts, "parameters", None) or []):
            n = p.get("name") if isinstance(p, dict) else p
            if n:
                param_names.add(n)
    for line_idx, ln in enumerate(lines):
        s = ln.strip()
        if not s.startswith("assert"):
            continue
        # If the line is `assert property (NAME);`, swap the body
        # for the named property body before applying the check.
        ref_m = _ASSERT_PROP_REF_RE.search(s)
        ref_name: Optional[str] = None
        if ref_m and ref_m.group(1) in property_bodies:
            ref_name = ref_m.group(1)
            body = property_bodies[ref_name]
        else:
            body = _extract_assertion_body(s)
        body = _strip_clock_and_disable(body).strip()
        if body.startswith("(") and body.endswith(")"):
            body = _strip_outer_parens(body)
        # After stripping, the body must NOT contain implication /
        # temporal / logical combinators — otherwise it's conditional
        # and not a bare equality.
        if re.search(r"\|[->=]+|##|\$(past|stable|rose|fell|onehot|isunknown)",
                     body):
            continue
        if "&&" in body or "||" in body:
            continue
        # Match the bare-equality shape on the cleaned body.
        m = re.match(
            r"\s*\(?\s*"
            r"([A-Za-z_]\w*(?:\.\w+)*)"
            r"(?:\[[^\]]+\])?"
            r"\s*==\s*"
            r"(\d+'?[bdhoBDHOxXzZ?]?[\w?]*|\{[^}]+\})"
            r"\s*\)?\s*$",
            body,
        )
        if not m:
            m = _BARE_EQ_BODY_RE.search(body)
            if not m:
                continue
        sig = m.group(1)
        if sig.isupper() and len(sig) > 1:
            continue
        if sig in param_names:
            continue
        # Mark the assert line for deletion.
        drop_idx.add(line_idx)
        # And the associated property declaration (if any).
        if ref_name and ref_name in decl_line_ranges:
            start, end = decl_line_ranges[ref_name]
            for j in range(start, end + 1):
                drop_idx.add(j)
        # Also drop the immediately-preceding `//` comment line that
        # described the dropped assertion.
        prev = line_idx - 1
        while prev >= 0 and not lines[prev].strip():
            prev -= 1
        if prev >= 0 and lines[prev].strip().startswith("//"):
            drop_idx.add(prev)
        removed += 1
        logger.info("Removed unconditional equality: %s", s[:90])

    # Second pass: rebuild kept by skipping marked indices.
    kept: List[str] = [ln for i, ln in enumerate(lines)
                       if i not in drop_idx]
    if removed:
        logger.info("Removed %d unconditional equality assertion(s).", removed)
    return "\n".join(kept)


def relax_strict_handshake_timing(sva_code: str) -> str:
    """No-op.

    Previously rewrote `(valid && !ready) |-> ... ##1 (valid && ready)`
    to `... |-> ##[1:$] (valid && ready)` to "relax" handshake timing.
    That transform was unsafe for two reasons:

      1. It silently changed the SEMANTICS of the emitted assertion
         (a real bug if the protocol genuinely required 1-cycle
         response — e.g., a fixed-latency pipeline stage).
      2. The matcher was case-specific (only 2-atom conjuncts,
         no hierarchical references, no bit-selects).

    We now leave such assertions alone.  If the consequent timing is
    genuinely wrong, the lint-feedback loop and the formal/simulation
    check will surface the violation; we should not make that decision
    silently here.
    """
    return sva_code


# ---------------------------------------------------------------------------
# Remove trivially true assertions
# ---------------------------------------------------------------------------

def remove_trivial_assertions(sva_code: str) -> str:
    """
    Remove assertions that are trivially true and add no verification value.

    Detects and removes:
    1. $bits() checks — $bits(signal) is a compile-time constant, always true.
    2. Identical ternary branches — (cond ? X : X) is always X, trivially true.
    3. Width-mismatched literals — e.g., 3'b0111 (4 bits in 3-bit literal).
    4. Empty assertion stubs — comment-only lines with no assert body.
    """
    lines = sva_code.splitlines()
    kept_lines: List[str] = []
    removed = 0

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        # Skip orphan comment stubs. Two patterns:
        #   a) Prefixed stubs: "// Assertion N:", "// Check 3 -", etc.
        #   b) Free-form descriptive comments ("// Structural check: …",
        #      "// New protocol checks …") with no assert within the next
        #      few non-blank lines — the LLM emitted a header but no body.
        # In both cases: if the next non-blank line is another comment or
        # EOF (not an `assert`), drop this comment.
        is_prefixed_stub = bool(re.match(
            r'^\s*//\s*(?:Assertion|Check|Test|Verify|Property)\s*\d*\s*[:.\-—]',
            stripped,
            re.IGNORECASE,
        ))
        is_plain_comment = stripped.startswith("//")
        if is_prefixed_stub or is_plain_comment:
            # Look ahead: if next non-blank line is not an assert, this
            # is an empty stub.
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines) or not lines[j].strip().startswith("assert"):
                removed += 1
                logger.debug("Removed empty stub: %s", stripped[:60])
                i += 1  # skip this comment line
                continue

        if stripped.startswith("assert") or "assert" in stripped.lower():
            remove = False
            reason = ""

            # 1. $bits() checks — compile-time constant.
            if re.search(r'\$bits\s*\(\s*\w+\s*\)\s*==\s*\d+', stripped):
                remove = True
                reason = "$bits() is compile-time constant"

            # 2. Identical ternary branches: (cond ? X : X)
            ternary = re.search(r'\?\s*\((.+?)\)\s*:\s*\((.+?)\)', stripped)
            if ternary and ternary.group(1).strip() == ternary.group(2).strip():
                remove = True
                reason = "identical ternary branches"

            # Also catch simpler form: cond ? val : val
            ternary2 = re.search(r'\?\s*(\S+)\s*:\s*(\S+)\)', stripped)
            if ternary2 and ternary2.group(1).strip().rstrip(')') == ternary2.group(2).strip().rstrip(')'):
                val1 = ternary2.group(1).strip().rstrip(')')
                val2 = ternary2.group(2).strip().rstrip(')')
                if val1 == val2:
                    remove = True
                    reason = "identical ternary branches"

            # 3. Width-mismatched literals: N'bXXX where len(XXX) > N
            for m in re.finditer(r"(\d+)'b([01]+)", stripped):
                width = int(m.group(1))
                bits = m.group(2)
                if len(bits) > width:
                    remove = True
                    reason = f"width mismatch: {m.group(0)} ({len(bits)} bits in {width}-bit literal)"
                    break

            if remove:
                removed += 1
                logger.info("Removed trivial assertion (%s): %s", reason, stripped[:80])
                # Also remove preceding comment.
                if kept_lines and kept_lines[-1].strip().startswith("//"):
                    kept_lines.pop()
                i += 1
                continue

        kept_lines.append(lines[i])
        i += 1

    if removed:
        logger.info("Removed %d trivially true assertion(s).", removed)
    return "\n".join(kept_lines)


# ---------------------------------------------------------------------------
# Verify (signal == constant) pairs against RTL
# ---------------------------------------------------------------------------

def verify_constant_signal_pairs(sva_code: str, facts: RTLFacts) -> str:
    """
    Drop assertions where ``(signal == constant)`` doesn't match the RTL.

    For each assertion equality of the form ``X == CONST`` where CONST is a
    distinctive multi-bit literal (e.g., ``32'h55005500``), check whether
    the RTL ever assigns CONST to X. If CONST is assigned to a different
    signal Y in the RTL (and only Y), the LLM has likely confused X with Y,
    so the assertion is dropped.

    This is design-agnostic — it uses ``facts.constant_signal_pairs``
    extracted via pyslang.

    Parameters
    ----------
    sva_code : str
    facts : RTLFacts
        Pre-extracted RTL facts (must include ``constant_signal_pairs``).

    Returns
    -------
    str
    """
    rtl_pairs = facts.constant_signal_pairs
    if not rtl_pairs:
        return sva_code

    logger.info(
        "Constant-signal verification: %d distinct literals from RTL.",
        len(rtl_pairs),
    )

    # Match (signal == literal) anywhere in the assertion body.
    _LITERAL_RE = r"\d{2,}'[bhdo][0-9a-fA-F_]+|\d+'h[0-9a-fA-F_]{2,}"
    _PAIR_RE = re.compile(
        r'\(?\s*(\w+(?:\[[^\]]+\])?)\s*==\s*(' + _LITERAL_RE + r')'
    )

    lines = sva_code.splitlines()
    kept: List[str] = []
    removed = 0

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("assert"):
            kept.append(line)
            continue

        # Strip the else $error clause to avoid matching in error msgs.
        body = re.split(r'\belse\s+\$\w+\s*\(', stripped, maxsplit=1)[0]

        bad = False
        for m in _PAIR_RE.finditer(body):
            sig_full = m.group(1)
            literal = m.group(2)
            # Strip the FIRST bit-select to get the base signal name —
            # use a non-greedy match so we don't gobble across multiple
            # `[...]` segments in lines like
            # `(sig[3] == 32'h... && other[5] == 32'h...)`.
            sig_base = re.sub(r'\[[^\]]*\]', '', sig_full, count=1)

            if literal in rtl_pairs:
                rtl_sigs = rtl_pairs[literal]
                # Only flag when the RTL maps this constant to exactly ONE
                # signal (avoids false positives when a constant is shared).
                if len(rtl_sigs) == 1 and sig_base not in rtl_sigs:
                    bad = True
                    expected = next(iter(rtl_sigs))
                    logger.info(
                        "Constant verification: %s == %s but RTL has %s == %s",
                        sig_full, literal, expected, literal,
                    )
                    break

        if bad:
            removed += 1
            if kept and kept[-1].strip().startswith("//"):
                kept.pop()
            continue
        kept.append(line)

    if removed:
        logger.info(
            "Constant-signal verification: removed %d mismatched assertion(s).",
            removed,
        )
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Validate signal widths
# ---------------------------------------------------------------------------

# Verilog literal: optional width, base char, payload.
# Captures: (width or '', base, payload)
_LITERAL_PARSE_RE = re.compile(r"(\d+)?'([bhdo])([0-9a-fA-F_]+)")


def _literal_width(literal: str) -> Optional[int]:
    """
    Parse a Verilog literal and return its declared width, or None if no
    explicit width is given (e.g., ``'b0`` has no width).
    """
    m = _LITERAL_PARSE_RE.match(literal.strip())
    if not m or not m.group(1):
        return None
    return int(m.group(1))


# Match (signal[bit_select] == LITERAL) or (signal == LITERAL) anywhere
# in an assertion body.
_WIDTH_CHECK_RE = re.compile(
    r'(\w+)(\[([^\]]+)\])?\s*==\s*(\d+\'[bhdo][0-9a-fA-F_]+)'
)


def validate_signal_widths(sva_code: str, facts: RTLFacts) -> str:
    """
    Drop assertions with width mismatches against ``facts.signal_widths``.

    Catches three categories:

    1. **Out-of-range bit selects:** ``sig[a:b]`` where ``a`` or ``b`` is
       beyond the declared width. Example: ``out_data[16:17]`` when
       ``out_data`` is 17 bits (max index 16, so ``[16:17]`` is invalid
       both in range syntax and out of bounds).

    2. **Bit-select index too high:** ``sig[N]`` where ``N >= width(sig)``.

    3. **Width-mismatched literal comparison:** ``sig == 32'h...`` where
       ``sig`` has fewer bits than the literal.

    Conservative — if width info is missing for a signal, the assertion is
    kept. Bit selects with non-numeric indices (e.g., parameter names) are
    skipped.
    """
    widths = facts.signal_widths
    if not widths:
        return sva_code

    lines = sva_code.splitlines()
    kept: List[str] = []
    removed = 0

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("assert"):
            kept.append(line)
            continue

        # Strip the else $error clause to avoid matching in messages.
        body = re.split(r'\belse\s+\$\w+\s*\(', stripped, maxsplit=1)[0]

        bad = False
        bad_reason = ""

        for m in _WIDTH_CHECK_RE.finditer(body):
            sig_name = m.group(1)
            bit_sel = m.group(3)  # text inside [] or None
            literal = m.group(4)

            sig_width = widths.get(sig_name)
            if sig_width is None:
                continue  # unknown — be conservative, skip

            # Check 1: bit-select range like sig[a:b].  Verilog allows
            # both `[msb:lsb]` (the common downto form) and `[lsb:msb]`
            # (the unpacked-array form).  We accept either, but flag a
            # range whose extremes exceed the signal's declared width.
            if bit_sel and ':' in bit_sel:
                parts = bit_sel.split(':')
                try:
                    a = int(parts[0].strip())
                    b = int(parts[1].strip())
                except ValueError:
                    continue  # non-numeric (parameter expr), skip
                hi, lo = max(a, b), min(a, b)
                if hi >= sig_width:
                    bad = True
                    bad_reason = (
                        f"out-of-range slice {sig_name}[{a}:{b}] on "
                        f"{sig_width}-bit signal (max valid index "
                        f"is {sig_width - 1})"
                    )
                    break

            # Check 2: single-bit index too high
            elif bit_sel:
                try:
                    idx = int(bit_sel.strip())
                except ValueError:
                    continue
                if idx >= sig_width:
                    bad = True
                    bad_reason = (
                        f"out-of-range index {sig_name}[{idx}] on "
                        f"{sig_width}-bit signal"
                    )
                    break

            # Check 3: width-mismatched literal compared with whole signal
            else:
                lit_w = _literal_width(literal)
                if lit_w is not None and lit_w > sig_width:
                    bad = True
                    bad_reason = (
                        f"literal width mismatch: {sig_name} "
                        f"({sig_width}-bit) == {literal} ({lit_w}-bit)"
                    )
                    break

        if bad:
            removed += 1
            logger.info("Width validation: dropped — %s", bad_reason)
            if kept and kept[-1].strip().startswith("//"):
                kept.pop()
            continue

        kept.append(line)

    if removed:
        logger.info(
            "Width validation: removed %d width-mismatched assertion(s).",
            removed,
        )
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Validate reset values
# ---------------------------------------------------------------------------

# Match `sig ==` (we then parse the RHS by hand to handle nested braces).
_RESET_EQ_LHS_RE = re.compile(r'(\w+)(?:\[[^\]]+\])?\s*==\s*')


def _read_value(text: str, start: int) -> Tuple[Optional[str], int]:
    """
    Parse a value expression starting at ``text[start]``. Handles three forms:

    - Verilog literal: ``32'h55005500``, ``1'b0``, ``2'b01``, etc.
    - Brace expression: ``{...}`` with proper nesting (``{N{1'b0}}``)
    - Bare identifier: ``cfg_is_int8``

    Returns ``(value_text, end_index)`` or ``(None, start)`` if no value
    is found.
    """
    # Skip leading whitespace.
    i = start
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text):
        return None, start

    ch = text[i]

    # Brace expression: track nesting depth.
    if ch == '{':
        depth = 0
        j = i
        while j < len(text):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
                if depth == 0:
                    return text[i:j + 1], j + 1
            j += 1
        return None, start  # unbalanced braces

    # Verilog literal: optional width digit(s), apostrophe, base, payload.
    m = re.match(r"(\d+'[bhdoBHDO][0-9a-fA-F_xXzZ?]+)", text[i:])
    if m:
        return m.group(1), i + m.end()

    # Bare identifier (single word).
    m = re.match(r"(\w+)", text[i:])
    if m:
        return m.group(1), i + m.end()

    return None, start


def remove_bus_slice_restatements(sva_code: str) -> str:
    """
    Drop assertions that are pure bus-slice decompositions.

    These are AST-generated passthrough assertions that restate RTL bus
    slicing as SVA, e.g.:

        assert (calc_addr[4:0] == accu_ctrl_pd[4:0]);
        assert (calc_data_16_0 == calc_data_all[43:0]);

    Both sides are slices of other signals with identical bit widths.
    These add no verification value — they're syntax-level restatements
    that would only fail if the synthesizer itself is broken.

    The filter is conservative: it drops ONLY when the entire assertion
    body matches the pattern ``LHS == RHS`` where:
    - RHS is a bit-select or bare signal
    - If both sides are slices, their widths match
    - No temporal operators, no multi-term logic

    Immediate assertions only. Concurrent (``assert property``) are
    preserved (they encode protocol properties, not restatements).
    """
    # Match the body: "LHS == RHS" optionally wrapped in parens,
    # followed by an else $error(...) clause and semicolon.
    # LHS can be: `ident` or `ident[range]`
    # RHS can be: `ident`, `ident[range]`, or (ident)
    body_re = re.compile(
        r"""^\s*assert\s*\(?\s*
            (?P<lhs>\w+(?:\[[^\]]+\])?)        # LHS: sig or sig[range]
            \s*==\s*
            \(?\s*
            (?P<rhs>\w+(?:\[[^\]]+\])?)        # RHS: sig or sig[range]
            \s*\)?\s*\)?\s*
            (?:else\s+\$\w+\([^)]*\))?\s*;\s*$
        """,
        re.VERBOSE,
    )

    def _slice_width(expr: str) -> Optional[int]:
        """Return the bit-width of a slice expression, or None if bare."""
        m = re.match(r"\w+\[(\d+)\s*:\s*(\d+)\]$", expr.strip())
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return abs(a - b) + 1
        m = re.match(r"\w+\[\d+\]$", expr.strip())
        if m:
            return 1
        # Bare identifier — width unknown.
        return None

    lines = sva_code.splitlines()
    kept: List[str] = []
    removed = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        m = body_re.match(line)
        if m:
            lhs = m.group("lhs")
            rhs = m.group("rhs")
            # Only drop if BOTH sides are slices with matching widths,
            # OR one side is a slice and the other is a bare signal
            # (bus-to-slice passthrough). Bare-to-bare is already handled
            # by the AST step 2 filter.
            lw = _slice_width(lhs)
            rw = _slice_width(rhs)
            is_pure_decomposition = False
            if lw is not None and rw is not None and lw == rw:
                is_pure_decomposition = True
            elif (lw is None and rw is not None) or (lw is not None and rw is None):
                # Bare signal vs slice — passthrough
                is_pure_decomposition = True

            if is_pure_decomposition:
                removed += 1
                # Drop preceding comment
                if kept and kept[-1].strip().startswith("//"):
                    kept.pop()
                i += 1
                continue

        kept.append(line)
        i += 1

    if removed:
        logger.info(
            "Bus-slice restatement filter: removed %d assertion(s).",
            removed,
        )
    return "\n".join(kept)


def validate_out_of_scope(
    sva_code: str,
    patterns: Optional[List[str]] = None,
    signal_names: Optional[Set[str]] = None,
) -> str:
    """
    Drop assertions that reference out-of-scope signals.

    Out-of-scope signals are typically clock-gating / DFT / power signals
    that should not appear in functional assertions. The set is built
    from two sources combined:

    1. ``signal_names`` — explicit set of signal names (typically from
       ``RTLFacts.out_of_scope_signals``, populated by structural
       detection). Match is exact.
    2. ``patterns`` — regex patterns applied to signal names. Useful as
       a manual override / safety-net for signals the structural detector
       misses.

    An assertion is dropped if ANY of its extracted signal identifiers
    matches the structural set OR any pattern.

    Parameters
    ----------
    sva_code : str
        Raw SVA code string.
    patterns : list of str, optional
        Regex patterns applied to signal names.
    signal_names : set of str, optional
        Exact signal names to match against.

    Returns
    -------
    str
        SVA code with out-of-scope assertions removed.
    """
    patterns = patterns or []
    signal_names = signal_names or set()

    if not patterns and not signal_names:
        return sva_code

    try:
        compiled = [re.compile(p) for p in patterns]
    except re.error as exc:
        logger.warning(
            "validate_out_of_scope: invalid regex pattern, skipping: %s",
            exc,
        )
        compiled = []

    lines = sva_code.splitlines()
    kept: List[str] = []
    removed = 0
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("assert"):
            sigs = _extract_signal_names(stripped)
            hit: Optional[str] = None
            for sig in sigs:
                if sig in signal_names:
                    hit = sig
                    break
                for pat in compiled:
                    if pat.search(sig):
                        hit = sig
                        break
                if hit:
                    break
            if hit:
                removed += 1
                logger.info(
                    "Out-of-scope: dropped assertion (matched '%s'): %s",
                    hit, stripped[:80],
                )
                # Drop the preceding // comment if present.
                if kept and kept[-1].strip().startswith("//"):
                    kept.pop()
                i += 1
                continue
        kept.append(lines[i])
        i += 1

    if removed:
        logger.info(
            "Out-of-scope validation: removed %d assertion(s) "
            "referencing forbidden signals.",
            removed,
        )
    return "\n".join(kept)


def validate_reset_values(sva_code: str, facts: RTLFacts) -> str:
    """
    Drop reset-value assertions where the asserted value (or asserted
    reset signal) contradicts the actual RTL reset extracted from
    sequential always blocks.

    Multi-clock-domain aware: ``facts.reset_signals`` may contain multiple
    reset names. Each register in ``facts.reset_values`` is tagged with
    its driving reset signal. The validator checks both the value AND
    that the assertion is using the right reset signal for that register.

    Detection:
    - Skip the assertion if its body doesn't reference any known reset
      signal (uses ``facts.reset_signals``, no regex naming heuristic).
    - Split on the first implication operator (``|->``, ``|=>``, ``||``)
      and analyze only the **consequent**. This avoids matching antecedent
      comparisons like ``(cfg_reg_en == 1'b1) |-> sig == val`` as reset
      claims.
    - Skip if the consequent contains another implication operator
      (complex nested temporal property — too risky to interpret).
    - For each ``sig == VALUE`` in the consequent, look up ``sig`` in
      ``facts.reset_values``. The lookup returns ``(reset_signal, value)``.
      Drop the assertion if either:
        1. The asserted value differs from the actual value
           (e.g., LLM says ``cfg_is_int16 == 1'b0`` but RTL says ``1'b1``)
        2. The asserted reset signal differs from the actual reset signal
           (e.g., LLM says ``!prstn |-> X == ...`` but X is reset by
           ``!nvdla_core_rstn`` — wrong clock domain)

    Conservative:
    - If a register isn't in ``reset_values``, the assertion is kept.
    - Only flags when BOTH the asserted and actual values are explicit
      Verilog literals (skips named constants like FSM state enums).
    """
    reset_vals = facts.reset_values
    reset_signals = facts.reset_signals
    if not reset_vals or not reset_signals:
        return sva_code

    def _norm(val: str) -> str:
        s = val.strip()
        while s.startswith('(') and s.endswith(')'):
            s = s[1:-1].strip()
        return re.sub(r'\s+', '', s)

    def _is_literal_or_replication(val: str) -> bool:
        s = val.strip()
        if re.match(r"^\d+'[bhdoBHDO][0-9a-fA-F_xXzZ?]+$", s):
            return True
        if s.startswith('{') and s.endswith('}'):
            inner = s[1:-1]
            return bool(re.match(r"^[\s\d'a-fA-FxXzZbBhHdDoO_{}]+$", inner))
        return False

    # Build a regex that matches `!<reset>` for any of the known resets.
    # Each name is escaped so special chars in signal names don't break.
    reset_alt = '|'.join(re.escape(r) for r in reset_signals)
    reset_re = re.compile(rf'!\s*({reset_alt})\b')

    lines = sva_code.splitlines()
    kept: List[str] = []
    removed = 0
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped.startswith("assert"):
            kept.append(line)
            i += 1
            continue

        # Strip the else $error clause.
        body = re.split(r'\belse\s+\$\w+\s*\(', stripped, maxsplit=1)[0]

        # Identify the asserted reset signal — must be one of the known
        # reset signals from facts. If multiple reset signals appear in
        # the body, take the first match (the predicate-side reset).
        rst_match = reset_re.search(body)
        if not rst_match:
            kept.append(line)
            i += 1
            continue
        asserted_reset = rst_match.group(1)

        # Extract the consequent — everything after the implication.
        consequent = body
        for op in ('|->', '|=>', '||'):
            if op in body:
                consequent = body.split(op, 1)[1]
                break

        # Conservative: skip complex nested temporal properties.
        if '|->' in consequent or '|=>' in consequent:
            kept.append(line)
            i += 1
            continue

        bad = False
        bad_reason = ""

        for m in _RESET_EQ_LHS_RE.finditer(consequent):
            sig_base = m.group(1)
            asserted_val, _end = _read_value(consequent, m.end())
            if asserted_val is None:
                continue

            entry = reset_vals.get(sig_base)
            if entry is None:
                continue
            actual_reset, actual_val = entry

            # Check 1: wrong reset signal (MCD error).
            # Only fires if RTL knows which reset clears this register
            # AND the assertion uses a different reset.
            if actual_reset is not None and actual_reset != asserted_reset:
                bad = True
                bad_reason = (
                    f"{sig_base} reset signal: asserted=!{asserted_reset} "
                    f"actual=!{actual_reset}"
                )
                break

            # Check 2: wrong reset value.
            # Only flag when both sides are explicit literals.
            if not (_is_literal_or_replication(asserted_val)
                    and _is_literal_or_replication(actual_val)):
                continue

            if _norm(asserted_val) != _norm(actual_val):
                bad = True
                bad_reason = (
                    f"{sig_base} reset value: asserted="
                    f"{asserted_val.strip()} actual={actual_val.strip()}"
                )
                break

        if bad:
            removed += 1
            logger.info("Reset value validation: dropped — %s", bad_reason)
            if kept and kept[-1].strip().startswith("//"):
                kept.pop()
            i += 1
            continue

        kept.append(line)
        i += 1

    if removed:
        logger.info(
            "Reset value validation: removed %d wrong-reset-value assertion(s).",
            removed,
        )
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# RTL data-flow check (case selector mismatch)
# ---------------------------------------------------------------------------

def _expression_identifiers(expr: str) -> Set[str]:
    """Extract identifier-like tokens from an expression, filtering literals."""
    # Remove Verilog literals and operators.
    clean = re.sub(r"\d+'[bhdo][0-9a-fA-F_]+", '', expr)
    ids = set(re.findall(r'\b([a-zA-Z_]\w*)\b', clean))
    _SV_KEYWORDS = {
        "if", "else", "case", "endcase", "default", "begin", "end",
        "wire", "reg", "logic", "input", "output", "assign", "always",
        "always_comb", "always_ff", "posedge", "negedge",
    }
    return ids - _SV_KEYWORDS


def check_case_selector_mismatch(
    sva_code: str,
    facts: RTLFacts,
) -> str:
    """
    Drop assertions where the LLM uses a raw input but the RTL case
    statement that drives the asserted signal uses a transformed version.

    Detection logic:
    1. Parse each conditional assertion ``!(X == val) || (Y == result)``
    2. Look up Y in the case-selector map → find the selector S that drives Y
    3. Extract identifiers from S → set of signals the case discriminates on
    4. If X is NOT in S's identifier set, BUT X is in the dependency closure
       of one of S's identifiers (i.e., that identifier is derived from X
       via a non-trivial expression), the assertion is suspect.

    Conservative — only fires when:
    - The case selector is non-trivial (contains operators or concatenation)
    - The transformation is non-identity (the dependent expression contains
      operators beyond simple wire passthrough)
    - Both X and the case selector identifiers exist in the RTL

    Parameters
    ----------
    sva_code : str
    facts : RTLFacts
        Pre-extracted RTL facts (must include ``signal_definitions`` and
        ``case_selectors``).

    Returns
    -------
    str
        SVA code with mismatched assertions removed.
    """
    signal_defs = facts.signal_definitions
    case_selectors = facts.case_selectors

    if not case_selectors:
        return sva_code

    logger.info(
        "Data-flow check: %d signal defs, %d case-driven signals.",
        len(signal_defs), len(case_selectors),
    )

    lines = sva_code.splitlines()
    kept: List[str] = []
    removed = 0

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("assert"):
            kept.append(line)
            continue

        parsed = _parse_assertion_eq(line)
        if not parsed:
            kept.append(line)
            continue

        ant_sigs, lhs_sig, _rhs = parsed
        if not ant_sigs:
            kept.append(line)
            continue

        # Strip bit selects from lhs.
        lhs_base = re.sub(r'\[.*\]', '', lhs_sig)

        # Look up the asserted signal in case-driven signals.
        if lhs_base not in case_selectors:
            kept.append(line)
            continue

        selectors = case_selectors[lhs_base]
        # Get the union of identifiers used in any selector.
        selector_ids: Set[str] = set()
        nontrivial_selector = False
        for sel in selectors:
            sel_ids = _expression_identifiers(sel)
            selector_ids |= sel_ids
            # Non-trivial = contains concatenation, operator, or multi-id.
            if len(sel_ids) > 1 or '{' in sel or any(
                op in sel for op in ['^', '&', '|', '+', '-', '~']
            ):
                nontrivial_selector = True

        if not nontrivial_selector:
            kept.append(line)
            continue

        # Check if any antecedent signal is NOT in the selector identifiers.
        # If the LLM uses signal X that isn't part of the case selector,
        # but X is referenced in the definition of one of the selector ids,
        # the LLM has likely confused the raw input with the transformed
        # version.
        suspect = False
        for ant_sig in ant_sigs:
            if ant_sig in selector_ids:
                continue  # Direct use — fine
            # Check if ant_sig appears in any selector identifier's definition.
            for sel_id in selector_ids:
                defn = signal_defs.get(sel_id)
                if not defn:
                    continue
                if re.search(r'\b' + re.escape(ant_sig) + r'\b', defn):
                    # ant_sig is part of how sel_id is computed.
                    # Check if the transformation is non-trivial.
                    # Trivial = direct passthrough (defn is just ant_sig).
                    defn_clean = defn.strip()
                    if defn_clean != ant_sig and any(
                        op in defn_clean for op in ['^', '&', '|', '+', '-', '~', '?', '{']
                    ):
                        suspect = True
                        logger.info(
                            "Data-flow mismatch: assertion uses '%s' but "
                            "case selector for '%s' is '%s' (where '%s' = '%s')",
                            ant_sig, lhs_base, ', '.join(selectors), sel_id, defn,
                        )
                        break
            if suspect:
                break

        if suspect:
            removed += 1
            if kept and kept[-1].strip().startswith("//"):
                kept.pop()
            continue

        kept.append(line)

    if removed:
        logger.info(
            "Data-flow check: removed %d case-selector mismatched assertion(s).",
            removed,
        )
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Signal existence validation
# ---------------------------------------------------------------------------

def _extract_signal_names(assertion: str) -> Set[str]:
    """
    Extract signal-like identifiers from an assertion string.

    Skips SystemVerilog keywords, system functions ($bits, $error, etc.),
    numeric literals, and common operators.

    Only analyzes the assertion **body** (before ``else $error(...)``)
    to avoid false positives from English words in error messages.
    Also handles the case where the LLM emits an unterminated string.
    """
    # Strip the else $error(...) clause entirely. This is the most robust
    # way to avoid leaking error-message words as fake signal names —
    # works even when the string is unterminated or has escaped quotes.
    clean = re.split(r'\belse\s+\$\w+\s*\(', assertion, maxsplit=1)[0]
    # Also strip any string literals that may remain (e.g., in $past args).
    clean = re.sub(r'"[^"]*"', '', clean)
    # Remove system functions ($bits, $past, etc.).
    clean = re.sub(r'\$\w+', '', clean)

    # Find all identifiers (word-like tokens).
    identifiers = set(re.findall(r'\b([a-zA-Z_]\w*)\b', clean))

    # Remove SV keywords and assertion syntax.
    _SV_KEYWORDS = {
        "assert", "property", "else", "error", "if", "iff", "disable",
        "posedge", "negedge", "or", "and", "not", "begin", "end",
        "always", "always_comb", "always_ff", "module", "endmodule",
        "input", "output", "wire", "reg", "logic", "assign",
        "case", "endcase", "default", "for", "generate",
        # Common assertion keywords
        "sequence", "endsequence", "cover", "assume",
    }
    identifiers -= _SV_KEYWORDS

    # Remove bit-width prefixes that look like identifiers (e.g., "1" from "1'b0").
    identifiers = {s for s in identifiers if not s.isdigit()}

    # Remove bit-literal fragments (e.g., "b011", "h0", "hff", "b0", "b1").
    # These come from Verilog literals like 3'b011, 16'hff.
    identifiers = {
        s for s in identifiers
        if not re.match(r'^[bhdo][0-9a-fA-F_]+$', s)
    }

    return identifiers


def validate_signals(
    sva_code: str,
    facts: RTLFacts,
    denylist_path: Optional[str] = None,
) -> str:
    """
    Remove assertions that reference signals not in the design.

    For each assertion, extract all signal-like identifiers and check
    whether they exist in ``facts.all_signals`` (which includes both
    port-level signals from the signal map and internal signals declared
    in the RTL source).

    We use a soft threshold: drop an assertion only if MORE THAN HALF of its
    signal references are unknown. This avoids false positives from signals
    that are legitimately in the design but not extracted by pyslang.

    Parameters
    ----------
    sva_code : str
        Raw SVA code string.
    facts : RTLFacts
        Pre-extracted RTL facts (must include ``all_signals``).
    denylist_path : str, optional
        If provided, every hallucinated signal name found in this call is
        merged into the JSON file at this path (counts incremented). The
        file is created if it doesn't exist. Used by the Stage 2
        hallucination knowledgebase — logging is always-on (cheap), the
        prompt-injection side is gated separately.

    Returns
    -------
    str
        SVA code with hallucinated-signal assertions removed.
    """
    known_signals = facts.all_signals
    if not known_signals:
        logger.debug("No known signals available — skipping signal validation.")
        return sva_code

    logger.info(
        "Signal validation: %d known signals from RTL + signal_map.",
        len(known_signals),
    )

    lines = sva_code.splitlines()
    kept_lines: List[str] = []
    removed = 0
    hallucinations_this_run: Dict[str, int] = {}

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if stripped.startswith("assert"):
            sig_names = _extract_signal_names(stripped)

            if sig_names:
                unknown = sig_names - known_signals
                known_ratio = 1.0 - (len(unknown) / len(sig_names)) if sig_names else 1.0

                if known_ratio < 0.5:
                    removed += 1
                    for name in unknown:
                        hallucinations_this_run[name] = (
                            hallucinations_this_run.get(name, 0) + 1
                        )
                    logger.info(
                        "Signal validation: dropped assertion (%.0f%% unknown signals: %s): %s",
                        (1 - known_ratio) * 100,
                        ", ".join(sorted(unknown)[:5]),
                        stripped[:80],
                    )
                    # Also skip the comment line above if it's the previous kept line.
                    if kept_lines and kept_lines[-1].strip().startswith("//"):
                        kept_lines.pop()
                    i += 1
                    continue

            kept_lines.append(lines[i])
        else:
            kept_lines.append(lines[i])
        i += 1

    if removed:
        logger.info("Signal validation: removed %d assertion(s) with hallucinated signals.", removed)

    # Merge new hallucinations into the persistent denylist file.
    if denylist_path and hallucinations_this_run:
        _merge_hallucination_denylist(denylist_path, hallucinations_this_run)

    return "\n".join(kept_lines)


def _merge_hallucination_denylist(
    path: str,
    new_counts: Dict[str, int],
) -> None:
    """
    Merge ``new_counts`` into the JSON denylist file at ``path``.

    Creates the file (and parent directories) if missing. Existing entries
    have their counts incremented; new names are added with the current
    count. The file format is a flat ``{name: count}`` dict — simple,
    diff-friendly, and easy to inspect by hand.
    """
    import json

    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create denylist dir %s: %s", p.parent, exc)
        return

    existing: Dict[str, int] = {}
    if p.exists():
        try:
            with open(p, "r") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    try:
                        existing[str(k)] = int(v)
                    except (TypeError, ValueError):
                        pass
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read denylist %s: %s", p, exc)

    for name, count in new_counts.items():
        existing[name] = existing.get(name, 0) + int(count)

    try:
        with open(p, "w") as f:
            json.dump(existing, f, indent=2, sort_keys=True)
        logger.info(
            "Hallucination denylist updated: %s (+%d new entries)",
            p, len(new_counts),
        )
    except OSError as exc:
        logger.warning("Could not write denylist %s: %s", p, exc)


# ---------------------------------------------------------------------------
# Semantic deduplication (AST vs LLM)
# ---------------------------------------------------------------------------

def _normalise_assertion_semantic(assertion: str) -> str:
    """
    Produce a canonical form of an assertion for semantic comparison.

    Normalises:
    - Whitespace
    - Error messages (stripped entirely)
    - Comments (stripped)
    - Operator spacing: removes spaces around ==, !=, ||, &&, |->
    - Parenthesis normalisation: strips redundant outer parens
    - Operand ordering: sorts commutative operands (==, !=, &&, ||)
    """
    s = assertion.strip()

    # Remove comment lines.
    s = re.sub(r'//.*$', '', s, flags=re.MULTILINE).strip()

    # Remove error message.
    s = re.sub(r'else\s+\$error\(.*$', '', s).strip()

    # Remove trailing semicolon.
    s = s.rstrip(';').strip()

    # Remove `assert property (` or `assert (` prefix.
    s = re.sub(r'^assert\s+property\s*\(', '(', s)
    s = re.sub(r'^assert\s*\(', '(', s)

    # Remove disable iff clause.
    s = re.sub(r'disable\s+iff\s*\([^)]*\)', '', s)

    # Remove clock edge: @(posedge clk) or @(negedge clk)
    s = re.sub(r'@\s*\(\s*(?:pos|neg)edge\s+\w+\s*\)', '', s)

    # Normalise whitespace.
    s = re.sub(r'\s+', ' ', s).strip()

    # Remove outer parens if the whole thing is wrapped.
    while s.startswith('(') and s.endswith(')'):
        inner = s[1:-1]
        # Check paren balance of inner.
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
            s = inner.strip()
        else:
            break

    # Normalise operator spacing.
    s = re.sub(r'\s*(==|!=|<=|>=|\|\||&&|\|->|\|=>|->)\s*', r' \1 ', s)
    s = re.sub(r'\s+', ' ', s).strip()

    # Remove all parentheses for canonical comparison.
    s = s.replace('(', '').replace(')', '')
    s = re.sub(r'\s+', ' ', s).strip()

    # Normalise negation: "! x" → "!x" (remove space after !).
    s = re.sub(r'!\s+', '!', s)

    # Split into tokens by ||, extract and sort each clause.
    # This makes "!A || B" and "B || !A" equivalent.
    clauses = [c.strip() for c in s.split('||')]
    clauses.sort()
    s = ' || '.join(clauses)

    return s.lower()


def semantic_deduplicate(sva_code: str) -> str:
    """
    Remove assertions that are semantically equivalent to earlier ones.

    Goes beyond string-based deduplication by normalising assertion bodies
    to a canonical form before comparison. This catches cases where the LLM
    regenerates an AST assertion with different formatting, error messages,
    or operand ordering.

    Parameters
    ----------
    sva_code : str
        Raw SVA code string.

    Returns
    -------
    str
        SVA code with semantic duplicates removed (keeps first occurrence).
    """
    lines = sva_code.splitlines()
    seen_sigs: Set[str] = set()
    kept_lines: List[str] = []
    removed = 0

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("assert"):
            sig = _normalise_assertion_semantic(stripped)

            if sig in seen_sigs:
                removed += 1
                logger.debug("Semantic dedup: %s", stripped[:80])
                # Also remove preceding comment.
                if kept_lines and kept_lines[-1].strip().startswith("//"):
                    kept_lines.pop()
                continue
            seen_sigs.add(sig)

        kept_lines.append(line)

    if removed:
        logger.info("Semantic deduplication: removed %d assertion(s).", removed)
    return "\n".join(kept_lines)


# ---------------------------------------------------------------------------
# Subsumption & contradiction detection
# ---------------------------------------------------------------------------

def _split_top_level_or(expr: str) -> Optional[Tuple[str, str]]:
    """
    Find top-level ``||`` in an expression (paren-aware).
    Returns (lhs, rhs) of the first split, or None if no top-level ``||``.
    """
    depth = 0
    i = 0
    while i < len(expr) - 1:
        ch = expr[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '|' and expr[i + 1] == '|' and depth == 0:
            return (expr[:i].strip(), expr[i + 2:].strip())
        i += 1
    return None


def _split_top_level_eq(expr: str) -> Optional[Tuple[str, str]]:
    """
    Find top-level ``==`` in an expression (paren-aware).
    Returns (lhs, rhs) or None.
    """
    depth = 0
    i = 0
    while i < len(expr) - 1:
        ch = expr[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '=' and expr[i + 1] == '=' and depth == 0:
            return (expr[:i].strip(), expr[i + 2:].strip())
        i += 1
    return None


# `_strip_outer_parens` is defined once near the top of this module
# (with the shared expression utilities) and reused by every detector
# below.  An older duplicate definition lived here; it has been removed
# so that any future change to the helper takes effect uniformly across
# all callers.


def _parse_assertion_eq(line: str) -> Optional[Tuple[Set[str], str, str]]:
    """
    Parse an assertion into (antecedent_signals, lhs, rhs) of an equality
    consequent.

    Handles both forms:
    - Implication: ``assert (!(A) || (X == Y)) else $error(...);``
    - Unconditional: ``assert (X == Y) else $error(...);``

    Returns None if the assertion isn't an equality form.
    """
    s = line.strip()
    if not s.startswith("assert") or s.startswith("assert property"):
        return None
    m = re.match(r'assert\s*\((.+)\)\s*else\s+\$error', s)
    if not m:
        return None

    body = _strip_outer_parens(m.group(1).strip())

    # Check for implication via top-level ||
    impl = _split_top_level_or(body)
    if impl:
        ant_part, conseq_part = impl
        ant_sigs = _extract_signal_names(ant_part)
        conseq = _strip_outer_parens(conseq_part)
    else:
        ant_sigs = set()  # unconditional
        conseq = body

    eq = _split_top_level_eq(conseq)
    if not eq:
        return None
    lhs, rhs = eq
    lhs_norm = re.sub(r'\s+', '', _strip_outer_parens(lhs))
    rhs_norm = re.sub(r'\s+', '', _strip_outer_parens(rhs))

    return (ant_sigs, lhs_norm, rhs_norm)


def remove_subsumed_and_contradicting(sva_code: str) -> str:
    """
    Remove assertions that are subsumed by or contradict stronger assertions.

    Three rules:

    **Rule A — Redundancy:** If a conditional assertion has the same
    ``(lhs, rhs)`` as an unconditional assertion, the conditional is
    implied by the unconditional → drop conditional.

    **Rule B — Over-generalization:** If two conditional assertions have
    the same ``(lhs, rhs)`` and one's antecedent signal set is a strict
    subset of the other's, the broader one is making a stronger claim
    that the narrower (more specific) one doesn't support. The narrower
    versions came from the AST extraction (per-value enumeration), so
    the broader version is the over-generalization → drop the broader.

    **Rule C — Contradiction:** If a conditional assertion claims
    ``lhs == rhs_b`` and an unconditional assertion claims
    ``lhs == rhs_a`` (different rhs), the conditional contradicts the
    unconditional → drop the conditional.

    All three rules are design-agnostic — they work by structural
    comparison of assertion forms, not by knowing signal names or design
    semantics.
    """
    lines = sva_code.splitlines()
    parsed: List[Tuple[int, Set[str], str, str]] = []

    for i, line in enumerate(lines):
        result = _parse_assertion_eq(line)
        if result:
            ant_sigs, lhs, rhs = result
            parsed.append((i, ant_sigs, lhs, rhs))

    # Build maps from unconditional assertions.
    uncond_pairs: Set[Tuple[str, str]] = set()
    uncond_lhs_to_rhs: Dict[str, Set[str]] = {}
    for line_idx, sigs, lhs, rhs in parsed:
        if not sigs:  # unconditional
            uncond_pairs.add((lhs, rhs))
            uncond_lhs_to_rhs.setdefault(lhs, set()).add(rhs)

    to_remove: Set[int] = set()

    # Rule A: conditional matching unconditional (lhs, rhs) → redundant.
    for line_idx, sigs, lhs, rhs in parsed:
        if sigs and (lhs, rhs) in uncond_pairs:
            to_remove.add(line_idx)
            logger.debug("Subsumed (redundant): line %d", line_idx)

    # Rule C: conditional with same lhs but different rhs → contradiction.
    for line_idx, sigs, lhs, rhs in parsed:
        if (sigs and lhs in uncond_lhs_to_rhs
                and rhs not in uncond_lhs_to_rhs[lhs]):
            to_remove.add(line_idx)
            logger.debug("Contradicting (cond vs uncond): line %d", line_idx)

    # Rule C2: unconditional vs unconditional contradiction.
    # Two unconditional assertions with same lhs but different rhs:
    # the first one wins (AST runs before LLM), drop the later ones.
    seen_uncond_lhs: Dict[str, Tuple[int, str]] = {}
    for line_idx, sigs, lhs, rhs in parsed:
        if sigs:
            continue
        if line_idx in to_remove:
            continue
        if lhs in seen_uncond_lhs:
            prev_idx, prev_rhs = seen_uncond_lhs[lhs]
            if prev_rhs != rhs:
                to_remove.add(line_idx)
                logger.debug(
                    "Contradicting (uncond vs uncond): line %d (rhs=%s) "
                    "conflicts with line %d (rhs=%s)",
                    line_idx, rhs, prev_idx, prev_rhs,
                )
        else:
            seen_uncond_lhs[lhs] = (line_idx, rhs)

    # Rule B: over-generalization within conditional groups.
    by_pair: Dict[Tuple[str, str], List[Tuple[int, Set[str]]]] = {}
    for line_idx, sigs, lhs, rhs in parsed:
        if sigs:
            by_pair.setdefault((lhs, rhs), []).append((line_idx, sigs))

    for pair, group in by_pair.items():
        if pair in uncond_pairs:
            continue  # handled by Rule A
        if len(group) < 2:
            continue
        for line_idx, sigs in group:
            for other_idx, other_sigs in group:
                if other_idx != line_idx and sigs < other_sigs:
                    to_remove.add(line_idx)
                    logger.debug("Over-generalized: line %d", line_idx)
                    break

    # Apply removal.
    kept: List[str] = []
    removed = 0
    for i, line in enumerate(lines):
        if i in to_remove:
            removed += 1
            # Also remove preceding comment.
            if kept and kept[-1].strip().startswith("//"):
                kept.pop()
            continue
        kept.append(line)

    if removed:
        logger.info(
            "Removed %d subsumed/contradicting assertion(s).", removed
        )
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# LLM self-review pass
# ---------------------------------------------------------------------------

def llm_self_review(
    agent: "SVAAgent",
    sva_code: str,
    signal_map: Dict[str, Any],
) -> str:
    """
    Ask the LLM to review and fix its own generated assertions.

    The LLM receives:
    - The full set of assertions (both AST and LLM-generated)
    - The signal map (valid signal names and widths)
    - Instructions to check each assertion for correctness

    The LLM is asked to:
    1. Remove assertions that duplicate what AST already covers
    2. Fix assertions with wrong signal names or widths
    3. Fix logically incorrect assertions
    4. Keep only assertions that add genuine value

    Parameters
    ----------
    agent : SVAAgent
        The agent instance (model already loaded).
    sva_code : str
        Current SVA code to review.
    signal_map : dict
        Signal map from design_info.

    Returns
    -------
    str
        Reviewed and corrected SVA code.
    """
    if agent.backend is None:
        logger.debug("No LLM backend — skipping self-review.")
        return sva_code

    # Build a concise signal reference (ports + internal RTL identifiers).
    sig_ref_lines = []
    for name, info in sorted(signal_map.items()):
        if "." not in name:  # Top-level signals only.
            w = info.get("width", "?")
            d = info.get("direction", "?")
            sig_ref_lines.append(f"  {name}: {d}, {w}-bit")

    # Also extract internal signal names from RTL so the LLM knows they exist.
    config = getattr(agent, 'config', None)
    rtl_dir = getattr(config, 'rtl_dir', '') if config else ''
    if rtl_dir:
        rtl_ids = _extract_rtl_identifiers(rtl_dir)
        # Only add identifiers not already in the signal map.
        port_names = set(signal_map.keys())
        internal_ids = sorted(rtl_ids - port_names)
        if internal_ids:
            sig_ref_lines.append("")
            sig_ref_lines.append("  Internal signals (from RTL, no width info):")
            # Cap to avoid token bloat.
            for name in internal_ids[:80]:
                sig_ref_lines.append(f"  {name}: internal")

    sig_ref = "\n".join(sig_ref_lines[:150])  # Cap total lines.

    # Split assertions into batches to avoid token truncation.
    assertion_entries = split_assertions(sva_code)
    if not assertion_entries:
        return sva_code

    batch_size = 15  # ~15 assertions per batch to stay within token limits
    batches = [
        assertion_entries[i:i + batch_size]
        for i in range(0, len(assertion_entries), batch_size)
    ]

    logger.info(
        "=== LLM self-review pass: %d assertions in %d batch(es) ===",
        len(assertion_entries), len(batches),
    )
    agent._current_phase = "self_review"

    all_reviewed: List[str] = []

    for batch_idx, batch in enumerate(batches, 1):
        # Format this batch as SVA text.
        batch_sva = reassemble_assertions(batch)

        review_prompt = (
            "You are reviewing SystemVerilog assertions for correctness.\n\n"
            "SIGNALS in this design:\n"
            f"{sig_ref}\n\n"
            "NOTE: The signal list above includes both port-level AND internal "
            "signals. Assertions may reference internal signals (wires, regs) "
            "that are valid in the design even if they are not top-level ports. "
            "Do NOT remove assertions just because a signal is listed as "
            "'internal' — those signals exist in the RTL.\n\n"
            "ASSERTIONS TO REVIEW:\n"
            "```systemverilog\n"
            f"{batch_sva}\n"
            "```\n\n"
            "For EACH assertion, check:\n"
            "1. Is the logic correct? (e.g., the condition actually tests "
            "what the comment says)\n"
            "2. Is it a useful check? Remove trivially true assertions "
            "(e.g., asserting a signal equals itself, or $bits() checks "
            "which are compile-time constants).\n"
            "3. Are temporal operators correct? (e.g., |=> is next-cycle, "
            "don't use it for combinational logic checks)\n\n"
            "Output the CORRECTED assertions in a ```systemverilog``` fence.\n"
            "Keep all good assertions unchanged. Fix broken ones. "
            "Remove unfixable ones.\n"
            "Preserve the original comments above each assertion.\n"
            "Do NOT add new assertions — only fix or remove existing ones.\n"
            f"End with <<SVA_COMPLETE>>."
        )

        messages = [
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": review_prompt},
        ]

        logger.info("--- Self-review batch %d/%d (%d assertions) ---",
                     batch_idx, len(batches), len(batch))

        max_steps = 5
        final_text = ""

        for step in range(1, max_steps + 1):
            logger.info("  Self-review step %d/%d", step, max_steps)
            response_text, tool_calls = agent._step(messages)

            if tool_calls:
                messages.append({"role": "assistant", "content": response_text})
                for call in tool_calls:
                    observation = agent._dispatch(call)
                    messages.append({
                        "role": "tool",
                        "name": call.get("name", ""),
                        "content": observation,
                    })
            else:
                final_text = response_text
                break

            if "<<SVA_COMPLETE>>" in response_text:
                final_text = response_text
                break
        else:
            messages.append({
                "role": "user",
                "content": (
                    "Output the reviewed assertions now in a "
                    "```systemverilog``` block. End with <<SVA_COMPLETE>>."
                ),
            })
            final_text, _ = agent._step(messages)

        batch_reviewed = agent._extract_sva(final_text)
        if batch_reviewed.strip():
            all_reviewed.append(batch_reviewed)
            logger.info("  Batch %d: %d → %d assertions.",
                        batch_idx, len(batch),
                        len(re.findall(r'^\s*assert\s', batch_reviewed, re.MULTILINE)))
        else:
            # LLM returned empty — keep original batch.
            logger.warning("  Batch %d: empty response — keeping originals.", batch_idx)
            all_reviewed.append(batch_sva)

    reviewed_sva = "\n\n".join(all_reviewed)
    original_count = len(re.findall(r'^\s*assert\s', sva_code, re.MULTILINE))
    reviewed_count = len(re.findall(r'^\s*assert\s', reviewed_sva, re.MULTILINE))
    logger.info(
        "Self-review complete: %d → %d assertions (%d removed/fixed).",
        original_count, reviewed_count,
        original_count - reviewed_count,
    )
    return reviewed_sva


# ---------------------------------------------------------------------------
# Reassemble passing assertions into a single SVA string
# ---------------------------------------------------------------------------

def reassemble_assertions(passed: List[Dict[str, str]]) -> str:
    """
    Join passing assertion entries back into a single SVA code string.

    Each entry's comment (if any) is placed above its assertion, separated
    by a blank line from the previous entry for readability.

    Parameters
    ----------
    passed : list of dict
        Each dict has "comment" and "assertion" keys.

    Returns
    -------
    str
        Reassembled SVA code.
    """
    blocks: List[str] = []
    for entry in passed:
        block_parts: List[str] = []
        if entry["comment"]:
            block_parts.append(entry["comment"])
        block_parts.append(entry["assertion"])
        blocks.append("\n".join(block_parts))

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Main lint-fix-relint loop
# ---------------------------------------------------------------------------

def run_lint_loop(
    agent: "SVAAgent",
    raw_sva: str,
    config: "PipelineConfig",
    facts: Optional[RTLFacts] = None,
    signal_map: Optional[Dict[str, Any]] = None,
    clock_signal: Optional[str] = None,
    reset_signal: Optional[str] = None,
    analysis_tracer: Optional[Any] = None,
) -> str:
    """
    Post-generation lint feedback loop.

    Phase 1: TRANSFORM — repair structurally broken assertions (no removal).
    Phase 2: REMOVE WRONG — trivially true, wrong-style assertions.
    Phase 3: VALIDATE & DEDUPLICATE — signal validation, all dedup passes.
    Phase 4: Lint loop — pyslang validation with LLM repair.

    Each function appears in exactly one phase. Transforms run first so
    that downstream dedup/subsumption sees the canonicalised forms (e.g.,
    a wrapped bare fragment now matches an existing assertion). String
    dedup runs before subsumption because subsumption groups assertions
    by signal set — duplicates would inflate group sizes and mask the
    subsumption relationship.

    Parameters
    ----------
    agent : SVAAgent
        The agent instance (model already loaded).
    raw_sva : str
        Raw SVA code from the initial generation pass.
    config : PipelineConfig
        Pipeline configuration.
    facts : RTLFacts, optional
        Pre-extracted structured RTL facts. If None, an empty ``RTLFacts``
        is created — post-processors that need facts will become no-ops.
    signal_map : dict, optional
        Signal map from design_info. Only used to drive the LLM self-review
        pass; structural validation uses ``facts.all_signals`` instead.
    clock_signal : str, optional
        Detected clock signal name (e.g., "nvdla_core_clk").
    reset_signal : str, optional
        Detected reset signal name (e.g., "nvdla_core_rstn").

    Returns
    -------
    str
        Cleaned SVA code containing only assertions that passed linting.
    """
    # Use an empty facts object if none was provided so all post-processors
    # can be called unconditionally.
    if facts is None:
        facts = RTLFacts()

    # Accumulate assertions that pass across all iterations.
    all_passed: List[Dict[str, str]] = []

    def _count(sva: str) -> int:
        return len(re.findall(r"^\s*assert\b", sva, re.MULTILINE))

    def _trace(phase: str, step: str, action: str, before: str, after: str,
               description: str = "") -> None:
        """Record a post-processor step in the analysis tracer."""
        if analysis_tracer is None:
            return
        b = _count(before)
        a = _count(after)
        delta = a - b
        if action == "transform":
            delta = 0  # transforms don't change count
        try:
            analysis_tracer.record(phase, step, action, delta, description)
        except Exception:
            pass

    # Seed the trace with the initial count before Phase 1.
    if analysis_tracer is not None:
        try:
            analysis_tracer.record(
                "1-generation", "agent_output", action="keep",
                delta=_count(raw_sva),
                description="Assertions from AST + LLM agent",
            )
        except Exception:
            pass

    # Phase 1: TRANSFORM — repair structurally broken assertions.
    # No removal, no dedup. Each fix only modifies assertions to a
    # canonical form so downstream phases can recognise duplicates.
    # When ``enable_deterministic_repair`` is False, all transforms are
    # skipped — failed assertions then either get dropped by Phase 2
    # validators or routed to the LLM via the lint feedback loop.
    if getattr(config, "enable_deterministic_repair", True):
        _before = raw_sva; raw_sva = fix_bare_property_fragments(raw_sva, clock_signal, reset_signal); _trace("2-phase1", "fix_bare_property_fragments", "transform", _before, raw_sva, "Wrap bare property fragments")
        _before = raw_sva; raw_sva = fix_immediate_implication(raw_sva); _trace("2-phase1", "fix_immediate_implication", "transform", _before, raw_sva, "|-> in immediate -> !(c)||(r)")
        _before = raw_sva; raw_sva = fix_double_negation(raw_sva); _trace("2-phase1", "fix_double_negation", "transform", _before, raw_sva, "Simplify double negation")
        _before = raw_sva; raw_sva = fix_immediate_and_form(raw_sva); _trace("2-phase1", "fix_immediate_and_form", "transform", _before, raw_sva, "AND-form to implication")
        _before = raw_sva; raw_sva = fix_condition_only_assertions(raw_sva, facts); _trace("2-phase1", "fix_condition_only_assertions", "transform", _before, raw_sva, "Repair condition-only via msg + facts; drop on validation failure")
        _before = raw_sva; raw_sva = fix_next_cycle_on_combinational(raw_sva, facts); _trace("2-phase1", "fix_next_cycle_on_combinational", "transform", _before, raw_sva, "|=> to |-> for comb signals")
        _before = raw_sva; raw_sva = fix_same_cycle_past_on_sequential(raw_sva, facts); _trace("2-phase1", "fix_same_cycle_past_on_sequential", "transform", _before, raw_sva, "|-> to |=> for $past on seq signals")
    else:
        logger.info("Phase 1 (deterministic repair) DISABLED by config — "
                    "skipping all 7 fix_* transforms.")
        _trace("2-phase1", "phase1_disabled", "remove", raw_sva, raw_sva,
               "Deterministic repair disabled (ablation)")

    # Phase 2: REMOVE WRONG — drop assertions that are structurally broken
    # or trivially true (compile-time constants, identical ternary, etc.).
    # 4 semantic-vacuity checks first — these catch patterns that parse
    # cleanly + use real signals but are semantically wrong (manual review
    # of SpecGuard outputs on cmacfull/rubik/sdp surfaced these).
    # Pre-pass: split lines that concatenate `endproperty` and the
    # following `assert property (NAME)` so per-line detectors see
    # the assert at the start of a line instead of buried mid-line.
    # Some pipelines (notably the security pass) emit them concatenated.
    _before = raw_sva; raw_sva = _split_property_assert_on_one_line(raw_sva); _trace("3-phase2", "_split_property_assert_on_one_line", "transform", _before, raw_sva, "Split combined property+assert onto separate lines")
    _before = raw_sva; raw_sva = remove_dead_code_under_disable(raw_sva); _trace("3-phase2", "remove_dead_code_under_disable", "remove", _before, raw_sva, "disable iff (X) cancels antecedent X")
    _before = raw_sva; raw_sva = relax_strict_handshake_timing(raw_sva); _trace("3-phase2", "relax_strict_handshake_timing", "transform", _before, raw_sva, "Relax exact-##1 handshake completion")
    _before = raw_sva; raw_sva = remove_missing_implication_assertions(raw_sva); _trace("3-phase2", "remove_missing_implication_assertions", "remove", _before, raw_sva, "assert (a && b) without |-> meant implication")
    _before = raw_sva; raw_sva = remove_unconditional_equality_assertions(raw_sva, facts); _trace("3-phase2", "remove_unconditional_equality_assertions", "remove", _before, raw_sva, "Bare (sig == LIT) without antecedent")
    _before = raw_sva; raw_sva = remove_tautological_implications(raw_sva); _trace("3-phase2", "remove_tautological_implications", "remove", _before, raw_sva, "(... && X && ...) -> X (any form)")
    _before = raw_sva; raw_sva = remove_vacuous_antecedent(raw_sva); _trace("3-phase2", "remove_vacuous_antecedent", "remove", _before, raw_sva, "Constant-false antecedent: never fires")
    _before = raw_sva; raw_sva = remove_self_comparison_assertions(raw_sva); _trace("3-phase2", "remove_self_comparison_assertions", "remove", _before, raw_sva, "(X OP X) trivially true / false")
    _before = raw_sva; raw_sva = remove_vacuous_body_assertions(raw_sva); _trace("3-phase2", "remove_vacuous_body_assertions", "remove", _before, raw_sva, "Body simplifies to constant-true; never fires")
    _before = raw_sva; raw_sva = validate_inside_widths(raw_sva, facts, mode="drop"); _trace("3-phase2", "validate_inside_widths", "remove", _before, raw_sva, "LHS-vs-literal width mismatch in inside-set")
    _before = raw_sva; raw_sva = validate_loop_var_scope(raw_sva, facts, mode="drop"); _trace("3-phase2", "validate_loop_var_scope", "remove", _before, raw_sva, "Loop-shaped identifier (i/j/k) not in design signal map")
    # Semantic-affinity validation runs in flag-only mode by default so
    # the pipeline operator can observe what gets marked before deciding
    # to enable drop-on-low-affinity.  Threshold 0.0 means we annotate
    # only the strongest signal: assertions whose signals NEVER co-occur
    # in any single RTL statement.  Switch to mode="drop" once the
    # false-positive rate has been characterised on the target designs.
    _before = raw_sva; raw_sva = validate_semantic_affinity(raw_sva, facts, threshold=0.0, mode="flag"); _trace("3-phase2", "validate_semantic_affinity", "annotate", _before, raw_sva, "Mark assertions whose signals never co-occur in RTL")
    # Implication-direction validation: flag (only flag for now) any
    # `A |-> B` whose dataflow direction runs the wrong way.  Same
    # rationale as semantic-affinity for staying in flag-only mode
    # while we observe true / false positive rates.
    _before = raw_sva; raw_sva = validate_implication_direction(raw_sva, facts, mode="flag"); _trace("3-phase2", "validate_implication_direction", "annotate", _before, raw_sva, "Mark assertions whose implication direction is reversed vs RTL dataflow")
    _before = raw_sva; raw_sva = validate_past_depth(raw_sva, facts, mode="flag"); _trace("3-phase2", "validate_past_depth", "annotate", _before, raw_sva, "Flag $past(d, N) where N != actual pipeline depth")
    _before = raw_sva; raw_sva = validate_cross_module_assertions(raw_sva, facts, mode="flag"); _trace("3-phase2", "validate_cross_module_assertions", "annotate", _before, raw_sva, "Flag assertions whose signals split across modules with no port-connection link")
    _before = raw_sva; raw_sva = validate_state_assertions(raw_sva, facts, mode="flag"); _trace("3-phase2", "validate_state_assertions", "annotate", _before, raw_sva, "Flag state-literal hallucinations against extracted FSM state set")
    _before = raw_sva; raw_sva = remove_trivial_assertions(raw_sva); _trace("3-phase2", "remove_trivial_assertions", "remove", _before, raw_sva, "Trivially true")
    # Coverage breadcrumb runs LAST so subsequent removers (like the
    # orphan-comment cleanup inside remove_trivial_assertions) can't
    # accidentally strip it.
    _before = raw_sva; raw_sva = annotate_doc_coverage(raw_sva, facts); _trace("3-phase2", "annotate_doc_coverage", "annotate", _before, raw_sva, "Append // DOC_COVERAGE summary listing uncovered documented properties")
    _before = raw_sva; raw_sva = remove_wrong_style_assertions(raw_sva, config); _trace("3-phase2", "remove_wrong_style_assertions", "remove", _before, raw_sva, "Wrong assertion style")
    _before = raw_sva; raw_sva = verify_constant_signal_pairs(raw_sva, facts); _trace("3-phase2", "verify_constant_signal_pairs", "remove", _before, raw_sva, "Wrong constant-signal pair")
    _before = raw_sva; raw_sva = validate_signal_widths(raw_sva, facts); _trace("3-phase2", "validate_signal_widths", "remove", _before, raw_sva, "Width mismatch")
    _before = raw_sva; raw_sva = validate_reset_values(raw_sva, facts); _trace("3-phase2", "validate_reset_values", "remove", _before, raw_sva, "Wrong reset values")

    # Step 4: drop assertions referencing out-of-scope signals.
    # Combines structurally-detected signals (facts.out_of_scope_signals)
    # with optional manual regex patterns from the YAML.
    _before = raw_sva
    raw_sva = validate_out_of_scope(
        raw_sva,
        patterns=getattr(config, "out_of_scope_patterns", []),
        signal_names=getattr(facts, "out_of_scope_signals", set()),
    )
    _trace("3-phase2", "validate_out_of_scope", "remove", _before, raw_sva, "Out-of-scope signal refs")

    # Step 5: drop pure bus-slice restatements.
    if getattr(config, "remove_bus_slice_restatements", True):
        _before = raw_sva; raw_sva = remove_bus_slice_restatements(raw_sva); _trace("3-phase2", "remove_bus_slice_restatements", "remove", _before, raw_sva, "Bus-slice decomposition")

    # Optional: RTL data-flow check for case selector mismatches.
    # Off by default; enable with `agent.use_dataflow_check: true`.
    if getattr(config, 'use_dataflow_check', False):
        _before = raw_sva; raw_sva = check_case_selector_mismatch(raw_sva, facts); _trace("3-phase2", "check_case_selector_mismatch", "remove", _before, raw_sva, "Wrong case selector")
    else:
        logger.info("Skipping data-flow check (disabled).")

    # Phase 3: VALIDATE & DEDUPLICATE.
    # Order within phase matters:
    #   1. validate_signals — drop hallucinated signal references first
    #      so dedup doesn't waste work on assertions about non-existent
    #      signals.
    #   2. deduplicate_assertions — exact string match dedup.
    #   3. semantic_deduplicate — canonical-form dedup (catches dups
    #      that string dedup missed due to whitespace/operator ordering).
    #   4. remove_subsumed_and_contradicting — logical subsumption rules
    #      (must run last because it groups by signal set, and duplicates
    #      would distort the group sizes).
    # Stage 2: per-(design, model) hallucination knowledgebase. Logging
    # is always-on (cheap and builds the dataset). Reading the resulting
    # denylist back into the prompt is gated by use_hallucination_denylist
    # in the agent.
    deny_path: Optional[str] = None
    try:
        deny_dir = Path(getattr(config, "hallucination_denylist_dir",
                                "indices/hallucinations"))
        design_key = Path(getattr(config, "rtl_dir", "")).name or "unknown_design"
        model_key = (
            (getattr(config, "model_id", "") or "unknown_model")
            .replace("/", "_")
            .replace(":", "_")
        )
        deny_path = str(deny_dir / f"{design_key}__{model_key}.json")
    except Exception as exc:
        logger.debug("Could not compute denylist path: %s", exc)

    _before = raw_sva; raw_sva = validate_signals(raw_sva, facts, denylist_path=deny_path); _trace("4-phase3", "validate_signals", "remove", _before, raw_sva, "Hallucinated signals")
    _before = raw_sva; raw_sva = deduplicate_assertions(raw_sva); _trace("4-phase3", "deduplicate_assertions", "remove", _before, raw_sva, "String-based dedup")
    _before = raw_sva; raw_sva = semantic_deduplicate(raw_sva); _trace("4-phase3", "semantic_deduplicate", "remove", _before, raw_sva, "Canonical-form dedup")
    _before = raw_sva; raw_sva = remove_subsumed_and_contradicting(raw_sva); _trace("4-phase3", "remove_subsumed_and_contradicting", "remove", _before, raw_sva, "Subsumed / contradicting")

    # Phase 4: LLM self-review — ask the LLM to check its own work.
    # Off by default; enable with `agent.use_self_review: true` in config.
    if getattr(config, 'use_self_review', False) and agent.backend is not None and signal_map:
        raw_sva = llm_self_review(agent, raw_sva, signal_map)
    else:
        logger.info("Skipping LLM self-review (disabled or no backend/signal map).")

    # The SVA string to split and lint on each iteration.  Initially the
    # full agent output; on subsequent iterations the agent's fix output.
    remaining_sva = raw_sva

    # Report tracks the full history across iterations.
    lint_report: Dict[str, Any] = {
        "iterations": [],
        "final_status": "INCOMPLETE",
        "total_refinement_iterations": 0,
    }

    logger.info(
        "Starting lint feedback loop (max %d refinement iterations) …",
        config.max_refinement_iterations,
    )

    iteration = 0
    for iteration in range(1, config.max_refinement_iterations + 1):
        logger.info("=== Lint iteration %d ===", iteration)

        # --- Split into individual assertions ---
        assertions = split_assertions(remaining_sva)
        if not assertions:
            logger.warning("No assertions found in SVA string — nothing to lint.")
            break

        logger.info("Split into %d assertion(s). Linting …", len(assertions))

        # --- Lint each assertion ---
        passed, failures = lint_all_assertions(
            assertions,
            reject_assert_property=config.reject_assert_property,
        )
        all_passed.extend(passed)

        # --- Log to trace if available ---
        if hasattr(agent, 'trace'):
            agent.trace.log_lint_iteration(
                iteration=iteration,
                total=len(assertions),
                passed=len(passed),
                failed=len(failures),
                failure_summaries=[
                    f["assertion"][:80] for f in failures[:10]
                ],
            )

        # --- Record this iteration in the report ---
        lint_report["iterations"].append({
            "iteration": iteration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_assertions": len(assertions),
            "passed": len(passed),
            "failed": len(failures),
            "failures": [
                {
                    "index": f["index"],
                    "assertion": f["assertion"],
                    "comment": f["comment"],
                    "error": f["error"],
                }
                for f in failures
            ],
        })
        save_lint_report(lint_report, config.lint_failures_file)

        # --- All passed? ---
        if not failures:
            logger.info("All assertions passed linting.")
            lint_report["final_status"] = "ALL_PASSED"
            break

        # --- Refinement: ask agent to fix failures ---
        # In ast_only mode the backend is None and refinement is impossible;
        # drop failing assertions and stop the loop instead of crashing.
        if agent.backend is None:
            logger.warning(
                "ast_only mode: no LLM backend for refinement. "
                "Dropping %d failing assertion(s) and stopping lint loop.",
                len(failures),
            )
            lint_report["final_status"] = "AST_ONLY_NO_REFINEMENT"
            break

        logger.info(
            "%d assertion(s) failed. Sending to agent for refinement …",
            len(failures),
        )
        remaining_sva = agent.refine_assertions(failures)

        if not remaining_sva or not remaining_sva.strip():
            logger.warning(
                "Agent returned empty refinement response — "
                "stopping lint loop."
            )
            lint_report["final_status"] = "EMPTY_REFINEMENT"
            break
    else:
        # Max iterations reached with failures still present.
        logger.warning(
            "Max refinement iterations (%d) reached. "
            "%d assertion(s) could not be fixed and will be dropped.",
            config.max_refinement_iterations,
            len(failures) if 'failures' in dir() else 0,
        )
        lint_report["final_status"] = "MAX_ITERATIONS_REACHED"

    lint_report["total_refinement_iterations"] = iteration
    save_lint_report(lint_report, config.lint_failures_file)

    # --- Reassemble the final passing assertions ---
    final_sva = reassemble_assertions(all_passed)

    if not final_sva.strip():
        logger.error(
            "No assertions survived linting. Check %s for details.",
            config.lint_failures_file,
        )
    else:
        logger.info(
            "Lint loop complete: %d assertion(s) in final output.",
            len(all_passed),
        )

    return final_sva
