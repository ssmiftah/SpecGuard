"""
Security pass: a dedicated pipeline stage that emits CWE-tagged SystemVerilog
assertions from a threat-model document.

Runs after the functional lint loop in main.py.  For each ``### CWE-NNNN``
section in the configured ``threat_model`` markdown file, the pass:

  1. Extracts the scenario text plus any signal-name hints (glob patterns like
     ``csb2pdp_*``).
  2. Resolves those hints against the design's signal map.
  3. Asks the LLM to propose ONE SVA property that detects the CWE scenario,
     wrapped in ``assert property`` with a ``disable iff (!rstn)`` guard and a
     ``$error`` message naming the CWE.
  4. Lints the proposed assertion through the same ``lint_single_assertion``
     gate used by the functional pipeline.
  5. Drops anything that fails lint or whose signal references are mostly
     hallucinated (``validate_signals``).
  6. Appends the surviving CWE-tagged assertions to the lint-clean SVA file
     under a ``// === Security assertions (CWE-tagged) ===`` banner.

Each LLM call is logged via ``agent.trace.log_step(phase="security_pass", …)``
so token usage rolls into the standard ``*_token_summary.json`` aggregate.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threat model parsing
# ---------------------------------------------------------------------------

# Matches "### CWE-NNNN (Optional title)"
_CWE_HEADER_RE = re.compile(
    r"^###\s+CWE[-\s]?(\d+)\s*(?:\(([^)]+)\))?\s*$",
    re.MULTILINE,
)
# Matches glob-style signal hints in scenario text:
#   csb2pdp_*, pdp_dp2wdma_*, mon_reg2dp_*, abuf_rd_*, abuf_*
# Allows underscore immediately before the literal '*'.
_SIGNAL_HINT_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_]*\*)"
)
# Matches concrete signal names backticked or in-line:
#   `nvdla_core_clk`, `csb2pdp_rdma_req_pvld`
_SIGNAL_LITERAL_RE = re.compile(
    r"`([A-Za-z][A-Za-z0-9_]+)`"
)


@dataclass
class Scenario:
    """One CWE scenario extracted from the threat model."""
    cwe_id: int
    cwe_title: str
    body: str
    signal_hints: List[str] = field(default_factory=list)
    literal_signals: List[str] = field(default_factory=list)

    @property
    def safe_property_name(self) -> str:
        """Property identifier safe for SystemVerilog."""
        title_slug = re.sub(
            r"[^a-z0-9]+", "_", self.cwe_title.lower()
        ).strip("_")
        return f"p_cwe_{self.cwe_id}_{title_slug or 'scenario'}"[:60]


def parse_threat_model(path: Path) -> List[Scenario]:
    """Parse a markdown threat-model file into a list of Scenario records.

    Recognises ``### CWE-NNNN (Title)`` headers; captures the body text
    until the next ``###`` or ``##`` heading.  Returns ``[]`` if the file
    does not exist or contains no recognisable CWE sections.
    """
    if not path.exists():
        logger.warning("Threat model not found: %s", path)
        return []

    text = path.read_text(encoding="utf-8", errors="replace")

    # Find all CWE header positions, then slice between consecutive headers.
    headers = list(_CWE_HEADER_RE.finditer(text))
    if not headers:
        logger.warning("No '### CWE-NNNN' sections found in %s", path)
        return []

    # Boundary marker for the last section's body.
    section_end_re = re.compile(r"^(##\s|###\s)", re.MULTILINE)

    scenarios: List[Scenario] = []
    for i, m in enumerate(headers):
        cwe_id = int(m.group(1))
        cwe_title = (m.group(2) or "").strip()
        body_start = m.end()
        # Body ends at the next `### ` or `## ` heading.
        next_match = section_end_re.search(text, body_start)
        body_end = next_match.start() if next_match else len(text)
        body = text[body_start:body_end].strip()

        signal_hints = sorted(set(_SIGNAL_HINT_RE.findall(body)))
        literal_signals = sorted(set(_SIGNAL_LITERAL_RE.findall(body)))

        scenarios.append(Scenario(
            cwe_id=cwe_id,
            cwe_title=cwe_title,
            body=body,
            signal_hints=signal_hints,
            literal_signals=literal_signals,
        ))

    logger.info(
        "Threat model %s: %d CWE scenario(s) parsed.",
        path.name, len(scenarios),
    )
    return scenarios


# ---------------------------------------------------------------------------
# Signal matching
# ---------------------------------------------------------------------------

def _glob_to_regex(pattern: str) -> re.Pattern:
    """Convert a glob-style pattern (e.g. 'csb2pdp_*') to a regex."""
    # Escape everything except '*' which becomes '.*'.
    parts = pattern.split("*")
    escaped = ".*".join(re.escape(p) for p in parts)
    return re.compile(rf"^{escaped}$")


def match_signals_to_scenario(
    scenario: Scenario,
    signal_map: Dict[str, Any],
    max_signals: int = 30,
) -> List[str]:
    """Return signal names from ``signal_map`` that match the scenario.

    Strategy:
      1. Glob-expand each ``signal_hint`` (e.g. ``csb2pdp_*``).
      2. Add any ``literal_signals`` that exist in the map verbatim.
      3. Cap the result at ``max_signals`` to keep the prompt compact.

    Returns ``[]`` if no signals match — caller should still build a prompt
    with a "no specific signals matched" fallback.
    """
    if not signal_map:
        return []

    matched: List[str] = []
    seen: set = set()
    all_keys = list(signal_map.keys())

    for pattern in scenario.signal_hints:
        rx = _glob_to_regex(pattern)
        for key in all_keys:
            if key in seen:
                continue
            if rx.match(key):
                matched.append(key)
                seen.add(key)
                if len(matched) >= max_signals:
                    return matched

    for lit in scenario.literal_signals:
        if lit in signal_map and lit not in seen:
            matched.append(lit)
            seen.add(lit)
            if len(matched) >= max_signals:
                return matched

    return matched


def _format_signal_summary(
    signals: List[str],
    signal_map: Dict[str, Any],
) -> str:
    """Compact one-line-per-signal summary for the prompt."""
    if not signals:
        return "  (no design signals matched the scenario hints)"
    out_lines: List[str] = []
    for s in signals:
        info = signal_map.get(s, {})
        direction = info.get("direction", "?")
        width = info.get("bit_width", info.get("width", "?"))
        out_lines.append(f"  - {s} ({direction}, {width}b)")
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SECURITY_SYSTEM_PROMPT = (
    "You are a hardware security verification expert. "
    "You write SystemVerilog Assertions (SVA) that detect violations of "
    "specific Common Weakness Enumeration (CWE) scenarios in RTL designs. "
    "You only reference signals present in the provided signal list. "
    "You produce concurrent assertions wrapped in `assert property (...)` "
    "with a `disable iff (!<reset>)` guard, and a `$error` message that "
    "names the CWE class (e.g., `\"CWE-1245: ...\"`). "
    "OUTPUT FORMAT: emit the SVA code ONLY. Do not show reasoning, "
    "do not add prose, do not wrap in code fences. Begin your reply with "
    "the literal token `property` and end with the assert statement's "
    "terminating semicolon."
)


def _build_messages(
    scenario: Scenario,
    signals: List[str],
    signal_map: Dict[str, Any],
    top_module: str,
    clock_signal: Optional[str],
    reset_signal: Optional[str],
) -> List[Dict[str, str]]:
    """Build the (system, user) message list for one scenario."""
    clk = clock_signal or "<unknown_clk>"
    rstn = reset_signal or "<unknown_rstn>"
    signals_block = _format_signal_summary(signals, signal_map)

    # Trim threat body to keep prompts bounded — threat sections are usually
    # short already, but cap defensively.
    body = scenario.body
    if len(body) > 1500:
        body = body[:1500] + "\n... (truncated)"

    user = (
        f"# Threat scenario\n"
        f"CWE-{scenario.cwe_id}"
        + (f" — {scenario.cwe_title}\n" if scenario.cwe_title else "\n")
        + f"\n{body}\n\n"
        f"# Top module\n  {top_module}\n"
        f"  clock : {clk}\n"
        f"  reset : {rstn} (active-low)\n\n"
        f"# Available signals (use ONLY these)\n{signals_block}\n\n"
        f"# Task\n"
        f"Propose ONE SystemVerilog assertion that detects a violation of "
        f"the CWE-{scenario.cwe_id} scenario above. Requirements:\n"
        f"  1. Use the form `property {scenario.safe_property_name}; "
        f"@(posedge {clk}) disable iff (!{rstn}) <body>; endproperty` "
        f"followed by `assert property ({scenario.safe_property_name}) "
        f"else $error(\"CWE-{scenario.cwe_id}: <description>\");`.\n"
        f"  2. Reference ONLY signals listed above. Do not invent signals.\n"
        f"  3. The `$error` string MUST start with `CWE-{scenario.cwe_id}:` "
        f"so the assertion is traceable to the threat class.\n"
        f"  4. Output ONLY the property + assert pair. No prose, no fences."
    )

    return [
        {"role": "system", "content": _SECURITY_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Response extraction
# ---------------------------------------------------------------------------

# Strip optional ``` code fences (with or without language tag).
_FENCE_OPEN_RE = re.compile(r"^\s*```[A-Za-z]*\s*\n", re.MULTILINE)
_FENCE_CLOSE_RE = re.compile(r"\n\s*```\s*$", re.MULTILINE)
# qwen3 may emit a hidden <think>...</think> block — strip it.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _extract_assertion_block(response_text: str) -> Optional[str]:
    """Pull the property+assert pair out of the model's response.

    Returns the cleaned SVA code, or ``None`` if no recognisable
    ``assert property`` appears in the response.
    """
    if not response_text:
        return None

    text = _THINK_BLOCK_RE.sub("", response_text).strip()
    text = _FENCE_OPEN_RE.sub("", text)
    text = _FENCE_CLOSE_RE.sub("", text).strip()

    if "assert property" not in text:
        return None

    # If a `property ... endproperty` block precedes the assert, keep the
    # whole region; otherwise return from the first `assert property`
    # onwards. Scan to the closing ';' after the assert statement.
    start = 0
    if "property " in text and "endproperty" in text:
        # Take from the first 'property ' that precedes the assert.
        m = re.search(r"\bproperty\s+\w+\s*;", text)
        if m:
            start = m.start()
    else:
        m = re.search(r"\bassert\s+property\b", text)
        if m:
            start = m.start()

    block = text[start:].strip()

    # Strip anything after the final ';' that closes the assert statement.
    # We want to keep multi-line bodies, so accept up to the LAST ';' in the
    # block — the model occasionally appends prose after the assertion.
    last_semi = block.rfind(";")
    if last_semi != -1:
        block = block[: last_semi + 1]

    return block.strip() or None


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_security_pass(
    agent: Any,
    lint_clean_sva: str,
    config: Any,
    facts: Optional[Any] = None,
    signal_map: Optional[Dict[str, Any]] = None,
    clock_signal: Optional[str] = None,
    reset_signal: Optional[str] = None,
) -> str:
    """Run the security pass and return the merged SVA code.

    No-ops (returns ``lint_clean_sva`` unchanged) when:
      - ``config.security_pass_enabled`` is False, or
      - ``config.threat_model`` is empty / file missing, or
      - the threat file has no parseable CWE sections.
    """
    if not getattr(config, "security_pass_enabled", False):
        return lint_clean_sva

    threat_path_str = getattr(config, "threat_model", "")
    if not threat_path_str:
        logger.info("security_pass_enabled but no threat_model configured.")
        return lint_clean_sva

    threat_path = Path(threat_path_str)
    if not threat_path.is_absolute():
        # Resolve relative to repo root (cwd at run time).
        threat_path = Path.cwd() / threat_path
    scenarios = parse_threat_model(threat_path)
    if not scenarios:
        return lint_clean_sva

    max_scenarios = int(getattr(config, "security_pass_max_scenarios", 0) or 0)
    if max_scenarios > 0:
        scenarios = scenarios[:max_scenarios]
        logger.info(
            "security_pass: capping at %d scenarios.", max_scenarios,
        )

    signal_map = signal_map or {}
    top_module = getattr(config, "top_module", "")

    # `run_lint_loop` is imported lazily inside the try block (avoids
    # import-time cycles and only loads when the pass actually runs).

    # Temporarily widen the backend's max_tokens for security calls.
    # qwen3 spends a large fraction of completion tokens inside hidden
    # `<think>` blocks; the functional pass's 2048-token ceiling is too
    # tight to leave room for both reasoning and the SVA body. Restored
    # in a finally block at the end of the function.
    backend_cfg = getattr(getattr(agent, "backend", None), "config", None)
    saved_max_tokens: Optional[int] = None
    SECURITY_MAX_TOKENS = 8192
    if backend_cfg is not None and hasattr(backend_cfg, "max_new_tokens"):
        saved_max_tokens = int(backend_cfg.max_new_tokens)
        if saved_max_tokens < SECURITY_MAX_TOKENS:
            backend_cfg.max_new_tokens = SECURITY_MAX_TOKENS
            logger.info(
                "security_pass: temporarily raised max_new_tokens "
                "%d -> %d for the security calls.",
                saved_max_tokens, SECURITY_MAX_TOKENS,
            )

    accepted_blocks: List[Tuple[Scenario, str]] = []
    rejected_summary: List[Tuple[int, str]] = []

    def _restore_backend_cap() -> None:
        if backend_cfg is not None and saved_max_tokens is not None:
            backend_cfg.max_new_tokens = saved_max_tokens

    for idx, scenario in enumerate(scenarios, start=1):
        signals = match_signals_to_scenario(scenario, signal_map)

        # Skip scenarios with no signal context — the LLM has nothing
        # concrete to anchor the property on and tends to emit degenerate
        # placeholders like `1'b1`. A future enhancement could fall back
        # to a relevance-ranked subset of the full signal map.
        if not signals:
            logger.info(
                "security_pass [%d/%d]: SKIP CWE-%d (%s) — "
                "no signals matched the scenario hints.",
                idx, len(scenarios), scenario.cwe_id,
                scenario.cwe_title or "untitled",
            )
            rejected_summary.append(
                (scenario.cwe_id, "no signals matched scenario hints")
            )
            continue

        messages = _build_messages(
            scenario,
            signals,
            signal_map,
            top_module=top_module,
            clock_signal=clock_signal,
            reset_signal=reset_signal,
        )

        logger.info(
            "security_pass [%d/%d]: CWE-%d (%s) — %d matched signal(s)",
            idx, len(scenarios), scenario.cwe_id,
            scenario.cwe_title or "untitled",
            len(signals),
        )

        try:
            response_text, _tool_calls, usage = agent.backend.generate(
                messages, []
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "security_pass [%d/%d]: backend call failed: %s",
                idx, len(scenarios), exc,
            )
            rejected_summary.append((scenario.cwe_id, f"backend error: {exc}"))
            continue

        # Log to the trace so token usage rolls into the global summary.
        if hasattr(agent, "trace"):
            agent.trace.log_step(
                phase="security_pass",
                step=idx,
                model_output=response_text,
                usage=usage,
                notes=f"CWE-{scenario.cwe_id}: {scenario.cwe_title}",
            )

        block = _extract_assertion_block(response_text)
        if not block:
            logger.info(
                "security_pass [%d/%d]: no parseable assertion in response.",
                idx, len(scenarios),
            )
            rejected_summary.append(
                (scenario.cwe_id, "no parseable assert property in response")
            )
            continue

        # No per-scenario lint gate. The merged-block lint loop below is
        # the single gate — its deterministic-repair phase is now
        # format-aware (skips concurrent property blocks) and its
        # LLM-feedback refinement can rescue single-shot syntax slips.
        accepted_blocks.append((scenario, block))
        logger.info(
            "security_pass [%d/%d]: candidate emitted — CWE-%d "
            "(lint loop will validate)", idx, len(scenarios),
            scenario.cwe_id,
        )

    if not accepted_blocks:
        _restore_backend_cap()
        logger.info(
            "security_pass: 0/%d scenarios produced lint-clean assertions.",
            len(scenarios),
        )
        return lint_clean_sva

    # Build the merged security block and route it through `run_lint_loop`
    # SEPARATELY from the functional block. Two reasons for the separate
    # invocation:
    #   1. Inter-block dedup (which strips `else $error(...)` for
    #      comparison) would silently drop a security assertion whose
    #      body matches a functional one but whose CWE-tagged $error
    #      message differs.
    #   2. Phase 1 transforms (fix_bare_property_fragments,
    #      fix_immediate_implication, etc.) are now format-aware and
    #      skip concurrent property/endproperty blocks. The merged-block
    #      lint loop therefore preserves CWE assertions while still
    #      benefiting from the LLM-feedback refinement on lint failures.
    raw_security_block = "\n\n".join(
        f"// CWE-{s.cwe_id}: {s.cwe_title}\n{block}"
        for s, block in accepted_blocks
    )

    # Augment facts.all_signals with the property/sequence names declared
    # inside the security block, so `validate_signals` (called from
    # within the lint loop) does not treat them as hallucinated signals
    # and drop the `assert property (NAME)` line.
    declared_names: set = set()
    for _scenario, _block in accepted_blocks:
        for m in re.finditer(r"\bproperty\s+(\w+)", _block):
            declared_names.add(m.group(1))
        for m in re.finditer(r"\bsequence\s+(\w+)", _block):
            declared_names.add(m.group(1))

    original_known = None
    if facts is not None and declared_names:
        original_known = facts.all_signals
        facts.all_signals = set(original_known) | declared_names

    try:
        from .lint_loop import run_lint_loop
        logger.info(
            "security_pass: routing %d candidate(s) through lint loop "
            "(deterministic repair + LLM refinement on lint failures).",
            len(accepted_blocks),
        )
        validated_security = run_lint_loop(
            agent, raw_security_block, config,
            facts=facts,
            signal_map=signal_map,
            clock_signal=clock_signal,
            reset_signal=reset_signal,
            analysis_tracer=None,
        )
    finally:
        # Restore facts.all_signals first, then the backend token cap.
        if facts is not None and original_known is not None:
            facts.all_signals = original_known
        _restore_backend_cap()

    if not validated_security.strip():
        logger.info(
            "security_pass: all %d candidate assertions dropped by the "
            "lint loop.", len(accepted_blocks),
        )
        return lint_clean_sva

    final_count = len(re.findall(
        r"\bassert\s+property\b|\bassert\s*\(",
        validated_security,
    ))

    banner = (
        "\n\n// " + "=" * 68 + "\n"
        "// === Security assertions (CWE-tagged) ===\n"
        f"// Generated by security_pass from {threat_path.name}\n"
        f"// {final_count} accepted of {len(scenarios)} scenarios "
        f"(after lint-loop refinement)\n"
        "// " + "=" * 68 + "\n\n"
    )

    merged = lint_clean_sva.rstrip() + banner + validated_security.rstrip() + "\n"

    logger.info(
        "security_pass: appended %d CWE-tagged assertion(s) "
        "(rejected at LLM stage: %d).",
        final_count, len(rejected_summary),
    )
    return merged
