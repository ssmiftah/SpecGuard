"""
ast_clustering.py
-----------------
Template-aware clustering and arithmetic compaction for AST-extracted
RTL patterns.

Motivation
==========
Wide-bus RTL designs (e.g. NVDLA cacc) emit one continuous assignment
per bit position::

    assign abuf_wr_elem_121 = cfg_is_wg[89]  ? calc_pout_121 : calc_pout_73;
    assign abuf_wr_elem_122 = cfg_is_wg[90]  ? calc_pout_122 : calc_pout_74;
    ...
    assign abuf_wr_elem_152 = cfg_is_wg[120] ? calc_pout_152 : calc_pout_104;

The AST extractor faithfully emits 32 assertions for this family. Each
one is correct, but they all share the same *template*::

    abuf_wr_elem_<X> == cfg_is_wg[<Y>] ? calc_pout_<Z> : calc_pout_<W>

where ``Y = X - 32``, ``Z = X``, and ``W = X - 48`` (linear functions
of the loop variable ``X``). Routing all 32 to the LLM for spec-vs-RTL
enrichment costs hundreds of thousands of tokens to confirm something
the spec only says once.

This module:

1. **Clusters** patterns by abstract template (numeric indices →
   placeholders).
2. **Derives** an arithmetic relationship between the loop variable
   and each non-loop slot (linear / stride / identity / bit-shift /
   integer division / modular).
3. **Symbolically verifies** the derived relationship by substituting
   each member's loop-variable value back and string-matching against
   the original assertion.
4. **Recursively partitions** when a single arithmetic pattern doesn't
   fit the whole cluster (e.g. branched-on-parity families): the
   largest consistent subset is compacted, the residual is recursed.
5. **Emits a packed concat-and-mux assertion** that semantically
   covers every cluster member in one statement.

Compaction is a *transformation*; the symbolic verifier is the
correctness gate. If verification fails for any reason, the cluster
falls back to representative + sibling tags.

Public API
==========
``cluster_and_compact(patterns, **opts) -> CompactionResult``

Output is a flat list of ``CompactionUnit``:

- ``CompactedUnit``: a packed-vector assertion that subsumes N members
- ``RepresentativeUnit``: one member acting as the spec-check proxy for
  its siblings (used when arithmetic doesn't compact)
- ``IndividualUnit``: a single pattern with no sibling relationship

The agent feeds CompactedUnit and RepresentativeUnit through the LLM
spec-check; siblings inherit the result via tagging.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

from .ast_assertions import RTLPattern

logger = logging.getLogger(__name__)


# =============================================================================
# Template normalisation
# =============================================================================

# Stash and restore Verilog typed literals so their internal digits
# don't get treated as indices.
#   8'h0       9'b101010   17'h10000   4'd3
_VERILOG_LITERAL_RE = re.compile(r"\d+'[bdhoBDHO][\w?xzXZ]+")

# Match a numeric suffix on an identifier.
#   calc_dlv_elem_3   →  calc_dlv_elem_  + index 3
#   cfg_is_int8       → no match (no trailing `_<N>`)
_IDENT_SUFFIX_RE = re.compile(r"(?<=[A-Za-z_])([A-Za-z_]\w*?)_(\d+)\b")

# Match a bare integer inside square brackets — used as a bit-select.
#   cfg_is_wg[89]   →  cfg_is_wg[  + index 89  + ]
# Range expressions like [a:b] or [i+1] are skipped (only pure ints).
_BRACKET_INDEX_RE = re.compile(r"\[(\d+)\]")


def _normalize_template(text: str) -> Tuple[str, List[int]]:
    """Extract the abstract template and the per-slot index list from
    an assertion body fragment.

    Two index sources are recognised:

    * **Identifier suffixes** — ``calc_dlv_elem_3`` becomes
      ``calc_dlv_elem_*`` with index ``3``.
    * **Bare integer bit-selects** — ``cfg_is_int8[66]`` becomes
      ``cfg_is_int8[*]`` with index ``66``.

    Verilog typed literals (``8'h0``, ``17'h10000``) are preserved
    verbatim — their digits are part of the value, not an index.

    Indices are returned in left-to-right textual order, matching
    where the placeholders appear in the template. Two members with
    the same template share an index list of identical length and
    slot semantics.

    Returns
    -------
    (template, indices) : (str, list of int)
    """
    # Stash literals so we don't touch their digits.
    literals: List[str] = []

    def _stash_literal(m: "re.Match[str]") -> str:
        literals.append(m.group(0))
        return f"\x00LIT{len(literals) - 1}\x00"

    work = _VERILOG_LITERAL_RE.sub(_stash_literal, text)

    # Walk the string left-to-right collecting indices in order, then
    # produce the templated string in a single pass. We can't just do
    # two independent regex passes because the textual order matters
    # for the index list.
    indices: List[int] = []

    def _replace(match: "re.Match[str]") -> str:
        # Either identifier suffix or bracket index — distinguish by
        # which group matched.
        if match.lastgroup == "ident":
            indices.append(int(match.group("ident_n")))
            return match.group("ident_pre") + "_*"
        else:
            indices.append(int(match.group("brk_n")))
            return "[*]"

    combined = re.compile(
        r"(?P<ident>(?P<ident_pre>[A-Za-z_]\w*?)_(?P<ident_n>\d+)\b)"
        r"|"
        r"(?P<brk>\[(?P<brk_n>\d+)\])"
    )
    templated = combined.sub(_replace, work)

    # Restore literals.
    for i, lit in enumerate(literals):
        templated = templated.replace(f"\x00LIT{i}\x00", lit)

    return templated, indices


# =============================================================================
# Arithmetic pattern derivation
# =============================================================================

# A SlotFormula expresses how slot k depends on the loop-variable X.
@dataclass(frozen=True)
class SlotFormula:
    """Closed-form integer function of a single loop variable.

    Encodes ``f(X) = (X >> shift) * mul + add``. This subsumes:

    * constant   : mul=0, shift=0, add=c            (no dependence on X)
    * identity   : mul=1, shift=0, add=0
    * offset     : mul=1, shift=0, add=k            (X + k)
    * stride     : mul=k, shift=0, add=b            (k*X + b)
    * div_offset : mul=1, shift=n, add=b            ((X >> n) + b)
    * scaled_div : mul=k, shift=n, add=b            ((X >> n) * k + b)

    Reduction (``X mod k``) and other non-monotonic formulas aren't
    representable here and are handled via recursive partitioning
    instead — when no single SlotFormula fits the whole cluster, the
    largest agreeing subset gets compacted and the residual is
    recursed.
    """
    shift: int  # right-shift bits applied to X first (0 = identity)
    mul: int    # multiplier (0 = constant; ignore X entirely)
    add: int    # constant offset

    def evaluate(self, x: int) -> int:
        if self.mul == 0:
            return self.add
        return ((x >> self.shift) * self.mul) + self.add

    def __str__(self) -> str:
        if self.mul == 0:
            return f"{self.add}"
        base = "X"
        if self.shift > 0:
            base = f"(X >> {self.shift})"
        if self.mul == 1:
            term = base
        elif self.mul == -1:
            term = f"-{base}"
        else:
            term = f"{self.mul}*{base}"
        if self.add == 0:
            return term
        sign = "+" if self.add >= 0 else "-"
        return f"{term} {sign} {abs(self.add)}"


# Stride / shift values to try when fitting a SlotFormula. Powers of
# 2 cover bit-packing (e.g. 8-wide → stride 8). Small primes cover
# typical permutations.
_TRY_MULS = (1, -1, 2, -2, 3, 4, -4, 8, -8, 16, 32, 64, 128)
_TRY_SHIFTS = (0, 1, 2, 3, 4, 5, 6, 7)


def _fit_slot_formula(
    samples: List[Tuple[int, int]],
) -> Optional[SlotFormula]:
    """Find the simplest SlotFormula that fits every (X, y) sample.

    Search order: constant → identity → offset → stride → shifted
    stride. Returns the first formula that maps every X to its
    corresponding y exactly.

    Returns None when no formula in the search space fits.
    """
    if not samples:
        return None

    # Constant — every y is the same.
    ys = [y for _, y in samples]
    if all(y == ys[0] for y in ys):
        return SlotFormula(shift=0, mul=0, add=ys[0])

    # Identity / offset family — mul=1, shift=0.
    if all(y - x == samples[0][1] - samples[0][0] for x, y in samples):
        offset = samples[0][1] - samples[0][0]
        return SlotFormula(shift=0, mul=1, add=offset)

    # Stride family — mul ∈ _TRY_MULS, shift=0.
    for mul in _TRY_MULS:
        if mul == 1:  # already handled above
            continue
        if mul == 0:  # constant case
            continue
        x0, y0 = samples[0]
        add = y0 - mul * x0
        if all(y == mul * x + add for x, y in samples):
            return SlotFormula(shift=0, mul=mul, add=add)

    # Shifted stride family — mul × (X >> shift) + add.
    for shift in _TRY_SHIFTS:
        if shift == 0:
            continue  # already covered
        for mul in _TRY_MULS:
            if mul == 0:
                continue
            x0, y0 = samples[0]
            add = y0 - mul * (x0 >> shift)
            if all(y == mul * (x >> shift) + add for x, y in samples):
                return SlotFormula(shift=shift, mul=mul, add=add)

    return None


# =============================================================================
# Cluster + compaction units
# =============================================================================

@dataclass
class ClusterMember:
    """A single RTLPattern, paired with its extracted index list."""
    pattern: RTLPattern
    indices: Tuple[int, ...]

    @property
    def loop_index(self) -> int:
        """The first slot is treated as the loop variable X. For LHS
        patterns this is invariably the LHS's bit index."""
        return self.indices[0] if self.indices else 0


@dataclass
class TemplateCluster:
    """All members sharing the same template + pattern_type."""
    template: str
    pattern_type: str
    members: List[ClusterMember] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class CompactedUnit:
    """One packed-vector assertion that subsumes N siblings.

    The compacted assertion is in concat-and-mux form::

        ({lhs_pack} == ((~{sel_pack}) & {else_pack})
                    | ({sel_pack} & {then_pack}))

    Plus a comment header recording the compaction provenance.
    """
    assertion_text: str
    description: str
    pattern_type: str
    member_count: int
    member_lines: List[int]
    template: str
    formulas: Dict[int, SlotFormula]  # slot index -> formula


@dataclass
class RepresentativeUnit:
    """One representative pattern + a list of siblings that inherit
    its spec-check result. Used when arithmetic compaction failed and
    we still want to suppress redundant LLM enrichment."""
    representative: ClusterMember
    siblings: List[ClusterMember]
    template: str


@dataclass
class IndividualUnit:
    """A pattern with no sibling relationship — handled normally."""
    member: ClusterMember


CompactionUnit = Union[CompactedUnit, RepresentativeUnit, IndividualUnit]


@dataclass
class CompactionStats:
    n_input_patterns: int = 0
    n_clusters: int = 0
    n_compacted_units: int = 0
    n_representative_units: int = 0
    n_individual_units: int = 0
    n_members_compacted: int = 0
    n_members_via_representative: int = 0
    largest_cluster: int = 0
    largest_compacted_cluster: int = 0


@dataclass
class CompactionResult:
    units: List[CompactionUnit]
    stats: CompactionStats


# =============================================================================
# Clustering — group patterns by template
# =============================================================================

# Pattern types we'll attempt to cluster + compact via TEMPLATE
# normalisation (numeric suffixes / bracket indices → placeholders).
# seq_reset and seq_func are also included so bit-walk families like
# `accu_ctrl_ram_sel_X <= 0` cluster naturally; for seq_reset we
# additionally apply VALUE clustering (see below) so unrelated signals
# that share a reset value can pack into one assertion.
_CLUSTERABLE_TYPES = {
    "ternary_mux", "comb_comparison", "direct_assign",
    "wire_passthrough", "case_branch", "seq_reset", "seq_func",
    "if_branch",
}


# =============================================================================
# Value-level clustering for seq_reset
# =============================================================================
#
# Empirical observation: in NVDLA cacc, 191 of 214 seq_reset patterns
# reset their signal to zero (with various Verilog literal widths).
# Template clustering catches only 19 of those because the LHS signal
# names don't share a numeric suffix; the other 172 stay as singletons
# even though semantically they all say "this signal resets to zero".
#
# We group those by (clock, reset_polarity, reset_is_async,
# canonical_rhs) where canonical_rhs strips literal width:
#
#     1'b0       → 'CONST_ZERO
#     8'h0       → 'CONST_ZERO
#     4'b0000    → 'CONST_ZERO
#     {16{1'b0}} → 'CONST_ZERO
#     8'hff      → '8hff       (preserved — non-zero specific value)
#     reg_value  → reg_value   (expression, not a literal)
#
# Members with the same canonical key pack into one concat-equality
# assertion: `{sig1, sig2, ..., sigN} == '0` (or the canonical value).

_LITERAL_ZERO_RE = re.compile(
    r"^\s*\d+\s*'[bdhoBDHO]\s*0+\s*$"
)
_REPLICATED_ZERO_RE = re.compile(
    r"^\s*\{\s*\d+\s*\{\s*\d+\s*'[bdhoBDHO]\s*0+\s*\}\s*\}\s*$"
)
_LITERAL_ONE_RE = re.compile(
    r"^\s*\d+\s*'[bdhoBDHO]\s*1+\s*$"
)
_REPLICATED_ONE_RE = re.compile(
    r"^\s*\{\s*\d+\s*\{\s*\d+\s*'[bdhoBDHO]\s*1+\s*\}\s*\}\s*$"
)


def _canonical_constant(rhs: str) -> Optional[str]:
    """Map a Verilog literal expression to a width-less canonical
    form, or None if the expression isn't a recognised constant.

    Recognised:
    * Literal zero of any width — returns ``"'0"``
    * Replicated zero ``{N{1'b0}}`` — returns ``"'0"``
    * Literal all-ones (any width N where all bits are 1) — ``"'1"``
    * Replicated all-ones — ``"'1"``

    Other literal values (e.g. ``8'hff`` for a one-byte not-all-ones)
    are returned with width preserved so different specific values
    don't collapse.
    """
    s = rhs.strip()
    if _LITERAL_ZERO_RE.match(s) or _REPLICATED_ZERO_RE.match(s):
        return "'0"
    if _REPLICATED_ONE_RE.match(s):
        return "'1"
    # General sized-literal: parse the value and check whether it equals
    # 2^width - 1 (all-ones).  Handles hex (``8'hff``), decimal
    # (``3'd7``) and octal (``2'o3``) — earlier code only recognised
    # binary all-ones.
    m = re.match(
        r"\s*(\d+)\s*'([bdhoBDHO])\s*([0-9a-fA-F_xXzZ]+)\s*$", s,
    )
    if m:
        width = int(m.group(1))
        base = m.group(2).lower()
        value = m.group(3).replace("_", "")
        if any(c in value for c in "xXzZ"):
            # Don't classify partial-X / partial-Z literals.
            return f"{width}'{base}{value}"
        try:
            int_val = int(value, {"b": 2, "d": 10, "h": 16, "o": 8}[base])
        except (ValueError, KeyError):
            return f"{width}'{base}{value}"
        if int_val == 0:
            return "'0"
        if width > 0 and int_val == (1 << width) - 1:
            return "'1"
        return f"{width}'{base}{value}"
    return None  # not a recognised constant


@dataclass
class ValueCluster:
    """All seq_reset members that reset to the same canonical value."""
    canonical_value: str
    clock: Optional[str]
    reset: Optional[str]
    reset_polarity: str
    reset_is_async: bool
    members: List[ClusterMember] = field(default_factory=list)


def cluster_seq_reset_by_value(
    patterns: List[RTLPattern],
) -> Tuple[List[ValueCluster], List[RTLPattern]]:
    """Bucket seq_reset patterns by (clock, reset_polarity, async,
    canonical_value).

    Patterns whose RHS is not a recognised constant (e.g. resets to a
    register value rather than a literal) are returned in the
    leftover list for normal template-based handling.
    """
    buckets: Dict[Tuple[str, str, str, bool, str], ValueCluster] = {}
    leftovers: List[RTLPattern] = []
    for p in patterns:
        if p.pattern_type != "seq_reset":
            leftovers.append(p)
            continue
        cv = _canonical_constant(p.rhs)
        if cv is None:
            leftovers.append(p)
            continue
        key = (
            p.clock or "", p.reset or "",
            p.reset_polarity, p.reset_is_async, cv,
        )
        cluster = buckets.setdefault(key, ValueCluster(
            canonical_value=cv,
            clock=p.clock, reset=p.reset,
            reset_polarity=p.reset_polarity,
            reset_is_async=p.reset_is_async,
        ))
        cluster.members.append(ClusterMember(pattern=p, indices=()))
    return list(buckets.values()), leftovers


def _make_packed_reset_unit(
    cluster: ValueCluster,
) -> Optional[CompactedUnit]:
    """Emit a single packed-concat assertion for a value-cluster of
    seq_reset patterns.

    Form (active-low async reset)::

        // Compacted from N seq_reset assertions resetting to <value>
        assert property (@(posedge clk) !rstn |->
                         {sig1, sig2, ..., sigN} == '0)
                         else $error("Reset value mismatch in N signals");
    """
    if cluster.size_below_min:
        return None
    members = cluster.members
    if not members:
        return None

    # Build a stable ordering — by source line, then by signal name.
    sorted_members = sorted(
        members,
        key=lambda m: (m.pattern.source_line, m.pattern.lhs),
    )
    lhs_pack = "{" + ", ".join(m.pattern.lhs for m in sorted_members) + "}"

    clk = cluster.clock or "clk"
    rst = cluster.reset or "rst_n"
    rst_cond = f"!{rst}" if cluster.reset_polarity == "low" else rst

    if cluster.reset_is_async:
        timing = f"({rst_cond}) |-> "
    else:
        timing = f"({rst_cond}) |=> "

    description = (
        f"Compacted {len(sorted_members)} seq_reset assertions "
        f"resetting to {cluster.canonical_value}"
    )

    assertion_text = (
        f"// {description}\n"
        f"// Members: {', '.join(m.pattern.lhs for m in sorted_members[:6])}"
        f"{', ...' if len(sorted_members) > 6 else ''}\n"
        f"assert property (@(posedge {clk}) {timing}"
        f"{lhs_pack} == {cluster.canonical_value}) "
        f'else $error("Packed reset-value mismatch across '
        f'{len(sorted_members)} signals (canonical={cluster.canonical_value})");'
    )

    return CompactedUnit(
        assertion_text=assertion_text,
        description=description,
        pattern_type="seq_reset",
        member_count=len(sorted_members),
        member_lines=[m.pattern.source_line for m in sorted_members],
        template=f"<value-cluster> reset_to={cluster.canonical_value}",
        formulas={},  # not arithmetic — value-level grouping
    )


# Property added to ValueCluster post-hoc so we don't pollute the dataclass
# with computed fields the user shouldn't see in the constructor.
def _value_cluster_size(self: ValueCluster) -> int:
    return len(self.members)
ValueCluster.size = property(_value_cluster_size)  # type: ignore


def _value_cluster_size_below_min(self: ValueCluster) -> bool:
    # Even small reset clusters (size 2-3) are worth packing because
    # the cost is just one extra concat element per member, and the
    # LLM-cost saving is N-1 batches saved.
    return len(self.members) < 2
ValueCluster.size_below_min = property(_value_cluster_size_below_min)  # type: ignore


# =============================================================================
# Condition-level clustering for seq_func
# =============================================================================
#
# NVDLA monitor-signal mirroring is the largest seq_func family in
# every design we've measured.  Dozens of signals share a single
# condition (e.g. ``mon_op_en_pos == 1'b1``) and update one-to-one to
# their reg2dp_<name> source on the next clock.  Template clustering
# only catches the small subset whose signal name has a numeric
# suffix; the bulk are template-singletons because their suffixes are
# arbitrary identifiers.
#
# Bucket by (clock, reset, polarity, condition).  Pack as one
# concat-equality assertion per bucket (when size >= 2).  The packed
# form preserves per-member semantics because $past distributes over
# concat: ``$past({a,b,c}) ≡ {$past(a),$past(b),$past(c)}``.

@dataclass
class ConditionCluster:
    """All seq_func members triggered by the same condition under the
    same clock + reset polarity."""
    clock: Optional[str]
    reset: Optional[str]
    reset_polarity: str
    condition: str
    members: List[ClusterMember] = field(default_factory=list)


def _normalize_condition(cond: str) -> str:
    """Canonicalise an if-condition for clustering purposes.

    * Strip surrounding whitespace and outer balanced parens.
    * Strip a redundant trailing ``== 1'b1`` / ``== 1'd1`` / ``== 1``
      (and the ``===`` strict-compare variants), which are no-ops on
      a 1-bit boolean.
    * Collapse runs of whitespace to single spaces.

    The transform is purely syntactic — no logical equivalence
    checks beyond the ``==1`` redundancy.  Without this, conditions
    like ``en``, ``(en)``, and ``(en) == 1'b1`` end up in three
    separate clusters even though they semantically fire on the
    same scenario.
    """
    s = cond.strip()

    def _strip_outer_parens(text: str) -> str:
        """Strip a single layer of balanced outer parentheses."""
        if len(text) < 2 or text[0] != "(" or text[-1] != ")":
            return text
        depth = 0
        for i, ch in enumerate(text):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(text) - 1:
                    return text  # parens don't wrap the whole expression
        return text[1:-1].strip()

    # Iterate paren-strip + ==1-strip until neither changes.  Without
    # the loop, ``(en) == 1'b1`` first paren-strips to itself (no
    # outer parens because of the trailing ``== 1'b1``), then strips
    # to ``(en)``, and we'd stop with a leftover paren.
    prev = None
    while prev != s:
        prev = s
        s = _strip_outer_parens(s)
        s = re.sub(r"\s*={2,3}\s*1'b1\s*$", "", s)
        s = re.sub(r"\s*={2,3}\s*1'd1\s*$", "", s)
        s = re.sub(r"\s*={2,3}\s*1\s*$", "", s)
        s = s.strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def cluster_seq_func_by_condition(
    patterns: List[RTLPattern],
) -> Tuple[List[ConditionCluster], List[RTLPattern]]:
    """Group seq_func patterns by (clock, reset, polarity, condition).

    Returns (clusters, leftovers).  Patterns without a condition are
    pushed to leftovers — they need normal template handling.

    The condition is normalised via :func:`_normalize_condition`
    before bucketing so equivalent conditions (e.g. ``en``,
    ``(en)``, ``(en) == 1'b1``) cluster together.
    """
    buckets: Dict[Tuple[str, str, str, str], ConditionCluster] = {}
    leftovers: List[RTLPattern] = []
    for p in patterns:
        if p.pattern_type != "seq_func":
            leftovers.append(p)
            continue
        if not p.condition:
            leftovers.append(p)
            continue
        norm_cond = _normalize_condition(p.condition)
        key = (p.clock or "", p.reset or "", p.reset_polarity, norm_cond)
        cluster = buckets.setdefault(key, ConditionCluster(
            clock=p.clock, reset=p.reset,
            reset_polarity=p.reset_polarity,
            # Keep the normalised form for clearer error messages.
            condition=norm_cond,
        ))
        cluster.members.append(ClusterMember(pattern=p, indices=()))
    return list(buckets.values()), leftovers


def _make_packed_seq_func_unit(
    cluster: ConditionCluster,
) -> Optional[CompactedUnit]:
    """Pack same-condition seq_func family into one concat-equality
    assertion::

        (cond) |=> {lhs1, lhs2, ..., lhsN}
                == $past({rhs1, rhs2, ..., rhsN})
    """
    members = cluster.members
    if len(members) < 2:
        return None

    sorted_members = sorted(
        members,
        key=lambda m: (m.pattern.source_line, m.pattern.lhs),
    )
    lhs_pack = "{" + ", ".join(m.pattern.lhs for m in sorted_members) + "}"
    rhs_pack = "{" + ", ".join(m.pattern.rhs for m in sorted_members) + "}"

    clk = cluster.clock or "clk"
    rst = cluster.reset or "rst_n"
    rst_cond = f"!{rst}" if cluster.reset_polarity == "low" else rst

    description = (
        f"Compacted {len(sorted_members)} seq_func assertions "
        f"sharing condition '{cluster.condition[:40]}'"
    )

    sample_lhs = ", ".join(m.pattern.lhs for m in sorted_members[:5])
    if len(sorted_members) > 5:
        sample_lhs += ", ..."

    assertion_text = (
        f"// {description}\n"
        f"// LHS: {sample_lhs}\n"
        f"assert property (@(posedge {clk}) disable iff ({rst_cond}) "
        f"({cluster.condition}) |=> {lhs_pack} == $past({rhs_pack})) "
        f'else $error("Packed seq_func mismatch across '
        f'{len(sorted_members)} signals under condition '
        f"{cluster.condition[:40]});"
        '"'
    )
    # Fix the closing quote/paren ordering above.
    assertion_text = (
        f"// {description}\n"
        f"// LHS: {sample_lhs}\n"
        f"assert property (@(posedge {clk}) disable iff ({rst_cond}) "
        f"({cluster.condition}) |=> {lhs_pack} == $past({rhs_pack})) "
        f'else $error("Packed seq_func mismatch across '
        f'{len(sorted_members)} signals");'
    )

    return CompactedUnit(
        assertion_text=assertion_text,
        description=description,
        pattern_type="seq_func",
        member_count=len(sorted_members),
        member_lines=[m.pattern.source_line for m in sorted_members],
        template=f"<condition-cluster> cond={cluster.condition[:40]}",
        formulas={},
    )


# =============================================================================
# Constant-value clustering for combinational direct_assign / wire_passthrough
# =============================================================================
#
# Same logic as seq_reset value-clustering but for combinational
# assigns where the RHS is a recognised constant.  Packs as an
# unconditional concat-equality (no clock or reset gating).

@dataclass
class CombValueCluster:
    """Combinational signals all assigned the same canonical literal."""
    canonical_value: str
    members: List[ClusterMember] = field(default_factory=list)


def cluster_comb_by_value(
    patterns: List[RTLPattern],
) -> Tuple[List[CombValueCluster], List[RTLPattern]]:
    """Bucket direct_assign + wire_passthrough patterns whose RHS
    is a recognised Verilog constant.

    Returns (clusters, leftovers).
    """
    buckets: Dict[str, CombValueCluster] = {}
    leftovers: List[RTLPattern] = []
    for p in patterns:
        if p.pattern_type not in ("direct_assign", "wire_passthrough"):
            leftovers.append(p)
            continue
        cv = _canonical_constant(p.rhs)
        if cv is None:
            leftovers.append(p)
            continue
        cluster = buckets.setdefault(cv, CombValueCluster(canonical_value=cv))
        cluster.members.append(ClusterMember(pattern=p, indices=()))
    return list(buckets.values()), leftovers


def _make_packed_comb_unit(
    cluster: CombValueCluster,
) -> Optional[CompactedUnit]:
    """Emit a single unconditional concat-equality for a combinational
    value-cluster::

        {sig1, sig2, ..., sigN} == '0
    """
    members = cluster.members
    if len(members) < 2:
        return None
    sorted_members = sorted(
        members,
        key=lambda m: (m.pattern.source_line, m.pattern.lhs),
    )
    lhs_pack = "{" + ", ".join(m.pattern.lhs for m in sorted_members) + "}"

    description = (
        f"Compacted {len(sorted_members)} combinational signals "
        f"all assigned to {cluster.canonical_value}"
    )
    assertion_text = (
        f"// {description}\n"
        f"// Members: {', '.join(m.pattern.lhs for m in sorted_members[:6])}"
        f"{', ...' if len(sorted_members) > 6 else ''}\n"
        f"assert ({lhs_pack} == {cluster.canonical_value}) "
        f'else $error("Packed combinational value mismatch across '
        f'{len(sorted_members)} signals (canonical={cluster.canonical_value})");'
    )
    return CompactedUnit(
        assertion_text=assertion_text,
        description=description,
        pattern_type="direct_assign",
        member_count=len(sorted_members),
        member_lines=[m.pattern.source_line for m in sorted_members],
        template=f"<comb-value-cluster> rhs={cluster.canonical_value}",
        formulas={},
    )


# =============================================================================
# Row clustering for case_branch
# =============================================================================
#
# When one case statement drives multiple LHS, the AST extractor
# emits one assertion per (selector, value, lhs) triple.  For a case
# with B branches × L lhs signals, that's B×L assertions.  Bucketing
# by (selector, condition) collapses the L lhs in each branch into a
# single per-branch concat-equality assertion.

@dataclass
class CaseRowCluster:
    """All case_branch patterns sharing the same (selector, condition)."""
    selector: str
    condition: str
    members: List[ClusterMember] = field(default_factory=list)


def cluster_case_branches_by_row(
    patterns: List[RTLPattern],
) -> Tuple[List[CaseRowCluster], List[RTLPattern]]:
    """Bucket case_branch patterns by (selector, condition).

    The ``default`` branch is excluded — ``default`` is a SystemVerilog
    case keyword, not a literal, so it can't be substituted into the
    `(sel == cond)` template form the existing
    `_pattern_to_skeleton()` already chooses to skip default for the
    same reason.
    """
    buckets: Dict[Tuple[str, str], CaseRowCluster] = {}
    leftovers: List[RTLPattern] = []
    for p in patterns:
        if p.pattern_type != "case_branch":
            leftovers.append(p)
            continue
        if not (p.selector and p.condition):
            leftovers.append(p)
            continue
        if p.condition.strip().lower() == "default":
            leftovers.append(p)  # caller will drop it (extractor skips too)
            continue
        key = (p.selector, p.condition)
        cluster = buckets.setdefault(key, CaseRowCluster(
            selector=p.selector, condition=p.condition,
        ))
        cluster.members.append(ClusterMember(pattern=p, indices=()))
    return list(buckets.values()), leftovers


@dataclass
class IfRowCluster:
    """All if_branch patterns sharing the same (selector, branch).

    ``selector`` is the if-condition expression, ``branch`` is
    ``"then"`` or ``"else"``.  Members all assigned simultaneously
    when the branch is taken.
    """
    selector: str
    branch: str  # "then" or "else"
    members: List[ClusterMember] = field(default_factory=list)


def cluster_if_branches_by_row(
    patterns: List[RTLPattern],
) -> Tuple[List[IfRowCluster], List[RTLPattern]]:
    """Bucket if_branch patterns by (selector, branch).

    Combinational mux semantics: every LHS in the same branch updates
    simultaneously when the if-condition (selector) holds.  Packing
    them into one concat-equality preserves per-LHS coverage.
    """
    buckets: Dict[Tuple[str, str], IfRowCluster] = {}
    leftovers: List[RTLPattern] = []
    for p in patterns:
        if p.pattern_type != "if_branch":
            leftovers.append(p)
            continue
        if not (p.selector and p.condition):
            leftovers.append(p)
            continue
        if p.condition not in ("then", "else"):
            leftovers.append(p)
            continue
        key = (p.selector, p.condition)
        cluster = buckets.setdefault(key, IfRowCluster(
            selector=p.selector, branch=p.condition,
        ))
        cluster.members.append(ClusterMember(pattern=p, indices=()))
    return list(buckets.values()), leftovers


def _make_packed_if_row_unit(
    cluster: IfRowCluster,
) -> Optional[CompactedUnit]:
    """Pack one if-branch row into a single multi-signal implication.

    For the then-branch::

        !(cond) || ({lhs1, ..., lhsN} == {rhs1, ..., rhsN})

    For the else-branch::

        (cond) || ({lhs1, ..., lhsN} == {rhs1, ..., rhsN})

    Both forms preserve per-member semantics via concat-position
    alignment, the same trick we use for case_row.
    """
    members = cluster.members
    if len(members) < 2:
        return None
    sorted_members = sorted(
        members,
        key=lambda m: (m.pattern.source_line, m.pattern.lhs),
    )
    lhs_pack = "{" + ", ".join(m.pattern.lhs for m in sorted_members) + "}"
    rhs_pack = "{" + ", ".join(m.pattern.rhs for m in sorted_members) + "}"

    cond = cluster.selector
    if cluster.branch == "then":
        body_guard = f"!({cond})"
        polarity_text = f"if ({cond})"
    else:
        body_guard = f"({cond})"
        polarity_text = f"if (!({cond}))"

    description = (
        f"Compacted {len(sorted_members)} if-branch assignments "
        f"in {cluster.branch}-branch of '{cond[:40]}'"
    )
    assertion_text = (
        f"// {description}\n"
        f"// {polarity_text} → {len(sorted_members)} signals update "
        f"in lockstep\n"
        f"assert ({body_guard} || ({lhs_pack} == {rhs_pack})) "
        f'else $error("Packed if-row mismatch in {cluster.branch}-branch '
        f'of {cond[:40]} (across {len(sorted_members)} LHS)");'
    )
    return CompactedUnit(
        assertion_text=assertion_text,
        description=description,
        pattern_type="if_branch",
        member_count=len(sorted_members),
        member_lines=[m.pattern.source_line for m in sorted_members],
        template=f"<if-row> sel={cluster.selector} branch={cluster.branch}",
        formulas={},
    )


def _make_packed_case_row_unit(
    cluster: CaseRowCluster,
) -> Optional[CompactedUnit]:
    """Pack one case-row into a single multi-signal implication::

        !(selector == condition) || ({lhs1, ..., lhsN} == {rhs1, ..., rhsN})

    All members come from the same physical case branch; their
    individual ``lhs == rhs`` checks compose lossless-ly into a
    concat-equality.
    """
    members = cluster.members
    if len(members) < 2:
        return None
    sorted_members = sorted(
        members,
        key=lambda m: (m.pattern.source_line, m.pattern.lhs),
    )
    lhs_pack = "{" + ", ".join(m.pattern.lhs for m in sorted_members) + "}"
    rhs_pack = "{" + ", ".join(m.pattern.rhs for m in sorted_members) + "}"

    description = (
        f"Compacted {len(sorted_members)} case-branch assignments "
        f"in row ({cluster.selector} == {cluster.condition})"
    )
    assertion_text = (
        f"// {description}\n"
        f"assert (!(({cluster.selector}) == {cluster.condition}) || "
        f"({lhs_pack} == {rhs_pack})) "
        f'else $error("Packed case-row mismatch when '
        f'{cluster.selector}=={cluster.condition} '
        f'(across {len(sorted_members)} LHS)");'
    )
    return CompactedUnit(
        assertion_text=assertion_text,
        description=description,
        pattern_type="case_branch",
        member_count=len(sorted_members),
        member_lines=[m.pattern.source_line for m in sorted_members],
        template=(
            f"<case-row> sel={cluster.selector} val={cluster.condition}"
        ),
        formulas={},
    )


def cluster_patterns(
    patterns: List[RTLPattern],
) -> List[TemplateCluster]:
    """Group patterns by (pattern_type, normalized template).

    Patterns whose pattern_type is not in ``_CLUSTERABLE_TYPES`` (e.g.
    seq_reset, seq_func) are returned as singleton clusters so
    downstream code can treat them uniformly.
    """
    by_key: Dict[Tuple[str, str], TemplateCluster] = {}
    for p in patterns:
        if p.pattern_type not in _CLUSTERABLE_TYPES:
            # Singleton — non-clusterable. Use a unique key to keep it
            # in its own bucket.
            key = (p.pattern_type, f"__singleton_{id(p)}__")
            by_key[key] = TemplateCluster(
                template=key[1], pattern_type=p.pattern_type,
                members=[ClusterMember(pattern=p, indices=())],
            )
            continue
        body = f"{p.lhs} == {p.rhs}"
        if p.condition is not None:
            body = f"({p.selector} == {p.condition}) -> {body}"
        template, indices = _normalize_template(body)
        key = (p.pattern_type, template)
        cluster = by_key.setdefault(
            key, TemplateCluster(template=template, pattern_type=p.pattern_type),
        )
        cluster.members.append(
            ClusterMember(pattern=p, indices=tuple(indices))
        )
    return list(by_key.values())


# =============================================================================
# Recursive compaction
# =============================================================================

def _derive_cluster_pattern(
    members: List[ClusterMember],
) -> Optional[Dict[int, SlotFormula]]:
    """For each non-loop slot k (k = 1, 2, ...), fit a SlotFormula
    expressing slot_k as a function of slot_0 (the loop variable).

    Slot 0 is treated as the loop variable itself (formula y = x).

    Returns a {slot_index -> formula} map, or None if any non-loop
    slot can't be fit by a single formula.
    """
    if not members or len(members[0].indices) == 0:
        return None
    n_slots = len(members[0].indices)
    formulas: Dict[int, SlotFormula] = {}
    for slot in range(n_slots):
        if slot == 0:
            formulas[0] = SlotFormula(shift=0, mul=1, add=0)  # identity
            continue
        samples = [(m.loop_index, m.indices[slot]) for m in members]
        formula = _fit_slot_formula(samples)
        if formula is None:
            return None
        formulas[slot] = formula
    return formulas


def _largest_consistent_subset(
    members: List[ClusterMember],
) -> Tuple[List[ClusterMember], List[ClusterMember],
           Optional[Dict[int, SlotFormula]]]:
    """Find the largest subset of cluster members for which a single
    SlotFormula per slot fits, plus the residual members.

    Strategy: pick a seed (first member), pair with every other member
    in turn to derive a candidate (slot_k formula) hypothesis, test
    against the full cluster. The hypothesis with the largest agreeing
    subset wins.

    Returns (agreeing, disagreeing, formulas_or_None).
    """
    if len(members) < 2:
        return members, [], _derive_cluster_pattern(members)

    seed = members[0]
    n_slots = len(seed.indices)
    if n_slots == 0:
        return members, [], None

    best_agreeing: List[ClusterMember] = [seed]
    best_disagreeing: List[ClusterMember] = list(members[1:])
    best_formulas: Optional[Dict[int, SlotFormula]] = None

    for candidate in members[1:]:
        # Fit each non-loop slot using just the seed + this candidate.
        # If any slot can't be fit by a single formula on these two,
        # skip this candidate.
        candidate_formulas: Dict[int, SlotFormula] = {
            0: SlotFormula(shift=0, mul=1, add=0)
        }
        ok = True
        for slot in range(1, n_slots):
            two_samples = [
                (seed.loop_index, seed.indices[slot]),
                (candidate.loop_index, candidate.indices[slot]),
            ]
            f = _fit_slot_formula(two_samples)
            if f is None:
                ok = False
                break
            candidate_formulas[slot] = f
        if not ok:
            continue

        # Evaluate this hypothesis against every member in the cluster.
        agreeing: List[ClusterMember] = []
        disagreeing: List[ClusterMember] = []
        for m in members:
            x = m.loop_index
            consistent = True
            for slot in range(n_slots):
                if candidate_formulas[slot].evaluate(x) != m.indices[slot]:
                    consistent = False
                    break
            if consistent:
                agreeing.append(m)
            else:
                disagreeing.append(m)

        if len(agreeing) > len(best_agreeing):
            best_agreeing = agreeing
            best_disagreeing = disagreeing
            best_formulas = candidate_formulas

    return best_agreeing, best_disagreeing, best_formulas


def recursive_compact(
    members: List[ClusterMember],
    *,
    min_compact_size: int = 5,
    max_depth: int = 4,
    _depth: int = 0,
) -> List[Tuple[List[ClusterMember], Optional[Dict[int, SlotFormula]]]]:
    """Recursively partition a cluster into compactable subsets.

    Returns a list of (subset, formulas) pairs:

    * ``formulas`` is a dict per non-loop slot when the subset
      compacts — caller can emit a single packed assertion.
    * ``formulas`` is None when the subset is too small to compact
      or no arithmetic relationship was found — caller emits each
      member individually.

    Termination: each recursion strictly reduces residual size; max
    depth bounds recursion depth as a safety measure.
    """
    if not members:
        return []
    if _depth >= max_depth:
        return [(members, None)]
    if len(members) < min_compact_size:
        return [(members, None)]

    agreeing, disagreeing, formulas = _largest_consistent_subset(members)

    units: List[Tuple[List[ClusterMember], Optional[Dict[int, SlotFormula]]]] = []

    if formulas is not None and len(agreeing) >= min_compact_size:
        units.append((agreeing, formulas))
        if disagreeing:
            units.extend(
                recursive_compact(
                    disagreeing,
                    min_compact_size=min_compact_size,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                )
            )
        return units

    # No agreeing subset large enough — emit everyone individually.
    return [(members, None)]


# =============================================================================
# Symbolic verification
# =============================================================================

def verify_compaction(
    agreeing: List[ClusterMember],
    formulas: Dict[int, SlotFormula],
) -> bool:
    """Substitute each member's loop-variable index back into the
    template and verify that the regenerated (lhs, rhs) match the
    member's original (lhs, rhs) exactly.

    Catches edge cases _largest_consistent_subset may have missed if
    the index extractor's slot ordering disagrees with the placeholder
    expansion ordering.

    Returns True iff every member round-trips correctly.
    """
    if not agreeing:
        return False
    template_member = agreeing[0]
    body = f"{template_member.pattern.lhs} == {template_member.pattern.rhs}"
    template_str, _ = _normalize_template(body)

    for m in agreeing:
        x = m.loop_index
        # Compute the predicted values for each slot.
        predicted = [formulas[k].evaluate(x) for k in range(len(m.indices))]
        if predicted != list(m.indices):
            return False
        # Reinstate the predicted values into the template.
        regenerated = _reinflate_template(template_str, predicted)
        original_body = f"{m.pattern.lhs} == {m.pattern.rhs}"
        # Normalise whitespace before comparing.
        if _flatten_ws(regenerated) != _flatten_ws(original_body):
            return False
    return True


def _reinflate_template(template: str, indices: List[int]) -> str:
    """Replace each `_*` and `[*]` placeholder with the corresponding
    integer from ``indices`` in left-to-right order."""
    result_chars: List[str] = []
    i = 0
    n = len(template)
    idx_pos = 0
    while i < n:
        ch = template[i]
        if ch == "_" and i + 1 < n and template[i + 1] == "*":
            # _* placeholder
            if idx_pos < len(indices):
                result_chars.append("_" + str(indices[idx_pos]))
                idx_pos += 1
            else:
                result_chars.append("_*")
            i += 2
        elif ch == "[" and i + 2 < n and template[i + 1] == "*" \
                and template[i + 2] == "]":
            if idx_pos < len(indices):
                result_chars.append("[" + str(indices[idx_pos]) + "]")
                idx_pos += 1
            else:
                result_chars.append("[*]")
            i += 3
        else:
            result_chars.append(ch)
            i += 1
    return "".join(result_chars)


_WS_RE = re.compile(r"\s+")


def _flatten_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


# =============================================================================
# Concat-and-mux assertion emitter
# =============================================================================

def _make_compacted_skeleton(
    agreeing: List[ClusterMember],
    formulas: Dict[int, SlotFormula],
) -> Optional[CompactedUnit]:
    """Produce the packed concat-and-mux assertion for a verified
    arithmetic cluster.

    Currently supported pattern shape: ``ternary_mux`` whose RHS has
    the canonical form ``sel ? then : else``, where each of (lhs, sel,
    then, else) is a single scalar identifier with an integer index
    suffix. This covers all cacc bit-walk families; other shapes fall
    through to ``None`` (caller emits as representative + siblings).

    The generated form::

        // Compacted from N siblings: lhs_X..lhs_Y
        // Loop variable X spans [low..high]; per-bit mux:
        //   lhs[X] == sel[X] ? then[X] : else[X]
        // Index relationships: sel = X-32; then = X; else = X-48
        assert (
          {<lhs_pack>}
          == ((~{<sel_pack>}) & {<else_pack>})
           | ({<sel_pack>} & {<then_pack>})
        ) else $error("Compacted bit-mux family <name> mismatch");
    """
    if not agreeing:
        return None
    seed = agreeing[0]
    if seed.pattern.pattern_type != "ternary_mux":
        # Other types may benefit from compaction too but their
        # emission rules differ; leave as None for now and let the
        # caller emit representative + siblings.
        return None

    # Parse the ternary RHS to extract the sel / then / else operands.
    parsed = _parse_ternary(seed.pattern.rhs)
    if parsed is None:
        return None
    sel_text, then_text, else_text = parsed

    # All four expressions (lhs, sel, then, else) should have been
    # captured as numeric-suffix slots in the index list. We expect
    # 4 indices per member: [lhs_idx, sel_idx, then_idx, else_idx].
    # If the count differs, the shape doesn't match what this emitter
    # knows how to pack.
    if len(seed.indices) != 4:
        return None

    # Sort agreeing members by loop_index descending so the concat is
    # naturally MSB-first (matches Verilog convention).
    sorted_members = sorted(agreeing, key=lambda m: -m.loop_index)
    xs = [m.loop_index for m in sorted_members]
    if not xs:
        return None
    low, high = min(xs), max(xs)

    # Build packed concatenations.
    lhs_template = re.sub(r"_\d+\b", "_*", seed.pattern.lhs)
    sel_template = re.sub(r"\[\d+\]", "[*]", sel_text)
    then_template = re.sub(r"_\d+\b", "_*", then_text)
    else_template = re.sub(r"_\d+\b", "_*", else_text)

    def _pack(template_str: str, slot_idx: int) -> str:
        items = []
        for m in sorted_members:
            value = m.indices[slot_idx]
            items.append(_substitute_first_placeholder(template_str, value))
        return "{" + ", ".join(items) + "}"

    lhs_pack = _pack(lhs_template, 0)
    sel_pack = _pack(sel_template, 1)
    then_pack = _pack(then_template, 2)
    else_pack = _pack(else_template, 3)

    family_name = re.sub(r"_\*$", "", lhs_template)
    member_lines = [m.pattern.source_line for m in sorted_members]

    description = (
        f"Compacted {len(agreeing)} sibling assertions for "
        f"{family_name}_<{low}..{high}>: per-bit mux "
        f"{family_name}[X] == sel[f(X)] ? then[g(X)] : else[h(X)]"
    )

    formula_str = "; ".join(
        f"slot{k}: {f}" for k, f in sorted(formulas.items())
    )

    assertion_text = (
        f"// Compacted from {len(agreeing)} sibling assertions "
        f"({family_name}_<{low}..{high}>)\n"
        f"// Index formulas — {formula_str}\n"
        f"assert (\n"
        f"  {lhs_pack}\n"
        f"  == ((~{sel_pack}) & {else_pack})\n"
        f"   | ({sel_pack} & {then_pack})\n"
        f') else $error("Compacted bit-mux family '
        f'{family_name} mismatch");'
    )

    return CompactedUnit(
        assertion_text=assertion_text,
        description=description,
        pattern_type=seed.pattern.pattern_type,
        member_count=len(agreeing),
        member_lines=member_lines,
        template=seed.pattern.lhs + " == " + seed.pattern.rhs,
        formulas=dict(formulas),
    )


def _substitute_first_placeholder(template_str: str, value: int) -> str:
    """Replace the first `_*` or `[*]` in ``template_str`` with the
    given integer value."""
    # Try identifier-suffix placeholder first.
    new_text, n = re.subn(r"_\*", f"_{value}", template_str, count=1)
    if n == 1:
        return new_text
    new_text, n = re.subn(r"\[\*\]", f"[{value}]", template_str, count=1)
    if n == 1:
        return new_text
    return template_str


# Match a top-level ternary `cond ? then : else`. We need to be
# careful with nested parentheses.
def _parse_ternary(rhs: str) -> Optional[Tuple[str, str, str]]:
    """Parse a top-level ternary expression. Returns
    (cond, then_branch, else_branch) or None when the RHS isn't a
    single top-level ternary.
    """
    s = rhs.strip()
    # Strip exactly one outer paren pair if present.
    if s.startswith("(") and s.endswith(")"):
        # Verify the parens are balanced top-level.
        depth = 0
        balanced_outer = True
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    balanced_outer = False
                    break
        if balanced_outer:
            s = s[1:-1].strip()

    # Find `?` at depth 0.
    depth = 0
    q_pos = -1
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "?" and depth == 0:
            q_pos = i
            break
    if q_pos < 0:
        return None
    cond = s[:q_pos].strip()

    # Find matching `:` at depth 0 after q_pos.
    depth = 0
    c_pos = -1
    for i in range(q_pos + 1, len(s)):
        ch = s[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == ":" and depth == 0:
            c_pos = i
            break
    if c_pos < 0:
        return None
    then_branch = s[q_pos + 1:c_pos].strip()
    else_branch = s[c_pos + 1:].strip()
    # Strip outer parens on cond if present.
    if cond.startswith("(") and cond.endswith(")"):
        cond = cond[1:-1].strip()
    return cond, then_branch, else_branch


# =============================================================================
# Top-level entry point
# =============================================================================

def cluster_and_compact(
    patterns: List[RTLPattern],
    *,
    min_compact_size: int = 5,
    max_depth: int = 4,
    enable_compaction: bool = True,
    enable_value_clustering: bool = True,
    value_cluster_min_size: int = 2,
) -> CompactionResult:
    """End-to-end: cluster patterns, recursively compact each cluster,
    verify, and produce a list of CompactionUnits.

    Two complementary clustering strategies run before recursive
    compaction:

    1. **Value clustering** for ``seq_reset`` patterns.  Bucket by
       (clock, reset_polarity, async, canonical_value) where the
       canonical value strips Verilog literal width (``8'h0`` → ``'0``).
       Same-value reset families pack into one concat-equality
       assertion: ``{sig1, sig2, ...} == '0``.  This catches the
       common case where dozens of unrelated signals all reset to
       zero — template clustering misses these because the LHS names
       don't share a numeric suffix.

    2. **Template clustering** for the remaining patterns.  Group by
       abstract template (``calc_dlv_elem_*`` / ``cfg_is_int8[*]``)
       then run recursive arithmetic compaction.  Catches bit-walks.

    Parameters
    ----------
    patterns : list of RTLPattern
    min_compact_size : int
        Skip arithmetic compaction for clusters smaller than this.
    max_depth : int
        Recursion-depth ceiling for ``recursive_compact``.
    enable_compaction : bool
        When False, every cluster falls through to representative +
        siblings (no concat-mux emission). Useful for ablations.
    enable_value_clustering : bool
        When False, seq_reset patterns skip value clustering and go
        through normal template clustering only.
    value_cluster_min_size : int
        Minimum value-cluster size to pack (default 2 — even tiny
        packs save an LLM round-trip).
    """
    stats = CompactionStats(n_input_patterns=len(patterns))

    units: List[CompactionUnit] = []

    # --- Phase 1: value-level clustering for seq_reset patterns ---
    if enable_value_clustering:
        value_clusters, leftover_after_seq_reset = cluster_seq_reset_by_value(
            [p for p in patterns if p.pattern_type == "seq_reset"]
        )
        non_seq_reset = [p for p in patterns if p.pattern_type != "seq_reset"]
        for vc in value_clusters:
            if vc.size >= value_cluster_min_size:
                packed = _make_packed_reset_unit(vc)
                if packed is not None:
                    units.append(packed)
                    stats.n_compacted_units += 1
                    stats.n_members_compacted += vc.size
                    stats.largest_compacted_cluster = max(
                        stats.largest_compacted_cluster, vc.size
                    )
                    continue
            # Too small to pack OR emit failed — fall through as
            # individuals via the template path.
            for m in vc.members:
                leftover_after_seq_reset.append(m.pattern)
        # Combine non-seq_reset patterns with the leftovers.
        patterns_after_phase1 = non_seq_reset + leftover_after_seq_reset

        # --- Phase 1b: condition-level clustering for seq_func ---
        sf_clusters, leftover_after_sf = cluster_seq_func_by_condition(
            [p for p in patterns_after_phase1
             if p.pattern_type == "seq_func"]
        )
        non_seq_func = [p for p in patterns_after_phase1
                        if p.pattern_type != "seq_func"]
        for sfc in sf_clusters:
            if len(sfc.members) >= value_cluster_min_size:
                packed = _make_packed_seq_func_unit(sfc)
                if packed is not None:
                    units.append(packed)
                    stats.n_compacted_units += 1
                    stats.n_members_compacted += len(sfc.members)
                    stats.largest_compacted_cluster = max(
                        stats.largest_compacted_cluster, len(sfc.members)
                    )
                    continue
            for m in sfc.members:
                leftover_after_sf.append(m.pattern)
        patterns_after_phase1b = non_seq_func + leftover_after_sf

        # --- Phase 1c: combinational value clustering ---
        comb_clusters, leftover_after_comb = cluster_comb_by_value(
            [p for p in patterns_after_phase1b
             if p.pattern_type in ("direct_assign", "wire_passthrough")]
        )
        non_comb = [
            p for p in patterns_after_phase1b
            if p.pattern_type not in ("direct_assign", "wire_passthrough")
        ]
        for cvc in comb_clusters:
            if len(cvc.members) >= value_cluster_min_size:
                packed = _make_packed_comb_unit(cvc)
                if packed is not None:
                    units.append(packed)
                    stats.n_compacted_units += 1
                    stats.n_members_compacted += len(cvc.members)
                    stats.largest_compacted_cluster = max(
                        stats.largest_compacted_cluster, len(cvc.members)
                    )
                    continue
            for m in cvc.members:
                leftover_after_comb.append(m.pattern)
        patterns_after_phase1c = non_comb + leftover_after_comb

        # --- Phase 1d: case-row clustering ---
        case_row_clusters, leftover_after_case = (
            cluster_case_branches_by_row(
                [p for p in patterns_after_phase1c
                 if p.pattern_type == "case_branch"]
            )
        )
        non_case = [p for p in patterns_after_phase1c
                    if p.pattern_type != "case_branch"]
        for crc in case_row_clusters:
            if len(crc.members) >= value_cluster_min_size:
                packed = _make_packed_case_row_unit(crc)
                if packed is not None:
                    units.append(packed)
                    stats.n_compacted_units += 1
                    stats.n_members_compacted += len(crc.members)
                    stats.largest_compacted_cluster = max(
                        stats.largest_compacted_cluster, len(crc.members)
                    )
                    continue
            for m in crc.members:
                leftover_after_case.append(m.pattern)
        patterns_after_phase1d = non_case + leftover_after_case

        # --- Phase 1e: if-row clustering ---
        if_row_clusters, leftover_after_if = (
            cluster_if_branches_by_row(
                [p for p in patterns_after_phase1d
                 if p.pattern_type == "if_branch"]
            )
        )
        non_if = [p for p in patterns_after_phase1d
                  if p.pattern_type != "if_branch"]
        for irc in if_row_clusters:
            if len(irc.members) >= value_cluster_min_size:
                packed = _make_packed_if_row_unit(irc)
                if packed is not None:
                    units.append(packed)
                    stats.n_compacted_units += 1
                    stats.n_members_compacted += len(irc.members)
                    stats.largest_compacted_cluster = max(
                        stats.largest_compacted_cluster, len(irc.members)
                    )
                    continue
            for m in irc.members:
                leftover_after_if.append(m.pattern)
        patterns_for_template = non_if + leftover_after_if
    else:
        patterns_for_template = list(patterns)

    # --- Phase 2: template + arithmetic clustering for everything else ---
    clusters = cluster_patterns(patterns_for_template)
    stats.n_clusters = len(clusters)
    if clusters:
        stats.largest_cluster = max(stats.largest_cluster,
                                    max(c.size for c in clusters))

    for cluster in clusters:
        if cluster.size == 1:
            units.append(IndividualUnit(member=cluster.members[0]))
            stats.n_individual_units += 1
            continue

        compact_attempts = (
            recursive_compact(
                cluster.members,
                min_compact_size=min_compact_size,
                max_depth=max_depth,
            )
            if enable_compaction
            else [(cluster.members, None)]
        )

        for subset, formulas in compact_attempts:
            if formulas is not None and verify_compaction(subset, formulas):
                compacted = _make_compacted_skeleton(subset, formulas)
                if compacted is not None:
                    units.append(compacted)
                    stats.n_compacted_units += 1
                    stats.n_members_compacted += len(subset)
                    stats.largest_compacted_cluster = max(
                        stats.largest_compacted_cluster, len(subset)
                    )
                    continue
            # Compaction failed (no formula, verification failed, or
            # the emitter doesn't know how to pack this shape) — fall
            # back to a representative.
            if len(subset) > 1:
                units.append(RepresentativeUnit(
                    representative=subset[0],
                    siblings=subset[1:],
                    template=cluster.template,
                ))
                stats.n_representative_units += 1
                stats.n_members_via_representative += len(subset) - 1
            else:
                units.append(IndividualUnit(member=subset[0]))
                stats.n_individual_units += 1

    logger.info(
        "AST clustering: %d patterns → %d clusters → "
        "%d compacted (%d members), %d representatives (%d siblings), "
        "%d individuals. Largest cluster %d (largest compacted %d).",
        stats.n_input_patterns, stats.n_clusters,
        stats.n_compacted_units, stats.n_members_compacted,
        stats.n_representative_units, stats.n_members_via_representative,
        stats.n_individual_units,
        stats.largest_cluster, stats.largest_compacted_cluster,
    )

    return CompactionResult(units=units, stats=stats)


# =============================================================================
# Bridging to AssertionSkeleton (the agent's existing currency)
# =============================================================================

def units_to_skeletons(
    result: CompactionResult,
) -> Tuple[List["AssertionSkeleton"], List["AssertionSkeleton"]]:
    """Convert a CompactionResult into two skeleton lists:

    * **llm_bound**: one skeleton per CompactedUnit + RepresentativeUnit
      representative + IndividualUnit. These all go through the LLM
      spec-vs-RTL conformance check (Policy B — uniform spec validation
      on every unique RTL template).
    * **direct_bound**: one skeleton per RepresentativeUnit *sibling*,
      tagged with a ``// SIBLING_OF <template>`` breadcrumb.  Siblings
      bypass the LLM because their representative carries the spec
      check on their behalf — they semantically encode the same
      behaviour, just at a different bit position / signal index.

    Skeletons returned here are AssertionSkeleton instances (the
    same dataclass the existing agent.py pipeline uses), so callers
    can drop them straight into format_skeletons_for_llm() /
    format_skeletons_as_sva() without changes.
    """
    from .ast_assertions import AssertionSkeleton, generate_skeletons

    llm_bound: List[AssertionSkeleton] = []
    direct_bound: List[AssertionSkeleton] = []

    def _pattern_skel(pattern: RTLPattern) -> Optional[AssertionSkeleton]:
        # Reuse the existing single-pattern → skeleton converter.  We
        # call generate_skeletons on a list-of-one so its internal
        # invariant generation is skipped (no group of patterns to
        # decode mutual exclusivity from).
        skels = generate_skeletons([pattern], is_combinational=False)
        return skels[0] if skels else None

    for unit in result.units:
        if isinstance(unit, CompactedUnit):
            # The packed assertion is already complete — wrap it in
            # an AssertionSkeleton so it slots into the existing
            # batching / formatting code.
            llm_bound.append(AssertionSkeleton(
                assertion_text=unit.assertion_text,
                pattern_type=unit.pattern_type + "_packed",
                source_line=unit.member_lines[0] if unit.member_lines else 0,
                source_text=(
                    f"<packed: {unit.member_count} members, "
                    f"template={unit.template[:80]}>"
                ),
                description=unit.description,
            ))
        elif isinstance(unit, RepresentativeUnit):
            # Representative goes to the LLM with a comment noting
            # the cluster it stands for; siblings emit directly with
            # a back-reference. Mirror baseline routing: trivial
            # pattern types (case_branch / wire_passthrough /
            # direct_assign) skip LLM validation entirely — for those,
            # both the representative and its siblings emit directly.
            rep_ptype = unit.representative.pattern.pattern_type
            rep_is_trivial = rep_ptype in {
                "case_branch", "wire_passthrough", "direct_assign",
            }
            rep_skel = _pattern_skel(unit.representative.pattern)
            if rep_skel is not None:
                rep_skel.description = (
                    f"[REPRESENTATIVE of {len(unit.siblings) + 1} "
                    f"sibling cluster] {rep_skel.description}"
                )
                if rep_is_trivial:
                    direct_bound.append(rep_skel)
                else:
                    llm_bound.append(rep_skel)
            for sib in unit.siblings:
                sib_skel = _pattern_skel(sib.pattern)
                if sib_skel is None:
                    continue
                rep_lhs = unit.representative.pattern.lhs
                tag = (
                    f"// SIBLING_OF {rep_lhs} — inherits spec-validation "
                    f"from representative ({len(unit.siblings) + 1} members "
                    f"in cluster)"
                )
                sib_skel.assertion_text = tag + "\n" + sib_skel.assertion_text
                direct_bound.append(sib_skel)
        else:  # IndividualUnit
            indiv_skel = _pattern_skel(unit.member.pattern)
            if indiv_skel is None:
                continue
            # Mirror the baseline (no-clustering) routing: trivial
            # pattern types (case_branch / wire_passthrough /
            # direct_assign) emit directly without LLM validation —
            # otherwise the clustering path sends every un-clustered
            # trivial pattern through spec_validation, blowing up the
            # LLM call count (rvv: 108 LLM-bound including 93 trivial
            # → 9 LLM calls vs baseline 2).
            ptype = unit.member.pattern.pattern_type
            if ptype in {"case_branch", "wire_passthrough",
                         "direct_assign"}:
                direct_bound.append(indiv_skel)
            else:
                llm_bound.append(indiv_skel)

    return llm_bound, direct_bound
