"""
analysis_export.py
------------------
Diagnostic artifact exports for the SVA pipeline.

Writes human-readable + machine-readable files documenting:
- Module instantiation hierarchy (ASCII tree + Mermaid graph)
- Full ``RTLFacts`` dump (JSON)
- Raw AST patterns grouped by type (text)
- Pre-LLM AST assertion skeletons (SV)
- Per-step assertion count trace (CSV, streaming)
- Per-module assertion count summary (CSV)

All artifacts are written to a single analysis directory. File writes are
best-effort: failures are logged but do not abort the pipeline.
"""

import csv
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def resolve_analysis_dir(config: Any) -> str:
    """Return the directory where analysis artifacts should be written."""
    explicit = getattr(config, "analysis_dir", "")
    if explicit:
        return explicit
    sva_path = getattr(config, "output_sva_file", "")
    if sva_path:
        base = Path(sva_path).with_suffix("")
        return str(base.with_name(base.name + "_analysis"))
    return "./analysis"


def ensure_dir(path: str) -> bool:
    """Create ``path`` (and parents). Return True on success."""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except OSError as exc:
        logger.warning("Could not create analysis dir %s: %s", path, exc)
        return False


# ---------------------------------------------------------------------------
# Hierarchy export
# ---------------------------------------------------------------------------

def export_hierarchy(
    out_dir: str,
    comp: Any,
    top_module: str,
    sink_checker=None,
) -> None:
    """
    Write hierarchy.txt (ASCII tree) + hierarchy.mmd (Mermaid).

    ``comp`` is a pyslang ``Compilation`` object.
    ``sink_checker`` is an optional callable ``inst -> bool`` used to
    tag sink modules in the tree output.
    """
    if comp is None:
        return

    try:
        top_insts = list(comp.getRoot().topInstances)
    except Exception as exc:
        logger.debug("Hierarchy export: getRoot failed: %s", exc)
        return

    target = None
    for inst in top_insts:
        if inst.body.name == top_module:
            target = inst
            break
    if target is None and top_insts:
        target = top_insts[0]
    if target is None:
        return

    tree_lines: List[str] = []
    edges: List[Tuple[str, str]] = []  # (parent_mod, child_mod) for Mermaid
    node_labels: Dict[str, str] = {}   # module_name -> display label

    visited: Set[int] = set()

    def _port_counts(inst):
        n_in = 0
        n_out = 0
        n_sig = 0
        try:
            for m in inst.body:
                t = type(m).__name__
                if t == "PortSymbol":
                    d = str(getattr(m, "direction", "")).lower()
                    if d.endswith(".in"):
                        n_in += 1
                    elif d.endswith(".out"):
                        n_out += 1
                elif t in {"NetSymbol", "VariableSymbol"}:
                    n_sig += 1
        except (AttributeError, TypeError):
            pass
        return n_in, n_out, n_sig

    def _fmt_node(inst, is_top: bool = False) -> str:
        mod_name = inst.body.name
        n_in, n_out, n_sig = _port_counts(inst)
        tags = []
        if is_top:
            tags.append("top")
        if sink_checker and sink_checker(inst):
            tags.append("SINK")
        parts = [f"{n_in}+{n_out} ports", f"{n_sig} nets"]
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        return f"{mod_name}  [{', '.join(parts)}]{tag_str}"

    def _walk(inst, prefix: str, is_last: bool, is_root: bool):
        iid = id(inst)
        if iid in visited:
            return
        visited.add(iid)

        mod_name = inst.body.name
        inst_name = getattr(inst, "name", "") or mod_name
        node_labels[mod_name] = mod_name

        if is_root:
            tree_lines.append(_fmt_node(inst, is_top=True))
            child_prefix = ""
        else:
            branch = "└── " if is_last else "├── "
            tree_lines.append(
                prefix + branch + inst_name + ": " + _fmt_node(inst)
            )
            child_prefix = prefix + ("    " if is_last else "│   ")

        # Collect child instances
        children = []
        try:
            for m in inst.body:
                if type(m).__name__ == "InstanceSymbol":
                    children.append(m)
        except (AttributeError, TypeError):
            pass

        for i, child in enumerate(children):
            last = (i == len(children) - 1)
            _walk(child, child_prefix, last, is_root=False)
            try:
                edges.append((mod_name, child.body.name))
            except AttributeError:
                pass

    _walk(target, "", True, is_root=True)

    # Write ASCII tree
    try:
        with open(Path(out_dir) / "hierarchy.txt", "w") as fh:
            fh.write("# Module instantiation hierarchy\n")
            fh.write(f"# Top: {top_module}\n")
            fh.write("# Format: module_name  [ports, nets, tags]\n\n")
            fh.write("\n".join(tree_lines))
            fh.write("\n")
    except OSError as exc:
        logger.warning("hierarchy.txt write failed: %s", exc)

    # Write Mermaid graph — deduplicate edges
    try:
        unique_edges = sorted(set(edges))
        with open(Path(out_dir) / "hierarchy.mmd", "w") as fh:
            fh.write("%% Module hierarchy (Mermaid)\n")
            fh.write("graph TD\n")
            for parent, child in unique_edges:
                fh.write(f"  {parent} --> {child}\n")
    except OSError as exc:
        logger.warning("hierarchy.mmd write failed: %s", exc)


# ---------------------------------------------------------------------------
# RTL facts JSON export
# ---------------------------------------------------------------------------

def export_rtl_facts_json(out_dir: str, facts: Any) -> None:
    """Dump RTLFacts as JSON (sets converted to sorted lists)."""
    if facts is None:
        return

    def _serialise(value):
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, dict):
            return {k: _serialise(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_serialise(v) for v in value]
        return value

    # Build dict explicitly so per_module recursion works cleanly.
    def _facts_to_dict(f):
        d = {}
        for attr in (
            "signal_definitions", "case_selectors", "combinational_signals",
            "signal_drive_kind", "constant_signal_pairs", "all_signals",
            "signal_widths", "signal_frequencies", "clock_signals",
            "reset_signals", "reset_polarity", "clock_reset_pairs",
            "reset_values", "signal_to_module", "module_hierarchy",
            "out_of_scope_signals", "is_complete", "parse_warnings",
        ):
            if hasattr(f, attr):
                d[attr] = _serialise(getattr(f, attr))
        # per_module is a dict of RTLFacts — recurse.
        pm = getattr(f, "per_module", None)
        if pm:
            d["per_module"] = {
                name: _facts_to_dict(sub) for name, sub in pm.items()
            }
        return d

    try:
        with open(Path(out_dir) / "rtl_facts.json", "w") as fh:
            json.dump(_facts_to_dict(facts), fh, indent=2,
                      default=str, sort_keys=True)
    except (OSError, TypeError) as exc:
        logger.warning("rtl_facts.json write failed: %s", exc)


# ---------------------------------------------------------------------------
# AST patterns + skeletons export
# ---------------------------------------------------------------------------

def export_ast_patterns(
    out_dir: str,
    patterns: List[Any],
) -> None:
    """Write ast_patterns.txt grouped by pattern_type."""
    if not patterns:
        return

    by_type: Dict[str, List[Any]] = {}
    for p in patterns:
        by_type.setdefault(p.pattern_type, []).append(p)

    try:
        with open(Path(out_dir) / "ast_patterns.txt", "w") as fh:
            fh.write(f"# AST patterns extracted ({len(patterns)} total)\n\n")
            for ptype in sorted(by_type.keys()):
                group = by_type[ptype]
                fh.write(f"=== {ptype} ({len(group)} patterns) ===\n")
                for p in group[:200]:  # cap per group to keep file readable
                    src = getattr(p, "source_text", "") or ""
                    src = src.replace("\n", " ")[:120]
                    line = getattr(p, "source_line", 0)
                    fh.write(f"  line {line}: {src}\n")
                if len(group) > 200:
                    fh.write(f"  ... +{len(group) - 200} more\n")
                fh.write("\n")
    except OSError as exc:
        logger.warning("ast_patterns.txt write failed: %s", exc)


def export_ast_skeletons(
    out_dir: str,
    skeletons: List[Any],
) -> None:
    """Dump AST-generated assertion skeletons (pre-LLM) as a .sv file."""
    if not skeletons:
        return

    try:
        with open(Path(out_dir) / "ast_skeletons.sv", "w") as fh:
            fh.write("// Pre-LLM AST-generated assertion skeletons\n")
            fh.write(f"// Total: {len(skeletons)}\n\n")
            for skel in skeletons:
                desc = getattr(skel, "description", "")
                if desc:
                    fh.write(f"// {desc}\n")
                fh.write(getattr(skel, "assertion_text", "") + "\n\n")
    except OSError as exc:
        logger.warning("ast_skeletons.sv write failed: %s", exc)


# ---------------------------------------------------------------------------
# AssertionTracer — streaming CSV of per-step counts
# ---------------------------------------------------------------------------

class AssertionTracer:
    """
    Streaming logger for per-step assertion counts.

    Each ``record()`` call appends one row to ``assertion_trace.csv`` and
    updates the running remaining-count. Crash-resilient (flushes on
    every write) and cheap.

    Usage:
        tracer = AssertionTracer(out_dir)
        tracer.record("0-extract", "ast_pattern_extraction",
                      action="extract", delta=2180,
                      description="Initial AST patterns")
        tracer.record("2-phase2", "validate_widths",
                      action="remove", delta=-1,
                      description="Width mismatches")
        tracer.close()
    """

    _COLUMNS = ["timestamp", "phase", "step", "action",
                "delta", "remaining", "description"]

    def __init__(self, out_dir: str):
        self.remaining = 0
        self.file = None
        self.writer = None
        self._opened = False
        self._path = str(Path(out_dir) / "assertion_trace.csv")
        try:
            self.file = open(self._path, "w", newline="")
            self.writer = csv.writer(self.file)
            self.writer.writerow(self._COLUMNS)
            self.file.flush()
            self._opened = True
        except OSError as exc:
            logger.warning("assertion_trace.csv open failed: %s", exc)

    def record(
        self,
        phase: str,
        step: str,
        action: str,
        delta: int,
        description: str = "",
    ) -> None:
        """Log one step."""
        if action in ("extract", "add"):
            self.remaining += delta
        elif action == "remove":
            # delta should be negative; accept positive counts defensively
            self.remaining -= abs(delta)
        elif action == "keep":
            self.remaining = delta  # set absolute count
        # 'transform' and 'pass_through' leave remaining unchanged.

        if not self._opened or self.writer is None:
            return
        try:
            self.writer.writerow([
                f"{time.time():.3f}",
                phase, step, action,
                delta, self.remaining,
                description,
            ])
            self.file.flush()
        except OSError:
            pass

    def close(self) -> None:
        if self.file is not None:
            try:
                self.file.close()
            except OSError:
                pass
            self.file = None


# ---------------------------------------------------------------------------
# Per-module assertion count CSV
# ---------------------------------------------------------------------------

_LINE_COMMENT_RE = re.compile(r"^\s*//")
_ASSERT_START_RE = re.compile(r"^\s*assert\b")


def export_per_module_csv(
    out_dir: str,
    rtl_dir: str,
    ast_skeletons: Optional[List[Any]],
    final_sva_path: Optional[str],
    facts: Any,
) -> None:
    """
    Write assertions_per_module.csv.

    Columns:
        module_name, depth_from_top, num_signals, num_ast_skeletons,
        num_final_assertions

    Mapping assertions → modules is done via ``source_line`` on AST
    skeletons + a pre-built file-line-range map. Final SVA assertions
    are attributed by signal names (any assertion referencing a signal
    declared in module M → counted toward M).
    """
    # Build module -> file map by scanning RTL files for ``module X (``
    mod_files: Dict[str, str] = {}
    try:
        for root, _, files in os.walk(rtl_dir):
            for fname in sorted(files):
                if Path(fname).suffix not in {".v", ".sv"}:
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read()
                    for m in re.finditer(r"^\s*module\s+(\w+)", content, re.MULTILINE):
                        mod_name = m.group(1)
                        if mod_name not in mod_files:
                            mod_files[mod_name] = fname
                except OSError:
                    continue
    except OSError:
        pass

    # AST skeleton counts per module: use skeleton.source_text (which
    # contains RTL snippets) to match against modules we know about.
    ast_counts: Dict[str, int] = {m: 0 for m in mod_files}
    if ast_skeletons:
        for skel in ast_skeletons:
            src = getattr(skel, "source_text", "") or ""
            # Cheap match: if any module's file name appears in source OR
            # if the skeleton text references signals known to the module.
            # Simplest: match by signal-to-module.
            # We use facts.signal_to_module if populated.
            matched = False
            if facts and getattr(facts, "signal_to_module", None):
                # Extract identifiers from skeleton source text + assertion
                text = (getattr(skel, "source_text", "") + " "
                        + getattr(skel, "assertion_text", ""))
                ids = set(re.findall(r"\b([a-zA-Z_]\w*)\b", text))
                for ident in ids:
                    mod = facts.signal_to_module.get(ident)
                    if mod:
                        ast_counts[mod] = ast_counts.get(mod, 0) + 1
                        matched = True
                        break
            if not matched:
                # Unmatched: count under "(unknown)"
                ast_counts["(unknown)"] = ast_counts.get("(unknown)", 0) + 1

    # Final assertion counts per module: parse the final SVA and look up
    # each assertion's signals in signal_to_module.
    final_counts: Dict[str, int] = {m: 0 for m in mod_files}
    try:
        if final_sva_path and os.path.exists(final_sva_path):
            with open(final_sva_path, "r") as fh:
                for line in fh:
                    if not _ASSERT_START_RE.match(line):
                        continue
                    # Strip error message
                    body = re.split(r"\belse\s+\$\w+\s*\(", line, maxsplit=1)[0]
                    ids = set(re.findall(r"\b([a-zA-Z_]\w*)\b", body))
                    if facts and getattr(facts, "signal_to_module", None):
                        for ident in ids:
                            mod = facts.signal_to_module.get(ident)
                            if mod:
                                final_counts[mod] = final_counts.get(mod, 0) + 1
                                break
    except OSError:
        pass

    # Per-module metadata (signals, depth)
    hierarchy = getattr(facts, "module_hierarchy", {}) or {}
    per_module = getattr(facts, "per_module", {}) or {}

    try:
        path = str(Path(out_dir) / "assertions_per_module.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([
                "module_name", "depth_from_top", "num_signals",
                "num_ast_skeletons", "num_final_assertions",
            ])
            all_modules = set(mod_files.keys()) | set(per_module.keys())
            for mod in sorted(all_modules):
                depth = hierarchy.get(mod, "")
                n_sig = ""
                if mod in per_module:
                    try:
                        n_sig = len(per_module[mod].all_signals)
                    except AttributeError:
                        pass
                w.writerow([
                    mod, depth, n_sig,
                    ast_counts.get(mod, 0),
                    final_counts.get(mod, 0),
                ])
            # Unknown bucket
            if ast_counts.get("(unknown)"):
                w.writerow([
                    "(unknown)", "", "",
                    ast_counts.get("(unknown)", 0),
                    0,
                ])
    except OSError as exc:
        logger.warning("assertions_per_module.csv write failed: %s", exc)
