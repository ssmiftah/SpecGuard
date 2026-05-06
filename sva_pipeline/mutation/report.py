"""
Mutation testing report generation.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def build_report(
    design_name: str,
    sva_file: str,
    mutants: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    simulator: str,
    sim_cycles: int,
) -> Dict[str, Any]:
    """
    Build a structured mutation testing report.

    Parameters
    ----------
    design_name : str
        Top module name.
    sva_file : str
        Path to the SVA assertions file.
    mutants : list of dict
        Generated mutant metadata (id, mutation, filename).
    results : list of dict
        Simulation results (id, status, error_msg, sim_time_ms).
    simulator : str
        Simulator name used.
    sim_cycles : int
        Number of simulation cycles.

    Returns
    -------
    dict
        Complete report ready for JSON serialisation.
    """
    # Build a result lookup by mutant ID.
    result_map = {r["id"]: r for r in results}

    # Count statuses.
    total = len(mutants)
    killed = sum(1 for r in results if r["status"] == "killed")
    survived = sum(1 for r in results if r["status"] == "survived")
    stillborn = sum(1 for r in results if r["status"] == "stillborn")
    timed_out = sum(1 for r in results if r["status"] == "timeout")
    errors = sum(1 for r in results if r["status"] == "error")

    # Mutation score: killed / (total - stillborn - errors).
    denominator = total - stillborn - errors
    mutation_score = killed / denominator if denominator > 0 else 0.0

    # Summary by operator.
    op_summary: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "killed": 0, "survived": 0, "stillborn": 0}
    )
    for m in mutants:
        mutation = m["mutation"]
        r = result_map.get(m["id"], {})
        status = r.get("status", "error")
        op_name = mutation.operator
        op_summary[op_name]["total"] += 1
        if status in op_summary[op_name]:
            op_summary[op_name][status] += 1

    # Detailed mutant list.
    mutant_details = []
    surviving_mutants = []
    for m in mutants:
        mutation = m["mutation"]
        r = result_map.get(m["id"], {})
        detail = {
            "id": m["id"],
            "operator": mutation.operator,
            "file": m["filename"],
            "line": mutation.line_no,
            "original": mutation.original.strip(),
            "mutated": mutation.mutated.strip(),
            "description": mutation.description,
            "status": r.get("status", "error"),
            "sim_time_ms": r.get("sim_time_ms", 0),
        }
        mutant_details.append(detail)
        if r.get("status") == "survived":
            surviving_mutants.append(detail)

    report = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "design": design_name,
            "sva_file": sva_file,
            "simulator": simulator,
            "sim_cycles": sim_cycles,
            "total_mutants": total,
            "stillborn": stillborn,
            "killed": killed,
            "survived": survived,
            "timed_out": timed_out,
            "errors": errors,
            "mutation_score": round(mutation_score, 4),
            "mutation_score_pct": f"{mutation_score:.1%}",
        },
        "summary_by_operator": dict(op_summary),
        "mutants": mutant_details,
        "surviving_mutants": surviving_mutants,
    }

    return report


def save_report(report: Dict[str, Any], path: str) -> None:
    """Write the mutation report to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    logger.info("Mutation report written to %s", path)


def print_summary(report: Dict[str, Any]) -> None:
    """Print a concise summary to stdout."""
    meta = report["metadata"]
    print()
    print("=" * 60)
    print("MUTATION TESTING RESULTS")
    print("=" * 60)
    print(f"  Design       : {meta['design']}")
    print(f"  Simulator    : {meta['simulator']}")
    print(f"  Total mutants: {meta['total_mutants']}")
    print(f"  Killed       : {meta['killed']}")
    print(f"  Survived     : {meta['survived']}")
    print(f"  Stillborn    : {meta['stillborn']}")
    print(f"  Mutation score: {meta['mutation_score_pct']}")
    print()

    # Operator breakdown.
    print("  By operator:")
    for op_name, counts in sorted(report["summary_by_operator"].items()):
        t = counts["total"]
        k = counts["killed"]
        s = counts["survived"]
        rate = k / (t - counts.get("stillborn", 0)) if (t - counts.get("stillborn", 0)) > 0 else 0
        print(f"    {op_name:<20s}  {k}/{t} killed ({rate:.0%})")

    # List surviving mutants.
    survivors = report.get("surviving_mutants", [])
    if survivors:
        print()
        print(f"  Surviving mutants ({len(survivors)}):")
        for s in survivors[:10]:  # Show first 10.
            print(f"    [{s['id']}] line {s['line']}: {s['description']}")
            print(f"         {s['original'][:70]}")
            print(f"       → {s['mutated'][:70]}")
        if len(survivors) > 10:
            print(f"    ... and {len(survivors) - 10} more (see report JSON)")

    print()
    print(f"  Full report: {meta.get('sva_file', 'mutation_report.json')}")
    print("=" * 60)
