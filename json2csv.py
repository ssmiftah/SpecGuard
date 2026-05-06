"""
json2csv.py
-----------
Convert pipeline JSON reports (lint and mutation) to CSV files.

Usage:
    python json2csv.py <report.json>              # auto-detects type
    python json2csv.py <report.json> -o output.csv  # custom output path
    python json2csv.py --all                      # convert all JSON reports in project

Produces separate CSV files for each section:
  Lint report:
    - <name>_lint_summary.csv    (iteration-level summary)
    - <name>_lint_failures.csv   (per-failure details)

  Mutation report:
    - <name>_mutation_summary.csv    (metadata + score)
    - <name>_mutation_operators.csv  (per-operator breakdown)
    - <name>_mutation_mutants.csv    (per-mutant details)
    - <name>_mutation_survivors.csv  (surviving mutants only)
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def convert_lint_report(data: Dict[str, Any], output_prefix: str) -> List[str]:
    """Convert a lint failures JSON to CSV files."""
    created = []

    # Summary CSV: one row per iteration.
    summary_path = f"{output_prefix}_lint_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Iteration", "Timestamp", "Total Assertions",
            "Passed", "Failed", "Status",
        ])
        for it in data.get("iterations", []):
            writer.writerow([
                it.get("iteration", ""),
                it.get("timestamp", ""),
                it.get("total_assertions", ""),
                it.get("passed", ""),
                it.get("failed", ""),
                "ALL_PASSED" if it.get("failed", 0) == 0 else "HAS_FAILURES",
            ])
        # Final row with overall status.
        writer.writerow([])
        writer.writerow(["Final Status", data.get("final_status", "")])
        writer.writerow(["Total Refinement Iterations", data.get("total_refinement_iterations", "")])
    created.append(summary_path)

    # Failures CSV: one row per failure across all iterations.
    all_failures = []
    for it in data.get("iterations", []):
        for fail in it.get("failures", []):
            all_failures.append({
                "iteration": it.get("iteration", ""),
                "index": fail.get("index", ""),
                "assertion": fail.get("assertion", "").replace("\n", " "),
                "error": fail.get("error", "").replace("\n", " ")[:200],
                "comment": fail.get("comment", "").replace("\n", " ")[:100],
            })

    if all_failures:
        failures_path = f"{output_prefix}_lint_failures.csv"
        with open(failures_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "iteration", "index", "assertion", "error", "comment",
            ])
            writer.writeheader()
            writer.writerows(all_failures)
        created.append(failures_path)

    return created


def convert_mutation_report(data: Dict[str, Any], output_prefix: str) -> List[str]:
    """Convert a mutation report JSON to CSV files."""
    created = []
    meta = data.get("metadata", {})

    # Summary CSV: single row with metadata.
    summary_path = f"{output_prefix}_mutation_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Field", "Value"])
        for key, value in meta.items():
            writer.writerow([key, value])
    created.append(summary_path)

    # Operators CSV: one row per operator.
    operators = data.get("summary_by_operator", {})
    if operators:
        ops_path = f"{output_prefix}_mutation_operators.csv"
        with open(ops_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Operator", "Total", "Killed", "Survived",
                "Stillborn", "Kill Rate",
            ])
            for op_name, counts in sorted(operators.items()):
                total = counts.get("total", 0)
                killed = counts.get("killed", 0)
                stillborn = counts.get("stillborn", 0)
                denominator = total - stillborn
                rate = f"{killed/denominator:.0%}" if denominator > 0 else "N/A"
                writer.writerow([
                    op_name,
                    total,
                    killed,
                    counts.get("survived", 0),
                    stillborn,
                    rate,
                ])
        created.append(ops_path)

    # Mutants CSV: one row per mutant.
    mutants = data.get("mutants", [])
    if mutants:
        mutants_path = f"{output_prefix}_mutation_mutants.csv"
        with open(mutants_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "ID", "Operator", "File", "Line",
                "Original", "Mutated", "Description",
                "Status", "Time (ms)",
            ])
            for m in mutants:
                writer.writerow([
                    m.get("id", ""),
                    m.get("operator", ""),
                    m.get("file", ""),
                    m.get("line", ""),
                    m.get("original", "").replace("\n", " ")[:100],
                    m.get("mutated", "").replace("\n", " ")[:100],
                    m.get("description", ""),
                    m.get("status", ""),
                    m.get("sim_time_ms", ""),
                ])
        created.append(mutants_path)

    # Survivors CSV: just the surviving mutants.
    survivors = data.get("surviving_mutants", [])
    if survivors:
        survivors_path = f"{output_prefix}_mutation_survivors.csv"
        with open(survivors_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "ID", "Operator", "Line", "Original", "Mutated", "Description",
            ])
            for s in survivors:
                writer.writerow([
                    s.get("id", ""),
                    s.get("operator", ""),
                    s.get("line", ""),
                    s.get("original", "").replace("\n", " ")[:100],
                    s.get("mutated", "").replace("\n", " ")[:100],
                    s.get("description", ""),
                ])
        created.append(survivors_path)

    return created


def convert_trace_report(data: Dict[str, Any], output_prefix: str) -> List[str]:
    """Convert a pipeline trace JSON to a detailed CSV."""
    created = []
    trace_path = f"{output_prefix}_trace.csv"

    steps = data.get("steps", [])
    with open(trace_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Step #", "Phase", "Step", "Timestamp",
            "Output Length", "Num Tool Calls", "Tools Used",
            "Assertions", "Notes",
            "Model Output Preview",
        ])
        for i, s in enumerate(steps, 1):
            tools = ", ".join(
                tc["name"] for tc in s.get("tool_calls", [])
            ) or "-"
            preview = s.get("model_output", "")[:200].replace("\n", " | ")
            writer.writerow([
                i,
                s.get("phase", ""),
                s.get("step", ""),
                s.get("timestamp", ""),
                s.get("model_output_length", 0),
                s.get("num_tool_calls", 0),
                tools,
                s.get("assertions_generated", 0),
                s.get("notes", ""),
                preview,
            ])

        # Summary rows.
        writer.writerow([])
        writer.writerow(["Pipeline Start", data.get("pipeline_start", "")])
        writer.writerow(["Pipeline End", data.get("pipeline_end", "")])
        writer.writerow(["Total Steps", data.get("total_steps", len(steps))])

    created.append(trace_path)
    return created


def detect_report_type(data: Dict[str, Any]) -> str:
    """Detect whether a JSON file is a lint, mutation, or trace report."""
    if "metadata" in data and "mutation_score" in data.get("metadata", {}):
        return "mutation"
    if "iterations" in data and "final_status" in data:
        return "lint"
    if "steps" in data and "pipeline_start" in data:
        return "trace"
    return "unknown"


def convert_file(json_path: str, output_prefix: str = None) -> List[str]:
    """Convert a single JSON report file to CSV(s)."""
    with open(json_path, "r") as f:
        data = json.load(f)

    if output_prefix is None:
        output_prefix = str(Path(json_path).with_suffix(""))

    report_type = detect_report_type(data)

    if report_type == "mutation":
        files = convert_mutation_report(data, output_prefix)
    elif report_type == "lint":
        files = convert_lint_report(data, output_prefix)
    elif report_type == "trace":
        files = convert_trace_report(data, output_prefix)
    else:
        print(f"  Unknown report type: {json_path}", file=sys.stderr)
        return []

    return files


def find_all_reports(root_dir: str) -> List[str]:
    """Find all lint and mutation JSON reports in the project."""
    reports = []
    for dirpath, _, filenames in os.walk(root_dir):
        # Skip hidden dirs and venv.
        if any(p.startswith(".") for p in dirpath.split(os.sep)):
            continue
        if ".venv" in dirpath or "RTL Cases" in dirpath:
            continue
        for fname in sorted(filenames):
            if fname.endswith(".json"):
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    if detect_report_type(data) != "unknown":
                        reports.append(fpath)
                except (json.JSONDecodeError, OSError):
                    pass
    return reports


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage:")
        print("  python json2csv.py <report.json>         # convert one file")
        print("  python json2csv.py <report.json> -o out   # custom output prefix")
        print("  python json2csv.py --all                  # convert all reports")
        sys.exit(0)

    if sys.argv[1] == "--all":
        root = os.path.dirname(os.path.abspath(__file__))
        reports = find_all_reports(root)
        if not reports:
            print("No JSON reports found.")
            sys.exit(0)

        print(f"Found {len(reports)} report(s):\n")
        total_csv = 0
        for rpath in reports:
            rel = os.path.relpath(rpath, root)
            rtype = detect_report_type(json.load(open(rpath)))
            files = convert_file(rpath)
            total_csv += len(files)
            print(f"  {rel} ({rtype})")
            for f in files:
                print(f"    → {os.path.relpath(f, root)}")

        print(f"\nConverted {len(reports)} JSON report(s) to {total_csv} CSV file(s).")

    else:
        json_path = sys.argv[1]
        output_prefix = None
        if "-o" in sys.argv:
            idx = sys.argv.index("-o")
            if idx + 1 < len(sys.argv):
                output_prefix = sys.argv[idx + 1]

        if not os.path.exists(json_path):
            print(f"File not found: {json_path}", file=sys.stderr)
            sys.exit(1)

        files = convert_file(json_path, output_prefix)
        if files:
            print(f"Created {len(files)} CSV file(s):")
            for f in files:
                print(f"  {f}")
        else:
            print("No CSV files created (unknown report type).")


if __name__ == "__main__":
    main()
