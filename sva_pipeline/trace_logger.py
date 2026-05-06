"""
trace_logger.py
---------------
Detailed step-by-step trace logging for the SVA pipeline.

Records every model interaction (prompt, response, tool calls, observations)
into a structured JSON file and a human-readable CSV file.

The trace captures what the LLM generated at each step and iteration,
enabling post-mortem analysis of the generation process.
"""

import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TraceLogger:
    """
    Records every step of the pipeline into a trace file.

    Usage:
        trace = TraceLogger("output/trace.json")
        trace.log_step(phase="planning", step=1, ...)
        trace.save()
    """

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.csv_path = str(Path(output_path).with_suffix(".csv"))
        self.steps: List[Dict[str, Any]] = []
        self.start_time = datetime.now(timezone.utc)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def log_step(
        self,
        phase: str,
        step: int,
        model_output: str = "",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_observations: Optional[List[Dict[str, str]]] = None,
        assertions_generated: int = 0,
        notes: str = "",
        usage: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Record one pipeline step.

        Parameters
        ----------
        phase : str
            Pipeline phase: "extraction", "generation", "planning",
            "execution", "direct", "refinement", "lint".
        step : int
            Step number within the phase.
        model_output : str
            Raw model output text.
        tool_calls : list of dict, optional
            Tool calls made: [{"name": "...", "arguments": {...}}]
        tool_observations : list of dict, optional
            Tool results: [{"tool": "...", "result": "..."}]
        assertions_generated : int
            Number of assertions in this step's output.
        notes : str
            Any additional context.
        usage : dict, optional
            Token usage for this LLM call:
            {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
        """
        u = usage or {}
        prompt_tokens = int(u.get("prompt_tokens", 0))
        completion_tokens = int(u.get("completion_tokens", 0))
        total_tokens = int(u.get("total_tokens", prompt_tokens + completion_tokens))

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "step": step,
            "model_output": model_output[:2000],  # cap to prevent huge files
            "model_output_length": len(model_output),
            "tool_calls": [
                {"name": tc.get("name", ""), "arguments": tc.get("arguments", {})}
                for tc in (tool_calls or [])
            ],
            "tool_observations": [
                {"tool": obs.get("tool", ""), "result": obs.get("result", "")[:500]}
                for obs in (tool_observations or [])
            ],
            "has_tool_calls": bool(tool_calls),
            "num_tool_calls": len(tool_calls or []),
            "assertions_generated": assertions_generated,
            "notes": notes,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        self.steps.append(entry)

        # Also log to the standard logger for real-time visibility.
        tc_summary = ", ".join(
            tc.get("name", "") for tc in (tool_calls or [])
        ) or "none"
        logger.info(
            "  [TRACE] %s step %d: %d chars output, tools=[%s] %s",
            phase, step, len(model_output), tc_summary, notes,
        )

    def log_lint_iteration(
        self,
        iteration: int,
        total: int,
        passed: int,
        failed: int,
        failure_summaries: Optional[List[str]] = None,
    ) -> None:
        """Record a lint loop iteration."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": "lint",
            "step": iteration,
            "model_output": "",
            "model_output_length": 0,
            "tool_calls": [],
            "tool_observations": [],
            "has_tool_calls": False,
            "num_tool_calls": 0,
            "assertions_generated": total,
            "notes": f"passed={passed}, failed={failed}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "lint_details": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "failures": (failure_summaries or [])[:10],
            },
        }
        self.steps.append(entry)

    def totals(self) -> Dict[str, int]:
        """Sum token usage across all steps."""
        totals = {"prompt_tokens": 0, "completion_tokens": 0,
                  "total_tokens": 0, "llm_calls": 0}
        for s in self.steps:
            pt = s.get("prompt_tokens", 0)
            ct = s.get("completion_tokens", 0)
            if pt > 0 or ct > 0:
                totals["llm_calls"] += 1
            totals["prompt_tokens"] += pt
            totals["completion_tokens"] += ct
            totals["total_tokens"] += s.get("total_tokens", pt + ct)
        return totals

    def save(self) -> None:
        """Write the trace to JSON and CSV files."""
        # JSON trace — full detail.
        trace_data = {
            "pipeline_start": self.start_time.isoformat(),
            "pipeline_end": datetime.now(timezone.utc).isoformat(),
            "total_steps": len(self.steps),
            "steps": self.steps,
        }

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)

        # CSV trace — one row per step, human-readable.
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Step #", "Phase", "Step", "Timestamp",
                "Output Length", "Tool Calls", "Tools Used",
                "Assertions",
                "Prompt Tokens", "Completion Tokens", "Total Tokens",
                "Notes",
                "Output Preview",
            ])
            for i, s in enumerate(self.steps, 1):
                tools = ", ".join(
                    tc["name"] for tc in s.get("tool_calls", [])
                ) or "-"
                preview = s.get("model_output", "")[:150].replace("\n", " ")
                writer.writerow([
                    i,
                    s["phase"],
                    s["step"],
                    s["timestamp"],
                    s["model_output_length"],
                    s["num_tool_calls"],
                    tools,
                    s["assertions_generated"],
                    s.get("prompt_tokens", 0),
                    s.get("completion_tokens", 0),
                    s.get("total_tokens", 0),
                    s["notes"],
                    preview,
                ])

        logger.info(
            "Trace saved: %s (%d steps), CSV: %s",
            self.output_path, len(self.steps), self.csv_path,
        )
