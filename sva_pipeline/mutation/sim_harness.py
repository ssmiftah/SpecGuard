"""
Simulation harness — compiles and runs mutants against SVA assertions.

Supports Verilator as the primary simulator.  Each mutant is compiled and
simulated in an isolated temp directory.  Assertion failures are detected
via exit codes and stderr scanning.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import MutationConfig

logger = logging.getLogger(__name__)


def run_all_mutants(
    mutants: List[Dict[str, Any]],
    support_files: List[str],
    testbench_source: str,
    sva_source: str,
    config: MutationConfig,
) -> List[Dict[str, Any]]:
    """
    Run all mutants in parallel and collect results.

    Parameters
    ----------
    mutants : list of dict
        Each has: id, mutation, source (mutated RTL text), filename.
    support_files : list of str
        Paths to support files (submodules) to compile alongside the mutant.
    testbench_source : str
        Testbench source text.
    sva_source : str
        SVA assertions source text.
    config : MutationConfig

    Returns
    -------
    list of dict
        Each has: id, status ("killed"/"survived"/"stillborn"/"timeout"),
        error_msg, sim_time_ms.
    """
    total = len(mutants)
    logger.info(
        "Running %d mutant(s) with %d worker(s) on %s …",
        total, config.max_workers, config.simulator,
    )

    results: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=config.max_workers) as pool:
        futures = {}
        for m in mutants:
            future = pool.submit(
                _run_one_mutant,
                mutant=m,
                support_files=support_files,
                testbench_source=testbench_source,
                sva_source=sva_source,
                config=config,
            )
            futures[future] = m["id"]

        for future in as_completed(futures):
            mutant_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "id": mutant_id,
                    "status": "error",
                    "error_msg": str(exc),
                    "sim_time_ms": 0,
                }
            results.append(result)

            # Progress logging.
            done = len(results)
            killed = sum(1 for r in results if r["status"] == "killed")
            if done % 10 == 0 or done == total:
                logger.info(
                    "  Progress: %d/%d done (%d killed so far)",
                    done, total, killed,
                )

    # Sort by mutant ID for deterministic output.
    results.sort(key=lambda r: r["id"])
    return results


def _run_one_mutant(
    mutant: Dict[str, Any],
    support_files: List[str],
    testbench_source: str,
    sva_source: str,
    config: MutationConfig,
) -> Dict[str, Any]:
    """
    Compile and simulate one mutant in an isolated temp directory.

    Returns a result dict with status and timing info.
    """
    mutant_id = mutant["id"]
    start_time = time.time()

    # Create isolated workspace.
    workdir = tempfile.mkdtemp(prefix=f"mutant_{mutant_id:04d}_")

    try:
        # Write the mutated DUT source (with SVA injected before endmodule).
        dut_path = os.path.join(workdir, mutant["filename"])

        # Write the testbench.
        tb_path = os.path.join(workdir, "tb_top.sv")
        with open(tb_path, "w") as fh:
            fh.write(testbench_source)

        # Inject SVA assertions directly into the DUT source before endmodule.
        # This gives assertions access to all internal signals without bind.
        top_module = mutant.get("top_module", "")
        dut_with_sva = _inject_sva_into_dut(mutant["source"], sva_source, top_module)
        with open(dut_path, "w") as fh:
            fh.write(dut_with_sva)

        # Symlink support files into the workspace.
        for sf in support_files:
            dst = os.path.join(workdir, os.path.basename(sf))
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(sf), dst)

        # Collect all source files.
        all_sources = [dut_path] + [
            os.path.join(workdir, os.path.basename(sf))
            for sf in support_files
        ] + [tb_path]

        # Dispatch to the configured simulator backend.
        if config.simulator == "xsim":
            return _run_xsim(mutant_id, workdir, all_sources, config, start_time)
        else:
            return _run_verilator(mutant_id, workdir, all_sources, config, start_time)

    except subprocess.TimeoutExpired:
        elapsed = int((time.time() - start_time) * 1000)
        return {
            "id": mutant_id,
            "status": "timeout",
            "error_msg": f"Timed out after {config.sim_timeout_sec}s",
            "sim_time_ms": elapsed,
        }
    except Exception as exc:
        elapsed = int((time.time() - start_time) * 1000)
        return {
            "id": mutant_id,
            "status": "error",
            "error_msg": str(exc),
            "sim_time_ms": elapsed,
        }
    finally:
        # Clean up workspace.
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Verilator backend
# ---------------------------------------------------------------------------

def _run_verilator(
    mutant_id: int, workdir: str, all_sources: List[str],
    config: MutationConfig, start_time: float,
) -> Dict[str, Any]:
    """Compile and simulate one mutant with Verilator."""
    compile_result = subprocess.run(
        [
            config.verilator_bin,
            "--binary", "--assert", "--timing",
            "--top-module", "tb_top",
            "-Wno-fatal", "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC",
            "-Wno-CASEINCOMPLETE", "-Wno-LATCH", "-Wno-INITIALDLY",
            "--Mdir", os.path.join(workdir, "obj_dir"),
            "-o", "sim_mutant",
            f"-I{workdir}",
        ] + all_sources,
        capture_output=True, text=True,
        timeout=config.sim_timeout_sec, cwd=workdir,
    )

    if compile_result.returncode != 0:
        return {
            "id": mutant_id, "status": "stillborn",
            "error_msg": (compile_result.stderr or compile_result.stdout or "")[:500],
            "sim_time_ms": int((time.time() - start_time) * 1000),
        }

    sim_binary = os.path.join(workdir, "obj_dir", "sim_mutant")
    if not os.path.exists(sim_binary):
        sim_binary = os.path.join(workdir, "obj_dir", "Vtb_top")

    sim_result = subprocess.run(
        [sim_binary],
        capture_output=True, text=True,
        timeout=config.sim_timeout_sec, cwd=workdir,
    )

    elapsed = int((time.time() - start_time) * 1000)
    output = sim_result.stderr + sim_result.stdout
    killed = (
        sim_result.returncode != 0
        or "%Error" in output or "Assertion" in output
        or "$error" in output or "FATAL" in output
    )
    return {
        "id": mutant_id,
        "status": "killed" if killed else "survived",
        "error_msg": output[:300] if killed else "",
        "sim_time_ms": elapsed,
    }


# ---------------------------------------------------------------------------
# Vivado xsim backend
# ---------------------------------------------------------------------------

def _run_xsim(
    mutant_id: int, workdir: str, all_sources: List[str],
    config: MutationConfig, start_time: float,
) -> Dict[str, Any]:
    """Compile and simulate one mutant with Vivado xsim."""
    # Resolve xsim tool paths.  Vivado tools are in the same bin/ directory.
    xsim_bin = config.xsim_bin
    bin_dir = os.path.dirname(xsim_bin) if os.path.dirname(xsim_bin) else ""

    def _tool(name: str) -> str:
        return os.path.join(bin_dir, name) if bin_dir else name

    # Step 1: xvlog — compile all sources.
    compile_result = subprocess.run(
        [_tool("xvlog"), "--sv"] + all_sources,
        capture_output=True, text=True,
        timeout=config.sim_timeout_sec, cwd=workdir,
    )

    if compile_result.returncode != 0:
        return {
            "id": mutant_id, "status": "stillborn",
            "error_msg": (compile_result.stderr or compile_result.stdout or "")[:500],
            "sim_time_ms": int((time.time() - start_time) * 1000),
        }

    # Step 2: xelab — elaborate with assertions enabled.
    elab_result = subprocess.run(
        [_tool("xelab"), "tb_top", "-s", "sim_snapshot",
         "--debug", "off", "-R",  # -R = run immediately after elaboration
         ],
        capture_output=True, text=True,
        timeout=config.sim_timeout_sec, cwd=workdir,
    )

    elapsed = int((time.time() - start_time) * 1000)
    output = elab_result.stderr + elab_result.stdout

    # xsim reports assertion failures as "ERROR" or "Assertion" in the transcript.
    killed = (
        elab_result.returncode != 0
        or "ERROR" in output or "Assertion" in output
        or "$error" in output or "FATAL" in output
    )
    return {
        "id": mutant_id,
        "status": "killed" if killed else "survived",
        "error_msg": output[:300] if killed else "",
        "sim_time_ms": elapsed,
    }


# ---------------------------------------------------------------------------
# SVA injection into DUT
# ---------------------------------------------------------------------------

def _inject_sva_into_dut(dut_source: str, sva_source: str, top_module: str) -> str:
    """
    Inject SVA assertions into the DUT source just before ``endmodule``.

    This places assertions inside the DUT module scope so they can reference
    internal signals directly — no hierarchical paths or bind needed.

    Strips header comments from the SVA source to keep the injected code clean.
    """
    import re

    # Separate concurrent vs immediate assertions.
    # Verilator requires:
    #   - `assert property (...)` at module scope (concurrent)
    #   - `assert (...)` inside `always_comb` (immediate / procedural)
    concurrent_lines = []
    immediate_lines = []
    in_concurrent = False

    for line in sva_source.splitlines():
        stripped = line.strip()

        # Skip file header comments.
        if stripped.startswith("//") and any(
            kw in stripped.lower()
            for kw in ["auto-generated", "validated with", "model:"]
        ):
            continue
        if not stripped:
            continue

        # Track multi-line concurrent assertions.
        if "assert property" in stripped:
            in_concurrent = True
            concurrent_lines.append(line)
        elif in_concurrent:
            concurrent_lines.append(line)
            # End of concurrent assertion when line ends with ;
            if stripped.endswith(";"):
                in_concurrent = False
                concurrent_lines.append("")  # blank separator
        elif stripped.startswith("assert"):
            # Immediate assertion.
            immediate_lines.append(line)
        elif stripped.startswith("//"):
            # Context comment — goes with whatever comes next.
            # Peek ahead logic is complex; just add to both.
            concurrent_lines.append(line)
            immediate_lines.append(line)
        else:
            # Continuation of something — add to immediate.
            immediate_lines.append(line)

    # Build injection block.
    parts = ["\n// === INJECTED SVA ASSERTIONS ==="]

    concurrent_body = "\n".join(concurrent_lines).strip()
    immediate_body = "\n".join(
        l for l in immediate_lines if l.strip()
    ).strip()

    if concurrent_body:
        parts.append("// Concurrent assertions (module scope)")
        parts.append(concurrent_body)
    if immediate_body:
        parts.append("// Immediate assertions (procedural scope)")
        parts.append("always_comb begin")
        parts.append(immediate_body)
        parts.append("end")

    injection = "\n".join(parts) + "\n\n"

    # Find the last `endmodule` and inject before it.
    endmodule_re = re.compile(r"^(\s*endmodule\b)", re.MULTILINE)
    matches = list(endmodule_re.finditer(dut_source))

    if not matches:
        return dut_source + "\n" + injection

    last_match = matches[-1]
    return (
        dut_source[:last_match.start()]
        + injection
        + dut_source[last_match.start():]
    )
