"""
Mutation testing framework for SVA assertions.

Entry point: ``run_mutation_testing(config, design_info)``
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

from .config import MutationConfig
from .mutant_generator import generate_mutants
from .testbench_gen import generate_testbench
from .sim_harness import run_all_mutants
from .report import build_report, save_report, print_summary

logger = logging.getLogger(__name__)


def run_mutation_testing(
    config: Any,          # PipelineConfig
    design_info: Any,     # DesignInfo from slang_frontend
) -> Dict[str, Any]:
    """
    Run the full mutation testing flow.

    1. Read the DUT source file and the generated SVA.
    2. Generate mutants using the configured operators.
    3. Generate or load a testbench.
    4. Compile and simulate each mutant.
    5. Produce and save the mutation report.

    Parameters
    ----------
    config : PipelineConfig
        Must have a ``mutation`` field of type MutationConfig.
    design_info : DesignInfo
        Design analysis results (for testbench generation).

    Returns
    -------
    dict
        The full mutation report.
    """
    mut_config: MutationConfig = config.mutation

    # ---- Identify the DUT file to mutate ----------------------------------
    if mut_config.dut_files:
        dut_filenames = mut_config.dut_files
    else:
        # Auto-detect: find the file containing the top module.
        dut_filenames = _find_dut_file(config.rtl_dir, config.top_module)

    if not dut_filenames:
        logger.error("No DUT file found for mutation testing.")
        return {"metadata": {"error": "No DUT file found"}}

    # Read the DUT source.
    dut_path = os.path.join(config.rtl_dir, dut_filenames[0])
    with open(dut_path, "r", encoding="utf-8") as fh:
        dut_source = fh.read()

    logger.info("Mutating DUT: %s", dut_path)

    # ---- Identify support files -------------------------------------------
    support_files = []
    if mut_config.support_files:
        for sf in mut_config.support_files:
            sfp = os.path.join(config.rtl_dir, sf)
            if os.path.exists(sfp):
                support_files.append(sfp)
    else:
        # Auto-detect: all .v/.sv files in rtl_dir except the DUT.
        for root, _, files in os.walk(config.rtl_dir):
            for fname in files:
                if Path(fname).suffix in {".v", ".sv"} and fname not in dut_filenames:
                    support_files.append(os.path.join(root, fname))

    logger.info("Support files: %d", len(support_files))

    # ---- Generate mutants -------------------------------------------------
    mutants = generate_mutants(
        dut_source=dut_source,
        dut_filename=dut_filenames[0],
        config=mut_config,
        signal_map=design_info.signal_map,
        top_module=config.top_module,
    )

    if not mutants:
        logger.warning("No mutants generated — nothing to test.")
        return {"metadata": {"error": "No mutants generated"}}

    # ---- Read the SVA assertions ------------------------------------------
    sva_path = config.output_sva_file
    if not os.path.exists(sva_path):
        logger.error("SVA file not found: %s", sva_path)
        return {"metadata": {"error": f"SVA file not found: {sva_path}"}}

    with open(sva_path, "r", encoding="utf-8") as fh:
        sva_source = fh.read()

    # ---- Generate or load testbench ---------------------------------------
    if mut_config.testbench and os.path.exists(mut_config.testbench):
        logger.info("Using user-provided testbench: %s", mut_config.testbench)
        with open(mut_config.testbench, "r") as fh:
            tb_source = fh.read()
    else:
        logger.info("Auto-generating testbench …")
        tb_source = generate_testbench(
            top_module=config.top_module,
            signal_map=design_info.signal_map,
            sva_file="sva_output.sv",
            clock_signal=design_info.clock_signal,
            reset_signal=design_info.reset_signal,
            sim_cycles=mut_config.sim_cycles,
        )

    # ---- Run simulations --------------------------------------------------
    results = run_all_mutants(
        mutants=mutants,
        support_files=support_files,
        testbench_source=tb_source,
        sva_source=sva_source,
        config=mut_config,
    )

    # ---- Build and save report --------------------------------------------
    report = build_report(
        design_name=config.top_module,
        sva_file=sva_path,
        mutants=mutants,
        results=results,
        simulator=mut_config.simulator,
        sim_cycles=mut_config.sim_cycles,
    )

    save_report(report, mut_config.report_file)
    print_summary(report)

    return report


def _find_dut_file(rtl_dir: str, top_module: str) -> list:
    """
    Find the file that contains the top module definition.
    """
    import re
    module_re = re.compile(rf"^\s*module\s+{re.escape(top_module)}\b", re.MULTILINE)

    for root, _, files in os.walk(rtl_dir):
        for fname in sorted(files):
            if Path(fname).suffix in {".v", ".sv"}:
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                        if module_re.search(fh.read()):
                            return [fname]
                except OSError:
                    continue
    return []
