"""
Mutation testing configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# All available mutation operator names.
ALL_OPERATORS = [
    "OP_REPLACE",
    "CONST_REPLACE",
    "SIGNAL_SWAP",
    "BITSLICE_MUT",
    "COND_NEGATE",
    "ASSIGN_DELETE",
    "SENSITIVITY_MUT",
]


@dataclass
class MutationConfig:
    """Configuration for the mutation testing framework."""

    # Master switch — when False, mutation testing is skipped entirely.
    enabled: bool = False

    # Which RTL files to mutate (filenames relative to rtl_dir).
    # If empty, auto-detected as the file containing top_module.
    dut_files: List[str] = field(default_factory=list)

    # Files to compile but NOT mutate (submodules, packages).
    # If empty, all files in rtl_dir except dut_files are support files.
    support_files: List[str] = field(default_factory=list)

    # Path to a user-provided testbench.  If empty, auto-generated.
    testbench: str = ""

    # Which mutation operators to apply (default: all).
    operators: List[str] = field(default_factory=lambda: list(ALL_OPERATORS))

    # Simulator selection: "verilator" or "xsim".
    # Verilator is faster but has limited SVA support (no double |=>).
    # xsim (Vivado) supports the full IEEE 1800-2017 SVA spec.
    simulator: str = "verilator"
    verilator_bin: str = "verilator"
    xsim_bin: str = "xsim"  # Vivado's xsim (usually on PATH after sourcing settings64.sh)

    # Simulation parameters.
    sim_cycles: int = 500
    sim_timeout_sec: int = 30

    # Limits.
    max_mutants: int = 200
    max_workers: int = 4

    # Output.
    report_file: str = "./mutation_report.json"
