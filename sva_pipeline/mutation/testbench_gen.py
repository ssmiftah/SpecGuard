"""
Auto-generate a testbench for mutation testing.

Produces a minimal SystemVerilog testbench that:
  1. Generates clock and reset signals (auto-detected names)
  2. Instantiates the DUT with all ports wired
  3. Includes the generated SVA assertions
  4. Drives random stimulus on all input ports
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_sva_bind(top_module: str, sva_source: str) -> str:
    """
    Wrap SVA assertions in a bind module so they can access DUT internals.

    Verilator's ``bind`` statement injects the assertion module into the
    DUT's scope, making internal wires/regs visible without hierarchical
    references.

    Parameters
    ----------
    top_module : str
        DUT module name to bind into.
    sva_source : str
        Raw SVA assertion code (contents of sva_output.sv).

    Returns
    -------
    str
        A SystemVerilog file containing the assertion module + bind statement.
    """
    # Strip header comments from the SVA file.
    sva_lines = []
    for line in sva_source.splitlines():
        stripped = line.strip()
        if stripped.startswith("//") and not stripped.startswith("// "):
            continue  # skip bare header comments
        sva_lines.append(line)
    sva_body = "\n".join(sva_lines)

    return f"""\
// Auto-generated bind wrapper for mutation testing
// Binds SVA assertions into {top_module} so internal signals are accessible.

module sva_assertions_bind;
{sva_body}
endmodule

bind {top_module} sva_assertions_bind sva_bind_inst();
"""


def generate_testbench(
    top_module: str,
    signal_map: Dict[str, Any],
    sva_file: str,
    clock_signal: Optional[str] = None,
    reset_signal: Optional[str] = None,
    sim_cycles: int = 500,
) -> str:
    """
    Generate a Verilator-compatible testbench.

    Parameters
    ----------
    top_module : str
        DUT module name.
    signal_map : dict
        Signal map from DesignInfo (module, direction, width, type).
    sva_file : str
        Path to the SVA assertions file (for `include).
    clock_signal : str, optional
        Auto-detected clock port name (e.g. "pclk").
    reset_signal : str, optional
        Auto-detected reset port name (e.g. "prstn").
    sim_cycles : int
        Number of clock cycles to simulate.

    Returns
    -------
    str
        Complete testbench source code.
    """
    # Separate ports by direction (only top-module ports, no hierarchical).
    inputs: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []

    for name, info in sorted(signal_map.items()):
        # Skip hierarchical signals (submodule ports).
        if "." in name:
            continue
        # Skip if not from the top module.
        if info.get("module") != top_module:
            continue

        port = {"name": name, "width": info.get("width", 1), "type": info.get("type", "data")}

        if info.get("direction") == "input":
            inputs.append(port)
        elif info.get("direction") == "output":
            outputs.append(port)

    # Determine clock and reset ports.
    clk_port = clock_signal or "clk"
    rst_port = reset_signal or "rst_n"

    # Is reset active-low? Check naming convention.
    rst_active_low = rst_port.endswith("n") or rst_port.endswith("rstn") or "rst_n" in rst_port

    lines = []
    lines.append(f"// Auto-generated testbench for mutation testing")
    lines.append(f"// DUT: {top_module}")
    lines.append(f"")
    lines.append(f"module tb_top;")
    lines.append(f"")

    # Clock generation.
    lines.append(f"  // Clock generation")
    lines.append(f"  logic {clk_port} = 0;")
    lines.append(f"  always #5 {clk_port} = ~{clk_port};")
    lines.append(f"")

    # Reset sequence.
    lines.append(f"  // Reset sequence")
    if rst_active_low:
        lines.append(f"  logic {rst_port} = 0;")
        lines.append(f"  initial begin")
        lines.append(f"    {rst_port} = 0;")
        lines.append(f"    #25;")
        lines.append(f"    {rst_port} = 1;")
        lines.append(f"  end")
    else:
        lines.append(f"  logic {rst_port} = 1;")
        lines.append(f"  initial begin")
        lines.append(f"    {rst_port} = 1;")
        lines.append(f"    #25;")
        lines.append(f"    {rst_port} = 0;")
        lines.append(f"  end")
    lines.append(f"")

    # Declare all non-clock/reset ports.
    lines.append(f"  // Port declarations")
    declared = {clk_port, rst_port}
    for port in inputs + outputs:
        if port["name"] in declared:
            continue
        declared.add(port["name"])
        w = port["width"]
        if w > 1:
            lines.append(f"  logic [{w-1}:0] {port['name']};")
        else:
            lines.append(f"  logic {port['name']};")
    lines.append(f"")

    # DUT instantiation.
    lines.append(f"  // DUT instantiation")
    lines.append(f"  {top_module} dut (")
    all_ports = []
    for port in inputs + outputs:
        all_ports.append(port["name"])
    # Also include clock and reset if they're in the signal map.
    all_port_names = sorted(set(all_ports) | {clk_port, rst_port})
    for i, pname in enumerate(all_port_names):
        comma = "," if i < len(all_port_names) - 1 else ""
        lines.append(f"    .{pname}({pname}){comma}")
    lines.append(f"  );")
    lines.append(f"")

    # SVA assertions are bound into the DUT using a bind statement.
    # This ensures internal signals (wires, regs) are accessible.
    # The bind file wraps the assertions in a module that is bound to the DUT.
    lines.append(f"  // SVA assertions bound into DUT (see sva_bind.sv)")
    lines.append(f"")

    # Random stimulus for input ports (excluding clock and reset).
    stimulus_ports = [
        p for p in inputs
        if p["name"] not in {clk_port, rst_port}
    ]

    if stimulus_ports:
        lines.append(f"  // Random stimulus")
        lines.append(f"  initial begin")
        lines.append(f"    // Wait for reset release")
        lines.append(f"    #30;")
        lines.append(f"    repeat ({sim_cycles}) begin")
        lines.append(f"      @(posedge {clk_port});")
        for port in stimulus_ports:
            if port["width"] == 1:
                lines.append(f"      {port['name']} <= $random & 1;")
            else:
                lines.append(f"      {port['name']} <= $random;")
        lines.append(f"    end")
        lines.append(f"    $finish;")
        lines.append(f"  end")
    else:
        lines.append(f"  initial begin")
        lines.append(f"    #({sim_cycles} * 10 + 50);")
        lines.append(f"    $finish;")
        lines.append(f"  end")

    lines.append(f"")
    lines.append(f"endmodule")
    lines.append(f"")

    tb_source = "\n".join(lines)
    logger.info(
        "Generated testbench: %d lines, %d stimulus ports, %d cycles.",
        len(lines), len(stimulus_ports), sim_cycles,
    )
    return tb_source
