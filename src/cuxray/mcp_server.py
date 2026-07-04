"""MCP server exposing cuxray to coding agents (stdio transport).

Install with `pip install 'cuxray[mcp]'`, run with `cuxray mcp`, register e.g.:

    claude mcp add cuxray -- cuxray mcp

All tools return the same JSON documents as the CLI's --json output
(schema `cuxray.schema/1`).
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from .archspec import lookup
from .diffgate import diff_reports, eval_gate, parse_gate
from .occupancy import compute, find_cliffs, sweep_block_sizes
from .report import build_report
from .toolchain import resolve

mcp = FastMCP(
    "cuxray",
    instructions=(
        "Hardware-free static analysis of CUDA kernel binaries: register "
        "pressure, spills (source-attributed, loop-weighted), occupancy "
        "what-ifs, and diffs between two builds. Works on .cubin files, host "
        "ELFs (.so/.o), Triton cache directories, and .ptx. No GPU needed. "
        "Everything reported is ground truth read from the binary or NVIDIA's "
        "published architecture tables — no estimates."
    ),
)


@mcp.tool()
def list_kernels(path: str) -> dict:
    """List kernels and headline resources (regs/smem/stack) in an artifact."""
    return build_report(path, resolve(quiet=True))


@mcp.tool()
def report(
    path: str,
    threads: Optional[int] = None,
    kernel_regex: Optional[str] = None,
    arch: Optional[str] = None,
) -> dict:
    """Full static report for an artifact: per-kernel registers, source-line
    register-pressure peaks, spill locations weighted by loop depth, and (if
    `threads` is given) occupancy with limiter and cliff analysis.
    `arch` is only needed for raw .ptx inputs (e.g. 'sm_120a')."""
    return build_report(path, resolve(quiet=True), threads=threads,
                        kernel_re=kernel_regex, arch=arch)


@mcp.tool()
def diff(
    old_path: str,
    new_path: str,
    threads: Optional[int] = None,
    kernel_regex: Optional[str] = None,
) -> dict:
    """Compare two artifacts kernel-by-kernel: regs, stack, smem, spills,
    pressure peak, occupancy. Use after editing a kernel to see exactly what
    the change cost or saved."""
    tc = resolve(quiet=True)
    old = build_report(old_path, tc, threads=threads, kernel_re=kernel_regex)
    new = build_report(new_path, tc, threads=threads, kernel_re=kernel_regex)
    return diff_reports(old, new, kernel_regex)


@mcp.tool()
def occupancy_whatif(
    arch: str,
    regs_per_thread: int,
    threads_per_block: int,
    smem_bytes: int = 0,
    sweep_blocks: bool = False,
) -> dict:
    """Static occupancy for a hypothetical kernel configuration — no binary
    needed. Returns blocks/SM, limiter, and cliffs (e.g. 'dropping to 128
    regs gains a block'). arch examples: sm_90, sm_120."""
    spec = lookup(arch)
    occ = compute(spec, regs_per_thread, threads_per_block, smem_static=smem_bytes)
    d = occ.to_dict()
    d["cliffs"] = find_cliffs(spec, occ)
    if sweep_blocks:
        d["sweep"] = [
            {"threads": o.threads_per_block, "occupancy_pct": o.occupancy_pct,
             "blocks_per_sm": o.blocks_per_sm}
            for o in sweep_block_sizes(spec, regs_per_thread, smem_static=smem_bytes)
        ]
    return d


@mcp.tool()
def gate(path: str, expression: str, kernel_regex: Optional[str] = None) -> dict:
    """Evaluate a CI gate expression against an artifact, e.g.
    'spill_instrs==0, regs<=168, occupancy(threads=256)>=25'.
    Returns {passed: bool, violations: [...]}."""
    doc = build_report(path, resolve(quiet=True), kernel_re=kernel_regex)
    violations = eval_gate(doc, parse_gate(expression))
    return {"passed": not violations, "violations": violations}


def run() -> None:
    mcp.run()
