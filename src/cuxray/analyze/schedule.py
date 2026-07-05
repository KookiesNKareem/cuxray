"""Per-loop cycle estimates from the compiler's embedded schedule.

The stall fields give the deterministic issue schedule ptxas computed:
summing them over a loop body yields the cycles the warp spends issuing and
stalling per iteration, excluding variable-latency waits (scoreboard joins
on memory results), which are counted separately. All outputs are ESTIMATES:
they assume single-warp execution with cache-hit memory behavior and are
lower bounds on real per-warp iteration time.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from ..parse.cfgdot import FunctionCFG
from ..parse.ctrl import Ctrl
from ..parse.sass import Function, Instruction


def loop_schedule(func: Function, cfg: Optional[FunctionCFG],
                  controls: dict[int, Ctrl],
                  bytes_per_iter: Optional[dict[str, int]] = None) -> list[dict]:
    """bytes_per_iter maps loop header -> est. global bytes per warp-iter
    (from roofline.loop_report); when given, rows gain a per-byte
    normalization so kernels of different widths compare directly."""
    if cfg is None or not cfg.loops:
        return []
    by_block: dict[str, list[Instruction]] = {}
    for i in func.instructions:
        by_block.setdefault(i.block or "", []).append(i)

    rows = []
    for header, members in cfg.loops.items():
        instrs = [i for b in members for i in by_block.get(b, [])]
        ctrl_pairs = [(i, controls.get(i.addr)) for i in instrs]
        known = [(i, c) for i, c in ctrl_pairs if c is not None]
        if not known:
            continue
        stall_cycles = sum(max(c.stall, 1) for _, c in known)
        waits = sum(1 for _, c in known if c.watdb)
        by_line: dict = defaultdict(int)
        by_op: dict = defaultdict(lambda: [0, 0])  # base opcode -> [count, cycles]
        for i, c in known:
            if i.line is not None:
                by_line[(i.file, i.line)] += max(c.stall, 1)
            slot = by_op[i.opcode.split(".")[0]]
            slot[0] += 1
            slot[1] += max(c.stall, 1)
        top = sorted(
            ({"file": f, "line": ln, "est_stall_cycles": v}
             for (f, ln), v in by_line.items()),
            key=lambda d: -d["est_stall_cycles"],
        )[:5]
        top_ops = sorted(
            ({"opcode": k, "count": v[0], "est_stall_cycles": v[1]}
             for k, v in by_op.items()),
            key=lambda d: -d["est_stall_cycles"],
        )[:6]
        lines = [i.line for i in instrs if i.line]
        row = {
            "header": header,
            "loop_depth": max(cfg.loop_depth.get(b, 0) for b in members),
            "line_span": [min(lines), max(lines)] if lines else None,
            "instructions": len(instrs),
            "est_issue_stall_cycles_per_iter": stall_cycles,
            "scoreboard_waits_per_iter": waits,
            "top_stall_lines": top,
            "top_stall_opcodes": top_ops,
            "coverage": round(len(known) / len(instrs), 3),
        }
        nbytes = (bytes_per_iter or {}).get(header)
        if nbytes:
            row["est_global_bytes_per_warp_iter"] = nbytes
            row["est_stall_cycles_per_512B"] = round(stall_cycles * 512 / nbytes, 1)
        rows.append(row)
    rows.sort(key=lambda r: (-r["loop_depth"], -r["est_issue_stall_cycles_per_iter"]))
    return rows
