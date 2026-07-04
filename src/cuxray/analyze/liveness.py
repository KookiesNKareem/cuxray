"""Register-pressure analysis from merged -gi/-plr disassembly."""

from __future__ import annotations

from collections import defaultdict

from ..parse.sass import Function


def pressure(func: Function) -> dict:
    """Pressure curve + peak + per-source-line aggregation for one kernel.

    Requires merge_liveness() to have been applied; instructions without
    liveness (no -plr row) are skipped.
    """
    curve = [(i.addr, i.live_gpr) for i in func.instructions if i.live_gpr is not None]
    if not curve:
        return {"available": False}

    peak_val = max(v for _, v in curve)
    peak_instrs = [i for i in func.instructions if i.live_gpr == peak_val]
    first = peak_instrs[0]

    by_line: dict[tuple[str | None, int | None], int] = defaultdict(int)
    for i in func.instructions:
        if i.live_gpr is None:
            continue
        key = (i.file, i.line)
        by_line[key] = max(by_line[key], i.live_gpr)

    per_line = [
        {"file": f, "line": ln, "max_live_gpr": v}
        for (f, ln), v in by_line.items()
        if ln is not None
    ]
    per_line.sort(key=lambda d: -d["max_live_gpr"])

    return {
        "available": True,
        "peak": {
            "live_gpr": peak_val,
            "addr": first.addr,
            "file": first.file,
            "line": first.line,
            "instruction_count_at_peak": len(peak_instrs),
        },
        "per_line": per_line[:20],
        "instructions_analyzed": len(curve),
    }
