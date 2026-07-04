"""Spill localization: STL/LDL instructions mapped to source lines and
weighted by loop depth from the CFG.

Access width from the opcode suffix gives exact bytes per executed
instruction (.128→16 B ... none→4 B), which is the same accounting ptxas
uses for its "bytes spill stores/loads" report (validated in tests against
ptxas -v on the spill fixture).
"""

from __future__ import annotations

from collections import defaultdict

from ..parse.sass import Function, is_spill

_WIDTH = {"128": 16, "64": 8, "32": 4, "16": 2, "8": 1, "U8": 1, "S8": 1, "U16": 2, "S16": 2}


def _bytes(opcode: str) -> int:
    for part in opcode.split(".")[1:]:
        if part in _WIDTH:
            return _WIDTH[part]
    return 4


def spill_map(func: Function, loop_depth: dict[str, int] | None = None) -> dict:
    loop_depth = loop_depth or {}
    rows: dict[tuple, dict] = defaultdict(
        lambda: {"stores": 0, "loads": 0, "store_bytes": 0, "load_bytes": 0,
                 "loop_depth": 0, "addrs": []}
    )
    tot_store_i = tot_load_i = tot_store_b = tot_load_b = 0
    max_depth = 0

    for i in func.instructions:
        if not is_spill(i):
            continue
        depth = loop_depth.get(i.block or "", 0)
        key = (i.file, i.line)
        row = rows[key]
        row["loop_depth"] = max(row["loop_depth"], depth)
        if len(row["addrs"]) < 4:
            row["addrs"].append(i.addr)
        b = _bytes(i.opcode)
        if i.opcode.startswith("STL"):
            row["stores"] += 1
            row["store_bytes"] += b
            tot_store_i += 1
            tot_store_b += b
        else:
            row["loads"] += 1
            row["load_bytes"] += b
            tot_load_i += 1
            tot_load_b += b
        max_depth = max(max_depth, depth)

    by_line = [
        {"file": f, "line": ln, **row}
        for (f, ln), row in rows.items()
    ]
    by_line.sort(key=lambda d: (-d["loop_depth"], -(d["stores"] + d["loads"])))

    return {
        "store_instructions": tot_store_i,
        "load_instructions": tot_load_i,
        "store_bytes": tot_store_b,
        "load_bytes": tot_load_b,
        "max_loop_depth": max_depth,
        "by_line": by_line,
    }
