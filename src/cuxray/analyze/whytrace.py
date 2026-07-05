"""Backward slice of an address computation with abstract lane-values.

Explains why an access is (or is not) analyzable: walks the defining
instructions of the address registers in reverse, reporting each step's
abstract value and pointing at the first place precision degrades.
"""

from __future__ import annotations

import re
from typing import Optional

from ..parse.sass import Function, Instruction
from . import lanevalue as lv
from .dataflow import State, _dest, _split_operands, analyze_ex, memory_operand, step

_SRC_REG = re.compile(r"\bU?R\d+\b")


def _defined_regs(instr: Instruction) -> set[str]:
    ops = _split_operands(instr.operands)
    d = _dest(ops)
    if d is None:
        return set()
    out = {d}
    m = re.match(r"^(U?R)(\d+)$", d)
    if not m:
        return out
    pre, n = m.group(1), int(m.group(2))
    width = 0
    if ".WIDE" in instr.opcode or ".64" in instr.opcode:
        width = 1
    elif ".128" in instr.opcode:
        width = 3
    for i in range(1, width + 1):
        out.add(f"{pre}{n + i}")
    return out


def slice_access(func: Function, block_dims: tuple[int, int, int],
                 target_addr: int, max_rows: int = 30) -> dict:
    """Chain of definitions feeding the memory operand at target_addr."""
    pre, _flow = analyze_ex(func, block_dims)
    tids = lv.tid_vectors(block_dims)
    by_addr = {i.addr: i for i in func.instructions}
    tgt = by_addr.get(target_addr)
    if tgt is None:
        raise ValueError(f"no instruction at {hex(target_addr)}")

    mem = memory_operand(tgt)
    want = set(_SRC_REG.findall(mem or ""))
    st0 = pre.get(target_addr)
    addr_val = None
    if st0 is not None and mem:
        from .dataflow import addr_value
        addr_val = addr_value(mem, st0)

    rows = []
    origin = None
    before = [i for i in func.instructions if i.addr < target_addr]
    for instr in reversed(before):
        if len(rows) >= max_rows or not want:
            break
        defined = _defined_regs(instr)
        hit = defined & want
        if not hit:
            continue
        st = pre.get(instr.addr)
        post = st.copy() if st else State()
        step(instr, post, tids)
        srcs = set(_SRC_REG.findall(",".join(_split_operands(instr.operands)[1:])))
        src_vals = {r: (st.get(r) if st else lv.varying()) for r in sorted(srcs)}
        out_val = post.get(next(iter(hit)))
        row = {
            "addr": instr.addr,
            "opcode": instr.opcode,
            "operands": instr.operands,
            "defines": sorted(hit),
            "value": repr(out_val),
            "sources": {r: repr(v) for r, v in src_vals.items()},
            "file": instr.file, "line": instr.line,
        }
        # the first (deepest-found) step whose output degrades while no
        # register source is itself VARYING is where precision was lost
        if out_val.kind == lv.VARYING and \
                not any(v.kind == lv.VARYING for v in src_vals.values()):
            row["degrades_here"] = True
            origin = instr.addr
        rows.append(row)
        want -= defined
        want |= {r for r in srcs if r not in ("RZ", "URZ")}

    return {
        "target": {"addr": target_addr, "opcode": tgt.opcode,
                   "operands": tgt.operands, "mem": mem,
                   "file": tgt.file, "line": tgt.line},
        "address_value": repr(addr_val) if addr_val is not None else None,
        "chain": rows,
        "first_degradation": origin,
        "unresolved_inputs": sorted(want),
    }


def why_kernel(func: Function, block_dims: tuple[int, int, int],
               target_addr: Optional[int] = None,
               loop_depth: Optional[dict[str, int]] = None) -> list[dict]:
    """Slices for the kernel's unanalyzed accesses (or one explicit addr)."""
    from .access import analyze_accesses
    if target_addr is not None:
        return [slice_access(func, block_dims, target_addr)]
    res = analyze_accesses(func, block_dims, loop_depth or {})
    out = []
    for a in res["unanalyzed"]:
        s = slice_access(func, block_dims, a["addr"])
        s["reason"] = a.get("reason")
        out.append(s)
    return out
