"""Static shared-memory bank-conflict and global-coalescing analysis (Layer B).

Model notes (kept deliberately explicit — every number here is defensible):

Shared memory: 32 banks × 4 bytes, bank = (addr >> 2) & 31. Lanes accessing
the same 4-byte word broadcast (no conflict); lanes hitting the same bank at
*different* words serialize. Wide accesses (.64/.128) are processed in 2/4
phases — one 4-byte word column per phase — so we report the max ways across
phases. Natural alignment (required by hardware) plus the uniform-shift
invariance of conflict counts means the unknown uniform base never affects
the answer: a uniform base only *rotates* the bank pattern.

Global memory: counted in 32-byte sectors (what NCU reports). The unknown
uniform base CAN change sector counts by boundary straddle, so we evaluate
all base offsets mod 32 and report the worst case, flagging when it differs
from the best.

Accesses we do NOT model in this version (listed as unanalyzed, with reason,
never guessed): LDSM/STSM matrix loads, LDGSTS/TMA async copies, generic
LD/ST (address space unknown), LDL/STL (local — covered by the spill map).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from ..parse.sass import Function, Instruction
from . import lanevalue as lv
from .dataflow import State, analyze

_WIDTH = {"128": 16, "64": 8, "32": 4, "U8": 1, "S8": 1, "U16": 2, "S16": 2}
_SCALE = re.compile(r"\.X(\d+)$")
_BRACKET = re.compile(r"\[([^\]]*)\]")


def _op_width(opcode: str) -> int:
    for part in opcode.split(".")[1:]:
        if part in _WIDTH:
            return _WIDTH[part]
    return 4


def _addr_value(mem: str, st: State) -> lv.Value:
    """Evaluate the inside of a [...] memory operand to a lane-value."""
    total = lv.const(0)
    for term in re.split(r"(?<!e)\+", mem.replace(" ", "")):  # split on +
        if not term:
            continue
        neg = term.startswith("-")
        term = term.lstrip("-")
        scale = 1
        m = _SCALE.search(term.split("+")[0])
        base = term
        for suffix in (".X4", ".X8", ".X16", ".X32", ".64", ".U32"):
            base = base.replace(suffix, "")
        if m:
            scale = int(m.group(1))
        if re.match(r"^(R\d+|RZ|UR\d+|URZ)$", base):
            v = st.get(base)
            if scale != 1:
                v = lv.mul(v, lv.const(scale))
        elif re.match(r"^0x[0-9a-fA-F]+$|^\d+$", base):
            v = lv.const(int(base, 0))
        else:
            return lv.varying()
        if neg:
            v = lv.sub(lv.const(0), v)
        total = lv.add(total, v)
    return total


def _memory_operand(instr: Instruction) -> Optional[str]:
    groups = _BRACKET.findall(instr.operands)
    if not groups:
        return None
    # desc[UR4][R2.64] → the descriptor bracket contains only a UR; take the
    # last bracket group that references an R register or immediate.
    for g in reversed(groups):
        if re.search(r"R\d+|RZ|0x", g):
            return g
    return groups[-1]


def bank_conflict_ways(vec: tuple[int, ...], width: int) -> tuple[int, bool]:
    """(max conflict ways, any_broadcast). vec = per-lane byte offsets.

    Wide accesses are serviced in transaction groups of consecutive lanes —
    whole warp for <=4 B, half-warps for 8 B, quarter-warps for 16 B — so a
    fully contiguous float4 load is conflict-free (each 8-lane group covers
    all 32 banks exactly once). Conflicts are counted within a group as the
    max number of DISTINCT 4-byte words mapped to one bank; same-word lanes
    broadcast.
    """
    group = max(128 // max(width, 4), 8) if width > 4 else 32
    words_per_lane = max(width // 4, 1)
    worst, broadcast = 1, False
    for g0 in range(0, len(vec), group):
        by_bank: dict[int, set[int]] = {}
        word_hits: dict[int, int] = {}
        for a in vec[g0:g0 + group]:
            for k in range(words_per_lane):
                word = (a >> 2) + k
                by_bank.setdefault(word & 31, set()).add(word)
                word_hits[word] = word_hits.get(word, 0) + 1
        worst = max(worst, max(len(w) for w in by_bank.values()))
        if any(c > 1 for c in word_hits.values()):
            broadcast = True
    return worst, broadcast


def sector_count(vec: tuple[int, ...], width: int) -> tuple[int, int]:
    """(worst, best) 32-byte sectors touched across unknown base alignments."""
    counts = []
    for base in range(0, 32, 4):
        sectors = set()
        for a in vec:
            for k in range(0, max(width, 4), 4):
                sectors.add((base + a + k) >> 5)
        counts.append(len(sectors))
    return max(counts), min(counts)


def _stride(vec: tuple[int, ...]) -> Optional[int]:
    diffs = {(vec[i + 1] - vec[i]) & 0xFFFFFFFF for i in range(len(vec) - 1)}
    if len(diffs) == 1:
        d = diffs.pop()
        return d - (1 << 32) if d > (1 << 31) else d
    return None


_SKIP_REASON = {
    "LDSM": "matrix load — per-lane semantics not modeled yet",
    "STSM": "matrix store — per-lane semantics not modeled yet",
    "LDGSTS": "async global→shared copy — not modeled yet",
    "UTMALDG": "TMA bulk copy — hardware-managed, conflict-free by design",
    "UTMASTG": "TMA bulk copy — hardware-managed, conflict-free by design",
    "LD": "generic address space — cannot tell shared from global",
    "ST": "generic address space — cannot tell shared from global",
}


def analyze_accesses(func: Function, block_dims: tuple[int, int, int],
                     loop_depth: Optional[dict[str, int]] = None) -> dict:
    loop_depth = loop_depth or {}
    pre = analyze(func, block_dims)
    accesses, unanalyzed = [], []

    for i in func.instructions:
        base = i.opcode.split(".")[0]
        space = {"LDS": "shared", "STS": "shared",
                 "LDG": "global", "STG": "global"}.get(base)
        if space is None:
            if base in _SKIP_REASON:
                unanalyzed.append({
                    "addr": i.addr, "opcode": i.opcode, "file": i.file,
                    "line": i.line, "reason": _SKIP_REASON[base],
                })
            continue

        entry = {
            "addr": i.addr, "opcode": i.opcode, "space": space,
            "width": _op_width(i.opcode), "file": i.file, "line": i.line,
            "block": i.block, "loop_depth": loop_depth.get(i.block or "", 0),
        }
        mem = _memory_operand(i)
        val = _addr_value(mem, pre.get(i.addr, State())) if mem else lv.varying()
        if val.kind == lv.VARYING:
            entry["verdict"] = "unknown"
            entry["reason"] = "address is data-dependent or flows through unmodeled instructions"
            unanalyzed.append(entry)
            continue

        vec = val.vec
        entry["stride"] = _stride(vec)
        if space == "shared":
            ways, bcast = bank_conflict_ways(vec, entry["width"])
            entry["verdict"] = "conflict" if ways > 1 else "clean"
            entry["conflict_ways"] = ways
            entry["broadcast"] = bcast
        else:
            worst, best = sector_count(vec, entry["width"])
            ideal = math.ceil(32 * entry["width"] / 32)
            entry["sectors_worst"] = worst
            entry["sectors_best"] = best
            entry["sectors_ideal"] = ideal
            entry["efficiency_pct"] = round(100.0 * ideal / worst, 1)
            entry["verdict"] = "coalesced" if worst <= ideal + 1 else "uncoalesced"
        accesses.append(entry)

    worst_ways = max((a.get("conflict_ways", 1) for a in accesses), default=1)
    uncoalesced = sum(1 for a in accesses if a.get("verdict") == "uncoalesced")
    conflicted = sum(1 for a in accesses if a.get("verdict") == "conflict")
    return {
        "block_dims": list(block_dims),
        "accesses": accesses,
        "unanalyzed": unanalyzed,
        "analyzed_count": len(accesses),
        "unanalyzed_count": len(unanalyzed),
        "worst_bank_conflict_ways": worst_ways,
        "conflicted_shared_accesses": conflicted,
        "uncoalesced_global_accesses": uncoalesced,
    }
