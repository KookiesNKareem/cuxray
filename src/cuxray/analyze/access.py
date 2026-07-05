"""Static shared-memory bank-conflict and global-coalescing analysis.

Shared memory model: 32 banks x 4 bytes, bank = (addr >> 2) & 31. Same-word
lanes broadcast; same-bank different-word lanes serialize. Wide accesses are
serviced in transaction groups of consecutive lanes (warp / half-warp /
quarter-warp for 4/8/16 B). Conflict counts are invariant under the unknown
uniform base (natural alignment assumed).

Global model: 32-byte sectors. The uniform base can change sector counts by
boundary straddle; all base offsets mod 32 are evaluated and worst/best
reported.

LDSM/STSM use the 8-lane phase-group model (16 B rows). LDGSTS is analyzed
as both a global read and a shared write. Unmodeled (reported, not guessed):
TMA bulk copies, generic LD/ST, LDL/STL (covered by the spill map).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from ..parse.sass import Function, Instruction
from . import lanevalue as lv
from .dataflow import State, addr_value, analyze_ex, memory_operand

_WIDTH = {"128": 16, "64": 8, "32": 4, "U8": 1, "S8": 1, "U16": 2, "S16": 2}


def _op_width(opcode: str) -> int:
    for part in opcode.split(".")[1:]:
        if part in _WIDTH:
            return _WIDTH[part]
    return 4


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


def _verified_fixes(vec: tuple[int, ...], width: int) -> list[dict]:
    """Conflict fixes verified clean under the bank model before being
    suggested. Zero-smem-cost fixes (swizzles) rank first."""
    fixes: list[dict] = []
    # XOR swizzle of the word index: free (no smem growth)
    for mask in (7, 15, 31):
        swizzled = tuple(
            ((((a >> 2) ^ ((a >> 7) & mask)) << 2) | (a & 3)) for a in vec
        )
        ways, _ = bank_conflict_ways(swizzled, width)
        if ways == 1:
            fixes.append({
                "kind": "swizzle",
                "smem_delta_per_row": 0,
                "description": (
                    f"XOR-swizzle the word index: idx ^ ((idx >> 5) & {mask}) "
                    "— verified clean, costs no shared memory"
                ),
            })
            break
    s = _stride(vec)
    if s and s > 0:
        # Padding: classic row-pitch fix. vec = lane * s → lane * (s + pad)
        for pad in (4, 8, 12):
            padded = tuple((a // s) * (s + pad) + (a % s) for a in vec)
            ways, _ = bank_conflict_ways(padded, width)
            if ways == 1:
                fixes.append({
                    "kind": "pad",
                    "stride": s,
                    "pad_bytes": pad,
                    "smem_delta_per_row": pad,
                    "description": (
                        f"pad the {s} B row pitch to {s + pad} B — verified clean "
                        f"(costs {pad} B shared memory per row)"
                    ),
                })
                break
    return fixes


def _stride(vec: tuple[int, ...]) -> Optional[int]:
    diffs = {(vec[i + 1] - vec[i]) & 0xFFFFFFFF for i in range(len(vec) - 1)}
    if len(diffs) == 1:
        d = diffs.pop()
        return d - (1 << 32) if d > (1 << 31) else d
    return None


_SKIP_REASON = {
    "UTMALDG": "TMA bulk copy — hardware-managed, conflict-free by design",
    "UTMASTG": "TMA bulk copy — hardware-managed, conflict-free by design",
    "LD": "generic address space — cannot tell shared from global",
    "ST": "generic address space — cannot tell shared from global",
}


def analyze_accesses(func: Function, block_dims: tuple[int, int, int],
                     loop_depth: Optional[dict[str, int]] = None,
                     keep_vecs: bool = False) -> dict:
    """keep_vecs=True attaches the raw per-lane vector to each access under
    '_vec' (for the layout solver); excluded from report JSON."""
    loop_depth = loop_depth or {}
    pre, flow = analyze_ex(func, block_dims)
    accesses, unanalyzed = [], []

    def make_entry(i, space, width=None):
        return {
            "addr": i.addr, "opcode": i.opcode, "space": space,
            "width": width if width is not None else _op_width(i.opcode),
            "file": i.file, "line": i.line, "block": i.block,
            "loop_depth": loop_depth.get(i.block or "", 0),
        }

    def eval_addr(i, mem_text):
        return (addr_value(mem_text, pre.get(i.addr, State()))
                if mem_text else lv.varying("no address operand"))

    def finish_shared(entry, vec):
        if keep_vecs:
            entry["_vec"] = vec
        entry["stride"] = _stride(vec)
        ways, bcast = bank_conflict_ways(vec, entry["width"])
        entry["verdict"] = "conflict" if ways > 1 else "clean"
        entry["conflict_ways"] = ways
        entry["broadcast"] = bcast
        if ways > 1:
            entry["fixes"] = _verified_fixes(vec, entry["width"])
        accesses.append(entry)

    def finish_global(entry, vec):
        if keep_vecs:
            entry["_vec"] = vec
        entry["stride"] = _stride(vec)
        worst, best = sector_count(vec, entry["width"])
        ideal = math.ceil(32 * entry["width"] / 32)
        entry["sectors_worst"] = worst
        entry["sectors_best"] = best
        entry["sectors_ideal"] = ideal
        entry["efficiency_pct"] = round(100.0 * ideal / worst, 1)
        entry["verdict"] = "coalesced" if worst <= ideal + 1 else "uncoalesced"
        accesses.append(entry)

    def unknown(entry, val):
        entry["verdict"] = "unknown"
        entry["reason"] = val.reason or "address not traceable"
        unanalyzed.append(entry)

    for i in func.instructions:
        base = i.opcode.split(".")[0]

        if base in ("LDSM", "STSM"):
            # ldmatrix/stmatrix: 8*num consecutive lanes each supply a 16 B
            # row address; hardware services them in 8-lane phase groups —
            # exactly the width-16 transaction-group model.
            last = i.opcode.split(".")[-1]
            num = int(last) if last.isdigit() else 1
            entry = make_entry(i, "shared", width=16)
            val = eval_addr(i, memory_operand(i))
            if val.kind == lv.VARYING:
                unknown(entry, val)
            else:
                finish_shared(entry, val.vec[:8 * num])
            continue

        if base == "LDGSTS":
            # async copy: a global read AND a shared write in one instruction
            groups = re.findall(r"\[([^\]]*)\]", i.operands)
            smem_dst = groups[0] if groups else None
            g_entry = make_entry(i, "global")
            gval = eval_addr(i, memory_operand(i))  # last R-bearing bracket = src
            if gval.kind == lv.VARYING:
                unknown(g_entry, gval)
            else:
                finish_global(g_entry, gval.vec)
            s_entry = make_entry(i, "shared")
            sval = eval_addr(i, smem_dst)
            if sval.kind == lv.VARYING:
                unknown(s_entry, sval)
            else:
                finish_shared(s_entry, sval.vec)
            continue

        space = {"LDS": "shared", "STS": "shared",
                 "LDG": "global", "STG": "global"}.get(base)
        if space is None:
            if base in _SKIP_REASON:
                unanalyzed.append({
                    "addr": i.addr, "opcode": i.opcode, "file": i.file,
                    "line": i.line, "reason": _SKIP_REASON[base],
                })
            continue

        entry = make_entry(i, space)
        val = eval_addr(i, memory_operand(i))
        if val.kind == lv.VARYING:
            unknown(entry, val)
        elif space == "shared":
            finish_shared(entry, val.vec)
        else:
            finish_global(entry, val.vec)

    worst_ways = max((a.get("conflict_ways", 1) for a in accesses), default=1)
    uncoalesced = sum(1 for a in accesses if a.get("verdict") == "uncoalesced")
    conflicted = sum(1 for a in accesses if a.get("verdict") == "conflict")

    # Aggregate per source site for reporting (raw lists get huge on CUTLASS)
    sites: dict[tuple, dict] = {}
    for a in accesses:
        key = (a["file"], a["line"], a["space"], a["verdict"])
        s = sites.setdefault(key, {
            "file": a["file"], "line": a["line"], "space": a["space"],
            "verdict": a["verdict"], "count": 0, "loop_depth": 0,
            "conflict_ways": 1, "efficiency_pct": None, "stride": a.get("stride"),
            "fixes": [],
        })
        if a.get("fixes") and not s["fixes"]:
            s["fixes"] = a["fixes"]
        s["count"] += 1
        s["loop_depth"] = max(s["loop_depth"], a["loop_depth"])
        if "conflict_ways" in a:
            s["conflict_ways"] = max(s["conflict_ways"], a["conflict_ways"])
        if "efficiency_pct" in a:
            cur = s["efficiency_pct"]
            s["efficiency_pct"] = a["efficiency_pct"] if cur is None else min(cur, a["efficiency_pct"])
    by_site = sorted(
        sites.values(),
        key=lambda s: (s["verdict"] in ("clean", "coalesced"), -s["loop_depth"],
                       -s["conflict_ways"], -s["count"]),
    )
    reasons: dict[str, int] = {}
    for u in unanalyzed:
        reasons[u["reason"]] = reasons.get(u["reason"], 0) + 1

    return {
        "block_dims": list(block_dims),
        "dataflow_converged": flow["converged"],
        "unreached_blocks": flow["unreached_blocks"],
        "accesses": accesses,
        "by_site": by_site,
        "unanalyzed": unanalyzed,
        "unanalyzed_by_reason": reasons,
        "analyzed_count": len(accesses),
        "unanalyzed_count": len(unanalyzed),
        "worst_bank_conflict_ways": worst_ways,
        "conflicted_shared_accesses": conflicted,
        "uncoalesced_global_accesses": uncoalesced,
    }
