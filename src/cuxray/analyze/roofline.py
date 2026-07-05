"""Per-loop static roofline: FLOP counts, memory traffic, and arithmetic
intensity per warp-iteration.

All values in this module are ESTIMATES derived from instruction inspection
and are labeled as such in output. Units are per WARP per loop iteration
(SASS executes per warp; per-iteration ratios are trip-count-invariant).
Traffic counts request bytes at 32-byte sector granularity; cache reuse may
reduce actual DRAM traffic, so byte figures are upper bounds on demand and
the derived AI is a lower bound.
"""

from __future__ import annotations

import re
from typing import Optional

from ..parse.cfgdot import FunctionCFG
from ..parse.sass import Function, Instruction

# FLOPs per warp per instruction for scalar/vector math ops.
# FMA counts as 2. H2 variants operate on 2 halves per lane.
_FLOP_OPS = {
    "FFMA": 2, "DFMA": 2, "FADD": 1, "DADD": 1, "FMUL": 1, "DMUL": 1,
    "FSEL": 0, "MUFU": 1, "FCHK": 0,
    "HFMA2": 4, "HADD2": 2, "HMUL2": 2,
}
_LANE = 32

_MMA_SHAPE = re.compile(r"\.(\d{3,5})\.")


def _mma_flops(opcode: str) -> Optional[int]:
    """FLOPs per warp for an MMA instruction, from its m/n/k shape digits.

    Shape strings concatenate m, n, k (e.g. 16816 = m16 n8 k16). Known
    encodings are decoded; unknown shapes return None and are counted
    separately so totals stay honest.
    """
    m = _MMA_SHAPE.search(opcode)
    if not m:
        return None
    digits = m.group(1)
    shapes = {
        "884": (8, 8, 4), "1688": (16, 8, 8), "16816": (16, 8, 16),
        "16832": (16, 8, 32), "1684": (16, 8, 4), "16864": (16, 8, 64),
        "168128": (16, 8, 128),
    }
    if digits not in shapes:
        return None
    mm, nn, kk = shapes[digits]
    return 2 * mm * nn * kk


def loop_report(func: Function, cfg: Optional[FunctionCFG],
                accesses: Optional[list[dict]] = None, spec=None) -> list[dict]:
    """Roofline rows for each natural loop, innermost-biased (loops sorted by
    depth descending). `accesses` (from analyze_accesses) supplies sector
    counts for traffic; without it, traffic falls back to access-width bytes
    (perfect-coalescing assumption, flagged). `spec` (ArchSpec) enables the
    SIMT-vs-tensor-core datapath crossover check per loop."""
    if cfg is None or not cfg.loops:
        return []
    by_block: dict[str, list[Instruction]] = {}
    for i in func.instructions:
        by_block.setdefault(i.block or "", []).append(i)
    acc_by_addr = {a["addr"]: a for a in (accesses or [])}

    rows = []
    for header, members in cfg.loops.items():
        instrs = [i for b in members for i in by_block.get(b, [])]
        if not instrs:
            continue
        flops = 0
        mma_flops = 0
        unknown_mma = 0
        for i in instrs:
            base = i.opcode.split(".")[0]
            if base in _FLOP_OPS:
                flops += _FLOP_OPS[base] * _LANE
            elif base in ("HMMA", "QMMA", "OMMA", "BMMA", "DMMA"):
                f = _mma_flops(i.opcode)
                if f is None:
                    unknown_mma += 1
                else:
                    mma_flops += f

        gbytes = gbytes_ideal = invariant_bytes = 0
        smem_wavefronts = smem_wavefronts_ideal = 0
        approximate_traffic = False
        for i in instrs:
            base = i.opcode.split(".")[0]
            a = acc_by_addr.get(i.addr)
            if base in ("LDG", "STG") or (base == "LDGSTS"):
                if a and "sectors_worst" in a:
                    gbytes += a["sectors_worst"] * 32
                    gbytes_ideal += a["sectors_ideal"] * 32
                    if a.get("block_invariant"):
                        invariant_bytes += a["sectors_worst"] * 32
                elif base != "LDGSTS":
                    width = 4
                    gbytes += width * _LANE
                    gbytes_ideal += width * _LANE
                    approximate_traffic = True
            if a and a.get("space") == "shared" and "conflict_ways" in a:
                groups = max(1, _LANE // max(128 // max(a["width"], 4), 8)) \
                    if a["width"] > 4 else 1
                smem_wavefronts += a["conflict_ways"] * groups
                smem_wavefronts_ideal += groups

        total_flops = flops + mma_flops
        lines = [i.line for i in instrs if i.line]
        row = {
            "header": header,
            "loop_depth": max(cfg.loop_depth.get(b, 0) for b in members),
            "instructions": len(instrs),
            "line_span": [min(lines), max(lines)] if lines else None,
            "est_flops_per_warp_iter": total_flops,
            "est_mma_flops_per_warp_iter": mma_flops,
            "unknown_mma_instructions": unknown_mma,
            "est_global_bytes_per_warp_iter": gbytes,
            "est_global_bytes_ideal": gbytes_ideal,
            "est_traffic_inflation": (round(gbytes / gbytes_ideal, 2)
                                      if gbytes_ideal else None),
            "est_smem_wavefronts_per_iter": smem_wavefronts,
            "est_smem_replay_factor": (round(smem_wavefronts / smem_wavefronts_ideal, 2)
                                       if smem_wavefronts_ideal else None),
            "est_arithmetic_intensity": (round(total_flops / gbytes, 3)
                                         if gbytes else None),
            "est_block_invariant_bytes_per_warp_iter": invariant_bytes,
            "approximate_traffic": approximate_traffic,
        }
        if spec is not None:
            from .crossover import analyze_loop
            xr = analyze_loop([i.opcode for i in instrs], spec,
                              row["est_arithmetic_intensity"])
            if xr:
                row["tensor_crossover"] = xr
        rows.append(row)
    rows.sort(key=lambda r: (-r["loop_depth"], -r["instructions"]))
    return rows


def classify(ai: Optional[float], peak_tflops: Optional[float],
             peak_gbs: Optional[float]) -> Optional[dict]:
    """Bound classification when the user supplies device peaks."""
    if ai is None or not peak_tflops or not peak_gbs:
        return None
    ridge = peak_tflops * 1000.0 / peak_gbs  # FLOP per byte
    return {
        "ridge_flop_per_byte": round(ridge, 2),
        "bound": "memory" if ai < ridge else "compute",
        "est_peak_fraction_if_memory_bound": round(min(1.0, ai / ridge), 3),
    }
