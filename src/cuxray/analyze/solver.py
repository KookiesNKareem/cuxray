"""Shared-memory layout solver.

Searches XOR swizzles under which every shared access in a kernel is
bank-conflict-free simultaneously. Candidates are verified against the
bank model in `access.bank_conflict_ways`; only verified layouts are
returned.

Swizzle convention (CUTLASS/cute `Swizzle<B,M,S>` applied to byte offsets):
B bits at position M+S are XORed onto bits at position M:

    addr' = addr ^ ((addr >> S) & (((1 << B) - 1) << M))

e.g. <3,4,3>: addr ^ ((addr >> 3) & 0x70).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Optional

from .access import bank_conflict_ways


@dataclass
class Pattern:
    vec: tuple[int, ...]
    width: int
    label: str = ""          # e.g. "bank_conflict.cu:19 LDS"
    ways_before: int = 1


@dataclass
class Solution:
    b: int
    m: int
    s: int
    per_pattern: list[dict]  # label, before, after

    @property
    def formula(self) -> str:
        mask = ((1 << self.b) - 1) << self.m
        return f"addr ^ ((addr >> {self.s}) & {hex(mask)})"

    @property
    def cutlass(self) -> str:
        return f"Swizzle<{self.b},{self.m},{self.s}>"

    @property
    def cute_type(self) -> str:
        return f"cute::Swizzle<{self.b},{self.m},{self.s}>"

    def cuda_snippet(self, elem_bytes: int = 4) -> str:
        mask = ((1 << self.b) - 1) << self.m
        lines = [
            "// Verified conflict-free by cuxray solve. Apply to the byte",
            "// offset within the shared-memory tile (both writes and reads).",
            "__device__ __forceinline__ unsigned cuxray_swizzle(unsigned byte_off) {",
            f"    return byte_off ^ ((byte_off >> {self.s}) & {hex(mask)}u);",
            "}",
        ]
        if elem_bytes and (1 << self.m) % elem_bytes == 0 and self.s >= 0:
            import math
            eshift = int(math.log2(elem_bytes))
            emask = mask >> eshift
            lines += [
                f"// Element-index form for {elem_bytes}-byte elements:",
                f"//   idx ^ ((idx >> {self.s}) & {hex(emask)}u)",
            ]
        lines += [
            f"// CUTLASS/cute: use {self.cute_type} in the tile layout.",
            "// Verify after applying: cuxray report <cubin> --threads <shape>",
        ]
        return "\n".join(lines)


def apply_swizzle(addr: int, b: int, m: int, s: int) -> int:
    return addr ^ ((addr >> s) & (((1 << b) - 1) << m))


def _clean(vec: tuple[int, ...], width: int, b: int, m: int, s: int) -> int:
    swizzled = tuple(apply_swizzle(a, b, m, s) for a in vec)
    ways, _ = bank_conflict_ways(swizzled, width)
    return ways


def solve(patterns: list[Pattern],
          b_range=range(1, 6), m_range=range(2, 8), s_range=range(1, 8),
          max_solutions: int = 3) -> list[Solution]:
    """Find swizzles under which every pattern is conflict-free.

    Constraints on the search: the swizzle must not disturb addressing
    below the access width (m must be >= log2(width) so a wide access's
    bytes stay contiguous), and source bits (m+s) must stay within the
    32-bit address. Results are ordered simplest-first (fewest bits, then
    smallest granule).
    """
    if not patterns:
        return []
    import math
    min_m = max(int(math.log2(max(p.width, 4))) for p in patterns)
    solutions: list[Solution] = []
    for b, m, s in product(b_range, m_range, s_range):
        if m < min_m or m + s + b > 32:
            continue
        per = []
        ok = True
        for p in patterns:
            after = _clean(p.vec, p.width, b, m, s)
            if after > 1:
                ok = False
                break
            per.append({"label": p.label, "before": p.ways_before, "after": after})
        if ok:
            solutions.append(Solution(b=b, m=m, s=s, per_pattern=per))
    solutions.sort(key=lambda sol: (sol.b, sol.m, sol.s))
    return solutions[:max_solutions]


def patterns_from_accesses(accesses: list[dict]) -> list[Pattern]:
    """Build solver input from analyze_accesses(..., keep_vecs=True) output —
    every SHARED access with a known lane vector participates (the layout
    transform applies to the whole tile, so clean accesses must stay clean)."""
    out = []
    for a in accesses:
        if a.get("space") != "shared" or "_vec" not in a:
            continue
        loc = f"{(a.get('file') or '?').rsplit('/', 1)[-1]}:{a.get('line')}"
        out.append(Pattern(
            vec=a["_vec"], width=a["width"],
            label=f"{loc} {a['opcode'].split('.')[0]}",
            ways_before=a.get("conflict_ways", 1),
        ))
    return out
