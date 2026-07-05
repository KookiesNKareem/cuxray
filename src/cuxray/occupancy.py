"""Static occupancy calculator — port of NVIDIA's cuda_occupancy.h algorithm.

Faithful to cuda_occupancy.h (cuda_cudart 13.3.29) for the warp/block/register
limits: registers are allocated per warp in units of `reg_alloc_granularity`
(256), drawn from per-subpartition register files (4 subpartitions/SM), with
the hardware launch check that rounds warp allocation up to the subpartition
count. Partitioned global caching is omitted (not supported on CC >= 7).

Shared memory follows the physical model rather than the header's per-block
limit check verbatim: allocated = round_up(static + dynamic + reserved,
granularity); blocks = carveout // allocated. Kernels using more than 48 KB
get an informational "requires opt-in" note. (The header's comparison of
reserved-inclusive allocation against the reserved-exclusive opt-in limit
would reject the documented per-block maximum; validate against ncu_occupancy
on real hardware before changing this.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .archspec import ArchSpec, KB

_INF = 1 << 30
_OPT_IN_THRESHOLD = 48 * KB


def _div_round_up(x: int, y: int) -> int:
    return (x + y - 1) // y


def _round_up(x: int, y: int) -> int:
    return y * _div_round_up(x, y)


@dataclass
class Occupancy:
    arch: str
    threads_per_block: int
    regs_per_thread: int
    smem_static: int
    smem_dynamic: int
    carveout_kb: Optional[int]
    warps_per_block: int
    blocks_per_sm: int
    active_warps: int
    max_warps: int
    occupancy_pct: float
    limiter: str                      # "registers" | "shared_memory" | "warps" | "blocks" | "none"
    limits: dict[str, int] = field(default_factory=dict)
    regs_allocated_per_block: int = 0
    smem_allocated_per_block: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["limits"] = dict(self.limits)
        d["notes"] = list(self.notes)
        return d


def _blocks_by_warps(spec: ArchSpec, threads_per_block: int) -> int:
    if threads_per_block > spec.max_threads_per_block:
        return 0
    warps = _div_round_up(threads_per_block, spec.warp_size)
    return spec.max_warps_per_sm // warps


def _blocks_by_regs(spec: ArchSpec, regs_per_thread: int, warps_per_block: int) -> tuple[int, int]:
    """Returns (max_blocks, regs_allocated_per_block)."""
    if regs_per_thread == 0:
        return _INF, 0
    if regs_per_thread > spec.max_regs_per_thread:
        return 0, 0
    regs_per_warp_alloc = _round_up(regs_per_thread * spec.warp_size, spec.reg_alloc_granularity)
    regs_alloc_cta = regs_per_warp_alloc * warps_per_block
    # Hardware launch check assumes allocation across all subpartitions at once
    regs_assumed_cta = regs_per_warp_alloc * _round_up(warps_per_block, spec.subpartitions)
    if spec.max_regs_per_block < max(regs_assumed_cta, regs_alloc_cta):
        return 0, regs_alloc_cta
    regs_per_subpartition = spec.regfile_per_sm // spec.subpartitions
    warps_per_subpartition = regs_per_subpartition // regs_per_warp_alloc
    warps_per_sm = warps_per_subpartition * spec.subpartitions
    return warps_per_sm // warps_per_block, regs_alloc_cta


def _effective_smem_per_sm(spec: ArchSpec, allocated_per_cta: int, carveout_kb: Optional[int]) -> int:
    if carveout_kb is None:
        return spec.smem_per_sm
    requested = carveout_kb * KB
    if requested >= allocated_per_cta:
        return requested
    # Volta+ align-up rule: bump to the next supported carveout that fits the CTA
    for c in spec.smem_carveouts_kb:
        if c * KB >= allocated_per_cta:
            return c * KB
    return spec.smem_per_sm


def _blocks_by_smem(
    spec: ArchSpec, smem_static: int, smem_dynamic: int, carveout_kb: Optional[int]
) -> tuple[int, int, list[str]]:
    """Returns (max_blocks, smem_allocated_per_block, notes)."""
    notes: list[str] = []
    usage = smem_static + smem_dynamic
    if usage > spec.smem_per_block_max:
        notes.append(
            f"shared memory {usage} B exceeds per-block maximum "
            f"{spec.smem_per_block_max} B for {spec.sm} — kernel cannot launch"
        )
        return 0, 0, notes
    if usage > _OPT_IN_THRESHOLD:
        notes.append(
            f"shared memory {usage} B > 48 KB requires opt-in via "
            "cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)"
        )
    allocated = _round_up(usage + spec.smem_reserved_per_block, spec.smem_alloc_granularity)
    per_sm = _effective_smem_per_sm(spec, allocated, carveout_kb)
    if carveout_kb is not None and per_sm != carveout_kb * KB:
        notes.append(f"carveout {carveout_kb} KB too small for block; aligned up to {per_sm // KB} KB")
    if allocated == 0:
        return _INF, 0, notes
    return per_sm // allocated, allocated, notes


def compute(
    spec: ArchSpec,
    regs_per_thread: int,
    threads_per_block: int,
    smem_static: int = 0,
    smem_dynamic: int = 0,
    carveout_kb: Optional[int] = None,
) -> Occupancy:
    warps_per_block = _div_round_up(threads_per_block, spec.warp_size)

    by_warps = _blocks_by_warps(spec, threads_per_block)
    by_blocks = spec.max_blocks_per_sm
    by_regs, regs_alloc = _blocks_by_regs(spec, regs_per_thread, warps_per_block)
    by_smem, smem_alloc, notes = _blocks_by_smem(spec, smem_static, smem_dynamic, carveout_kb)

    limits = {
        "warps": by_warps,
        "blocks": by_blocks,
        "registers": by_regs,
        "shared_memory": by_smem,
    }
    blocks = min(limits.values())
    # Name ALL binding resources — on ties, relieving only one gains nothing,
    # and saying so prevents wasted optimization effort.
    binding = [name for name in ("registers", "shared_memory", "warps", "blocks")
               if limits[name] == blocks]
    limiter = "+".join(binding) if binding else "none"
    active_warps = blocks * warps_per_block
    max_warps = spec.max_warps_per_sm
    pct = 100.0 * active_warps / max_warps if max_warps else 0.0

    if threads_per_block > spec.max_threads_per_block:
        notes.append(
            f"block size {threads_per_block} exceeds max {spec.max_threads_per_block}"
        )

    return Occupancy(
        arch=spec.sm,
        threads_per_block=threads_per_block,
        regs_per_thread=regs_per_thread,
        smem_static=smem_static,
        smem_dynamic=smem_dynamic,
        carveout_kb=carveout_kb,
        warps_per_block=warps_per_block,
        blocks_per_sm=0 if blocks <= 0 else min(blocks, _INF),
        active_warps=max(active_warps, 0),
        max_warps=max_warps,
        occupancy_pct=round(max(pct, 0.0), 1),
        limiter=limiter,
        limits={k: (v if v < _INF else -1) for k, v in limits.items()},
        regs_allocated_per_block=regs_alloc,
        smem_allocated_per_block=smem_alloc,
        notes=notes,
    )


def find_cliffs(spec: ArchSpec, base: Occupancy) -> list[dict]:
    """Nearest points where blocks/SM changes, in each actionable direction."""
    cliffs: list[dict] = []
    b = base.blocks_per_sm

    def occ(**over) -> Occupancy:
        kw = dict(
            regs_per_thread=base.regs_per_thread,
            threads_per_block=base.threads_per_block,
            smem_static=base.smem_static,
            smem_dynamic=base.smem_dynamic,
            carveout_kb=base.carveout_kb,
        )
        kw.update(over)
        return compute(spec, **kw)

    if base.regs_per_thread > 0:
        for r in range(base.regs_per_thread - 1, 0, -1):
            o = occ(regs_per_thread=r)
            if o.blocks_per_sm > b:
                cliffs.append({
                    "kind": "gain", "resource": "registers",
                    "at": r, "delta": r - base.regs_per_thread,
                    "blocks_per_sm": o.blocks_per_sm, "occupancy_pct": o.occupancy_pct,
                })
                break
        for r in range(base.regs_per_thread + 1, spec.max_regs_per_thread + 1):
            o = occ(regs_per_thread=r)
            if o.blocks_per_sm < b:
                cliffs.append({
                    "kind": "loss", "resource": "registers",
                    "at": r, "delta": r - base.regs_per_thread,
                    "blocks_per_sm": o.blocks_per_sm, "occupancy_pct": o.occupancy_pct,
                })
                break

    usage = base.smem_static + base.smem_dynamic
    if usage > 0:
        step = spec.smem_alloc_granularity
        for s in range(usage - step, -1, -step):
            o = occ(smem_static=s, smem_dynamic=0)
            if o.blocks_per_sm > b:
                cliffs.append({
                    "kind": "gain", "resource": "shared_memory",
                    "at": s, "delta": s - usage,
                    "blocks_per_sm": o.blocks_per_sm, "occupancy_pct": o.occupancy_pct,
                })
                break
    return cliffs


def smem_headroom(spec: ArchSpec, base: Occupancy) -> Optional[int]:
    """Largest extra shared-memory bytes per block before blocks/SM drops.
    None when blocks/SM is 0 or shared memory is already the sole binding
    constraint with no room."""
    if base.blocks_per_sm <= 0:
        return None
    step = spec.smem_alloc_granularity
    usage = base.smem_static + base.smem_dynamic
    extra = 0
    while extra <= spec.smem_per_block_max - usage:
        o = compute(spec, base.regs_per_thread, base.threads_per_block,
                    smem_static=base.smem_static,
                    smem_dynamic=base.smem_dynamic + extra + step,
                    carveout_kb=base.carveout_kb)
        if o.blocks_per_sm < base.blocks_per_sm:
            return extra
        extra += step
    return extra


def sweep_block_sizes(
    spec: ArchSpec,
    regs_per_thread: int,
    smem_static: int = 0,
    smem_dynamic: int = 0,
    carveout_kb: Optional[int] = None,
    step: int = 32,
) -> list[Occupancy]:
    return [
        compute(spec, regs_per_thread, t, smem_static, smem_dynamic, carveout_kb)
        for t in range(step, spec.max_threads_per_block + 1, step)
    ]
