"""Per-compute-capability hardware capacities and allocation rules.

Sources (both retrieved 2026-07-04):
- Capacities: CUDA C++ Programming Guide, Table 27 "Technical Specifications
  per Compute Capability" (docs.nvidia.com/cuda/cuda-c-programming-guide/),
  plus the per-CC shared-memory carveout lists from section 20 prose.
- Allocation rules: cuda_occupancy.h from cuda_cudart 13.3.29 (NVIDIA redist):
  cudaOccRegAllocationGranularity (256, per warp), cudaOccSMemAllocationGranularity
  (128 B on CC >= 8), cudaOccSubPartitionsPerMultiprocessor (4),
  cudaOccMaxBlocksPerMultiprocessor (per-CC switch).

cuxray supports CC 7.5+ (Turing through consumer Blackwell). Older CCs would
need the 256 B smem granularity path and partitioned-global-caching logic that
were deliberately left out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

KB = 1024


@dataclass(frozen=True)
class ArchSpec:
    cc: tuple[int, int]
    name: str
    max_threads_per_sm: int
    max_blocks_per_sm: int
    smem_per_sm: int                  # bytes, max carveout
    smem_per_block_max: int           # bytes, opt-in maximum a block can address
    smem_carveouts_kb: tuple[int, ...]  # supported carveout settings, KB
    smem_reserved_per_block: int      # bytes reserved for system use (added to usage)
    warp_size: int = 32
    max_threads_per_block: int = 1024
    regfile_per_sm: int = 65536       # 32-bit registers
    max_regs_per_block: int = 65536
    max_regs_per_thread: int = 255
    reg_alloc_granularity: int = 256  # registers, allocated per warp
    smem_alloc_granularity: int = 128 # bytes (CC >= 8; 7.x uses 256)
    subpartitions: int = 4

    @property
    def max_warps_per_sm(self) -> int:
        return self.max_threads_per_sm // self.warp_size

    @property
    def sm(self) -> str:
        return f"sm_{self.cc[0]}{self.cc[1]}"


_CARVEOUTS_228 = (0, 8, 16, 32, 64, 100, 132, 164, 196, 228)
_CARVEOUTS_164 = (0, 8, 16, 32, 64, 100, 132, 164)
_CARVEOUTS_100 = (0, 8, 16, 32, 64, 100)

SPECS: dict[tuple[int, int], ArchSpec] = {}


def _add(spec: ArchSpec) -> None:
    SPECS[spec.cc] = spec


_add(ArchSpec((7, 5), "Turing", 1024, 16, 64 * KB, 64 * KB, (32, 64), 0,
              smem_alloc_granularity=256))
_add(ArchSpec((8, 0), "Ampere (A100)", 2048, 32, 164 * KB, 163 * KB, _CARVEOUTS_164, 1 * KB))
_add(ArchSpec((8, 6), "Ampere (GA10x)", 1536, 16, 100 * KB, 99 * KB, _CARVEOUTS_100, 1 * KB))
_add(ArchSpec((8, 7), "Ampere (Orin)", 1536, 16, 164 * KB, 163 * KB, _CARVEOUTS_164, 1 * KB))
_add(ArchSpec((8, 9), "Ada Lovelace", 1536, 24, 100 * KB, 99 * KB, _CARVEOUTS_100, 1 * KB))
_add(ArchSpec((9, 0), "Hopper", 2048, 32, 228 * KB, 227 * KB, _CARVEOUTS_228, 1 * KB))
_add(ArchSpec((10, 0), "Blackwell (B100/B200)", 2048, 32, 228 * KB, 227 * KB, _CARVEOUTS_228, 1 * KB))
_add(ArchSpec((10, 1), "Blackwell (GB10)", 2048, 24, 228 * KB, 227 * KB, _CARVEOUTS_228, 1 * KB))
_add(ArchSpec((10, 3), "Blackwell Ultra (B300)", 2048, 32, 228 * KB, 227 * KB, _CARVEOUTS_228, 1 * KB))
_add(ArchSpec((11, 0), "Blackwell (Thor)", 1536, 24, 228 * KB, 227 * KB, _CARVEOUTS_228, 1 * KB))
_add(ArchSpec((12, 0), "Blackwell (RTX 50 / RTX PRO)", 1536, 24, 100 * KB, 99 * KB, _CARVEOUTS_100, 1 * KB))
_add(ArchSpec((12, 1), "Blackwell (consumer, 12.1)", 1536, 24, 100 * KB, 99 * KB, _CARVEOUTS_100, 1 * KB))


# Approximate per-SM, per-clock MAC throughput of the SIMT datapath vs the
# tensor cores, by precision. These are ORDER-OF-MAGNITUDE figures whose
# RATIO (tensor / SIMT) is the load-bearing quantity — it sets where a
# tensor-core implementation overtakes a SIMT one as arithmetic-per-byte
# (e.g. batch) grows. Absolute values vary by SKU/clock; the ratio is
# stable within an architecture. "simt.int8" counts dp4a as 4 MACs/instr on
# the integer pipe; tensor int8 is the IMMA rate. None = no tensor cores /
# precision unsupported.
_MAC_RATES: dict[tuple[int, int], dict] = {
    (7, 5): {"simt": {"fp32": 64, "fp16": 128, "int8": 256},
             "tensor": {"fp16": 512, "int8": 1024}},
    (8, 0): {"simt": {"fp32": 64, "fp16": 128, "int8": 256},
             "tensor": {"fp16": 1024, "int8": 2048}},   # A100
    (8, 6): {"simt": {"fp32": 128, "fp16": 128, "int8": 256},
             "tensor": {"fp16": 512, "int8": 1024}},    # GA10x consumer
    (8, 9): {"simt": {"fp32": 128, "fp16": 128, "int8": 256},
             "tensor": {"fp16": 512, "int8": 1024}},    # Ada
    (9, 0): {"simt": {"fp32": 128, "fp16": 256, "int8": 512},
             "tensor": {"fp16": 2048, "int8": 4096}},   # Hopper
}


def mac_rates(spec: "ArchSpec") -> dict:
    """Approximate per-SM per-clock MAC rates ({'simt':{...},'tensor':{...}})
    for the arch, or {} when unmodeled. See _MAC_RATES for the caveats."""
    r = _MAC_RATES.get(spec.cc)
    if r is None and spec.cc[0] >= 10:            # Blackwell: reuse Hopper-ish
        r = _MAC_RATES[(9, 0)]
    return r or {}


_SM_RE = re.compile(r"^(?:sm_?|compute_?)?(\d+)(\d)(a|f)?$")


def lookup(arch: str | tuple[int, int]) -> ArchSpec:
    """Resolve 'sm_120', 'sm_120a', '120', '12.0', (12, 0) → ArchSpec."""
    if isinstance(arch, tuple):
        cc = arch
    else:
        s = arch.strip().lower()
        if "." in s:
            major, minor = s.split(".")
            cc = (int(major), int(minor))
        else:
            m = _SM_RE.match(s)
            if not m:
                raise KeyError(f"unrecognized architecture: {arch!r}")
            cc = (int(m.group(1)), int(m.group(2)))
    if cc not in SPECS:
        known = ", ".join(sorted(f"{a}.{b}" for a, b in SPECS))
        raise KeyError(f"unsupported compute capability {cc[0]}.{cc[1]} (known: {known})")
    return SPECS[cc]
