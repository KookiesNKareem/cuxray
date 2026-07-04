"""Parser for `ptxas -v` / `nvcc --resource-usage` compiler output.

Recorded format (CUDA 13.3, see tests/fixtures/recorded/ptxas_v.*):

    ptxas info    : Overriding maximum register limit 256 for '<name>' with  32 of maxrregcount option
    ptxas info    : 0 bytes gmem
    ptxas info    : Compiling entry function '<mangled>' for 'sm_120a'
    ptxas info    : Function properties for <mangled>
        208 bytes stack frame, 548 bytes spill stores, 556 bytes spill loads
    ptxas info    : Used 32 registers, used 0 barriers[, 8192 bytes smem]

Older toolkits emit variations (cmem segments, "Used X registers, Y bytes
smem" without barriers); the regexes are tolerant of field presence/order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

_COMPILING = re.compile(r"Compiling entry function '([^']+)' for '(\w+)'")
_PROPS_FOR = re.compile(r"Function properties for (\S+)")
_OVERRIDE = re.compile(r"Overriding maximum register limit \d+ for '([^']+)' with\s+(\d+)")
_STACK = re.compile(r"(\d+) bytes stack frame")
_SPILL_ST = re.compile(r"(\d+) bytes spill stores")
_SPILL_LD = re.compile(r"(\d+) bytes spill loads")
_REGS = re.compile(r"Used (\d+) registers")
_BARRIERS = re.compile(r"used (\d+) barriers")
_SMEM = re.compile(r"(\d+) bytes smem")
_GMEM = re.compile(r"(\d+) bytes gmem")


@dataclass
class PtxasKernel:
    name: str
    arch: Optional[str] = None
    regs: Optional[int] = None
    stack_frame: int = 0
    spill_stores: int = 0
    spill_loads: int = 0
    barriers: Optional[int] = None
    smem: int = 0
    maxrregcount: Optional[int] = None


def parse(text: str) -> dict[str, PtxasKernel]:
    kernels: dict[str, PtxasKernel] = {}
    overrides: dict[str, int] = {}
    current: Optional[PtxasKernel] = None

    def get(name: str) -> PtxasKernel:
        if name not in kernels:
            kernels[name] = PtxasKernel(name=name)
        return kernels[name]

    for line in text.splitlines():
        m = _OVERRIDE.search(line)
        if m:
            overrides[m.group(1)] = int(m.group(2))
            continue
        m = _COMPILING.search(line)
        if m:
            current = get(m.group(1))
            current.arch = m.group(2)
            continue
        m = _PROPS_FOR.search(line)
        if m:
            current = get(m.group(1))
            continue
        if current is None:
            continue
        m = _STACK.search(line)
        if m:
            current.stack_frame = int(m.group(1))
        m = _SPILL_ST.search(line)
        if m:
            current.spill_stores = int(m.group(1))
        m = _SPILL_LD.search(line)
        if m:
            current.spill_loads = int(m.group(1))
        m = _REGS.search(line)
        if m:
            current.regs = int(m.group(1))
        m = _BARRIERS.search(line)
        if m:
            current.barriers = int(m.group(1))
        m = _SMEM.search(line)
        if m:
            current.smem = int(m.group(1))

    for name, cap in overrides.items():
        if name in kernels:
            kernels[name].maxrregcount = cap
    return kernels
