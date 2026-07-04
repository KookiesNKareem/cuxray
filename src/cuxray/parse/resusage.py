"""Parser for `cuobjdump --dump-resource-usage` output.

Recorded format (CUDA 13.3, tests/fixtures/recorded/resusage.*):

    Resource usage:
     Common:
      GLOBAL:0
     Function _Z6spillyPKfPfii:
      REG:32 STACK:208 SHARED:0 LOCAL:0 CONSTANT[0]:920 TEXTURE:0 SURFACE:0 SAMPLER:0

Works on any cubin — including binaries we did not compile. STACK is the
per-thread local-memory frame (spills + local arrays live there).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_FUNCTION = re.compile(r"^\s*Function\s+(\S+?):\s*$")
_PAIR = re.compile(r"([A-Z]+(?:\[\d+\])?):(\d+)")


@dataclass
class ResourceUsage:
    name: str
    reg: int = 0
    stack: int = 0
    shared: int = 0
    local: int = 0
    constant: int = 0  # summed across banks
    raw: dict[str, int] = field(default_factory=dict)


def parse(text: str) -> dict[str, ResourceUsage]:
    kernels: dict[str, ResourceUsage] = {}
    current: ResourceUsage | None = None
    for line in text.splitlines():
        m = _FUNCTION.match(line)
        if m:
            current = ResourceUsage(name=m.group(1))
            kernels[current.name] = current
            continue
        if current is None:
            continue
        pairs = _PAIR.findall(line)
        if not pairs:
            continue
        for key, val in pairs:
            v = int(val)
            current.raw[key] = current.raw.get(key, 0) + v
            base = key.split("[")[0]
            if base == "REG":
                current.reg = v
            elif base == "STACK":
                current.stack = v
            elif base == "SHARED":
                current.shared = v
            elif base == "LOCAL":
                current.local = v
            elif base == "CONSTANT":
                current.constant += v
    return kernels
