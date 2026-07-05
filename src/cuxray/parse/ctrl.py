"""Control-field decoder for Volta-through-Hopper SASS (128-bit encoding).

Each instruction's high 64-bit word carries the compiler's static schedule
at bits [41, 62):

    ctrl   = (hi >> 41) & 0x1FFFFF
    stall  = ctrl[0:4]    cycles before the next instruction may issue
    yield  = ctrl[4]      scheduler hint
    wrtdb  = ctrl[5:8]    scoreboard set when this result lands (7 = none)
    readdb = ctrl[8:11]   scoreboard set when operands are consumed (7 = none)
    watdb  = ctrl[11:17]  wait mask over scoreboards 0-5
    reuse  = ctrl[17:21]  operand reuse-cache flags

Layout verified empirically per architecture (see tests): dependent-FFMA
chains show the documented 4-cycle FP32 stall, and load/consumer pairs show
matching wrtdb/watdb scoreboard indices. Supported: sm_80-sm_90a. Blackwell
(sm_100+) uses an unverified encoding and is refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

SUPPORTED_CC_MAJORS = (8, 9)

_PAIR = re.compile(
    r"/\*([0-9a-f]{4,})\*/\s+(.*?);\s*/\* (0x[0-9a-f]+) \*/\s*\n"
    r"\s*/\* (0x[0-9a-f]+) \*/"
)
_FUNC = re.compile(r"Function : (\S+)")


@dataclass(frozen=True)
class Ctrl:
    stall: int
    yield_: int
    wrtdb: int   # 7 = none
    readdb: int  # 7 = none
    watdb: int   # 6-bit wait mask
    reuse: int


def decode_word(hi: int) -> Ctrl:
    ctrl = (hi >> 41) & 0x1FFFFF
    return Ctrl(
        stall=ctrl & 0xF,
        yield_=(ctrl >> 4) & 1,
        wrtdb=(ctrl >> 5) & 7,
        readdb=(ctrl >> 8) & 7,
        watdb=(ctrl >> 11) & 0x3F,
        reuse=(ctrl >> 17) & 0xF,
    )


def parse_sass_controls(text: str) -> dict[str, dict[int, Ctrl]]:
    """{function: {addr: Ctrl}} from `cuobjdump -sass` output."""
    out: dict[str, dict[int, Ctrl]] = {}
    cur: Optional[dict[int, Ctrl]] = None
    pos = 0
    events = sorted(
        [(m.start(), "f", m) for m in _FUNC.finditer(text)]
        + [(m.start(), "i", m) for m in _PAIR.finditer(text)]
    )
    for _, kind, m in events:
        if kind == "f":
            cur = {}
            out[m.group(1)] = cur
        elif cur is not None:
            cur[int(m.group(1), 16)] = decode_word(int(m.group(4), 16))
    return out


def arch_supported(arch: Optional[str]) -> bool:
    if not arch:
        return False
    m = re.match(r"sm_(\d+)", arch)
    return bool(m) and int(m.group(1)) // 10 in SUPPORTED_CC_MAJORS
