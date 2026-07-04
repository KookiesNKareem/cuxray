"""Parsers for nvdisasm text output.

Two invocations, joined by instruction address (nvdisasm refuses to print
line-info and life ranges together — "Options --print-line-info and its
variants are ignored when printing register life range"):

  `nvdisasm -c -gi cubin`   → instructions with //## File "...", line N markers
  `nvdisasm -c -plr -gi`    → instructions annotated with live-register columns:
      /*0050*/  IMAD R7, R7, UR4, R0 ;  // | 3 v :   x | 1 ^ | 2  v :  |
      groups are GPR | PRED | UGPR; the first integer in each group is the
      occupied-register count for that class at this instruction.

Format fixtures: tests/fixtures/recorded/nvdisasm_{gi,plr}.* (CUDA 13.3,
sm_90 + sm_120a).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

_SECTION = re.compile(r"^\s*\.section\s+\.text\.(\S+?),")
_FILELINE = re.compile(r'^\s*//## File "([^"]+)", line (\d+)')
_INSTR = re.compile(
    r"^\s*/\*([0-9a-fA-F]+)\*/\s+(?:(@!?\S+)\s+)?([A-Za-z][\w.]*)\s*(.*?)\s*;"
)
_LABEL = re.compile(r"^([.\w$]+):\s*(?://.*)?$")
_TARGET = re.compile(r"^\s*\.target\s+(\S+)")
_PLR_COUNT = re.compile(r"(\d+)")  # search, not match: tolerant of leading glyph drift


@dataclass
class Instruction:
    addr: int
    opcode: str
    operands: str
    predicate: Optional[str] = None
    file: Optional[str] = None
    line: Optional[int] = None
    block: Optional[str] = None   # label of the containing basic block
    live_gpr: Optional[int] = None
    live_pred: Optional[int] = None
    live_ugpr: Optional[int] = None


@dataclass
class Function:
    name: str
    instructions: list[Instruction] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)  # in order of appearance


@dataclass
class Disassembly:
    target: Optional[str] = None
    functions: dict[str, Function] = field(default_factory=dict)


def parse_gi(text: str) -> Disassembly:
    """Parse `nvdisasm -c -gi` output: instructions + source lines + blocks."""
    dis = Disassembly()
    func: Optional[Function] = None
    cur_file: Optional[str] = None
    cur_line: Optional[int] = None
    cur_block: Optional[str] = None

    for raw in text.splitlines():
        m = _TARGET.match(raw)
        if m:
            dis.target = m.group(1)
            continue
        m = _SECTION.match(raw)
        if m:
            func = Function(name=m.group(1))
            dis.functions[func.name] = func
            cur_file = cur_line = None
            cur_block = func.name  # entry block is named after the function in -cfg
            continue
        if func is None:
            continue
        m = _FILELINE.match(raw)
        if m:
            cur_file, cur_line = m.group(1), int(m.group(2))
            continue
        m = _LABEL.match(raw.strip())
        if m:
            label = m.group(1)
            # skip the function's own symbol and .text.* section labels;
            # real block labels are .L_x_N style (plus the entry = func name)
            if label != func.name and not label.startswith(".text."):
                cur_block = label
                func.labels.append(label)
            continue
        m = _INSTR.match(raw)
        if m:
            func.instructions.append(Instruction(
                addr=int(m.group(1), 16),
                predicate=m.group(2),
                opcode=m.group(3),
                operands=m.group(4).strip(),
                file=cur_file,
                line=cur_line,
                block=cur_block,
            ))
    return dis


def parse_plr(text: str) -> dict[str, dict[int, tuple[int, int, int]]]:
    """Parse `nvdisasm -c -plr` output.

    Returns {function_name: {addr: (live_gpr, live_pred, live_ugpr)}}.
    """
    out: dict[str, dict[int, tuple[int, int, int]]] = {}
    cur: Optional[dict[int, tuple[int, int, int]]] = None

    for raw in text.splitlines():
        m = _SECTION.match(raw)
        if m:
            cur = {}
            out[m.group(1)] = cur
            continue
        if cur is None:
            continue
        m = _INSTR.match(raw)
        if not m:
            continue
        addr = int(m.group(1), 16)
        # annotation: everything after '//', split on '|' → group columns
        counts = [0, 0, 0]
        if "//" in raw:
            ann = raw.split("//", 1)[1]
            groups = [g for g in ann.split("|") if g.strip(" +-.─")]
            for i, g in enumerate(groups[:3]):
                cm = _PLR_COUNT.search(g)
                counts[i] = int(cm.group(1)) if cm else 0
        cur[addr] = (counts[0], counts[1], counts[2])
    return out


def merge_liveness(dis: Disassembly, plr: dict[str, dict[int, tuple[int, int, int]]]) -> None:
    """Attach live-register counts to a -gi disassembly, joining on address."""
    for name, func in dis.functions.items():
        table = plr.get(name)
        if not table:
            continue
        for instr in func.instructions:
            counts = table.get(instr.addr)
            if counts:
                instr.live_gpr, instr.live_pred, instr.live_ugpr = counts


_SPILL_OPS = ("STL", "LDL")


def is_spill(instr: Instruction) -> bool:
    op = instr.opcode.split(".")[0]
    return op in _SPILL_OPS


_SMEM_PREFIXES = ("LDS", "STS", "LDGSTS", "UTMA")  # LDS covers LDSM, STS covers STSM


def uses_shared_memory(func: Function) -> bool:
    """True if the kernel touches shared memory at all (loads/stores/matrix
    loads/async copies/TMA into smem). Used to flag kernels that allocate
    shared memory *dynamically*: they access smem while the binary records no
    static allocation — the size is a host-side launch parameter no static
    tool can recover."""
    return any(i.opcode.startswith(_SMEM_PREFIXES) for i in func.instructions)


def uses_register_reallocation(func: Function) -> bool:
    """True if the kernel redistributes registers between warpgroups at
    runtime (PTX setmaxnreg → SASS USETMAXREG on sm_90+). For such kernels
    the recorded per-thread REG count is the post-reallocation *maximum*,
    not the launch allocation — naive occupancy math from it is pessimistic
    (often "0 blocks" for kernels that ship in production, e.g. FA3)."""
    return any("SETMAXREG" in i.opcode for i in func.instructions)
