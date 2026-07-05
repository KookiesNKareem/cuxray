"""Forward dataflow over SASS assigning lane-values to registers.

Iterates basic blocks to a fixpoint, interpreting the address-forming subset
of SASS with the lanevalue domain. Unrecognized instructions make their
destinations VARYING.

Seeds:
  S2R  Rd, SR_TID.X/Y/Z      → exact per-lane vectors (needs block shape)
  S2R/S2UR SR_CTAID.*        → uniform-unknown
  LDC / LDCU / MOV imm / CS2R → uniform / constant
  LDG / LDS / LD (any load)   → VARYING (data-dependent)

Interpreted ops: MOV, IMAD(.WIDE/.HI…partially), IADD/IADD3, LEA, SHF, SHL,
SHR, LOP3.LUT, AND/OR/XOR via LOP3 or PLOP, ISETP (ignored), PRMT/F ops →
VARYING. Predicated writes join old and new values (still exact when both
agree; VARYING otherwise). UR registers tracked in the same map under their
"UR<n>" names; RZ/URZ are constant zero.
"""

from __future__ import annotations

import re
from typing import Optional

from ..parse.sass import Function, Instruction
from . import lanevalue as lv

_REG = re.compile(r"^-?~?\|?(R\d+|RZ|UR\d+|URZ|PT|!?P\d+)\|?(\.\w+)*$")
_IMM = re.compile(r"^-?0x[0-9a-fA-F]+$|^-?\d+$")
_SCALE = re.compile(r"\.X(\d+)$")
_BRACKET = re.compile(r"\[([^\]]*)\]")


def memory_operand(instr: Instruction) -> Optional[str]:
    """The inside of the address bracket, skipping TMA/LDG descriptor brackets."""
    groups = _BRACKET.findall(instr.operands)
    if not groups:
        return None
    for g in reversed(groups):
        if re.search(r"R\d+|RZ|0x", g):
            return g
    return groups[-1]


def addr_value(mem: str, st: "State") -> lv.Value:
    """Evaluate the inside of a [...] memory operand to a lane-value."""
    total = lv.const(0)
    for term in re.split(r"(?<!e)\+", mem.replace(" ", "")):
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
            return lv.varying(f"unsupported address term {base[:20]!r}")
        if neg:
            v = lv.sub(lv.const(0), v)
        total = lv.add(total, v)
    return total


def _split_operands(text: str) -> list[str]:
    """Split an operand string on top-level commas (brackets protected)."""
    out, depth, cur = [], 0, []
    for ch in text:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur).strip())
    return [o for o in out if o]


class State:
    def __init__(self, regs: Optional[dict] = None,
                 consts: Optional[dict[int, int]] = None):
        self.regs: dict[str, lv.Value] = regs or {}
        self.consts: dict[int, int] = consts or {}  # resolved c[0x0][slot]

    def get(self, name: str) -> lv.Value:
        if name in ("RZ", "URZ"):
            return lv.const(0)
        return self.regs.get(name, lv.varying())

    def set(self, name: str, val: lv.Value) -> None:
        if name not in ("RZ", "URZ", "PT"):
            self.regs[name] = val

    def copy(self) -> "State":
        return State(dict(self.regs), self.consts)

    def join_with(self, other: "State") -> bool:
        """Merge other into self; True if self changed."""
        changed = False
        for k in set(self.regs) | set(other.regs):
            a = self.regs.get(k, lv.varying())
            b = other.regs.get(k, lv.varying())
            j = lv.join(a, b)
            if j != a:
                self.regs[k] = j
                changed = True
        return changed


def _operand_value(op: str, st: State) -> lv.Value:
    op = op.strip()
    neg = op.startswith("-") and not _IMM.match(op)
    inv = op.startswith("~")
    core = op.lstrip("-~").split(".")[0].strip("|")
    if _IMM.match(op):
        return lv.const(int(op, 0))
    if re.match(r"^(R\d+|RZ|UR\d+|URZ)$", core):
        v = st.get(core)
        if neg:
            v = lv.sub(lv.const(0), v)
        if inv:
            v = lv.xor(v, lv.const(0xFFFFFFFF))
        return v
    if core.startswith("c[") or op.startswith("c["):
        # bank 0 low slots are the launch config; blockDim is caller-known
        m = re.match(r"^c\[0x0\]\[(0x[0-9a-fA-F]+)\]$",
                     (core if core.startswith("c[") else op).replace(" ", ""))
        if m and int(m.group(1), 16) in st.consts:
            v = lv.const(st.consts[int(m.group(1), 16)])
            if neg:
                v = lv.sub(lv.const(0), v)
            if inv:
                v = lv.xor(v, lv.const(0xFFFFFFFF))
            return v
        return lv.uniform_unknown()  # kernel param / other uniform
    return lv.varying()


def _dest(operands: list[str]) -> Optional[str]:
    if not operands:
        return None
    m = re.match(r"^(R\d+|UR\d+)$", operands[0].split(".")[0])
    return m.group(1) if m else None


_TID = re.compile(r"SR_TID\.?(X|Y|Z)")
_CTAID = re.compile(r"SR_CTAID\.?(X|Y|Z)")


_KNOWN_BASES = {
    "S2R", "S2UR", "LDC", "LDCU", "MOV", "CS2R", "IMAD", "IADD3", "IADD",
    "VIADD", "LEA", "SHF", "SHL", "SHR", "LOP3", "PRMT", "R2UR", "SEL",
}


def step(instr: Instruction, st: State, tids: dict[str, lv.Value]) -> None:
    op = instr.opcode
    base = op.split(".")[0]
    # Uniform-datapath ops (ULDC, UIADD3, ULEA, UMOV, ULOP3, USHF, ...) mirror
    # their regular counterparts on UR registers.
    if base.startswith("U") and base[1:] in _KNOWN_BASES:
        base = base[1:]
    ops = _split_operands(instr.operands)
    d = _dest(ops)
    # Integer ops may carry predicate outputs (LEA Rd, P0, ...) or use
    # extended-carry variants (.X with a per-lane carry-in). Drop predicate
    # operands from the source list; treat carry-in variants conservatively.
    _pred = re.compile(r"^!?U?P(T|\d+)$")
    srcs = [o for o in ops[1:] if not _pred.match(o.split(".")[0])]
    has_carry_in = ".X" in op.split(".")  # e.g. IMAD.X, LEA.HI.X, IADD3.X

    def assign(val: lv.Value) -> None:
        if d is None:
            return
        # Predicated write: merge with the previous value — except on the
        # first write, where the "previous value" is uninitialized garbage
        # the compiler never lets live paths read (predicated pairs always
        # cover the complement); joining against it would poison both halves.
        if instr.predicate and d in st.regs:
            val = lv.join(st.get(d), val)
            if val.kind == lv.VARYING and not val.reason:
                val = lv.varying("predicated write")
        st.set(d, val)

    if base in ("S2R", "S2UR"):
        src = ops[1] if len(ops) > 1 else ""
        m = _TID.search(src)
        if m:
            assign(tids[m.group(1).lower()])
        elif _CTAID.search(src):
            assign(lv.uniform_unknown(ctaid=True))
        elif "SR_" in src:
            assign(lv.uniform_unknown())
        else:
            assign(lv.varying())
        return

    if base in ("LDC", "LDCU"):
        assign(lv.uniform_unknown())
        if ".64" in op and d:
            st.set(_next_reg(d), lv.uniform_unknown())
        return

    if base in ("MOV", "CS2R"):
        assign(_operand_value(ops[1], st) if len(ops) > 1 else lv.varying())
        return

    if base == "R2UR":
        # Regular→uniform register copy; hardware requires the source to be
        # warp-uniform, so even when our analysis lost track of the source
        # value, "unknown uniform" is sound.
        if len(ops) > 1:
            src = _operand_value(ops[1], st)
            assign(src if src.is_uniform else lv.uniform_unknown())
        return

    if base == "IMAD":
        # IMAD Rd, Ra, Rb, Rc  →  Rd = Ra*Rb + Rc ; .WIDE writes a 64-bit pair
        if len(srcs) >= 3 and ".HI" not in op and not has_carry_in:
            a, b, c = (_operand_value(o, st) for o in srcs[:3])
            assign(lv.add(lv.mul(a, b), c))
            if ".WIDE" in op and d:
                # high word: lane-varying carries are negligible for real
                # offsets; model as uniform-unknown
                st.set(_next_reg(d), lv.uniform_unknown())
            return
        assign(lv.varying())
        return

    if base == "SEL":
        # SEL Rd, Ra, Rb, P — per-lane select. The result is one of the two
        # sources lane-wise, so their join is a sound abstraction (exact when
        # both agree; uniform-shift-aware otherwise).
        if len(srcs) >= 2:
            a = _operand_value(srcs[0], st)
            b = _operand_value(srcs[1], st)
            j = lv.join(a, b)
            if j.kind == lv.VARYING and not j.reason:
                j = lv.varying("predicated select")
            assign(j)
            return
        assign(lv.varying())
        return

    if base in ("IADD3", "IADD", "VIADD"):
        if has_carry_in:
            assign(lv.varying())
            return
        vals = [_operand_value(o, st) for o in srcs]
        res = vals[0] if vals else lv.varying()
        for v in vals[1:]:
            res = lv.add(res, v)
        assign(res)
        return

    if base == "LEA":
        # LEA Rd[, Pc], Ra, Rb[, sh]  →  Rd = (Ra << sh) + Rb
        if len(srcs) >= 2 and not has_carry_in and ".HI" not in op:
            a = _operand_value(srcs[0], st)
            b = _operand_value(srcs[1], st)
            sh = (lv.const(int(srcs[2], 0))
                  if len(srcs) > 2 and _IMM.match(srcs[2]) else lv.const(0))
            assign(lv.add(lv.shl(a, sh), b))
            return
        if ".HI" in op and not has_carry_in and len(srcs) >= 3 \
                and _IMM.match(srcs[-1]) and int(srcs[-1], 0) & 31:
            # LEA.HI Rd, Ra, Rb, Rc, N → Rb + hi32({Rc:Ra} << N)
            #   = Rb + ((Ra >> (32-N)) | (Rc << N)); the two parts occupy
            # disjoint bits, so | == +. Covers the signed-div idiom
            # x + (x >> 31) that fastdiv/index math compiles to.
            n = int(srcs[-1], 0) & 31
            a = _operand_value(srcs[0], st)
            b = _operand_value(srcs[1], st)
            c = _operand_value(srcs[2], st)
            hi = lv.add(lv.shr(a, lv.const(32 - n)), lv.shl(c, lv.const(n)))
            assign(lv.add(b, hi))
            return
        if ".HI" in op:
            vals = [_operand_value(o, st) for o in srcs if not _IMM.match(o)]
            assign(lv.uniform_unknown()
                   if vals and all(v.is_uniform for v in vals) else lv.varying())
            return
        assign(lv.varying())
        return

    if base in ("SHF", "USHF"):
        # Funnel shift of {ops[3]:ops[1]} by ops[2]. With an RZ filler it is a
        # plain shift: .L/.R take the source from the low word (ops[1]);
        # .L.HI/.R.HI take it from the high word (ops[3]).
        if len(ops) >= 4:
            sh = _operand_value(ops[2], st)
            lo, hi = ops[1].split(".")[0], ops[3].split(".")[0]
            if ".HI" in op and lo in ("RZ", "URZ"):
                a = _operand_value(ops[3], st)
                assign(lv.shl(a, sh) if ".L" in op else lv.shr(a, sh))
                return
            if ".HI" not in op and hi in ("RZ", "URZ"):
                a = _operand_value(ops[1], st)
                assign(lv.shl(a, sh) if ".L" in op else lv.shr(a, sh))
                return
        assign(lv.varying())
        return

    if base in ("SHL",):
        assign(lv.shl(_operand_value(ops[1], st), _operand_value(ops[2], st))
               if len(ops) >= 3 else lv.varying())
        return
    if base in ("SHR",):
        assign(lv.shr(_operand_value(ops[1], st), _operand_value(ops[2], st))
               if len(ops) >= 3 else lv.varying())
        return

    if base in ("LOP3", "ULOP3", "PLOP3"):
        # LOP3.LUT Rd, Ra, Rb, Rc, lut, ...
        if len(ops) >= 5 and _IMM.match(ops[4]):
            a, b, c = (_operand_value(o, st) for o in ops[1:4])
            assign(lv.lop3(a, b, c, int(ops[4], 0)))
            return
        assign(lv.varying())
        return

    if base == "PRMT":
        assign(lv.varying("unmodeled PRMT"))
        return

    if base.startswith(("LD", "ATOM", "RED", "TLD", "SULD")):
        if d:
            if base.startswith("LDSM"):
                # ldmatrix distributes fragments across lanes — result is
                # lane-varying even when the address is uniform
                val = lv.varying("data-dependent (matrix load result)")
            elif base.startswith("LD"):
                # A load at a warp-uniform address returns identical data to
                # every lane — provably uniform. (Atomics excluded: serialized
                # RMW returns different pre-values per lane even at one address.)
                mem = memory_operand(instr)
                addr = addr_value(mem, st) if mem else lv.varying()
                if addr.kind != lv.VARYING and addr.is_uniform:
                    val = lv.uniform_unknown(ctaid=addr.ctaid)
                else:
                    val = lv.varying("data-dependent (load result)", addr.ctaid)
            else:
                val = lv.varying("data-dependent (atomic result)")
            assign(val)
            if ".64" in op:
                st.set(_next_reg(d), val)
            elif ".128" in op:
                for i in range(1, 4):
                    st.set(_next_reg(d, i), val)
        return

    if base.startswith(("ST", "BRA", "EXIT", "BAR", "NOP", "ISETP", "USETP",
                        "DEPBAR", "MEMBAR", "BSYNC", "BSSY", "BREAK", "YIELD",
                        "WARPSYNC", "FENCE", "ERRBAR", "RET", "CALL", "JMP")):
        return  # no GPR destination we track

    # Anything else that names a destination register: unknown semantics
    if d is not None:
        assign(lv.varying(f"unmodeled {base}"))


def _next_reg(reg: str, offset: int = 1) -> str:
    m = re.match(r"^(U?R)(\d+)$", reg)
    return f"{m.group(1)}{int(m.group(2)) + offset}" if m else reg


def analyze(func: Function, block_dims: tuple[int, int, int],
            max_iters: int = 8) -> dict[int, State]:
    """Fixpoint dataflow. Returns the state *before* each instruction,
    keyed by address."""
    pre, _info = analyze_ex(func, block_dims, max_iters)
    return pre


def analyze_ex(func: Function, block_dims: tuple[int, int, int],
               max_iters: int = 8) -> tuple[dict[int, State], dict]:
    """Like analyze(), plus {"converged": bool, "unreached_blocks": int} so
    callers can surface incomplete analysis instead of silently degrading."""
    tids = lv.tid_vectors(block_dims)

    # Partition instructions into blocks by label (matches -cfg node names)
    blocks: list[list[Instruction]] = []
    block_of_label: dict[str, int] = {}
    cur: list[Instruction] = []
    cur_label = func.name
    for i in func.instructions:
        if i.block != cur_label:
            blocks.append(cur)
            block_of_label[cur_label] = len(blocks) - 1
            cur, cur_label = [], i.block
        cur.append(i)
    blocks.append(cur)
    block_of_label[cur_label] = len(blocks) - 1

    entry: list[Optional[State]] = [None] * len(blocks)
    bx, by, bz = block_dims
    entry[0] = State(consts={0x0: bx, 0x4: by, 0x8: bz})  # ntid.{x,y,z}
    pre: dict[int, State] = {}

    converged = False
    for _ in range(max_iters):
        changed = False
        for bi, insns in enumerate(blocks):
            if entry[bi] is None:
                continue
            st = entry[bi].copy()
            for i in insns:
                pre[i.addr] = st.copy()
                step(i, st, tids)
            # propagate: fallthrough + branch targets named in operands
            succs = []
            if bi + 1 < len(blocks):
                last = insns[-1] if insns else None
                if not (last and last.opcode in ("EXIT", "BRA", "RET", "JMP")
                        and not last.predicate):
                    succs.append(bi + 1)
            for i in insns:
                if i.opcode.startswith(("BRA", "JMP", "CALL", "BSSY")):
                    m = re.search(r"`?\(?(\.L_\w+)\)?", i.operands)
                    if m and m.group(1) in block_of_label:
                        succs.append(block_of_label[m.group(1)])
            for s in succs:
                if entry[s] is None:
                    entry[s] = st.copy()
                    changed = True
                elif entry[s].join_with(st):
                    changed = True
        if not changed:
            converged = True
            break
    unreached = sum(1 for e in entry if e is None)
    return pre, {"converged": converged, "unreached_blocks": unreached}
