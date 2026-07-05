"""Lane-value abstract domain for static access-pattern analysis.

Bank conflicts and coalescing depend only on differences between the 32
lanes' addresses within a warp; warp-uniform terms cancel. Values:

  PURE(vec)    exact per-lane values, no uniform part. Closed under all
               integer ops elementwise (constants are PURE with equal lanes).
  MIXED(vec)   exact per-lane offsets plus an unknown warp-uniform part.
               Closed under linear ops (+, -, *const, <<const); nonlinear
               ops on a nonzero vec degrade to VARYING.
  VARYING      lane-dependence unknown; carries an attribution reason.

The lane vectors model warp 0 of a caller-supplied block shape. Other warps
differ only by uniform tid offsets when blockDim.x % 32 == 0; callers emit a
note otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

WARP = 32
_MASK32 = 0xFFFFFFFF

PURE = "pure"
MIXED = "mixed"
VARYING = "varying"


@dataclass(frozen=True)
class Value:
    kind: str
    vec: Optional[tuple[int, ...]] = None  # per-lane values, len 32
    reason: Optional[str] = None           # why VARYING (attribution only)
    ctaid: bool = False                     # depends on blockIdx (taint)

    def __repr__(self) -> str:
        if self.kind == VARYING:
            return f"VARYING({self.reason})" if self.reason else "VARYING"
        head = ",".join(str(v) for v in self.vec[:4])
        return f"{self.kind.upper()}[{head},...]"

    @property
    def is_scalar(self) -> bool:
        """PURE with identical lanes — usable as a shift amount/multiplier."""
        return self.kind == PURE and len(set(self.vec)) == 1

    @property
    def scalar(self) -> int:
        return self.vec[0]

    @property
    def is_uniform(self) -> bool:
        """Same value in every lane (exact, or exact-offset + unknown uniform)."""
        return self.kind != VARYING and len(set(self.vec)) == 1


def varying(reason: Optional[str] = None, ctaid: bool = True) -> Value:
    # unknown values are conservatively assumed block-dependent
    return Value(VARYING, reason=reason, ctaid=ctaid)


def _first_reason(*vals: Value) -> Optional[str]:
    for v in vals:
        if v.kind == VARYING and v.reason:
            return v.reason
    return None


def const(c: int) -> Value:
    return Value(PURE, tuple([c & _MASK32] * WARP))


def uniform_unknown(ctaid: bool = False) -> Value:
    return Value(MIXED, tuple([0] * WARP), ctaid=ctaid)


def pure(vec: Sequence[int]) -> Value:
    return Value(PURE, tuple(v & _MASK32 for v in vec))


def tid_vectors(block_dims: tuple[int, int, int]) -> dict[str, Value]:
    """Per-lane tid.x/y/z for warp 0 of a block of the given shape."""
    bx, by, _bz = block_dims
    xs, ys, zs = [], [], []
    for lane in range(WARP):
        xs.append(lane % bx)
        ys.append((lane // bx) % by if by else 0)
        zs.append(lane // (bx * by) if by else 0)
    return {"x": pure(xs), "y": pure(ys), "z": pure(zs)}


def _lift(a: Value, b: Value) -> Optional[str]:
    if a.kind == VARYING or b.kind == VARYING:
        return None
    return MIXED if MIXED in (a.kind, b.kind) else PURE


def add(a: Value, b: Value) -> Value:
    kind = _lift(a, b)
    if kind is None:
        return varying(_first_reason(a, b), a.ctaid or b.ctaid)
    # uniform-unknown parts add to a single uniform-unknown part
    return Value(kind, tuple((x + y) & _MASK32 for x, y in zip(a.vec, b.vec)),
                 ctaid=a.ctaid or b.ctaid)


def sub(a: Value, b: Value) -> Value:
    kind = _lift(a, b)
    if kind is None:
        return varying(_first_reason(a, b), a.ctaid or b.ctaid)
    # (u1+v1)-(u2+v2): uniform parts collapse to one unknown uniform; exact vecs subtract
    return Value(kind, tuple((x - y) & _MASK32 for x, y in zip(a.vec, b.vec)),
                 ctaid=a.ctaid or b.ctaid)


def mul(a: Value, b: Value) -> Value:
    t = a.ctaid or b.ctaid
    if a.kind == VARYING or b.kind == VARYING:
        return varying(_first_reason(a, b), t)
    if a.kind == PURE and b.kind == PURE:
        return Value(PURE, tuple((x * y) & _MASK32 for x, y in zip(a.vec, b.vec)), ctaid=t)
    # MIXED * scalar-const is linear: (u + v)*c = u*c + v*c
    for m, s in ((a, b), (b, a)):
        if m.kind == MIXED and s.is_scalar:
            c = s.scalar
            return Value(MIXED, tuple((x * c) & _MASK32 for x in m.vec), ctaid=t)
    # uniform * uniform stays uniform
    if a.is_uniform and b.is_uniform:
        return uniform_unknown(t)
    return varying("nonlinear lane arithmetic", t)


def shl(a: Value, b: Value) -> Value:
    t = a.ctaid or b.ctaid
    if a.kind == VARYING or b.kind == VARYING:
        return varying(_first_reason(a, b), t)
    if a.kind == PURE and b.kind == PURE:
        return Value(PURE, tuple((x << (y & 31)) & _MASK32 for x, y in zip(a.vec, b.vec)), ctaid=t)
    if a.kind == MIXED and b.is_scalar:  # linear
        c = b.scalar & 31
        return Value(MIXED, tuple((x << c) & _MASK32 for x in a.vec), ctaid=t)
    if a.is_uniform and b.is_uniform:
        return uniform_unknown(t)
    return varying("nonlinear lane arithmetic", t)


def _nonlinear(op):
    def f(a: Value, b: Value) -> Value:
        t = a.ctaid or b.ctaid
        if a.kind == PURE and b.kind == PURE:
            return Value(PURE, tuple(op(x, y) & _MASK32 for x, y in zip(a.vec, b.vec)), ctaid=t)
        if a.kind != VARYING and b.kind != VARYING and a.is_uniform and b.is_uniform:
            return uniform_unknown(t)
        return varying(_first_reason(a, b) or "nonlinear lane arithmetic", t)
    return f


shr = _nonlinear(lambda x, y: x >> (y & 31))
and_ = _nonlinear(lambda x, y: x & y)
or_ = _nonlinear(lambda x, y: x | y)
xor = _nonlinear(lambda x, y: x ^ y)


def lop3(a: Value, b: Value, c: Value, lut: int) -> Value:
    """3-input bitwise LUT (SASS LOP3.LUT) — how swizzle XORs often compile."""
    t = a.ctaid or b.ctaid or c.ctaid
    if all(v.kind == PURE for v in (a, b, c)):
        out = []
        for x, y, z in zip(a.vec, b.vec, c.vec):
            r = 0
            for bit in range(32):
                idx = (((x >> bit) & 1) << 2) | (((y >> bit) & 1) << 1) | ((z >> bit) & 1)
                r |= ((lut >> idx) & 1) << bit
            out.append(r)
        return Value(PURE, tuple(out), ctaid=t)
    if all(v.kind != VARYING and v.is_uniform for v in (a, b, c)):
        return uniform_unknown(t)
    return varying(_first_reason(a, b, c) or "nonlinear lane arithmetic", t)


def join(a: Value, b: Value) -> Value:
    """Control-flow merge. Stable under repetition: a VARYING left operand
    is returned as-is (reason preserved) so fixpoint iteration converges."""
    t = a.ctaid or b.ctaid
    if a.kind == VARYING:
        return a if a.ctaid == t else Value(VARYING, reason=a.reason, ctaid=t)
    if b.kind == VARYING:
        return varying(b.reason or "control-flow merge", t)
    if a.vec == b.vec:
        if MIXED in (a.kind, b.kind) or t != a.ctaid:
            return Value(MIXED if MIXED in (a.kind, b.kind) else a.kind, a.vec, ctaid=t)
        return a
    # Same lane pattern up to a uniform shift is still MIXED with that pattern
    d0 = (a.vec[0] - b.vec[0]) & _MASK32
    if all(((x - y) & _MASK32) == d0 for x, y in zip(a.vec, b.vec)):
        base = tuple((x - a.vec[0]) & _MASK32 for x in a.vec)
        return Value(MIXED, base, ctaid=t)
    return varying("control-flow merge", t)
