"""Lane-value abstract domain for static access-pattern analysis (Layer B).

The key observation that makes this tractable and *exact where it matters*:
shared-memory bank conflicts and global coalescing depend only on the
DIFFERENCES between the 32 lanes' addresses within a warp. Warp-uniform terms
(base pointers, loop counters, block indices, kernel params) shift every lane
equally and cancel out of every difference — bank-conflict multiplicity is
invariant under uniform shifts (a uniform base rotates the bank pattern, it
never changes collision counts), and 128-byte-line counts change by at most
the boundary straddle we report explicitly.

So a register's abstract value is one of:

  PURE(vec)    exact per-lane values for the representative warp, with NO
               unknown uniform part (constants are PURE with equal entries).
               Closed under *all* integer ops elementwise — including the
               shifts/AND/XOR used by swizzled layouts, which is what lets
               cuxray prove CUTLASS-style swizzles conflict-free instead of
               giving up.
  MIXED(vec)   exact per-lane offsets PLUS an unknown warp-uniform part.
               Closed under +, -, *const, <<const (linear ops); anything
               nonlinear on the uniform part (>>, &, ^, *lane) degrades to
               VARYING. MIXED(zero-vec) is "uniform, unknown".
  VARYING      lane-dependence unknown (data-dependent loads, predicated
               writes, unsupported ops). Analyses must say "can't analyze",
               never guess.

The representative warp is warp 0 of a block shape supplied by the user
(--threads "256" or "32,8"). Warps other than 0 differ only by uniform tid
offsets when blockDim.x % 32 == 0 (the overwhelmingly common case); callers
should note when that doesn't hold.
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

    def __repr__(self) -> str:
        if self.kind == VARYING:
            return "VARYING"
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


def varying() -> Value:
    return Value(VARYING)


def const(c: int) -> Value:
    return Value(PURE, tuple([c & _MASK32] * WARP))


def uniform_unknown() -> Value:
    return Value(MIXED, tuple([0] * WARP))


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
        return varying()
    # uniform-unknown parts add to a single uniform-unknown part
    return Value(kind, tuple((x + y) & _MASK32 for x, y in zip(a.vec, b.vec)))


def sub(a: Value, b: Value) -> Value:
    kind = _lift(a, b)
    if kind is None:
        return varying()
    # (u1+v1)-(u2+v2): uniform parts collapse to one unknown uniform; exact vecs subtract
    return Value(kind, tuple((x - y) & _MASK32 for x, y in zip(a.vec, b.vec)))


def mul(a: Value, b: Value) -> Value:
    if a.kind == VARYING or b.kind == VARYING:
        return varying()
    if a.kind == PURE and b.kind == PURE:
        return Value(PURE, tuple((x * y) & _MASK32 for x, y in zip(a.vec, b.vec)))
    # MIXED * scalar-const is linear: (u + v)*c = u*c + v*c
    for m, s in ((a, b), (b, a)):
        if m.kind == MIXED and s.is_scalar:
            c = s.scalar
            return Value(MIXED, tuple((x * c) & _MASK32 for x in m.vec))
    # uniform * uniform stays uniform
    if a.is_uniform and b.is_uniform:
        return uniform_unknown()
    return varying()


def shl(a: Value, b: Value) -> Value:
    if a.kind == VARYING or b.kind == VARYING:
        return varying()
    if a.kind == PURE and b.kind == PURE:
        return Value(PURE, tuple((x << (y & 31)) & _MASK32 for x, y in zip(a.vec, b.vec)))
    if a.kind == MIXED and b.is_scalar:  # linear
        c = b.scalar & 31
        return Value(MIXED, tuple((x << c) & _MASK32 for x in a.vec))
    if a.is_uniform and b.is_uniform:
        return uniform_unknown()
    return varying()


def _nonlinear(op):
    def f(a: Value, b: Value) -> Value:
        if a.kind == PURE and b.kind == PURE:
            return Value(PURE, tuple(op(x, y) & _MASK32 for x, y in zip(a.vec, b.vec)))
        if a.kind != VARYING and b.kind != VARYING and a.is_uniform and b.is_uniform:
            return uniform_unknown()
        return varying()
    return f


shr = _nonlinear(lambda x, y: x >> (y & 31))
and_ = _nonlinear(lambda x, y: x & y)
or_ = _nonlinear(lambda x, y: x | y)
xor = _nonlinear(lambda x, y: x ^ y)


def lop3(a: Value, b: Value, c: Value, lut: int) -> Value:
    """3-input bitwise LUT (SASS LOP3.LUT) — how swizzle XORs often compile."""
    if all(v.kind == PURE for v in (a, b, c)):
        out = []
        for x, y, z in zip(a.vec, b.vec, c.vec):
            r = 0
            for bit in range(32):
                idx = (((x >> bit) & 1) << 2) | (((y >> bit) & 1) << 1) | ((z >> bit) & 1)
                r |= ((lut >> idx) & 1) << bit
            out.append(r)
        return Value(PURE, tuple(out))
    if all(v.kind != VARYING and v.is_uniform for v in (a, b, c)):
        return uniform_unknown()
    return varying()


def join(a: Value, b: Value) -> Value:
    """Control-flow merge."""
    if a.kind == VARYING or b.kind == VARYING:
        return varying()
    if a.vec == b.vec:
        return Value(MIXED, a.vec) if MIXED in (a.kind, b.kind) else a
    # Same lane pattern up to a uniform shift is still MIXED with that pattern
    d0 = (a.vec[0] - b.vec[0]) & _MASK32
    if all(((x - y) & _MASK32) == d0 for x, y in zip(a.vec, b.vec)):
        base = tuple((x - a.vec[0]) & _MASK32 for x in a.vec)
        return Value(MIXED, base)
    return varying()
