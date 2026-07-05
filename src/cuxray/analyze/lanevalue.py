"""Lane-value abstract domain for static access-pattern analysis.

Bank conflicts and coalescing depend only on differences between the 32
lanes' addresses within a warp; warp-uniform terms cancel. Values:

  PURE(vec)    exact per-lane values, no uniform part. Closed under all
               integer ops elementwise (constants are PURE with equal lanes).
  MIXED(vec)   exact per-lane offsets plus an unknown warp-uniform part U
               with a tracked alignment guarantee U % 2**ualign == 0.
               Closed under linear ops (+, -, *const, <<const). Right
               shifts by s <= ualign split exactly ((U+v)>>s = U>>s + v>>s),
               and bitwise ops against a PURE operand below 2**ualign only
               rewrite the low bits, so XOR-swizzled indices stay traceable.
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
    ualign: int = 0                         # MIXED: unknown part % 2**ualign == 0

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


def _al(v: Value) -> int:
    """Alignment of the unknown-uniform part; PURE has none (exact)."""
    return 32 if v.kind == PURE else v.ualign


def _tz(x: int) -> int:
    return 32 if x == 0 else (x & -x).bit_length() - 1


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
                 ctaid=a.ctaid or b.ctaid, ualign=min(_al(a), _al(b)) if kind == MIXED else 0)


def sub(a: Value, b: Value) -> Value:
    kind = _lift(a, b)
    if kind is None:
        return varying(_first_reason(a, b), a.ctaid or b.ctaid)
    # (u1+v1)-(u2+v2): uniform parts collapse to one unknown uniform; exact vecs subtract
    return Value(kind, tuple((x - y) & _MASK32 for x, y in zip(a.vec, b.vec)),
                 ctaid=a.ctaid or b.ctaid, ualign=min(_al(a), _al(b)) if kind == MIXED else 0)


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
            return Value(MIXED, tuple((x * c) & _MASK32 for x in m.vec), ctaid=t,
                         ualign=min(32, m.ualign + _tz(c)))
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
        return Value(MIXED, tuple((x << c) & _MASK32 for x in a.vec), ctaid=t,
                     ualign=min(32, a.ualign + c))
    if a.is_uniform and b.is_uniform:
        return uniform_unknown(t)
    return varying("nonlinear lane arithmetic", t)


def shr(a: Value, b: Value) -> Value:
    t = a.ctaid or b.ctaid
    if a.kind == PURE and b.kind == PURE:
        return Value(PURE, tuple((x >> (y & 31)) & _MASK32 for x, y in zip(a.vec, b.vec)), ctaid=t)
    if a.kind == MIXED and b.is_scalar and (b.scalar & 31) <= a.ualign:
        # U % 2**s == 0 makes the shift split exactly: (U+v)>>s = U>>s + v>>s
        s = b.scalar & 31
        return Value(MIXED, tuple(x >> s for x in a.vec), ctaid=t, ualign=a.ualign - s)
    if a.kind != VARYING and b.kind != VARYING and a.is_uniform and b.is_uniform:
        return uniform_unknown(t)
    return varying(_first_reason(a, b) or "nonlinear lane arithmetic", t)


def _bitop(op, and_mask: bool):
    """AND/OR/XOR. Against a PURE operand entirely below the MIXED side's
    2**ualign, only bits the unknown part cannot reach change: the result is
    U + newvec (or plain newvec for AND, which masks the unknown part off)."""
    def f(a: Value, b: Value) -> Value:
        t = a.ctaid or b.ctaid
        if a.kind == PURE and b.kind == PURE:
            return Value(PURE, tuple(op(x, y) & _MASK32 for x, y in zip(a.vec, b.vec)), ctaid=t)
        for m, p in ((a, b), (b, a)):
            if m.kind == MIXED and p.kind == PURE and m.ualign > 0 \
                    and max(p.vec) < (1 << m.ualign):
                lim = 1 << m.ualign
                low = tuple(op(x % lim, y) for x, y in zip(m.vec, p.vec))
                if and_mask:
                    return Value(PURE, low, ctaid=t)
                vec = tuple((x - x % lim + lo) & _MASK32 for x, lo in zip(m.vec, low))
                return Value(MIXED, vec, ctaid=t, ualign=m.ualign)
        if a.kind == MIXED and b.kind == MIXED and min(a.ualign, b.ualign) > 0:
            # low bits evaluate exactly; high bits stay one unknown uniform,
            # provided each side's known high part is lane-invariant
            al = min(a.ualign, b.ualign)
            lim = 1 << al
            if all(len({x - x % lim for x in v.vec}) == 1 for v in (a, b)):
                vec = tuple(op(x % lim, y % lim) for x, y in zip(a.vec, b.vec))
                return Value(MIXED, vec, ctaid=t, ualign=al)
        if a.kind != VARYING and b.kind != VARYING and a.is_uniform and b.is_uniform:
            return uniform_unknown(t)
        return varying(_first_reason(a, b) or "nonlinear lane arithmetic", t)
    return f


and_ = _bitop(lambda x, y: x & y, and_mask=True)
or_ = _bitop(lambda x, y: x | y, and_mask=False)
xor = _bitop(lambda x, y: x ^ y, and_mask=False)


def _lut_decompositions() -> dict[int, tuple[int, str, str]]:
    """LUT -> (outer operand index, outer op, inner op) for every 3-input LUT
    expressible as outer(v[i], inner(v[j], v[k])). Compilers emit LOP3 to fuse
    two binary ops, so this covers most real LUTs."""
    py = {"&": lambda x, y: x & y, "|": lambda x, y: x | y, "^": lambda x, y: x ^ y}
    table: dict[int, tuple[int, str, str]] = {}
    for i in range(3):
        j, k = [t for t in range(3) if t != i]
        for o_name, o_fn in py.items():
            for h_name, h_fn in py.items():
                lut = 0
                for idx in range(8):
                    bits = ((idx >> 2) & 1, (idx >> 1) & 1, idx & 1)  # (a, b, c)
                    lut |= (o_fn(bits[i], h_fn(bits[j], bits[k])) & 1) << idx
                table.setdefault(lut, (i, o_name, h_name))
    return table


_LUT_DECOMP = _lut_decompositions()


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
    vals = (a, b, c)
    # Fusing two binary ops is why LOP3 exists; un-fusing lets the binary
    # rules (mask purity, low-bit rewrites, alignment splits) apply exactly.
    dec = _LUT_DECOMP.get(lut)
    if dec is not None and any(v.kind == MIXED for v in vals):
        i, o_name, h_name = dec
        j, k = [t for t in range(3) if t != i]
        ops = {"&": and_, "|": or_, "^": xor}
        res = ops[o_name](vals[i], ops[h_name](vals[j], vals[k]))
        if res.kind != VARYING:
            return res
    mixed = [v for v in vals if v.kind == MIXED]
    if mixed and all(v.kind in (PURE, MIXED) for v in vals):
        al = min(v.ualign for v in mixed)
        # Every operand's bits above 2**al must be lane-invariant (PURE:
        # scalar or below the alignment; MIXED: known high part uniform, the
        # unknown part is uniform by definition); then the LUT output above
        # 2**al is a single unknown uniform.
        lim0 = 1 << al
        ok = all((v.is_scalar or max(v.vec) < lim0) if v.kind == PURE
                 else len({x - x % lim0 for x in v.vec}) == 1
                 for v in vals)
        if al > 0 and ok:
            lim = 1 << al
            low = []
            for x, y, z in zip(*(v.vec for v in vals)):
                r = 0
                for bit in range(al):
                    idx = ((((x % lim) >> bit) & 1) << 2) | ((((y % lim) >> bit) & 1) << 1) | (((z % lim) >> bit) & 1)
                    r |= ((lut >> idx) & 1) << bit
                low.append(r)
            return Value(MIXED, tuple(low), ctaid=t, ualign=al)
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
            return Value(MIXED if MIXED in (a.kind, b.kind) else a.kind, a.vec, ctaid=t,
                         ualign=min(_al(a), _al(b)) if MIXED in (a.kind, b.kind) else 0)
        return a
    # Same lane pattern up to a uniform shift is still MIXED with that pattern
    d0 = (a.vec[0] - b.vec[0]) & _MASK32
    if all(((x - y) & _MASK32) == d0 for x, y in zip(a.vec, b.vec)):
        base = tuple((x - a.vec[0]) & _MASK32 for x in a.vec)
        # unknown part is one of {a.vec[0]+Ua, b.vec[0]+Ub}
        al = min(min(_tz(a.vec[0]), _al(a)), min(_tz(b.vec[0]), _al(b)))
        return Value(MIXED, base, ctaid=t, ualign=al)
    return varying("control-flow merge", t)
