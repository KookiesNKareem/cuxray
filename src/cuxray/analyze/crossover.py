"""Datapath analysis: is a loop's arithmetic on the SIMT lanes or the tensor
cores, and — if SIMT — where would tensor cores overtake it?

The most expensive mistake a static optimizer can make is to declare a SIMT
kernel "clean" without noticing that its whole *architecture* is capped:
memory-bound at batch 1, it looks perfect, but as arithmetic-per-byte grows
(more columns / larger batch) it becomes compute-bound on the slow datapath
while a tensor-core implementation keeps scaling. This module surfaces that
ceiling from the static op mix, so it can be flagged before any hardware run.
"""

from __future__ import annotations

from typing import Optional

from ..archspec import ArchSpec, mac_rates

# Opcode families. Tensor-core MMAs vs the SIMT fused-multiply-adds that do
# the same MACs one lane at a time.
_TENSOR = ("HMMA", "IMMA", "OMMA", "BMMA", "MMA", "WGMMA", "HGMMA", "IGMMA")
_SIMT_FP = ("FFMA", "FMUL", "FADD", "HFMA", "HMUL", "HADD", "DFMA", "FMNMX")
_SIMT_INT = ("IDP", "DP4A", "DP2A", "IMAD", "IMMA.")  # IDP/dp4a = int8 dot


def datapath(opcodes: list[str]) -> str:
    """'tensor' | 'simt-fp' | 'simt-int' | 'mixed' | 'none' for a loop body."""
    bases = [o.split(".")[0] for o in opcodes]
    has_tensor = any(b in _TENSOR for b in bases)
    has_fp = any(b in _SIMT_FP for b in bases)
    # dp4a shows up as IDP; treat IMAD as int MAC only if it dominates
    has_int = any(b in ("IDP", "DP4A", "DP2A") for b in bases)
    if has_tensor:
        return "tensor" if not (has_fp or has_int) else "mixed"
    if has_int and has_fp:
        return "mixed"
    if has_int:
        return "simt-int"
    if has_fp:
        return "simt-fp"
    return "none"


def _precision(dp: str, opcodes: list[str]) -> Optional[str]:
    """Precision of the MAC-carrying op — what a tensor-core rewrite would
    use. dp4a → int8 (even in a 'mixed' loop that accumulates in float, which
    every real dp4a kernel does); else the FMA width."""
    bases = [o.split(".")[0] for o in opcodes]
    if any(b in ("IDP", "DP4A", "DP2A") for b in bases):
        return "int8"
    if any(b in ("HFMA", "HMUL", "HADD") for b in bases):
        return "fp16"
    if any(b in ("FFMA", "FMUL", "FADD", "DFMA") for b in bases):
        return "fp32"
    return None


def analyze_loop(opcodes: list[str], spec: Optional[ArchSpec],
                 arithmetic_intensity: Optional[float] = None) -> Optional[dict]:
    """If the loop does its MACs on the SIMT datapath, estimate how much
    tensor-core headroom exists. Returns None when not applicable (already
    tensor-core, no MACs, or arch unmodeled)."""
    dp = datapath(opcodes)
    if dp in ("tensor", "none") or spec is None:
        return None
    rates = mac_rates(spec)
    if not rates:
        return None
    prec = _precision(dp, opcodes)
    if prec is None:
        return None
    simt = rates.get("simt", {}).get(prec)
    tensor = rates.get("tensor", {}).get(prec)
    if not simt or not tensor:
        return None
    ratio = tensor / simt
    return {
        "datapath": dp,
        "precision": prec,
        "simt_mac_per_sm_clk": simt,
        "tensor_mac_per_sm_clk": tensor,
        "tensor_speedup_ceiling": round(ratio, 1),
        "arithmetic_intensity": arithmetic_intensity,
        # a decode/GEMV kernel's AI grows ~linearly with batch; once it
        # exceeds the SIMT compute ridge it is compute-bound on the slow path
        "note": (
            f"MACs run on the SIMT {prec} datapath (dp4a/FMA). On {spec.sm} "
            f"the tensor cores do ~{ratio:.0f}x more {prec} MACs/clock. This "
            "is fine while memory-bound (low arithmetic-per-byte / batch 1), "
            f"but as arithmetic-per-byte grows the loop becomes SIMT-compute-"
            f"bound and a tensor-core implementation would be up to ~{ratio:.0f}x "
            "faster. Measure across your batch sizes to find the crossover."
        ),
    }
