"""Static wall-clock estimate for a kernel launch (calibrated roofline).

A binary can't reveal the problem size — traffic and trip counts depend on
runtime arguments (K, M, ...). So the division of labor is:

  * the CALLER supplies the launch's total work — DRAM bytes and, for a
    compute-bound kernel, the MAC count — which for a GEMM/GEMV/attention is
    a one-line formula in the problem dims;
  * cuxray supplies the ideal roofline time from that work, and a
    two-constant calibration captures how this kernel-family realizes it on a
    given device:

        t = t_ideal / e_sat + t_fixed

    - t_ideal  = max(bytes / achievable_bw, macs / datapath_peak)
    - e_sat    the saturating efficiency once the launch amortizes overhead.
      Fitted, not physical: it also absorbs error in the caller's byte/MAC
      estimate and the peak table, so values slightly above 1.0 occur.
    - t_fixed  (µs): fixed per-launch cost — launch, any fused pre/post pass,
      un-amortized latency.

`fit()` recovers (e_sat, t_fixed) by least squares. IMPORTANT: the constants
are per (kernel-family, device), NOT per device — a GEMV, an FMA loop, and a
cuBLAS GEMM on the same A100 fit very different constants and do NOT
cross-apply (cross-family error runs 100-400%). Calibrate a family once,
then predict other sizes of THAT family on THAT device with no GPU.

Recorded validation is LEAVE-ONE-OUT (held out — see tests/test_perfmodel.py):
memory-bound W4A8 decode GEMV on A100 + A5000, compute-bound FP32 FMA and
tensor-core cuBLAS fp16 GEMM on A100. Held-out error ~2-37% by family, worst
at the smallest overhead-bound sizes. NOT validated: mixed memory+compute
near the roofline crossover, low-occupancy / latency-bound kernels, atomics,
divergence, irregular/uncoalesced access, reductions/stencils, attention,
non-GEMM tensor code, FMA/tensor on non-A100. Relative ordering of
same-family variants is more robust than the absolute number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Device:
    sms: int
    clock_ghz: float               # sustained GHz
    achievable_gbs: float          # measured streaming bandwidth, not spec
    cc: Optional[tuple] = None     # (major, minor) for the MAC-peak table


@dataclass
class Work:
    dram_bytes: float              # unique DRAM traffic for the launch
    macs: int = 0                  # multiply-accumulates (0 → memory-only)
    precision: str = "int8"
    datapath: str = "simt-int"     # 'tensor' picks the tensor-core peak


@dataclass
class Calibration:
    e_sat: float = 0.95            # saturating efficiency (device-transferable)
    t_fixed_us: float = 0.0        # fixed per-launch cost (device-specific)


def ideal_us(work: Work, dev: Device) -> dict:
    """Device-relative roofline floor, before efficiency/overhead."""
    t_mem = work.dram_bytes / (dev.achievable_gbs * 1e9) * 1e6
    t_compute = 0.0
    if work.macs:
        peak = _datapath_peak_macs_per_us(dev, work.precision, work.datapath)
        if peak:
            t_compute = work.macs / peak
    return {"t_ideal_us": max(t_mem, t_compute),
            "t_mem_ideal_us": t_mem, "t_compute_ideal_us": t_compute,
            "bound": "compute" if t_compute > t_mem else "memory"}


def predict(work: Work, dev: Device, calib: Calibration) -> dict:
    base = ideal_us(work, dev)
    t = base["t_ideal_us"] / max(0.02, calib.e_sat) + calib.t_fixed_us
    return {"us": round(t, 3), **{k: round(v, 3) if isinstance(v, float) else v
                                  for k, v in base.items()}}


def fit(samples: list[tuple[float, float]]) -> Calibration:
    """Least-squares (e_sat, t_fixed) from [(t_ideal_us, measured_us), ...].

    Fits measured = a*t_ideal + b with a = 1/e_sat, b = t_fixed."""
    n = len(samples)
    if n < 2:
        raise ValueError("need >= 2 samples to calibrate")
    sx = sum(x for x, _ in samples)
    sy = sum(y for _, y in samples)
    sxx = sum(x * x for x, _ in samples)
    sxy = sum(x * y for x, y in samples)
    denom = n * sxx - sx * sx
    if denom == 0:
        raise ValueError("degenerate calibration (all t_ideal equal)")
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    return Calibration(e_sat=round(1.0 / a, 4) if a > 0 else 0.95,
                       t_fixed_us=round(b, 3))


def _datapath_peak_macs_per_us(dev: Device, precision: str,
                               datapath: str) -> Optional[float]:
    from ..archspec import SPECS, lookup, mac_rates
    if dev.cc is not None:
        spec = SPECS.get(tuple(dev.cc))
    else:  # no cc given: pick the closest-SM known Ampere+ arch (rough)
        spec = lookup("sm_80" if dev.sms >= 100 else "sm_86")
    if spec is None:
        return None
    rates = mac_rates(spec)
    if not rates:
        return None
    kind = "tensor" if datapath == "tensor" else "simt"
    rate = rates.get(kind, {}).get(precision)
    return rate * dev.sms * dev.clock_ghz * 1e3 if rate else None
