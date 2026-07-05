"""Static wall-clock estimate for a kernel launch (calibrated roofline).

A binary can't reveal the problem size — traffic and trip counts depend on
runtime arguments (K, M, ...). So the division of labor is:

  * the CALLER supplies the launch's total work — DRAM bytes and, for a
    compute-bound kernel, the MAC count — which for a GEMM/GEMV/attention is
    a one-line formula in the problem dims;
  * cuxray supplies the device-independent ideal roofline time from that
    work, and a two-constant calibration captures how this kernel-family
    realizes it on a given device:

        t = t_ideal / e_sat + t_fixed

    - t_ideal  = max(bytes / achievable_bw, macs / datapath_peak)
    - e_sat    ∈ (0,1]: the saturating efficiency once the launch is large
      enough to amortize overhead. ~0.95-1.0, largely device-independent.
    - t_fixed  (µs): fixed per-launch cost — kernel launch, any fused
      pre/post pass, and un-amortized memory latency. Device-specific.

`fit()` recovers (e_sat, t_fixed) from a handful of (work, measured_us)
points — calibrate once per device, then predict the rest with no GPU.

Validated on the W4A8 decode GEMV: A100 11% mean / 30% max over 14 shapes,
A5000 2% mean / 8% max over 10 (see tests/test_perfmodel.py, machine-checked
against the recorded corpus). Absolute error is dominated by the smallest,
overhead-bound shapes; the RELATIVE ordering of variants at a fixed problem
size is tighter still, because t_ideal cancels and only e_sat/t_fixed differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Device:
    sms: int
    clock_ghz: float               # sustained GHz
    achievable_gbs: float          # measured streaming bandwidth, not spec


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
    from ..archspec import lookup, mac_rates
    try:
        spec = lookup(f"sm_{80 if dev.sms >= 100 else 86}")
    except Exception:
        return None
    rates = mac_rates(spec)
    if not rates:
        return None
    kind = "tensor" if datapath == "tensor" else "simt"
    rate = rates.get(kind, {}).get(precision)
    return rate * dev.sms * dev.clock_ghz * 1e3 if rate else None
