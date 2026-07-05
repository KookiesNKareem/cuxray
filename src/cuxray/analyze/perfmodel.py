"""Roofline floor for a kernel launch.

A binary can't reveal the problem size — traffic and MAC counts depend on
runtime arguments (K, M, ...). So the caller supplies the launch's total
work (DRAM bytes and, for compute, the MAC count — a one-line formula in the
problem dims), and cuxray reports the roofline FLOOR: the fastest that work
can run on this device, and whether it is memory- or compute-bound.

    t_floor = max(bytes / achievable_bw, macs / datapath_peak)

This is a lower bound with no free parameters — a real kernel runs slower by
its efficiency. cuxray deliberately does NOT predict that efficiency: it
would require per-(kernel-family, device) hardware calibration, which
contradicts the hardware-free premise and is a soft regression next to the
tool's validated facts. Use the floor to bound what's possible and to see
which resource binds; use `advise` (datapath crossover) to see whether the
kernel's math even runs on the datapath that can reach the floor.

The peaks come from archspec.mac_rates (validated: A100 FP32 9.74 TMAC/s and
tensor fp16 156 TMAC/s land on the hardware measurements). achievable_bw
should be a MEASURED streaming bandwidth, not the spec sheet.
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


def ideal_us(work: Work, dev: Device) -> dict:
    """Roofline floor (µs) and the binding resource. A true lower bound."""
    t_mem = work.dram_bytes / (dev.achievable_gbs * 1e9) * 1e6
    t_compute = 0.0
    if work.macs:
        peak = _datapath_peak_macs_per_us(dev, work.precision, work.datapath)
        if peak:
            t_compute = work.macs / peak
    return {"t_ideal_us": max(t_mem, t_compute),
            "t_mem_ideal_us": t_mem, "t_compute_ideal_us": t_compute,
            "bound": "compute" if t_compute > t_mem else "memory"}


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
