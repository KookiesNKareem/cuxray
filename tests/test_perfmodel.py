"""Perf model: the ROOFLINE FLOOR is exact arithmetic on the caller's work +
the validated peak table; the calibrated wall-clock ESTIMATE is empirical and
is reported here as leave-one-out (held-out) residuals, NOT in-sample fit —
so the error numbers are honest. The recorded corpus spans memory-bound /
compute-bound / tensor-core on A100 (+ A5000 for memory-bound); it does NOT
cover mixed, low-occupancy, irregular, or attention kernels.
"""
import json
from pathlib import Path

import pytest

from cuxray.analyze.perfmodel import Calibration, Device, Work, fit, ideal_us, predict

CORPUS = json.loads(
    (Path(__file__).parent / "fixtures" / "recorded" / "perfmodel_corpus.json").read_text()
)


def _work(s):
    return Work(dram_bytes=s["bytes"], macs=s.get("macs", 0),
                precision=s.get("precision", "int8"),
                datapath=s.get("datapath", "simt-int"))


def _loo_errors(case):
    """Leave-one-out: fit on all-but-one, predict the held-out point."""
    dev = Device(**case["device"])
    samples = case["samples"]
    ideals = [ideal_us(_work(s), dev)["t_ideal_us"] for s in samples]
    errs = []
    for i in range(len(samples)):
        train = [(ideals[j], samples[j]["us"]) for j in range(len(samples)) if j != i]
        calib = fit(train)
        pred = predict(_work(samples[i]), dev, calib)["us"]
        errs.append(abs(pred - samples[i]["us"]) / samples[i]["us"] * 100)
    return errs


@pytest.mark.parametrize("case", CORPUS["cases"], ids=[c["name"] for c in CORPUS["cases"]])
def test_held_out_estimate_within_recorded_bound(case):
    errs = _loo_errors(case)
    mean = sum(errs) / len(errs)
    # bounds are the HELD-OUT (leave-one-out) tolerances recorded per case —
    # looser than in-sample, and that is the honest number
    assert mean <= case["loo_mean_err_pct"], f"{case['name']}: LOO mean {mean:.1f}%"
    assert max(errs) <= case["loo_max_err_pct"], f"{case['name']}: LOO max {max(errs):.0f}%"


def test_calibration_does_not_transfer_across_families():
    """Documents the real limitation: constants are per (family, device).
    A tensor-GEMM calibration mispredicts a GEMV badly on the same device."""
    cases = {c["name"].split(",")[0]: c for c in CORPUS["cases"]}
    gemv = cases["memory-bound W4A8 decode GEMV"]
    gemm = cases["tensor-core fp16 GEMM (cuBLAS)"]
    dev = Device(**gemv["device"])
    gi = [ideal_us(_work(s), dev)["t_ideal_us"] for s in gemm["samples"]]
    gemm_calib = fit(list(zip(gi, [s["us"] for s in gemm["samples"]])))
    errs = [abs(predict(_work(s), dev, gemm_calib)["us"] - s["us"]) / s["us"] * 100
            for s in gemv["samples"]]
    assert sum(errs) / len(errs) > 30   # cross-family transfer is bad, as expected


def test_roofline_floor_is_exact_arithmetic():
    # the floor has no free parameters — pure work / peak
    dev = Device(sms=108, clock_ghz=1.41, achievable_gbs=1682)
    r = ideal_us(Work(dram_bytes=1682e9 * 1e-6 * 10), dev)   # exactly 10 µs of BW
    assert abs(r["t_mem_ideal_us"] - 10.0) < 1e-6
    assert r["bound"] == "memory"


def test_all_kernel_types_present():
    dps = {s.get("datapath", "simt-int")
           for c in CORPUS["cases"] for s in c["samples"]}
    assert {"simt-int", "simt-fp", "tensor"} <= dps


def test_datapath_peaks_land_near_hardware():
    from cuxray.analyze.perfmodel import _datapath_peak_macs_per_us
    dev = Device(sms=108, clock_ghz=1.41, achievable_gbs=1682, cc=(8, 0))
    fp32 = _datapath_peak_macs_per_us(dev, "fp32", "simt-fp")
    tc = _datapath_peak_macs_per_us(dev, "fp16", "tensor")
    assert 9.0e6 <= fp32 <= 10.5e6      # ~9.74 TMAC/s FP32
    assert 1.4e8 <= tc <= 1.7e8         # ~156 TMAC/s tensor fp16
