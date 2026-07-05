"""The perf model's accuracy claim, machine-checked against the recorded
hardware corpus — across kernel TYPES (memory-bound, compute-bound,
tensor-core) and DEVICES (A100, A5000). For each case: build Work, compute
t_ideal (which validates the per-arch peaks), fit (e_sat, t_fixed), assert
the residual stays within the recorded bound."""
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


@pytest.mark.parametrize("case", CORPUS["cases"], ids=[c["name"] for c in CORPUS["cases"]])
def test_calibrated_model_accuracy(case):
    dev = Device(**case["device"])
    ideals = [ideal_us(_work(s), dev)["t_ideal_us"] for s in case["samples"]]
    meas = [s["us"] for s in case["samples"]]
    calib = fit(list(zip(ideals, meas)))
    assert 0.4 <= calib.e_sat <= 1.2, (case["name"], calib)
    errs = [abs(predict(_work(s), dev, calib)["us"] - s["us"]) / s["us"] * 100
            for s in case["samples"]]
    mean = sum(errs) / len(errs)
    assert mean <= case["max_mean_err_pct"], f"{case['name']}: mean {mean:.1f}%"
    assert max(errs) <= case["max_single_err_pct"], f"{case['name']}: max {max(errs):.0f}%"


def test_all_kernel_types_covered():
    dps = {s.get("datapath", "simt-int")
           for c in CORPUS["cases"] for s in c["samples"]}
    assert {"simt-int", "simt-fp", "tensor"} <= dps


def test_fit_recovers_known_line():
    c = fit([(i, i / 0.8 + 5) for i in (10, 20, 40, 80)])
    assert abs(c.e_sat - 0.8) < 1e-3 and abs(c.t_fixed_us - 5) < 1e-3


def test_relative_ranking_ignores_fixed_work():
    dev = Device(sms=108, clock_ghz=1.41, achievable_gbs=1682)
    w = Work(dram_bytes=30e6)
    fast = predict(w, dev, Calibration(e_sat=0.95, t_fixed_us=6))["us"]
    slow = predict(w, dev, Calibration(e_sat=0.70, t_fixed_us=6))["us"]
    assert fast < slow


def test_datapath_peaks_land_near_hardware():
    # FP32 FMA peak on A100 ≈ 9.74 TMAC/s; tensor fp16 ≈ 156 TMAC/s.
    from cuxray.analyze.perfmodel import _datapath_peak_macs_per_us
    dev = Device(sms=108, clock_ghz=1.41, achievable_gbs=1682)
    fp32 = _datapath_peak_macs_per_us(dev, "fp32", "simt-fp")   # MAC/µs
    tc = _datapath_peak_macs_per_us(dev, "fp16", "tensor")
    assert 9.0e6 <= fp32 <= 10.5e6      # ~9.74 TMAC/s
    assert 1.4e8 <= tc <= 1.7e8         # ~156 TMAC/s (312 TFLOP/s)
