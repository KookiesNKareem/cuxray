"""The perf model's accuracy claim, machine-checked against the recorded
hardware corpus: fit (e_sat, t_fixed) per device, assert the residual stays
within the bound recorded for that device."""
import json
from pathlib import Path

import pytest

from cuxray.analyze.perfmodel import Calibration, Device, Work, fit, ideal_us, predict

CORPUS = json.loads(
    (Path(__file__).parent / "fixtures" / "recorded" / "perfmodel_corpus.json").read_text()
)


def _bytes(M, K):
    return M * K / 2 + K + M * (K // 128) * 2 + M * 2


@pytest.mark.parametrize("dev", CORPUS["devices"], ids=[d["name"] for d in CORPUS["devices"]])
def test_calibrated_model_accuracy(dev):
    device = Device(sms=108, clock_ghz=1.41, achievable_gbs=dev["achievable_gbs"])
    ideals = [ideal_us(Work(dram_bytes=_bytes(M, K)), device)["t_ideal_us"]
              for M, K, _ in dev["samples"]]
    meas = [m for *_, m in dev["samples"]]
    calib = fit(list(zip(ideals, meas)))
    assert 0.5 <= calib.e_sat <= 1.2, calib      # sane efficiency
    errs = []
    for (M, K, m), idl in zip(dev["samples"], ideals):
        p = predict(Work(dram_bytes=_bytes(M, K)), device, calib)["us"]
        errs.append(abs(p - m) / m * 100)
    mean = sum(errs) / len(errs)
    assert mean <= dev["max_mean_err_pct"], f"mean {mean:.1f}% > {dev['max_mean_err_pct']}%"
    assert max(errs) <= dev["max_single_err_pct"], f"max {max(errs):.0f}%"


def test_fit_recovers_known_line():
    # measured = ideal/0.8 + 5  → e_sat 0.8, t_fixed 5
    samples = [(i, i / 0.8 + 5) for i in (10, 20, 40, 80)]
    c = fit(samples)
    assert abs(c.e_sat - 0.8) < 1e-3 and abs(c.t_fixed_us - 5) < 1e-3


def test_relative_ranking_ignores_fixed_work():
    # two variants, same problem (same t_ideal), different efficiency →
    # ranking depends only on the calibration, not the work
    dev = Device(sms=108, clock_ghz=1.41, achievable_gbs=1682)
    w = Work(dram_bytes=30e6)
    fast = predict(w, dev, Calibration(e_sat=0.95, t_fixed_us=6))["us"]
    slow = predict(w, dev, Calibration(e_sat=0.70, t_fixed_us=6))["us"]
    assert fast < slow


def test_compute_bound_uses_mac_peak():
    dev = Device(sms=108, clock_ghz=1.41, achievable_gbs=1682)
    # tiny traffic, huge MAC count → compute-bound
    w = Work(dram_bytes=1000, macs=10**12, precision="int8", datapath="simt-int")
    r = ideal_us(w, dev)
    assert r["bound"] == "compute" and r["t_compute_ideal_us"] > 0
