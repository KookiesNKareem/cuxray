"""Roofline floor: exact arithmetic on the caller's work + the validated peak
table. The confident, honest claim is that the floor is a true LOWER BOUND —
no real kernel beats it. Checked against the recorded hardware corpus across
memory-bound / compute-bound / tensor-core kernels on A100 (+ A5000). cuxray
does not predict the efficiency gap above the floor (that was cut as
overreach — it needs per-family hardware calibration)."""
import json
from pathlib import Path

import pytest

from cuxray.analyze.perfmodel import Device, Work, ideal_us

CORPUS = json.loads(
    (Path(__file__).parent / "fixtures" / "recorded" / "perfmodel_corpus.json").read_text()
)


def _work(s):
    return Work(dram_bytes=s["bytes"], macs=s.get("macs", 0),
                precision=s.get("precision", "int8"),
                datapath=s.get("datapath", "simt-int"))


@pytest.mark.parametrize("case", CORPUS["cases"], ids=[c["name"] for c in CORPUS["cases"]])
def test_floor_is_a_true_lower_bound(case):
    """The roofline floor must never exceed a measured time — that's what
    makes it a floor. Also validates the byte/MAC math and the peaks aren't
    over-optimistic across kernel types and devices."""
    dev = Device(**case["device"])
    for s in case["samples"]:
        floor = ideal_us(_work(s), dev)["t_ideal_us"]
        assert floor <= s["us"] * 1.02, (   # 2% slack for measurement noise
            f"{case['name']}: floor {floor:.1f} > measured {s['us']}")


def test_floor_is_reachable_not_absurdly_low(case=None):
    """Sanity: the best-case efficiency across the corpus should be plausible
    (a good kernel gets within ~2x of the floor), so the floor isn't a
    vacuous underestimate."""
    best = {}
    for c in CORPUS["cases"]:
        dev = Device(**c["device"])
        effs = [ideal_us(_work(s), dev)["t_ideal_us"] / s["us"] for s in c["samples"]]
        best[c["name"]] = max(effs)
    for name, e in best.items():
        assert 0.4 <= e <= 1.02, f"{name}: best efficiency {e:.2f} implausible"


def test_roofline_floor_is_exact_arithmetic():
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
