"""Machine-checked hardware calibration.

Replays each record in recorded/calibration.json: recomputes the cuxray
static prediction from the recorded disassembly (no GPU) and asserts it
lands within the record's tolerance of the value measured on real hardware.
This turns the README's "validated within X% of hardware" claims into
CI-enforced facts.
"""

import json
from pathlib import Path

import pytest

from cuxray.analyze.schedule import loop_schedule
from cuxray.parse import cfgdot, ctrl, sass

REC = Path(__file__).parent / "fixtures" / "recorded"
RECORDS = json.loads((REC / "calibration.json").read_text())["records"]


def _predict_sched_cycles(rec: dict) -> float:
    dis = sass.parse_gi((REC / rec["recorded_gi"]).read_text())
    cfg = cfgdot.parse((REC / rec["recorded_cfg"]).read_text())
    controls = ctrl.parse_sass_controls((REC / rec["recorded_sass"]).read_text())
    func = dis.functions[rec["kernel"]]
    rows = loop_schedule(func, cfg.get(rec["kernel"]),
                         controls.get(rec["kernel"], {}))
    match = [r for r in rows if r["header"] == rec["loop_label"]]
    assert match, f"loop {rec['loop_label']} not found for {rec['id']}"
    return match[0]["est_issue_stall_cycles_per_iter"]


_PREDICTORS = {"sched_cycles_per_iter": _predict_sched_cycles}


@pytest.mark.parametrize("rec", RECORDS, ids=[r["id"] for r in RECORDS])
def test_prediction_within_tolerance_of_hardware(rec):
    predictor = _PREDICTORS.get(rec["kind"])
    assert predictor, f"no predictor registered for kind {rec['kind']}"
    predicted = predictor(rec)

    # the fixture records the prediction cuxray made when captured; the model
    # must not silently drift away from that recorded value
    assert predicted == rec["predicted"], (
        f"{rec['id']}: prediction drifted from recorded "
        f"{rec['predicted']} to {predicted}"
    )
    measured = rec["measured"]
    tol = rec["tolerance_pct"] / 100.0
    rel = abs(predicted - measured) / measured
    assert rel <= tol, (
        f"{rec['id']}: predicted {predicted} vs measured {measured} "
        f"({rel:.1%} > {rec['tolerance_pct']}% tolerance)"
    )
