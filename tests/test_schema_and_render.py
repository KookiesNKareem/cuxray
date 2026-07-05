"""Schema contract + rendering-layer tests, all against committed fixtures."""

import json
from pathlib import Path

import pytest
from rich.console import Console

from cuxray.render import render_diff, render_ls, render_report
from cuxray.diffgate import diff_reports

ROOT = Path(__file__).parent.parent
SAMPLE = json.loads(
    (Path(__file__).parent / "fixtures" / "recorded" / "sample_report.json").read_text()
)


def test_sample_report_conforms_to_schema():
    jsonschema = pytest.importorskip("jsonschema")
    from importlib.resources import files
    schema = json.loads(files("cuxray.schema").joinpath("cuxray.schema.1.json").read_text())
    jsonschema.validate(SAMPLE, schema)


def test_schema_file_is_valid_jsonschema():
    jsonschema = pytest.importorskip("jsonschema")
    from importlib.resources import files
    schema = json.loads(files("cuxray.schema").joinpath("cuxray.schema.1.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)


def _rendered(fn, *args) -> str:
    console = Console(record=True, width=120)
    fn(*args, console)
    return console.export_text()


class TestRender:
    def test_report_renders_key_facts(self):
        out = _rendered(render_report, SAMPLE)
        assert "bank conflict" in out          # col_conflict kernel
        assert "access patterns clean" in out  # xor_swizzle kernel
        assert "occupancy @32 thr" in out
        assert "fix:" in out                   # verified suggestion rendered

    def test_ls_renders(self):
        out = _rendered(render_ls, SAMPLE)
        assert "col_conflict" in out

    def test_diff_polarity_colors(self):
        old = json.loads(json.dumps(SAMPLE))
        new = json.loads(json.dumps(SAMPLE))
        k = new["units"][0]["kernels"][0]
        k["resources"]["regs"] += 8
        k["occupancy"]["occupancy_pct"] -= 10.0
        d = diff_reports(old, new)
        console = Console(record=True, width=140)
        render_diff(d, console)
        text = console.export_text()
        assert "occupancy_pct" in text
        # regs up AND occupancy down are BOTH regressions on one kernel
        assert d["regressions"] == 1

    def test_diff_schema_mismatch_raises(self):
        other = json.loads(json.dumps(SAMPLE))
        other["schema"] = "cuxray.schema/2"
        with pytest.raises(ValueError, match="schema mismatch"):
            diff_reports(SAMPLE, other)

    def test_diff_regression_polarity(self):
        old = json.loads(json.dumps(SAMPLE))
        improved = json.loads(json.dumps(SAMPLE))
        k = improved["units"][0]["kernels"][0]
        k["occupancy"]["occupancy_pct"] += 10.0   # improvement only
        d = diff_reports(old, improved)
        assert d["changed"] >= 1
        assert d["regressions"] == 0  # higher occupancy is not a regression


def test_commands_schema_valid_json_and_covers_advise():
    import json
    from importlib.resources import files
    text = files("cuxray.schema").joinpath("cuxray.commands.1.json").read_text()
    doc = json.loads(text)  # must parse
    assert doc["advise"]["properties"]["schema"]["const"] == "cuxray.advise/1"
    assert doc["solve"]["properties"]["results"]["items"]["properties"]["groups"]
    assert doc["sched"]["properties"]["results"]


def test_advise_json_matches_schema_shape():
    from cuxray.advise import advise
    k = {
        "resources": {"regs": 60},
        "spills": {"store_instructions": 2, "load_instructions": 1,
                   "store_bytes": 8, "load_bytes": 4,
                   "by_line": [{"file": "k.cu", "line": 3, "loop_depth": 0}]},
        "occupancy": {"occupancy_pct": 50.0, "limiter": "registers",
                      "threads_per_block": 256, "cliffs": []},
        "access": {"analyzed_count": 5, "unanalyzed_count": 0,
                   "dataflow_converged": True, "unreached_blocks": 0,
                   "conflicted_shared_accesses": 0,
                   "uncoalesced_global_accesses": 0,
                   "block_invariant_read_bytes": 0},
        "roofline": [],
    }
    for a in advise(k):
        assert set(a) >= {"severity", "title", "detail", "confidence"}
        assert a["severity"] in ("high", "medium", "low")
        assert a["confidence"] in ("high", "medium", "low", "unknown")
