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
