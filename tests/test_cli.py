"""CLI wiring tests (click.testing.CliRunner) — no toolchain required.

build_report/resolve are monkeypatched with the committed sample report
fixture; the occupancy command runs for real (pure Python)."""

import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from cuxray import cli
from cuxray.report import parse_block_dims

SAMPLE = json.loads(
    (Path(__file__).parent / "fixtures" / "recorded" / "sample_report.json").read_text()
)


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.setattr(cli, "_toolchain", lambda: object())
    monkeypatch.setattr(cli, "build_report", lambda path, tc, **kw: SAMPLE)
    return CliRunner()


class TestParseBlockDims:
    def test_forms(self):
        assert parse_block_dims("256") == ((256, 1, 1), 256)
        assert parse_block_dims("32,8") == ((32, 8, 1), 256)
        assert parse_block_dims("16,4,2") == ((16, 4, 2), 128)
        assert parse_block_dims(None) == (None, None)

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            parse_block_dims("banana")


class TestReport:
    def test_json_shape(self, runner, tmp_path):
        f = tmp_path / "x.cubin"
        f.write_bytes(b"\x7fELF")
        res = runner.invoke(cli.main, ["report", str(f), "--json"])
        assert res.exit_code == 0, res.output
        doc = json.loads(res.output)
        assert doc["schema"] == "cuxray.schema/1"

    def test_output_file(self, runner, tmp_path):
        f = tmp_path / "x.cubin"
        f.write_bytes(b"\x7fELF")
        out = tmp_path / "r.json"
        res = runner.invoke(cli.main, ["report", str(f), "-o", str(out)])
        assert res.exit_code == 0
        assert json.loads(out.read_text())["schema"] == "cuxray.schema/1"

    def test_invalid_regex_exit_2(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "_toolchain", lambda: object())

        def boom(path, tc, **kw):
            raise re.error("bad regex")
        monkeypatch.setattr(cli, "build_report", boom)
        f = tmp_path / "x.cubin"
        f.write_bytes(b"\x7fELF")
        res = CliRunner().invoke(cli.main, ["report", str(f), "--kernel", "["])
        assert res.exit_code == 2
        assert "invalid --kernel regex" in res.output


class TestGate:
    def test_pass_and_fail_exit_codes(self, runner, tmp_path):
        f = tmp_path / "x.cubin"
        f.write_bytes(b"\x7fELF")
        ok = runner.invoke(cli.main, ["gate", str(f), "regs<=64"])
        assert ok.exit_code == 0, ok.output
        bad = runner.invoke(cli.main, ["gate", str(f), "bank_ways<=1"])
        assert bad.exit_code == 1  # sample has a 32-way conflict kernel

    def test_syntax_error_exit_2(self, runner, tmp_path):
        f = tmp_path / "x.cubin"
        f.write_bytes(b"\x7fELF")
        res = runner.invoke(cli.main, ["gate", str(f), "frobs<=1"])
        assert res.exit_code == 2

    def test_missing_access_metric_is_gate_error(self, tmp_path, monkeypatch):
        stripped = json.loads(json.dumps(SAMPLE))
        for u in stripped["units"]:
            for k in u["kernels"]:
                k["access"] = None
        monkeypatch.setattr(cli, "_toolchain", lambda: object())
        monkeypatch.setattr(cli, "build_report", lambda path, tc, **kw: stripped)
        f = tmp_path / "x.cubin"
        f.write_bytes(b"\x7fELF")
        res = CliRunner().invoke(cli.main, ["gate", str(f), "bank_ways<=2"])
        assert res.exit_code == 2
        assert "requires access analysis" in res.output


class TestDiff:
    def test_self_diff_clean_and_fail_on_change(self, runner, tmp_path):
        f = tmp_path / "x.cubin"
        f.write_bytes(b"\x7fELF")
        res = runner.invoke(cli.main, ["diff", str(f), str(f), "--json"])
        assert res.exit_code == 0
        d = json.loads(res.output)
        assert d["changed"] == 0 and d["regressions"] == 0
        res = runner.invoke(cli.main, ["diff", str(f), str(f), "--fail-on-change"])
        assert res.exit_code == 0  # no changes → still 0


class TestOccupancy:
    def test_runs_without_toolchain(self):
        res = CliRunner().invoke(cli.main, [
            "occupancy", "--arch", "sm_120", "--regs", "168",
            "--threads", "256", "--json"])
        assert res.exit_code == 0, res.output
        d = json.loads(res.output)
        assert d["blocks_per_sm"] == 1 and d["limiter"] == "registers"

    def test_block_shape_form(self):
        res = CliRunner().invoke(cli.main, [
            "occupancy", "--arch", "sm_90", "--regs", "32",
            "--threads", "32,8", "--json"])
        assert res.exit_code == 0
        assert json.loads(res.output)["threads_per_block"] == 256
