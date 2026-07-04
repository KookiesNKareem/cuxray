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
    monkeypatch.setattr(cli, "_toolchain", lambda *a, **k: object())
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
        monkeypatch.setattr(cli, "_toolchain", lambda *a, **k: object())

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
        monkeypatch.setattr(cli, "_toolchain", lambda *a, **k: object())
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


class TestSarif:
    def test_gate_writes_valid_sarif(self, runner, tmp_path):
        f = tmp_path / "x.cubin"
        f.write_bytes(b"\x7fELF")
        sarif_path = tmp_path / "out.sarif"
        res = runner.invoke(cli.main, ["gate", str(f), "bank_ways<=1",
                                       "--sarif", str(sarif_path)])
        assert res.exit_code == 1  # sample has a conflicted kernel
        doc = json.loads(sarif_path.read_text())
        assert doc["version"] == "2.1.0"
        results = doc["runs"][0]["results"]
        assert results and results[0]["ruleId"] == "cuxray/bank_ways"
        assert results[0]["level"] == "error"


class TestCache:
    def test_cache_roundtrip(self, tmp_path, monkeypatch):
        from cuxray import report as report_mod
        monkeypatch.setenv("CUXRAY_CACHE", str(tmp_path))
        calls = []

        def fake_analyze(unit, tc, **kw):
            calls.append(1)
            return {"label": unit.label, "arch": "sm_90", "cubin_sha256": "x",
                    "kernels": []}
        monkeypatch.setattr(report_mod, "analyze_unit", fake_analyze)

        from cuxray.ingest import CubinUnit
        cub = tmp_path / "k.cubin"
        cub.write_bytes(b"\x7fELF" + b"\0" * 64)
        unit = CubinUnit(cubin=cub, label="k.cubin", source=cub)
        tc = object.__new__(type("T", (), {"nvdisasm": "nd", "cuobjdump": "co"}))

        d1 = report_mod._analyze_unit_cached(unit, tc, True, level="full")
        d2 = report_mod._analyze_unit_cached(unit, tc, True, level="full")
        assert len(calls) == 1                      # second hit came from cache
        assert d2.get("from_cache") is True
        # different params → miss
        report_mod._analyze_unit_cached(unit, tc, True, level="resources")
        assert len(calls) == 2
        # cache bypass → recompute
        report_mod._analyze_unit_cached(unit, tc, False, level="full")
        assert len(calls) == 3


class TestBudget:
    def test_budget_per_kernel_rules(self, runner, tmp_path):
        f = tmp_path / "x.cubin"
        f.write_bytes(b"\x7fELF")
        budget = tmp_path / "b.json"
        budget.write_text(json.dumps({
            "default": "spill_instrs==0",
            "kernels": [
                {"match": "col_conflict", "gate": "bank_ways<=32"},
                {"match": "xor_swizzle", "gate": "bank_ways<=1"},
            ],
        }))
        res = runner.invoke(cli.main, ["gate", str(f), "--budget", str(budget)])
        assert res.exit_code == 0, res.output  # each kernel within its own budget

        strict = tmp_path / "s.json"
        strict.write_text(json.dumps({
            "kernels": [{"match": "col_conflict", "gate": "bank_ways<=2"}],
        }))
        res = runner.invoke(cli.main, ["gate", str(f), "--budget", str(strict)])
        assert res.exit_code == 1  # col_conflict is 32-way

    def test_expr_and_budget_mutually_exclusive(self, runner, tmp_path):
        f = tmp_path / "x.cubin"
        f.write_bytes(b"\x7fELF")
        res = runner.invoke(cli.main, ["gate", str(f)])
        assert res.exit_code == 2


class TestBlockDimsStrict:
    def test_rejects_zero_negative_and_extra_dims(self):
        for bad in ("0", "-1", "32,0", "32,8,1,9", "2048"):
            res = CliRunner().invoke(cli.main, [
                "occupancy", "--arch", "sm_90", "--regs", "32",
                "--threads", bad])
            assert res.exit_code == 2, bad
