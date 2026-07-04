"""Analyzer tests: pressure, spill map (incl. the ptxas cross-validation),
diff, and gate — all from recorded fixtures, no toolchain required."""

from pathlib import Path

import pytest

from cuxray.analyze.liveness import pressure
from cuxray.analyze.spillmap import spill_map
from cuxray.diffgate import GateSyntaxError, diff_reports, eval_gate, parse_gate
from cuxray.parse import cfgdot, ptxasv, sass

REC = Path(__file__).parent / "fixtures" / "recorded"


def load_spill_func():
    dis = sass.parse_gi((REC / "nvdisasm_gi.spill.sm_120a.txt").read_text())
    plr = sass.parse_plr((REC / "nvdisasm_plr.spill.sm_120a.txt").read_text())
    sass.merge_liveness(dis, plr)
    cfg = cfgdot.parse((REC / "nvdisasm_cfg.spill.sm_120a.dot").read_text())
    name = "_Z6spillyPKfPfii"
    return dis.functions[name], cfg[name].loop_depth


def test_spill_bytes_match_ptxas_exactly():
    """Our SASS-width accounting must reproduce ptxas -v byte counts.

    This is the load-bearing validation: it means spill bytes are reportable
    for cubins cuxray did not compile.
    """
    func, depths = load_spill_func()
    sm = spill_map(func, depths)
    pk = ptxasv.parse((REC / "ptxas_v.spill.sm_120a.txt").read_text())["_Z6spillyPKfPfii"]
    assert sm["store_bytes"] == pk.spill_stores == 548
    assert sm["load_bytes"] == pk.spill_loads == 556


def test_spill_map_loop_weighting():
    func, depths = load_spill_func()
    sm = spill_map(func, depths)
    assert sm["max_loop_depth"] == 1
    # hot-loop lines rank first
    assert sm["by_line"][0]["loop_depth"] == 1
    assert sm["by_line"][0]["line"] in (13, 14)
    depth0 = [r for r in sm["by_line"] if r["loop_depth"] == 0]
    assert depth0, "prologue spills present but unranked"


def test_pressure_saxpy():
    dis = sass.parse_gi((REC / "nvdisasm_gi.saxpy.sm_120a.txt").read_text())
    plr = sass.parse_plr((REC / "nvdisasm_plr.saxpy.sm_120a.txt").read_text())
    sass.merge_liveness(dis, plr)
    p = pressure(dis.functions["_Z5saxpyifPKfPf"])
    assert p["available"]
    assert p["peak"]["live_gpr"] == 6
    assert p["peak"]["line"] == 4
    assert p["per_line"][0]["max_live_gpr"] == 6


def _mini_doc(regs=32, spills=(135, 137, 548, 556), arch="sm_120a", peak=30):
    return {
        "units": [{
            "label": "u", "arch": arch,
            "kernels": [{
                "name": "k", "demangled": "k",
                "resources": {"regs": regs, "stack_frame": 0, "smem_static": 0,
                              "shared_section": 0, "local": 0, "constant": 0},
                "pressure": {"available": True, "peak": {"live_gpr": peak}},
                "spills": {"store_instructions": spills[0], "load_instructions": spills[1],
                           "store_bytes": spills[2], "load_bytes": spills[3],
                           "max_loop_depth": 1, "by_line": []},
                "occupancy": None, "notes": [],
            }],
        }],
    }


class TestGate:
    def test_parse_and_pass(self):
        clauses = parse_gate("regs<=64, spill_instrs==272")
        assert not eval_gate(_mini_doc(), clauses)

    def test_violation(self):
        v = eval_gate(_mini_doc(), parse_gate("spill_instrs==0"))
        assert len(v) == 1 and "272" in v[0]["reason"]

    def test_occupancy_clause_computes(self):
        v = eval_gate(_mini_doc(regs=168), parse_gate("occupancy(threads=256)>=25"))
        assert len(v) == 1  # 168 regs on sm_120 @256 → 16.7%
        assert not eval_gate(_mini_doc(regs=32), parse_gate("occupancy(threads=256)>=25"))

    def test_syntax_errors(self):
        for bad in ("regs<", "frobs<=1", "occupancy>=25", ""):
            with pytest.raises(GateSyntaxError):
                parse_gate(bad)


class TestDiff:
    def test_diff_detects_changes(self):
        d = diff_reports(_mini_doc(regs=32), _mini_doc(regs=48, spills=(0, 0, 0, 0), peak=46))
        assert d["changed"] == 1
        metrics = {c["metric"]: c for c in d["kernels"][0]["changes"]}
        assert metrics["regs"]["delta"] == 16
        assert metrics["spill_bytes_total"]["delta"] == -1104
        assert metrics["pressure_peak"]["delta"] == 16

    def test_identical_reports_no_changes(self):
        d = diff_reports(_mini_doc(), _mini_doc())
        assert d["changed"] == 0
        assert d["added"] == d["removed"] == []


def test_register_reallocation_detection():
    from cuxray.parse.sass import Function, Instruction, uses_register_reallocation
    plain = Function(name="k", instructions=[Instruction(addr=0, opcode="FFMA", operands="")])
    assert not uses_register_reallocation(plain)
    ws = Function(name="k", instructions=[
        Instruction(addr=0, opcode="USETMAXREG.DEALLOC.CTAPOOL", operands="..."),
        Instruction(addr=16, opcode="FFMA", operands=""),
    ])
    assert uses_register_reallocation(ws)
