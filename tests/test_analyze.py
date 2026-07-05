"""Analyzer tests: pressure, spill map (incl. the ptxas cross-validation),
diff, and gate — all from recorded fixtures, no toolchain required."""

from pathlib import Path

import pytest

from cuxray.analyze.liveness import pressure
from cuxray.analyze.spillmap import spill_map
from cuxray.diffgate import GateSyntaxError, diff_reports, eval_gate, parse_gate
from cuxray.parse import cfgdot, ptxasv, sass

REC = Path(__file__).parent / "fixtures" / "recorded"


def load_spill_func(arch="sm_120a"):
    dis = sass.parse_gi((REC / f"nvdisasm_gi.spill.{arch}.txt").read_text())
    plr = sass.parse_plr((REC / f"nvdisasm_plr.spill.{arch}.txt").read_text())
    sass.merge_liveness(dis, plr)
    cfg = cfgdot.parse((REC / f"nvdisasm_cfg.spill.{arch}.dot").read_text())
    name = "_Z6spillyPKfPfii"
    return dis.functions[name], cfg[name].loop_depth


@pytest.mark.parametrize("arch", ["sm_90", "sm_120a"])
def test_spill_bytes_match_ptxas_exactly(arch):
    """Our SASS-width accounting must reproduce ptxas -v byte counts.

    This is the load-bearing validation: it means spill bytes are reportable
    for cubins cuxray did not compile.
    """
    func, depths = load_spill_func(arch)
    sm = spill_map(func, depths)
    pk = ptxasv.parse((REC / f"ptxas_v.spill.{arch}.txt").read_text())["_Z6spillyPKfPfii"]
    assert sm["store_bytes"] == pk.spill_stores > 0
    assert sm["load_bytes"] == pk.spill_loads > 0


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


def test_shared_memory_detection():
    from cuxray.parse.sass import Function, Instruction, uses_shared_memory
    none = Function(name="k", instructions=[Instruction(addr=0, opcode="LDG.E", operands="")])
    assert not uses_shared_memory(none)
    for op in ("LDS.64", "STS.128", "LDSM.16.M88.4", "LDGSTS.E.BYPASS.128", "UTMALDG.2D"):
        f = Function(name="k", instructions=[Instruction(addr=0, opcode=op, operands="")])
        assert uses_shared_memory(f), op


def test_pareto_marking():
    from cuxray.tune import mark_pareto
    rows = [
        {"cap": 32, "occupancy_pct": 100.0, "spill_bytes": 1104},
        {"cap": 64, "occupancy_pct": 83.3, "spill_bytes": 0},
        {"cap": 48, "occupancy_pct": 83.3, "spill_bytes": 500},   # dominated by 64
        {"cap": 255, "occupancy_pct": 50.0, "spill_bytes": 0},    # dominated by 64
    ]
    mark_pareto(rows)
    assert [r["cap"] for r in rows if r["pareto"]] == [32, 64]


def test_matrix_expansion_and_pareto():
    from cuxray.tunematrix import expand_matrix, mark_pareto
    combos = expand_matrix({"BM": ["64", "128"], "STAGES": ["2", "3"]})
    assert len(combos) == 4 and {"BM": "64", "STAGES": "3"} in combos
    variants = [
        {"config": {"BM": "64"}, "kernels": [{"occupancy_pct": 50, "spill_bytes": 0, "bank_ways": 1, "uncoalesced": 0}]},
        {"config": {"BM": "128"}, "kernels": [{"occupancy_pct": 50, "spill_bytes": 400, "bank_ways": 1, "uncoalesced": 0}]},
        {"config": {"BM": "256"}, "kernels": [{"occupancy_pct": 25, "spill_bytes": 0, "bank_ways": 8, "uncoalesced": 0}]},
    ]
    mark_pareto(variants)
    assert variants[0]["pareto"] and not variants[1]["pareto"] and not variants[2]["pareto"]
