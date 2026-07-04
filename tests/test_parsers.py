"""Parser tests against recorded tool output (CUDA 13.3, sm_90 + sm_120a).

These run with no toolchain installed — the recordings ARE the contract.
Regenerate with tests/fixtures/build.sh when bumping toolkit versions.
"""

from pathlib import Path

import pytest

from cuxray.parse import cfgdot, ptxasv, resusage, sass

REC = Path(__file__).parent / "fixtures" / "recorded"


def rec(name: str) -> str:
    return (REC / name).read_text()


class TestPtxasV:
    def test_spill_kernel(self):
        k = ptxasv.parse(rec("ptxas_v.spill.sm_120a.txt"))["_Z6spillyPKfPfii"]
        assert k.regs == 32
        assert k.stack_frame == 208
        assert k.spill_stores == 548
        assert k.spill_loads == 556
        assert k.maxrregcount == 32
        assert k.arch == "sm_120a"

    def test_multi_kernel(self):
        ks = ptxasv.parse(rec("ptxas_v.launch_bounds.sm_90.txt"))
        assert set(ks) == {"_Z5plainPKfPfi", "_Z7boundedPKfPfi"}
        assert all(k.regs == 10 for k in ks.values())

    def test_smem_line(self):
        k = ptxasv.parse(rec("ptxas_v.tiled_matmul.sm_120a.txt"))["_Z12tiled_matmulPKfS0_Pfi"]
        assert k.smem == 8192
        assert k.barriers == 1


class TestResUsage:
    def test_spill(self):
        r = resusage.parse(rec("resusage.spill.sm_120a.txt"))["_Z6spillyPKfPfii"]
        assert (r.reg, r.stack, r.shared, r.local) == (32, 208, 0, 0)
        assert r.constant == 920

    def test_shared_includes_reserved_kb(self):
        # ptxas reports 8192 user smem; the section adds the 1 KB system reserve
        r = resusage.parse(rec("resusage.tiled_matmul.sm_120a.txt"))
        assert r["_Z12tiled_matmulPKfS0_Pfi"].shared == 9216


class TestSass:
    def test_gi_saxpy(self):
        dis = sass.parse_gi(rec("nvdisasm_gi.saxpy.sm_120a.txt"))
        assert dis.target == "sm_120a"
        f = dis.functions["_Z5saxpyifPKfPf"]
        assert f.instructions[0].opcode == "LDC"
        assert f.instructions[0].line == 2
        assert f.instructions[0].file.endswith("saxpy.cu")
        opcodes = {i.opcode for i in f.instructions}
        assert {"S2R", "IMAD", "FFMA", "EXIT"} <= opcodes

    def test_plr_merge_peak(self):
        dis = sass.parse_gi(rec("nvdisasm_gi.saxpy.sm_120a.txt"))
        plr = sass.parse_plr(rec("nvdisasm_plr.saxpy.sm_120a.txt"))
        sass.merge_liveness(dis, plr)
        f = dis.functions["_Z5saxpyifPKfPf"]
        peak = max(i.live_gpr for i in f.instructions if i.live_gpr is not None)
        assert peak == 6  # hand-verified against the -plr rendering

    def test_spill_detection_lines_and_blocks(self):
        dis = sass.parse_gi(rec("nvdisasm_gi.spill.sm_120a.txt"))
        f = list(dis.functions.values())[0]
        spills = [i for i in f.instructions if sass.is_spill(i)]
        assert len(spills) == 272
        assert {i.line for i in spills} == {10, 13, 14, 18}
        assert ".L_x_2" in {i.block for i in spills}

    def test_predicated_instruction(self):
        dis = sass.parse_gi(rec("nvdisasm_gi.saxpy.sm_120a.txt"))
        f = dis.functions["_Z5saxpyifPKfPf"]
        pred = [i for i in f.instructions if i.predicate]
        assert pred and pred[0].predicate == "@P0"


class TestCfg:
    def test_loop_depths(self):
        cfg = cfgdot.parse(rec("nvdisasm_cfg.spill.sm_120a.dot"))
        c = cfg["_Z6spillyPKfPfii"]
        assert c.loop_depth[".L_x_2"] == 1        # the hot loop
        assert c.loop_depth["_Z6spillyPKfPfii"] == 0  # entry not in loop

    def test_straightline_kernel_no_loops(self):
        cfg = cfgdot.parse(rec("nvdisasm_cfg.saxpy.sm_120a.dot"))
        for f in cfg.values():
            assert all(d == 0 for d in f.loop_depth.values())

    def test_matmul_has_loop(self):
        cfg = cfgdot.parse(rec("nvdisasm_cfg.tiled_matmul.sm_120a.dot"))
        f = list(cfg.values())[0]
        assert max(f.loop_depth.values(), default=0) >= 1
