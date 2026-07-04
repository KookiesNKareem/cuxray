"""Layer B tests: lane-value domain, bank/sector math, and end-to-end
verdicts on the compiled bank_conflict fixtures (both arches)."""

from pathlib import Path

import pytest

from cuxray.analyze import lanevalue as lv
from cuxray.analyze.access import analyze_accesses, bank_conflict_ways, sector_count
from cuxray.parse import sass

REC = Path(__file__).parent / "fixtures" / "recorded"


class TestDomain:
    def test_pure_closed_under_everything(self):
        a = lv.pure(range(32))
        b = lv.pure([x * 4 for x in range(32)])
        assert lv.add(a, b).kind == lv.PURE
        assert lv.mul(a, b).kind == lv.PURE
        assert lv.xor(a, b).kind == lv.PURE
        assert lv.shr(a, lv.const(2)).kind == lv.PURE

    def test_mixed_linear_only(self):
        m = lv.add(lv.uniform_unknown(), lv.pure(range(32)))
        assert m.kind == lv.MIXED
        assert lv.mul(m, lv.const(4)).kind == lv.MIXED       # linear: ok
        assert lv.shl(m, lv.const(2)).kind == lv.MIXED       # linear: ok
        assert lv.shr(m, lv.const(2)).kind == lv.VARYING     # nonlinear
        assert lv.xor(m, lv.const(8)).kind == lv.VARYING     # nonlinear

    def test_uniform_stays_uniform_through_nonlinear(self):
        u = lv.uniform_unknown()
        assert lv.shr(u, lv.const(3)).is_uniform
        assert lv.xor(u, u).is_uniform

    def test_lop3_xor_lut(self):
        a, b = lv.pure(range(32)), lv.pure([7] * 32)
        out = lv.lop3(a, b, lv.const(0), 0x3C)  # a XOR b
        assert out.vec == tuple(x ^ 7 for x in range(32))

    def test_join_uniform_shift(self):
        a = lv.pure([x * 4 for x in range(32)])
        b = lv.pure([x * 4 + 100 for x in range(32)])
        j = lv.join(a, b)
        assert j.kind == lv.MIXED  # same lane pattern, uniform offset
        c = lv.pure([x * 8 for x in range(32)])
        assert lv.join(a, c).kind == lv.VARYING


class TestBankMath:
    def test_column_stride_128(self):
        ways, _ = bank_conflict_ways(tuple(l * 128 for l in range(32)), 4)
        assert ways == 32

    def test_contiguous(self):
        ways, _ = bank_conflict_ways(tuple(l * 4 for l in range(32)), 4)
        assert ways == 1

    def test_padded_stride_132(self):
        ways, _ = bank_conflict_ways(tuple(l * 132 for l in range(32)), 4)
        assert ways == 1

    def test_broadcast_same_word(self):
        ways, bcast = bank_conflict_ways(tuple([64] * 32), 4)
        assert ways == 1 and bcast

    def test_contiguous_float4_is_clean(self):
        # 8-lane transaction groups each cover all 32 banks exactly once
        ways, _ = bank_conflict_ways(tuple(l * 16 for l in range(32)), 16)
        assert ways == 1

    def test_column_float4_is_8way(self):
        ways, _ = bank_conflict_ways(tuple(l * 128 for l in range(32)), 16)
        assert ways == 8

    def test_two_way(self):
        # stride 8 B: lanes l and l+16 share a bank at different words
        ways, _ = bank_conflict_ways(tuple(l * 8 for l in range(32)), 4)
        assert ways == 2

    def test_half_warp_repeat_is_broadcast_not_conflict(self):
        ways, bcast = bank_conflict_ways(tuple((l % 16) * 8 for l in range(32)), 4)
        assert ways == 1 and bcast


class TestSectorMath:
    def test_contiguous_float(self):
        worst, best = sector_count(tuple(l * 4 for l in range(32)), 4)
        assert best == 4 and worst == 5  # straddle when base not 32B-aligned

    def test_fully_strided(self):
        worst, best = sector_count(tuple(l * 128 for l in range(32)), 4)
        assert worst == best == 32


EXPECT = {
    "_Z12col_conflictPKfPfi": dict(conflict_ways=32, conflicted=True),
    "_Z12padded_cleanPKfPfi": dict(conflict_ways=1, conflicted=False),
    "_Z11xor_swizzlePKfPfi": dict(conflict_ways=1, conflicted=False),
    "_Z14broadcast_readPKfPfi": dict(conflict_ways=1, conflicted=False),
}


@pytest.mark.parametrize("arch", ["sm_90", "sm_120a"])
class TestEndToEnd:
    def _run(self, arch):
        dis = sass.parse_gi((REC / f"nvdisasm_gi.bank_conflict.{arch}.txt").read_text())
        return {name: analyze_accesses(f, (32, 1, 1)) for name, f in dis.functions.items()}

    def test_full_coverage(self, arch):
        for name, res in self._run(arch).items():
            assert res["unanalyzed_count"] == 0, (name, res["unanalyzed"][:2])

    def test_shared_verdicts(self, arch):
        results = self._run(arch)
        for name, exp in EXPECT.items():
            res = results[name]
            assert res["worst_bank_conflict_ways"] == exp["conflict_ways"], name
            assert (res["conflicted_shared_accesses"] > 0) == exp["conflicted"], name

    def test_strided_global_uncoalesced(self, arch):
        res = self._run(arch)["_Z14strided_globalPKfPfii"]
        bad = [a for a in res["accesses"] if a["verdict"] == "uncoalesced"]
        assert bad and all(a["sectors_worst"] == 32 for a in bad)
        assert all(a["efficiency_pct"] == 12.5 for a in bad)


@pytest.mark.parametrize("arch", ["sm_90", "sm_120a"])
class TestMatrixAndAsync:
    def _run(self, arch):
        dis = sass.parse_gi((REC / f"nvdisasm_gi.ldsm_async.{arch}.txt").read_text())
        return {n: analyze_accesses(f, (32, 1, 1)) for n, f in dis.functions.items()}

    def test_full_coverage(self, arch):
        for name, res in self._run(arch).items():
            assert res["unanalyzed_count"] == 0, (name, res["unanalyzed"][:2])

    def test_ldsm_verdicts(self, arch):
        results = self._run(arch)
        for name, res in results.items():
            ldsm = [a for a in res["accesses"] if a["opcode"].startswith("LDSM")]
            assert ldsm
            if "row_major" in name:
                assert all(a["conflict_ways"] == 8 for a in ldsm), name
            else:
                assert all(a["conflict_ways"] == 1 for a in ldsm), name

    def test_ldgsts_dual_analysis(self, arch):
        for name, res in self._run(arch).items():
            sides = {a["space"]: a["verdict"] for a in res["accesses"]
                     if a["opcode"].startswith("LDGSTS")}
            assert sides.get("global") == "coalesced", name
            assert sides.get("shared") == "clean", name
