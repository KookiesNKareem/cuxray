"""Layer B tests: lane-value domain, bank/sector math, and end-to-end
verdicts on the compiled bank_conflict fixtures (both arches)."""

from pathlib import Path

import pytest

from cuxray.analyze import lanevalue as lv
from cuxray.analyze.access import (analyze_accesses, bank_conflict_ways,
                                   ideal_sectors, sector_count)
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


class TestAlignmentTracking:
    """MIXED values whose unknown-uniform part U satisfies U % 2**a == 0
    stay traceable through the shr/and/xor chains XOR swizzles compile to."""

    def _loop_index(self):
        # c = lane; loop back-edge c += 32 — the classic strided induction
        c0 = lv.pure(range(32))
        return lv.join(c0, lv.add(c0, lv.const(32)))

    def test_join_records_alignment(self):
        c = self._loop_index()
        assert c.kind == lv.MIXED and c.ualign == 5

    def test_linear_ops_scale_alignment(self):
        c = self._loop_index()
        assert lv.mul(c, lv.const(32)).ualign == 10
        assert lv.shl(c, lv.const(4)).ualign == 9
        assert lv.add(c, c).ualign == 5

    def test_shr_splits_below_alignment(self):
        va = lv.shr(lv.mul(self._loop_index(), lv.const(32)), lv.const(4))
        assert va.kind == lv.MIXED and va.ualign == 6
        assert va.vec == tuple(2 * l for l in range(32))

    def test_shr_past_alignment_degrades(self):
        c = self._loop_index()
        assert lv.shr(c, lv.const(6)).kind == lv.VARYING

    def test_and_mask_below_alignment_is_pure(self):
        bit = lv.and_(lv.shr(self._loop_index(), lv.const(3)), lv.const(1))
        assert bit.kind == lv.PURE
        assert bit.vec == tuple((l >> 3) & 1 for l in range(32))

    def test_xor_swizzle_stays_traceable(self):
        # gemv_v7's Swizzle<1,4,3> in int4-index form: v ^ ((v >> 3) & 1)
        va = lv.shr(lv.mul(self._loop_index(), lv.const(32)), lv.const(4))
        sw = lv.xor(va, lv.and_(lv.shr(va, lv.const(3)), lv.const(1)))
        assert sw.kind == lv.MIXED
        assert sw.vec == tuple(v ^ ((v >> 3) & 1) for v in (2 * l for l in range(32)))

    def test_lop3_through_aligned_mixed(self):
        va = lv.shr(lv.mul(self._loop_index(), lv.const(32)), lv.const(4))
        bit = lv.pure([((2 * l) >> 3) & 1 for l in range(32)])
        out = lv.lop3(va, bit, lv.const(0), 0x3C)  # a XOR b
        assert out.kind == lv.MIXED
        assert out.vec == tuple(v ^ ((v >> 3) & 1) for v in (2 * l for l in range(32)))

    def test_lop3_fused_or_and_as_emitted(self):
        # gemv_v7's swizzle as ptxas emits it: LOP3 0xf8 = a | (b & c) with
        # a = va (MIXED, align 6) and c = va >> 3 (MIXED, align 3)
        va = lv.shr(lv.mul(self._loop_index(), lv.const(32)), lv.const(4))
        out = lv.lop3(va, lv.const(1), lv.shr(va, lv.const(3)), 0xF8)
        assert out.kind == lv.MIXED
        assert out.vec == tuple(v | (1 & (v >> 3)) for v in (2 * l for l in range(32)))

    def test_zero_alignment_unchanged(self):
        m = lv.add(lv.uniform_unknown(), lv.pure(range(32)))
        assert lv.shr(m, lv.const(2)).kind == lv.VARYING
        assert lv.xor(m, lv.const(8)).kind == lv.VARYING


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

    def test_partial_broadcast_efficiency_capped(self):
        # 8 lanes per word (e.g. xscale[k/128]): footprint is 32 B → 1 ideal
        # sector; efficiency must never exceed 100%
        vec = tuple((l // 4) * 4 for l in range(32))
        ideal = ideal_sectors(vec, 4)
        worst, _ = sector_count(vec, 4)
        assert ideal == 1 and worst >= ideal

    def test_distinct_lanes_ideal_unchanged(self):
        assert ideal_sectors(tuple(l * 4 for l in range(32)), 4) == 4
        assert ideal_sectors(tuple(l * 16 for l in range(32)), 16) == 16


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


class TestSolver:
    def _patterns(self, arch="sm_120a"):
        from cuxray.analyze.solver import patterns_from_accesses
        dis = sass.parse_gi((REC / f"nvdisasm_gi.bank_conflict.{arch}.txt").read_text())
        res = analyze_accesses(dis.functions["_Z12col_conflictPKfPfi"], (32, 1, 1),
                               keep_vecs=True)
        return patterns_from_accesses(res["accesses"])

    def test_solver_fixes_col_conflict_jointly(self):
        from cuxray.analyze.solver import solve
        pats = self._patterns()
        assert any(p.ways_before == 32 for p in pats)   # column reads
        assert any(p.ways_before == 1 for p in pats)    # row writes must STAY clean
        sols = solve(pats)
        assert sols, "no joint swizzle found for the classic case"
        best = sols[0]
        assert all(pp["after"] == 1 for pp in best.per_pattern)
        assert "Swizzle<" in best.cutlass and "addr ^" in best.formula

    def test_solution_verifies_under_direct_simulation(self):
        # Re-verify the returned swizzle independently against the bank model
        from cuxray.analyze.solver import apply_swizzle, solve
        pats = self._patterns()
        best = solve(pats)[0]
        for p in pats:
            vec = tuple(apply_swizzle(a, best.b, best.m, best.s) for a in p.vec)
            ways, _ = bank_conflict_ways(vec, p.width)
            assert ways == 1

    def test_no_patterns_no_solutions(self):
        from cuxray.analyze.solver import solve
        assert solve([]) == []

    def test_min_granule_respects_width(self):
        # 16B accesses must not be swizzled below 16B granularity (m >= 4)
        from cuxray.analyze.solver import Pattern, solve
        pats = [Pattern(vec=tuple(l * 128 for l in range(32)), width=16)]
        for sol in solve(pats):
            assert sol.m >= 4

    def test_solution_emits_appliable_code(self):
        from cuxray.analyze.solver import solve
        best = solve(self._patterns())[0]
        snip = best.cuda_snippet()
        assert "__device__" in snip and "byte_off ^" in snip
        assert best.cute_type.startswith("cute::Swizzle<")

    def test_grouped_global_when_one_swizzle_suffices(self):
        from cuxray.analyze.solver import solve_grouped
        g = solve_grouped(self._patterns())
        assert g["global"], "classic case has a single global swizzle"
        assert len(g["groups"]) == 1 and not g["unsolved"]

    def test_grouped_splits_independent_tiles(self):
        # two tiles needing different swizzles: a stride-8B tile (2-way at
        # width 4) and a column tile (32-way). No single swizzle cleans both
        # granularities, but each is solvable alone.
        from cuxray.analyze.solver import Pattern, solve_grouped
        tile_a = Pattern(vec=tuple(l * 8 for l in range(32)), width=4,
                         ways_before=2, site="k.cu:10")
        tile_b = Pattern(vec=tuple(l * 128 for l in range(32)), width=4,
                         ways_before=32, site="k.cu:20")
        g = solve_grouped([tile_a, tile_b])
        if g["global"]:
            return  # a shared swizzle is even better; nothing to assert
        sites = {s for grp in g["groups"] for s in grp["sites"]}
        assert sites == {"k.cu:10", "k.cu:20"}
        for grp in g["groups"]:
            assert grp["solutions"], grp["sites"]


class TestBlockInvariance:
    def test_x_reads_flagged_invariant_weight_reads_not(self):
        # col_conflict: x[j*N+tx] has no blockIdx term → block-invariant;
        # strided_global: x[i*32+it] with i = ctaid*bd+tid → block-dependent
        dis = sass.parse_gi((REC / "nvdisasm_gi.bank_conflict.sm_120a.txt").read_text())
        res = analyze_accesses(dis.functions["_Z12col_conflictPKfPfi"], (32, 1, 1))
        inv = [a for a in res["accesses"]
               if a["space"] == "global" and a.get("block_invariant")]
        assert inv, "block-invariant x reads not detected"
        assert res["block_invariant_read_bytes"] > 0

        res2 = analyze_accesses(dis.functions["_Z14strided_globalPKfPfii"], (32, 1, 1))
        loads = [a for a in res2["accesses"]
                 if a["space"] == "global" and a["opcode"].startswith("LDG")]
        assert loads and not any(a.get("block_invariant") for a in loads)


class TestDataflowStep:
    """Direct step() semantics for patterns real compilers emit."""

    def _tids(self):
        return lv.tid_vectors((32, 1, 1))

    def test_predicated_first_write_is_strong(self):
        from cuxray.analyze.dataflow import State, step
        st = State()
        step(sass.Instruction(0, "MOV", "R8, 0x4", predicate="@!P0"), st, self._tids())
        assert st.get("R8").kind == lv.PURE  # uninit old value must not poison
        step(sass.Instruction(16, "MOV", "R8, 0x8", predicate="@P0"), st, self._tids())
        assert st.get("R8").kind == lv.MIXED  # overwrite still joins

    def test_lea_hi_signed_div_idiom(self):
        from cuxray.analyze.dataflow import State, step
        st = State()
        st.set("R4", lv.pure(range(32)))
        step(sass.Instruction(0, "LEA.HI", "R10, R4, R4, RZ, 0x1"), st, self._tids())
        # x + (x >> 31) == x for non-negative lane values
        assert st.get("R10").vec == tuple(range(32))

    def test_shf_r_hi_source_in_high_word(self):
        from cuxray.analyze.dataflow import State, step
        st = State()
        st.set("R28", lv.pure(range(0, 64, 2)))
        step(sass.Instruction(0, "SHF.R.U32.HI", "R32, RZ, 0x3, R28"), st, self._tids())
        assert st.get("R32").vec == tuple((2 * l) >> 3 for l in range(32))

    def test_laneid_exact_unknown_sr_uniform(self):
        from cuxray.analyze.dataflow import State, step
        st = State()
        step(sass.Instruction(0, "S2R", "R0, SR_LANEID"), st, self._tids())
        assert st.get("R0").vec == tuple(range(32))
        step(sass.Instruction(16, "S2UR", "UR5, SR_CgaCtaId"), st, self._tids())
        assert st.get("UR5").is_uniform

    def test_unmodeled_op_uniform_inputs_uniform(self):
        from cuxray.analyze.dataflow import State, step
        st = State()
        st.set("R7", lv.uniform_unknown())
        step(sass.Instruction(0, "IMAD.HI.U32", "R4, R7, R7, RZ"), st, self._tids())
        assert st.get("R4").is_uniform

    def test_single_path_definition_survives_merge(self):
        from cuxray.analyze.dataflow import State
        a = State({"R4": lv.pure(range(32))})
        b = State({})
        a.join_with(b)
        assert a.get("R4").kind == lv.PURE

    def test_prmt_exact_byte_permute(self):
        from cuxray.analyze.dataflow import State, step
        st = State()
        st.set("R2", lv.pure(range(32)))          # bytes: [l,0,0,0]
        st.set("R3", lv.pure([0x11223344] * 32))
        step(sass.Instruction(0, "PRMT", "R4, R2, 0x5410, R3"), st, self._tids())
        # 0x5410: bytes 0,1 from R2, bytes 4,5 (R3 low half) above
        assert st.get("R4").vec == tuple(l | (0x3344 << 16) for l in range(32))

    def test_grid_dims_resolve_singleton_ctaid(self):
        from cuxray.analyze.dataflow import analyze_ex
        fn = sass.Function("k", [
            sass.Instruction(0, "S2R", "R0, SR_CTAID.Y", block="k"),
            sass.Instruction(16, "S2R", "R1, SR_CTAID.X", block="k"),
        ])
        pre, _ = analyze_ex(fn, (32, 1, 1), grid_dims=(128, 1, 1))
        st = pre[16].copy()
        from cuxray.analyze.dataflow import step
        step(fn.instructions[1], st, lv.tid_vectors((32, 1, 1)))
        assert pre[16].get("R0").is_scalar          # gridDim.y == 1 → const 0
        assert not pre[16].get("R0").ctaid
        assert st.get("R1").ctaid                   # gridDim.x > 1 → tainted
