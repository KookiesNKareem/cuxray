"""Occupancy engine tests.

Expected values are hand-derived from the cuda_occupancy.h algorithm
(cuda_cudart 13.3.29) and Programming Guide Table 27. TODO(gpu-box): add a
cross-check test against NCU's ncu_occupancy Python module on real hardware.
"""

from cuxray.archspec import lookup
from cuxray.occupancy import compute, find_cliffs, sweep_block_sizes


def test_lookup_aliases():
    assert lookup("sm_120").cc == (12, 0)
    assert lookup("sm_120a").cc == (12, 0)
    assert lookup("120").cc == (12, 0)
    assert lookup("12.0").cc == (12, 0)
    assert lookup((9, 0)).name == "Hopper"
    assert lookup("sm_90").max_warps_per_sm == 64
    assert lookup("sm_120").max_warps_per_sm == 48
    assert lookup("sm_120").max_blocks_per_sm == 24


def test_saxpy_like_sm90_warp_limited():
    # 16 regs, 256 threads, no smem on Hopper:
    # regs/warp alloc = round_up(512, 256) = 512; warps/subpartition = 16384//512 = 32
    # → 128 warps by regs → 16 blocks; warp slots: 64//8 = 8 blocks → warp-limited, 100%
    o = compute(lookup("sm_90"), regs_per_thread=16, threads_per_block=256)
    assert o.blocks_per_sm == 8
    assert o.limiter == "warps"
    assert o.occupancy_pct == 100.0


def test_moe_like_sm120_register_limited():
    # 168 regs, 256 threads on sm_120:
    # regs/warp = round_up(168*32=5376, 256) = 5376; warps/sub = 16384//5376 = 3
    # → 12 warps/SM by regs → 12//8 = 1 block → 8/48 warps = 16.7%
    o = compute(lookup("sm_120"), regs_per_thread=168, threads_per_block=256)
    assert o.blocks_per_sm == 1
    assert o.limiter == "registers"
    assert o.active_warps == 8
    assert o.occupancy_pct == 16.7


def test_classic_calculator_case_sm86():
    # 37 regs, 128 threads, 8 KB smem on 8.6:
    # regs: round_up(37*32=1184,256)=1280; 16384//1280=12 warps/sub → 48 warps → 12 blocks
    # warps: 48//4 = 12 blocks; block slots: 16
    # smem: round_up(8192+1024,128)=9216; 102400//9216 = 11 → smem-limited
    o = compute(lookup("sm_86"), regs_per_thread=37, threads_per_block=128,
                smem_static=8192)
    assert o.limits["registers"] == 12
    assert o.limits["warps"] == 12
    assert o.limits["shared_memory"] == 11
    assert o.blocks_per_sm == 11
    assert o.limiter == "shared_memory"
    assert o.active_warps == 44


def test_zero_regs_zero_smem_block_slot_limited():
    o = compute(lookup("sm_120"), regs_per_thread=0, threads_per_block=32)
    # 48 warp slots / 1 warp = 48 by warps, but only 24 block slots
    assert o.blocks_per_sm == 24
    assert o.limiter == "blocks"


def test_oversized_block_is_zero():
    o = compute(lookup("sm_120"), regs_per_thread=32, threads_per_block=2048)
    assert o.blocks_per_sm == 0


def test_regs_over_255_cannot_launch():
    o = compute(lookup("sm_120"), regs_per_thread=256, threads_per_block=128)
    assert o.blocks_per_sm == 0


def test_smem_over_block_max_cannot_launch():
    o = compute(lookup("sm_120"), regs_per_thread=32, threads_per_block=128,
                smem_dynamic=100 * 1024)
    assert o.blocks_per_sm == 0
    assert any("cannot launch" in n for n in o.notes)


def test_smem_optin_note():
    o = compute(lookup("sm_120"), regs_per_thread=32, threads_per_block=128,
                smem_dynamic=64 * 1024)
    assert any("opt-in" in n for n in o.notes)


def test_max_smem_block_fits_exactly_one():
    # 99 KB usage + 1 KB reserved = 100 KB = full sm_120 carveout → exactly 1 block
    o = compute(lookup("sm_120"), regs_per_thread=32, threads_per_block=128,
                smem_dynamic=99 * 1024)
    assert o.limits["shared_memory"] == 1
    assert o.blocks_per_sm == 1


def test_register_cliff_gain_detected():
    # At 168 regs/256 threads on sm_120 (1 block), dropping to 128 regs:
    # round_up(128*32,256)=4096; 16384//4096=4 warps/sub → 16 warps → 2 blocks.
    base = compute(lookup("sm_120"), regs_per_thread=168, threads_per_block=256)
    cliffs = find_cliffs(lookup("sm_120"), base)
    gains = [c for c in cliffs if c["kind"] == "gain" and c["resource"] == "registers"]
    assert gains, cliffs
    g = gains[0]
    assert g["blocks_per_sm"] == 2
    # highest reg count that yields 2 blocks: warps/sub must be >= 4 →
    # regs/warp <= 4096 → regs <= 128
    assert g["at"] == 128


def test_sweep_shapes():
    res = sweep_block_sizes(lookup("sm_90"), regs_per_thread=32)
    assert len(res) == 32  # 32..1024 step 32
    best = max(r.occupancy_pct for r in res)
    assert best == 100.0


def test_carveout_alignment_note():
    # Ask for 32 KB carveout with a 60 KB block → aligned up to 64 KB
    o = compute(lookup("sm_120"), regs_per_thread=32, threads_per_block=128,
                smem_dynamic=60 * 1024, carveout_kb=32)
    assert any("aligned up" in n for n in o.notes)
    assert o.limits["shared_memory"] == 1
