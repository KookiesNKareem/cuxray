"""Ranking and confidence behavior of the advise synthesizer."""

from cuxray.advise import advise


def _kernel(**over):
    k = {
        "resources": {"regs": 40},
        "spills": {"store_instructions": 0, "load_instructions": 0,
                   "store_bytes": 0, "load_bytes": 0, "by_line": []},
        "occupancy": {"occupancy_pct": 100.0, "limiter": "warps",
                      "threads_per_block": 256, "cliffs": []},
        "access": {"analyzed_count": 10, "unanalyzed_count": 0,
                   "dataflow_converged": True, "unreached_blocks": 0,
                   "conflicted_shared_accesses": 0,
                   "uncoalesced_global_accesses": 0,
                   "block_invariant_read_bytes": 0},
        "roofline": [],
    }
    k.update(over)
    return k


def test_clean_kernel_no_actions():
    assert advise(_kernel()) == []


def test_spills_rank_first_and_high_confidence():
    k = _kernel(spills={"store_instructions": 4, "load_instructions": 2,
                        "store_bytes": 16, "load_bytes": 8,
                        "by_line": [{"file": "k.cu", "line": 12, "loop_depth": 1}]},
                occupancy={"occupancy_pct": 50.0, "limiter": "registers",
                           "threads_per_block": 256,
                           "cliffs": [{"kind": "gain", "resource": "registers",
                                       "at": 40, "delta": -8, "blocks_per_sm": 3,
                                       "occupancy_pct": 75.0}]})
    actions = advise(k)
    assert actions[0]["title"].startswith("eliminate register spills")
    assert actions[0]["confidence"] == "high"
    # the register-cut cliff is present but ranks after spills
    assert any("cut registers" in a["title"] for a in actions)


def test_confidence_degrades_with_coverage():
    k = _kernel(access={"analyzed_count": 2, "unanalyzed_count": 8,
                        "dataflow_converged": False, "unreached_blocks": 1,
                        "conflicted_shared_accesses": 1,
                        "worst_bank_conflict_ways": 4,
                        "uncoalesced_global_accesses": 0,
                        "block_invariant_read_bytes": 0})
    actions = advise(k)
    bank = next(a for a in actions if "bank conflict" in a["title"])
    assert bank["confidence"] == "low"


def test_issue_bound_loop_flagged():
    k = _kernel(roofline=[{"loop_depth": 1, "line_span": [5, 20],
                           "est_stall_cycles_per_512B": 90,
                           "est_global_bytes_per_warp_iter": 512}])
    actions = advise(k)
    assert any("issue" in a["title"] or "latency-bound" in a["title"]
               for a in actions)


def test_grid_invariant_action_uses_fraction():
    k = _kernel(grid_traffic={"grid_blocks": 4096, "invariant_fraction": 0.47,
                              "invariant_bytes_per_warp_iter": 512,
                              "worst_amplification": 1925.0})
    actions = advise(k)
    stage = next(a for a in actions if "block-invariant" in a["title"])
    assert "47%" in stage["detail"] and "4096" in stage["detail"]


def test_high_severity_before_medium():
    k = _kernel(
        occupancy={"occupancy_pct": 33.0, "limiter": "registers",
                   "threads_per_block": 256,
                   "cliffs": [{"kind": "gain", "resource": "registers",
                               "at": 40, "delta": -20, "blocks_per_sm": 2,
                               "occupancy_pct": 66.0}]},
        roofline=[{"loop_depth": 1, "line_span": [1, 5],
                   "est_stall_cycles_per_512B": 90,
                   "est_global_bytes_per_warp_iter": 512}])
    sevs = [a["severity"] for a in advise(k)]
    assert sevs == sorted(sevs, key=lambda s: {"high": 0, "medium": 1, "low": 2}[s])
