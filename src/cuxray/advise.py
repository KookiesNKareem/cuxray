"""Synthesize a report into a ranked list of design actions.

Every other analysis pass produces facts; this merges them into concrete,
prioritized recommendations with an estimated impact and a confidence level
derived from the coverage each fact was computed under. It reads only the
report JSON — no toolchain access — so it is cheap and deterministic.
"""

from __future__ import annotations

from typing import Optional

# Impact is a rough ordering key (higher = address first), not a physical
# unit. It blends how much the fix can move a memory-bound decode kernel
# against how certain the underlying signal is.
_SEV = {"high": 3, "medium": 2, "low": 1}


def _confidence(access: Optional[dict]) -> tuple[str, list[str]]:
    """Confidence in the access/roofline-derived advice and the caveats."""
    if not access:
        return "unknown", ["no block shape — pass --threads for access analysis"]
    caveats = []
    conf = "high"
    if not access.get("dataflow_converged", True):
        conf, c = "low", "dataflow did not converge"
        caveats.append(c)
    unreached = access.get("unreached_blocks") or 0
    if unreached:
        conf = "low"
        caveats.append(f"{unreached} basic block(s) unreached")
    analyzed = access.get("analyzed_count", 0)
    unana = access.get("unanalyzed_count", 0)
    if analyzed + unana:
        cov = analyzed / (analyzed + unana)
        if cov < 0.5:
            conf = "low" if conf != "low" else conf
            caveats.append(f"only {cov:.0%} of accesses traced")
        elif cov < 0.9 and conf == "high":
            conf = "medium"
            caveats.append(f"{cov:.0%} of accesses traced")
    return conf, caveats


def advise(kernel: dict, arch: Optional[str] = None,
           weight: float = 1.0) -> list[dict]:
    """Ranked action list for one kernel report entry.

    `weight` (default 1.0) scales each finding's estimated impact — pass a
    kernel's share of end-to-end runtime so `impact` ranks findings by
    recoverable *whole-program* time, not just per-kernel severity."""
    actions: list[dict] = []
    r = kernel.get("resources") or {}
    occ = kernel.get("occupancy") or {}
    access = kernel.get("access")
    roofline = kernel.get("roofline") or []
    spills = kernel.get("spills") or {}
    conf, caveats = _confidence(access)

    def add(severity, title, detail, impact=0.0, **extra):
        actions.append({
            "severity": severity, "title": title, "detail": detail,
            "confidence": extra.pop("confidence", conf),
            "evidence": extra.pop("evidence", []),
            # impact = a rough recoverable-cost proxy (higher = more to gain),
            # scaled by the kernel's runtime weight; used for ranking
            "impact": round(impact * weight, 2),
            **extra,
        })

    # 1. Spills — always first; they are unconditionally bad and certain.
    if spills.get("store_instructions") or spills.get("load_instructions"):
        by_line = spills.get("by_line") or []
        where = ""
        if by_line and by_line[0].get("line"):
            f = (by_line[0].get("file") or "?").rsplit("/", 1)[-1]
            where = f" (hottest at {f}:{by_line[0]['line']})"
        # impact ~ local-memory bytes moved; spill loads on the hot path hurt
        sbytes = (spills.get("store_bytes", 0) or 0) + (spills.get("load_bytes", 0) or 0)
        add("high", "eliminate register spills",
            f"{spills['store_instructions']} spill stores / "
            f"{spills['load_instructions']} loads to local memory{where}; "
            "raise -maxrregcount or cut live state",
            impact=sbytes, confidence="high",
            evidence=["spill byte accounting (validated vs ptxas)"])

    # 2. Occupancy cliffs — a register cut that unlocks another block/SM.
    for c in occ.get("cliffs", []):
        if c.get("kind") == "gain":
            gain = c["occupancy_pct"] - occ.get("occupancy_pct", 0)
            add("high" if gain >= 25 else "medium",
                f"cut {c['resource']} to {c['at']} ({c['delta']:+})",
                f"unlocks {c['blocks_per_sm']} blocks/SM "
                f"({occ.get('occupancy_pct')}% → {c['occupancy_pct']}%); "
                f"current limiter is {occ.get('limiter')}",
                impact=gain * 4,   # occupancy points recovered, weighted
                confidence="high",
                evidence=["occupancy model (validated vs cuda_occupancy.h "
                          "+ runtime API)"])

    # 3. Bank conflicts — solve suggests the verified swizzle separately.
    if access and access.get("conflicted_shared_accesses"):
        ways = access.get("worst_bank_conflict_ways") or 1
        n = access["conflicted_shared_accesses"]
        add("high" if ways >= 4 else "medium",
            "remove shared-memory bank conflicts",
            f"{n} conflicted access(es), worst {ways}-way; run "
            "`cuxray solve` for a verified swizzle",
            impact=n * (ways - 1) * 8,   # extra wavefronts per access
            evidence=["lane-value bank model"])

    # 4. Uncoalesced global — restructure or stage.
    if access and access.get("uncoalesced_global_accesses"):
        n = access["uncoalesced_global_accesses"]
        add("medium", "coalesce or stage global accesses",
            f"{n} uncoalesced global access(es); reorder indexing or stage "
            "through shared memory",
            impact=n * 16, evidence=["sector model"])

    # 5. Block-invariant re-reads — grid-level traffic amplification.
    gt = kernel.get("grid_traffic")
    if gt and gt.get("invariant_fraction", 0) >= 0.2:
        add("medium", "stage block-invariant reads",
            f"{gt['invariant_fraction']:.0%} of loop traffic is re-read by "
            f"every block; L2 usually absorbs it but on a miss that portion "
            f"moves up to {gt['grid_blocks']}× — do more rows/block or "
            "__ldcs the streaming side to spare L2",
            impact=gt["invariant_fraction"] * 100,
            confidence="low" if conf == "high" else conf,
            evidence=["grid-level invariant-traffic accounting"])
    elif access and (access.get("block_invariant_read_bytes") or 0) >= 4096:
        add("low", "consider staging block-invariant reads",
            f"{access['block_invariant_read_bytes']} B/block of reads are "
            "block-invariant; pass --grid for amplification impact",
            impact=10, evidence=["block-invariant read detection"])

    # 6. Scheduler-per-byte bound (the campaign's key signal).
    for lp in roofline:
        pb = lp.get("est_stall_cycles_per_512B")
        if pb is not None and pb > 40:
            span = (f"lines {lp['line_span'][0]}-{lp['line_span'][1]}"
                    if lp.get("line_span") else lp.get("header", "loop"))
            add("medium", "loop is issue/latency-bound, not bandwidth-bound",
                f"{span}: ~{pb} stall cycles per 512 B streamed — the ALU/"
                "issue schedule, not DRAM, gates this loop; shorten dependency "
                "chains or raise ILP (independent accumulators)",
                impact=pb, confidence="medium",
                evidence=["embedded scheduler stall bits (sm_80-sm_90a)"])
            break

    # 7. Tensor-core headroom — the architecture-ceiling check. A SIMT kernel
    # can look flawless yet be capped: fine at batch 1, beaten by tensor cores
    # as arithmetic-per-byte grows. Surface that before it costs hardware time.
    xr = _crossover(kernel, arch)
    if xr:
        add("medium", "SIMT datapath caps this loop — tensor cores scale past it",
            xr["note"],
            impact=xr["tensor_speedup_ceiling"] * 5, confidence="medium",
            crossover=xr,
            evidence=["static op-mix + per-arch MAC-rate model (approximate)"])

    def sort_key(a):
        return (-_SEV.get(a["severity"], 0), -a.get("impact", 0),
                -_SEV.get(a["confidence"], 0))

    actions.sort(key=sort_key)
    return actions


def _crossover(kernel: dict, arch: Optional[str]) -> Optional[dict]:
    # loop_report attaches "tensor_crossover" to roofline rows when a spec was
    # available; deepest (innermost) loop first.
    for lp in kernel.get("roofline") or []:
        if lp.get("tensor_crossover"):
            return lp["tensor_crossover"]
    return None
