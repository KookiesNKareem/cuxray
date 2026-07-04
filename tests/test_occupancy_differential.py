"""Differential test: cuxray's occupancy engine vs NVIDIA's reference.

tests/fixtures/recorded/occ_reference.csv holds 4344 configs (all supported
architectures × edge grid + deterministic random sweep) evaluated by NVIDIA's
own cuda_occupancy.h (13.3.29) via tests/tools/occ_harness.cpp. The Python
engine must reproduce activeBlocksPerMultiprocessor for every row.

Regenerate with tests/tools/gen_occ_reference.py after bumping toolkits.
"""

import csv
from pathlib import Path

import pytest

from cuxray.archspec import lookup
from cuxray.occupancy import compute

REF = Path(__file__).parent / "fixtures" / "recorded" / "occ_reference.csv"


def load_rows():
    with open(REF) as f:
        return list(csv.DictReader(f))


ROWS = load_rows()


def test_reference_fixture_is_substantial():
    assert len(ROWS) > 4000
    assert len({(r["major"], r["minor"]) for r in ROWS}) == 12  # all supported CCs


def test_engine_matches_nvidia_reference():
    mismatches = []
    for r in ROWS:
        spec = lookup((int(r["major"]), int(r["minor"])))
        occ = compute(
            spec,
            regs_per_thread=int(r["regs"]),
            threads_per_block=int(r["threads"]),
            smem_static=int(r["smemStatic"]),
            smem_dynamic=int(r["smemDyn"]),
        )
        expected = int(r["activeBlocks"]) if int(r["err"]) == 0 else 0
        if occ.blocks_per_sm != expected:
            mismatches.append((
                f"sm_{r['major']}{r['minor']}", r["threads"], r["regs"],
                r["smemStatic"], r["smemDyn"],
                f"ref={expected}", f"cuxray={occ.blocks_per_sm}",
                f"ref_limits regs={r['limRegs']} smem={r['limSmem']} "
                f"warps={r['limWarps']} blocks={r['limBlocks']}",
                f"our_limits={occ.limits}",
            ))
    assert not mismatches, (
        f"{len(mismatches)}/{len(ROWS)} mismatches; first 10:\n"
        + "\n".join(str(m) for m in mismatches[:10])
    )


def test_allocated_regs_match_reference():
    bad = []
    for r in ROWS:
        if int(r["err"]) != 0 or int(r["regs"]) == 0:
            continue
        spec = lookup((int(r["major"]), int(r["minor"])))
        occ = compute(spec, int(r["regs"]), int(r["threads"]),
                      int(r["smemStatic"]), int(r["smemDyn"]))
        if occ.blocks_per_sm > 0 and occ.regs_allocated_per_block != int(r["allocRegs"]):
            bad.append((r["major"], r["minor"], r["regs"], r["threads"],
                        r["allocRegs"], occ.regs_allocated_per_block))
    assert not bad, f"{len(bad)} allocation mismatches; first 5: {bad[:5]}"
