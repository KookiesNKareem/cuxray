#!/usr/bin/env python3
"""Generate the occupancy differential-reference fixture.

Runs NVIDIA's cuda_occupancy.h (via occ_harness) over a deterministic sweep of
configs for every supported architecture and writes
tests/fixtures/recorded/occ_reference.csv with both inputs and reference
outputs. test_occupancy_differential.py then asserts cuxray's Python engine
reproduces every row — no toolchain needed at test time.

Usage: gen_occ_reference.py <path-to-occ_harness-binary>
"""

import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cuxray.archspec import SPECS  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "recorded" / "occ_reference.csv"


def configs():
    rng = random.Random(20260704)  # deterministic fixture
    for (major, minor), spec in sorted(SPECS.items()):
        smem_max_block = spec.smem_per_block_max
        cases = []
        # Edge grid
        for threads in (32, 64, 128, 256, 512, 1024):
            for regs in (0, 16, 32, 40, 64, 96, 128, 168, 255):
                cases.append((threads, regs, 0, 0))
        # Smem edges (static and dynamic)
        for smem in (1, 48 * 1024, smem_max_block, smem_max_block - 128):
            cases.append((128, 32, smem, 0))
            cases.append((128, 32, 0, smem))
        # Random sweep
        for _ in range(300):
            threads = rng.randrange(32, 1025, 32)
            regs = rng.randint(0, 255)
            smem_s = rng.choice([0, 0, rng.randint(0, smem_max_block)])
            smem_d = rng.choice([0, 0, rng.randint(0, smem_max_block - smem_s)])
            cases.append((threads, regs, smem_s, smem_d))
        for threads, regs, smem_s, smem_d in cases:
            yield (major, minor, spec.max_threads_per_sm, spec.smem_per_sm,
                   spec.smem_per_block_max, spec.smem_reserved_per_block,
                   threads, regs, smem_s, smem_d)


def main():
    harness = sys.argv[1]
    rows = list(configs())
    stdin = "\n".join(" ".join(map(str, r)) for r in rows) + "\n"
    proc = subprocess.run([harness], input=stdin, capture_output=True, text=True,
                          check=True)
    outs = proc.stdout.strip().splitlines()
    assert len(outs) == len(rows), (len(outs), len(rows))
    with open(OUT, "w") as f:
        f.write("major,minor,maxThrSM,smemSM,smemBlkOptin,reserved,threads,regs,"
                "smemStatic,smemDyn,err,activeBlocks,limRegs,limSmem,limWarps,"
                "limBlocks,allocRegs,allocSmem\n")
        for r, o in zip(rows, outs):
            f.write(",".join(map(str, r)) + "," + ",".join(o.split()) + "\n")
    print(f"wrote {len(rows)} reference rows to {OUT}")


if __name__ == "__main__":
    main()
