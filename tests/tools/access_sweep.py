#!/usr/bin/env python3
"""Layer B production sweep: coverage + robustness over real cubins.

For every kernel in the sampled cubins, run the access analysis with a probe
block shape and report:
  - crash-free rate (must be 100%)
  - coverage: analyzed vs unanalyzable accesses, and why
  - flagged conflicts (candidates for manual review — production CUTLASS
    kernels are usually swizzled clean, so a flood of 32-way flags would
    smell like false positives)

Usage: access_sweep.py <dir-or-cubin>... [--sample N] [--threads X[,Y]]
"""

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cuxray.analyze.access import analyze_accesses      # noqa: E402
from cuxray.parse import cfgdot, sass                   # noqa: E402
from cuxray.toolchain import resolve                    # noqa: E402


def main():
    argv = sys.argv[1:]
    sample, dims = 0, (128, 1, 1)
    if "--sample" in argv:
        i = argv.index("--sample")
        sample = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if "--threads" in argv:
        i = argv.index("--threads")
        parts = [int(p) for p in argv[i + 1].split(",")] + [1, 1]
        dims = (parts[0], parts[1], parts[2])
        argv = argv[:i] + argv[i + 2:]

    cubins: list[Path] = []
    for a in argv:
        p = Path(a)
        cubins.extend(sorted(p.rglob("*.cubin")) if p.is_dir() else [p])
    if sample and len(cubins) > sample:
        rng = random.Random(20260705)
        by_arch: dict[str, list[Path]] = {}
        for c in cubins:
            tag = next((t for t in c.name.split(".") if t.startswith("sm_")), "?")
            by_arch.setdefault(tag, []).append(c)
        per = max(1, sample // len(by_arch))
        cubins = [c for v in by_arch.values() for c in rng.sample(v, min(per, len(v)))]

    tc = resolve(quiet=True)
    crashes, kernels = 0, 0
    verdicts: Counter = Counter()
    reasons: Counter = Counter()
    conflicts: list[tuple] = []

    for ci, cub in enumerate(cubins):
        try:
            dis = sass.parse_gi(tc.run("nvdisasm", ["-c", "-gi", str(cub)]))
            try:
                cfg = cfgdot.parse(tc.run("nvdisasm", ["-cfg", str(cub)]))
            except Exception:
                cfg = {}
            for name, func in dis.functions.items():
                kernels += 1
                depths = cfg[name].loop_depth if name in cfg else {}
                res = analyze_accesses(func, dims, depths)
                for a in res["accesses"]:
                    verdicts[(a["space"], a["verdict"])] += 1
                for r, n in res["unanalyzed_by_reason"].items():
                    reasons[r.split("—")[0].strip()[:52]] += n
                if res["worst_bank_conflict_ways"] > 2:
                    conflicts.append((cub.name, name[:60],
                                      res["worst_bank_conflict_ways"]))
        except Exception as e:
            crashes += 1
            print(f"CRASH {cub.name}: {type(e).__name__}: {str(e)[:120]}")
        if (ci + 1) % 10 == 0:
            print(f"  ... {ci + 1}/{len(cubins)} cubins")

    total = sum(verdicts.values()) + sum(reasons.values())
    print(f"\n{len(cubins)} cubins, {kernels} kernels, {crashes} crashes")
    print(f"{total} memory accesses seen; "
          f"{sum(verdicts.values())} analyzed ({100 * sum(verdicts.values()) / max(total, 1):.1f}%)")
    for k, n in verdicts.most_common():
        print(f"  {k[0]:>7}/{k[1]:<12} {n}")
    print("unanalyzed, by reason:")
    for r, n in reasons.most_common(8):
        print(f"  {n:>7}  {r}")
    print(f"\nkernels flagged >2-way bank conflict: {len(conflicts)}")
    for c in conflicts[:12]:
        print(f"  {c[2]:>2}-way  {c[0][:40]}  {c[1]}")
    sys.exit(1 if crashes else 0)


if __name__ == "__main__":
    main()
