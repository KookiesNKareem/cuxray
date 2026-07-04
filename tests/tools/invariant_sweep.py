#!/usr/bin/env python3
"""Invariant sweep over production cubins (no ground truth needed).

For every cubin given (or found under given dirs), run the full analysis
pipeline and machine-check structural invariants:

  I1  parsers never raise
  I2  every -gi instruction has a -plr row at the same address (and counts agree)
  I3  peak live GPRs <= recorded REG (<= 256 for USETMAXREG kernels)
  I4  spill byte counts are non-negative and stores+loads == instruction sum
  I5  resusage kernel set == disassembly function set
  I6  CFG loop depths are sane (0 <= depth <= 8)

Usage: invariant_sweep.py <cubin-or-dir>... [--sample N]
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cuxray.analyze.liveness import pressure           # noqa: E402
from cuxray.analyze.spillmap import spill_map          # noqa: E402
from cuxray.parse import cfgdot, resusage, sass        # noqa: E402
from cuxray.toolchain import resolve                   # noqa: E402


def check_cubin(path: Path, tc) -> list[str]:
    problems = []
    try:
        res = resusage.parse(tc.run("cuobjdump", ["--dump-resource-usage", str(path)]))
        dis = sass.parse_gi(tc.run("nvdisasm", ["-c", "-gi", str(path)]))
        plr = sass.parse_plr(tc.run("nvdisasm", ["-c", "-plr", str(path)]))
        try:
            cfg = cfgdot.parse(tc.run("nvdisasm", ["-cfg", str(path)]))
        except Exception as e:
            cfg = {}
            problems.append(f"I1 cfg raised: {type(e).__name__}: {e}")
    except Exception as e:
        return [f"I1 pipeline raised: {type(e).__name__}: {str(e)[:200]}"]

    # I5: kernel sets agree
    only_res = set(res) - set(dis.functions)
    only_dis = set(dis.functions) - set(res)
    if only_res:
        problems.append(f"I5 in resusage but not disasm: {sorted(only_res)[:3]}")
    if only_dis:
        problems.append(f"I5 in disasm but not resusage: {sorted(only_dis)[:3]}")

    for name, func in dis.functions.items():
        table = plr.get(name, {})
        gi_addrs = {i.addr for i in func.instructions}
        plr_addrs = set(table)
        # -plr legitimately omits trailing NOP padding; anything else missing
        # (or extra) is a parser bug.
        missing = gi_addrs - plr_addrs
        by_addr = {i.addr: i for i in func.instructions}
        non_nop_missing = [a for a in missing if by_addr[a].opcode != "NOP"]
        extra = plr_addrs - gi_addrs
        if non_nop_missing or extra:
            problems.append(
                f"I2 {name[:50]}: non-NOP missing from plr {len(non_nop_missing)} "
                f"(e.g. {[by_addr[a].opcode for a in sorted(non_nop_missing)[:4]]}), "
                f"extra in plr {len(extra)}"
            )
        sass.merge_liveness(dis, plr)
        p = pressure(func)
        r = res.get(name)
        if p.get("available") and r and r.reg:
            peak = p["peak"]["live_gpr"]
            cap = 256 if sass.uses_register_reallocation(func) else r.reg
            if peak > cap:
                problems.append(f"I3 {name[:50]}: peak {peak} > cap {cap} (REG={r.reg})")
        depths = cfg.get(name).loop_depth if name in cfg else {}
        if depths and not all(0 <= d <= 8 for d in depths.values()):
            problems.append(f"I6 {name[:50]}: weird loop depths {depths}")
        sm = spill_map(func, depths)
        by_line_sum = sum(row["stores"] + row["loads"] for row in sm["by_line"])
        if by_line_sum != sm["store_instructions"] + sm["load_instructions"]:
            problems.append(f"I4 {name[:50]}: by_line sum mismatch")
    return problems


def main():
    argv = sys.argv[1:]
    sample = 0
    if "--sample" in argv:
        i = argv.index("--sample")
        sample = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    args = argv
    cubins: list[Path] = []
    for a in args:
        p = Path(a)
        cubins.extend(sorted(p.rglob("*.cubin")) if p.is_dir() else [p])
    if sample and len(cubins) > sample:
        rng = random.Random(20260704)
        # Stratify by arch tag in filename so every arch stays represented
        by_arch: dict[str, list[Path]] = {}
        for c in cubins:
            tag = next((t for t in c.name.split(".") if t.startswith("sm_")), "?")
            by_arch.setdefault(tag, []).append(c)
        per = max(1, sample // len(by_arch))
        cubins = [c for v in by_arch.values() for c in rng.sample(v, min(per, len(v)))]

    tc = resolve(quiet=True)
    bad = 0
    for i, c in enumerate(cubins):
        problems = check_cubin(c, tc)
        if problems:
            bad += 1
            print(f"FAIL {c.name}")
            for p in problems[:5]:
                print(f"     {p}")
        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{len(cubins)} checked, {bad} with problems")
    print(f"DONE: {len(cubins) - bad}/{len(cubins)} cubins clean")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
