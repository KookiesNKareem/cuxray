"""Register-cap tuning via recompilation.

Recompiles a PTX kernel with ptxas at a ladder of --maxrregcount values and
analyzes each resulting cubin (registers, spills, occupancy). No GPU is
involved; spill behavior under a cap requires re-allocation and cannot be
derived from a single binary.
"""

from __future__ import annotations

import concurrent.futures as cf
import re
import tempfile
from pathlib import Path
from typing import Optional

from .analyze.spillmap import spill_map
from .archspec import lookup
from .occupancy import compute
from .parse import resusage, sass
from .toolchain import Toolchain

DEFAULT_CAPS = (24, 32, 40, 48, 64, 80, 96, 128, 168, 255)

_PTX_TARGET = re.compile(r"^\s*\.target\s+(sm_\w+)", re.M)


def _analyze_cap(ptx: Path, arch: str, cap: Optional[int], tc: Toolchain,
                 threads: Optional[int], smem_dynamic: int, workdir: Path) -> dict:
    out = workdir / f"cap_{cap or 'none'}.cubin"
    args = ["--gpu-name", arch, "--generate-line-info", "-o", str(out), str(ptx)]
    if cap:
        args = ["--maxrregcount", str(cap)] + args
    tc.run("ptxas", args)

    res = resusage.parse(tc.run("cuobjdump", ["--dump-resource-usage", str(out)]))
    dis = sass.parse_gi(tc.run("nvdisasm", ["-c", "-gi", str(out)]))

    rows = []
    for name, func in dis.functions.items():
        r = res.get(name)
        sm = spill_map(func, {})
        row = {
            "cap": cap,
            "kernel": name,
            "regs": r.reg if r else None,
            "stack": r.stack if r else None,
            "spill_instrs": sm["store_instructions"] + sm["load_instructions"],
            "spill_bytes": sm["store_bytes"] + sm["load_bytes"],
            "spill_top_line": (sm["by_line"][0]["line"] if sm["by_line"] else None),
        }
        if threads and r and r.reg is not None:
            occ = compute(lookup(arch), r.reg, threads,
                          smem_static=max(0, (r.shared or 0) - 1024) if r.shared else 0,
                          smem_dynamic=smem_dynamic)
            row["blocks_per_sm"] = occ.blocks_per_sm
            row["occupancy_pct"] = occ.occupancy_pct
            row["limiter"] = occ.limiter
        rows.append(row)
    return {"cap": cap, "kernels": rows}


def mark_pareto(rows: list[dict]) -> None:
    """A row is Pareto-optimal if no other row has occupancy >= AND spill
    bytes <= with at least one strict. Rows lacking occupancy are skipped."""
    scored = [r for r in rows if r.get("occupancy_pct") is not None]
    for r in scored:
        r["pareto"] = not any(
            o is not r
            and o["occupancy_pct"] >= r["occupancy_pct"]
            and o["spill_bytes"] <= r["spill_bytes"]
            and (o["occupancy_pct"] > r["occupancy_pct"]
                 or o["spill_bytes"] < r["spill_bytes"])
            for o in scored
        )


def sweep_regcaps(
    ptx: str | Path,
    tc: Toolchain,
    arch: Optional[str] = None,
    caps: tuple = DEFAULT_CAPS,
    threads: Optional[int] = None,
    smem_dynamic: int = 0,
) -> dict:
    ptx = Path(ptx)
    if arch is None:
        m = _PTX_TARGET.search(ptx.read_text(errors="replace"))
        if not m:
            raise ValueError(f"{ptx}: no .target in PTX; pass --arch")
        arch = m.group(1)

    with tempfile.TemporaryDirectory(prefix="cuxray_tune_") as td:
        workdir = Path(td)
        cap_list: list[Optional[int]] = [None] + [c for c in caps]
        with cf.ThreadPoolExecutor(max_workers=min(8, len(cap_list))) as ex:
            results = list(ex.map(
                lambda c: _analyze_cap(ptx, arch, c, tc, threads,
                                       smem_dynamic, workdir),
                cap_list,
            ))

    # regroup per kernel
    by_kernel: dict[str, list[dict]] = {}
    for res in results:
        for row in res["kernels"]:
            by_kernel.setdefault(row["kernel"], []).append(row)
    for rows in by_kernel.values():
        rows.sort(key=lambda r: (r["cap"] is None, r["cap"] or 0))
        mark_pareto(rows)

    return {
        "schema": "cuxray.tune/1",
        "arch": arch,
        "threads": threads,
        "kernels": [
            {"kernel": k, "rows": rows} for k, rows in sorted(by_kernel.items())
        ],
    }
