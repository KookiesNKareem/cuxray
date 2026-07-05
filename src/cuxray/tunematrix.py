"""Build-matrix tuning: compile a kernel across a Cartesian product of
preprocessor defines and rank the instantiations statically.

Compilation uses the user's nvcc (external; not part of the fetched
toolchain) since -D sweeps require the CUDA C++ front end. Each variant is
analyzed for registers, spills, occupancy, and access-pattern health; rows
are Pareto-marked. No GPU is involved.
"""

from __future__ import annotations

import concurrent.futures as cf
import itertools
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .analyze.access import analyze_accesses
from .analyze.spillmap import spill_map
from .archspec import lookup
from .occupancy import compute
from .parse import cfgdot, resusage, sass
from .toolchain import Toolchain, ToolchainError

def _reserved(arch: str) -> int:
    from .archspec import lookup
    try:
        return lookup(arch).smem_reserved_per_block
    except KeyError:
        return 1024



def find_nvcc() -> str:
    for cand in ("nvcc",):
        path = shutil.which(cand)
        if path:
            return path
    import os
    for var in ("CUDA_HOME", "CUDA_PATH"):
        home = os.environ.get(var)
        if home and (Path(home) / "bin" / "nvcc").exists():
            return str(Path(home) / "bin" / "nvcc")
    raise ToolchainError(
        "nvcc not found — `cuxray tune` compiles CUDA C++ and needs a local "
        "CUDA toolkit (the auto-fetched tools cover binaries and PTX only)"
    )


def expand_matrix(defines: dict[str, list[str]]) -> list[dict[str, str]]:
    if not defines:
        return [{}]
    keys = sorted(defines)
    return [dict(zip(keys, combo))
            for combo in itertools.product(*(defines[k] for k in keys))]


def _compile_and_analyze(nvcc: str, src: Path, arch: str, combo: dict,
                         extra_flags: list[str], tc: Toolchain,
                         block_dims, threads: Optional[int],
                         smem_dynamic: int, workdir: Path) -> dict:
    tag = "_".join(f"{k}{v}" for k, v in sorted(combo.items())) or "base"
    out = workdir / f"{tag}.cubin"
    cmd = [nvcc, "-cubin", f"-arch={arch}", "-lineinfo", "-o", str(out),
           *[f"-D{k}={v}" for k, v in combo.items()], *extra_flags, str(src)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"config": combo, "error": proc.stderr.strip()[-400:]}

    res = resusage.parse(tc.run("cuobjdump", ["--dump-resource-usage", str(out)]))
    dis = sass.parse_gi(tc.run("nvdisasm", ["-c", "-gi", str(out)]))
    try:
        cfg = cfgdot.parse(tc.run("nvdisasm", ["-cfg", str(out)]))
    except Exception:
        cfg = {}

    kernels = []
    for name, func in dis.functions.items():
        r = res.get(name)
        sm = spill_map(func, cfg[name].loop_depth if name in cfg else {})
        row = {
            "kernel": name,
            "regs": r.reg if r else None,
            "smem": (max(0, (r.shared or 0) - _reserved(arch))
                     if r and r.shared else 0),
            "spill_bytes": sm["store_bytes"] + sm["load_bytes"],
            "hot_spills": sum(x["stores"] + x["loads"]
                              for x in sm["by_line"] if x["loop_depth"] >= 1),
        }
        if block_dims:
            acc = analyze_accesses(func, block_dims,
                                   cfg[name].loop_depth if name in cfg else {})
            row["bank_ways"] = acc["worst_bank_conflict_ways"]
            row["uncoalesced"] = acc["uncoalesced_global_accesses"]
        if threads and r and r.reg is not None:
            occ = compute(lookup(arch), r.reg, threads,
                          smem_static=row["smem"], smem_dynamic=smem_dynamic)
            row["blocks_per_sm"] = occ.blocks_per_sm
            row["occupancy_pct"] = occ.occupancy_pct
        kernels.append(row)
    return {"config": combo, "kernels": kernels}


def mark_pareto(variants: list[dict]) -> None:
    """Pareto over (occupancy max, spill_bytes min, bank_ways min,
    uncoalesced min), aggregated per variant across its kernels."""
    def score(v):
        ks = v.get("kernels") or []
        if not ks:
            return None
        occ = min((k.get("occupancy_pct") or 0) for k in ks)
        return (occ,
                -sum(k["spill_bytes"] for k in ks),
                -max((k.get("bank_ways") or 1) for k in ks),
                -sum((k.get("uncoalesced") or 0) for k in ks))
    scored = [(v, score(v)) for v in variants if score(v) is not None]
    for v, sc in scored:
        v["pareto"] = not any(
            o is not v and all(so >= ss for so, ss in zip(osc, sc))
            and any(so > ss for so, ss in zip(osc, sc))
            for o, osc in scored
        )


def sweep_matrix(
    src: str | Path,
    tc: Toolchain,
    arch: str,
    defines: dict[str, list[str]],
    extra_flags: Optional[list[str]] = None,
    block_dims=None,
    threads: Optional[int] = None,
    smem_dynamic: int = 0,
) -> dict:
    nvcc = find_nvcc()
    combos = expand_matrix(defines)
    with tempfile.TemporaryDirectory(prefix="cuxray_tune_") as td:
        with cf.ThreadPoolExecutor(max_workers=min(8, len(combos))) as ex:
            variants = list(ex.map(
                lambda c: _compile_and_analyze(
                    nvcc, Path(src), arch, c, extra_flags or [], tc,
                    block_dims, threads, smem_dynamic, Path(td)),
                combos,
            ))
    mark_pareto(variants)
    return {
        "schema": "cuxray.tunematrix/1",
        "arch": arch,
        "source": str(src),
        "variants": variants,
        "failed": sum(1 for v in variants if "error" in v),
    }
