"""Assemble the cuxray report document (JSON schema `cuxray.schema/1`).

Schema shape (frozen at v0.1 — additive changes only after that):

{
  "schema": "cuxray.schema/1",
  "cuxray_version": "0.1.0",
  "artifact": {"path": str, "sha256": str},
  "toolchain": {...},
  "units": [{
      "label": str, "arch": "sm_120a", "cubin_sha256": str,
      "kernels": [{
          "name": str, "demangled": str,
          "resources": {"regs": int, "stack_frame": int, "shared_section": int,
                         "smem_static": int, "local": int, "constant": int},
          "pressure": {...},   # analyze.liveness.pressure()
          "spills": {...},     # analyze.spillmap.spill_map()
          "occupancy": {...} | null,   # occupancy.Occupancy.to_dict() + "cliffs"
          "notes": [str],
      }],
  }],
}
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from . import SCHEMA_VERSION, __version__
from .analyze.liveness import pressure
from .analyze.spillmap import spill_map
from .archspec import lookup
from .ingest import CubinUnit, ingest
from .occupancy import compute, find_cliffs
from .parse import cfgdot, resusage, sass
from .toolchain import Toolchain


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _demangle(names: list[str]) -> dict[str, str]:
    exe = shutil.which("c++filt")
    if not exe or not names:
        return {n: n for n in names}
    proc = subprocess.run([exe], input="\n".join(names), capture_output=True, text=True)
    if proc.returncode != 0:
        return {n: n for n in names}
    demangled = proc.stdout.splitlines()
    if len(demangled) != len(names):
        return {n: n for n in names}
    return dict(zip(names, demangled))


def analyze_unit(
    unit: CubinUnit,
    tc: Toolchain,
    threads: Optional[int] = None,
    carveout_kb: Optional[int] = None,
    kernel_re: Optional[str] = None,
) -> dict:
    cubin = str(unit.cubin)
    res = resusage.parse(tc.run("cuobjdump", ["--dump-resource-usage", cubin]))
    dis = sass.parse_gi(tc.run("nvdisasm", ["-c", "-gi", cubin]))
    plr = sass.parse_plr(tc.run("nvdisasm", ["-c", "-plr", cubin]))
    sass.merge_liveness(dis, plr)
    try:
        cfg = cfgdot.parse(tc.run("nvdisasm", ["-cfg", cubin]))
    except Exception:
        cfg = {}

    arch = dis.target or unit.arch
    spec = None
    try:
        spec = lookup(arch) if arch else None
    except KeyError:
        pass

    names = sorted(set(res) | set(dis.functions))
    if kernel_re:
        pat = re.compile(kernel_re)
        names = [n for n in names if pat.search(n)]
    demangled = _demangle(names)

    kernels = []
    for name in names:
        notes: list[str] = []
        r = res.get(name)
        func = dis.functions.get(name)

        shared_section = r.shared if r else 0
        smem_static = shared_section
        if spec and shared_section > 0 and spec.smem_reserved_per_block:
            smem_static = max(0, shared_section - spec.smem_reserved_per_block)
            notes.append(
                f"smem_static = shared section {shared_section} B minus "
                f"{spec.smem_reserved_per_block} B system-reserved"
            )

        k: dict = {
            "name": name,
            "demangled": demangled.get(name, name),
            "resources": {
                "regs": r.reg if r else None,
                "stack_frame": r.stack if r else None,
                "shared_section": shared_section,
                "smem_static": smem_static,
                "local": r.local if r else None,
                "constant": r.constant if r else None,
            },
            "pressure": {"available": False},
            "spills": None,
            "occupancy": None,
            "notes": notes,
        }

        if func:
            if not any(i.file for i in func.instructions):
                notes.append(
                    "no source-line info in this cubin — compile with -lineinfo "
                    "for file:line attribution"
                )
            k["pressure"] = pressure(func)
            depths = cfg.get(name).loop_depth if name in cfg else {}
            k["spills"] = spill_map(func, depths)

        if spec and threads and r and r.reg is not None:
            occ = compute(spec, r.reg, threads, smem_static=smem_static)
            d = occ.to_dict()
            d["cliffs"] = find_cliffs(spec, occ)
            k["occupancy"] = d

        kernels.append(k)

    return {
        "label": unit.label,
        "arch": arch,
        "cubin_sha256": _sha256(unit.cubin),
        "kernels": kernels,
    }


def build_report(
    path: str | Path,
    tc: Toolchain,
    threads: Optional[int] = None,
    carveout_kb: Optional[int] = None,
    kernel_re: Optional[str] = None,
    arch: Optional[str] = None,
) -> dict:
    p = Path(path)
    units = ingest(p, tc, arch=arch)
    return {
        "schema": SCHEMA_VERSION,
        "cuxray_version": __version__,
        "artifact": {"path": str(p), "sha256": _sha256(p) if p.is_file() else None},
        "toolchain": tc.describe(),
        "units": [
            analyze_unit(u, tc, threads=threads, carveout_kb=carveout_kb, kernel_re=kernel_re)
            for u in units
        ],
    }
