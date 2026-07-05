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
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from . import SCHEMA_VERSION, __version__
from .analyze.access import analyze_accesses
from .analyze.roofline import classify, loop_report
from .analyze.liveness import pressure
from .analyze.spillmap import spill_map
from .archspec import lookup
from .ingest import CubinUnit, IngestError, ingest
from .occupancy import compute, find_cliffs, smem_headroom
from .parse import cfgdot, elf, resusage, sass
from .toolchain import Toolchain


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_block_dims(threads: Optional[str]) -> tuple[Optional[tuple[int, int, int]], Optional[int]]:
    """'256' → ((256,1,1), 256); '32,8' → ((32,8,1), 256); None → (None, None).

    Requires 1-3 comma-separated positive integers; total <= 1024.
    """
    if threads is None:
        return None, None
    parts = [int(p) for p in str(threads).split(",")]
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"--threads takes 1-3 dimensions, got {len(parts)}: {threads!r}")
    if any(p < 1 for p in parts):
        raise ValueError(f"--threads dimensions must be positive: {threads!r}")
    while len(parts) < 3:
        parts.append(1)
    dims = (parts[0], parts[1], parts[2])
    total = dims[0] * dims[1] * dims[2]
    if total > 1024:
        raise ValueError(f"--threads total {total} exceeds the 1024 threads/block limit")
    return dims, total


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
    level: str = "full",
    fast: bool = False,
    smem_dynamic: Optional[int] = None,
    block_dims: Optional[tuple[int, int, int]] = None,
    peak_tflops: Optional[float] = None,
    peak_gbs: Optional[float] = None,
) -> dict:
    cubin = str(unit.cubin)
    data = unit.cubin.read_bytes()
    res = resusage.parse(tc.run("cuobjdump", ["--dump-resource-usage", cubin]))

    pat = re.compile(kernel_re) if kernel_re else None

    dis = sass.Disassembly()
    cfg: dict = {}
    if level == "full":
        # Restrict disassembly to matching kernels via symbol indices —
        # 38 s → 1.4 s on production-size cubins.
        fun_args: list[str] = []
        if pat:
            funcs = elf.functions(data)
            dem = _demangle([n for _, n in funcs])
            idxs = [str(i) for i, name in funcs
                    if pat.search(name) or pat.search(dem.get(name, name))]
            if not idxs:
                level = "resources"  # nothing matches; skip disassembly
            else:
                fun_args = ["-fun", ",".join(idxs)]
        if level == "full":
            dis = sass.parse_gi(tc.run("nvdisasm", ["-c", "-gi", *fun_args, cubin]))
            if not fast:
                plr = sass.parse_plr(tc.run("nvdisasm", ["-c", "-plr", *fun_args, cubin]))
                sass.merge_liveness(dis, plr)
            try:
                cfg = cfgdot.parse(tc.run("nvdisasm", ["-cfg", *fun_args, cubin]))
            except Exception:
                cfg = {}

    arch = dis.target or elf.sm_arch(data) or unit.arch
    spec = None
    try:
        spec = lookup(arch) if arch else None
    except KeyError:
        pass

    names = sorted(set(res) | set(dis.functions))
    demangled = _demangle(names)
    if pat:
        names = [n for n in names
                 if pat.search(n) or pat.search(demangled.get(n, n))]

    # Per-kernel block shape from binary metadata when the user gave none:
    # .reqntid is exact; .maxntid (__launch_bounds__) is an upper bound we
    # use as the assumed launch shape, with a note.
    meta_dims = elf.launch_dims(data) if (block_dims is None and level == "full") else {}

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
            "access": None,
            "roofline": None,
            "notes": notes,
        }

        realloc = False
        k_dims, k_threads = block_dims, threads
        if func:
            if not any(i.file for i in func.instructions):
                notes.append(
                    "no source-line info in this cubin — compile with -lineinfo "
                    "for file:line attribution"
                )
            k["pressure"] = pressure(func)
            if (not fast and func.instructions
                    and not k["pressure"].get("available")):
                notes.append(
                    "liveness output could not be parsed — possible nvdisasm "
                    "format drift; register-pressure data unavailable"
                )
            depths = cfg.get(name).loop_depth if name in cfg else {}
            k["spills"] = spill_map(func, depths)
            realloc = sass.uses_register_reallocation(func)
            if realloc:
                notes.append(
                    f"dynamic register reallocation (USETMAXREG) detected — "
                    f"REG={r.reg if r else '?'} is the post-reallocation maximum, "
                    "not the launch allocation; occupancy from it is pessimistic"
                )
            if (smem_dynamic is None and smem_static == 0
                    and sass.uses_shared_memory(func)):
                notes.append(
                    "kernel uses shared memory but none is statically allocated "
                    "— dynamic smem of unknown launch-time size; occupancy "
                    "assumes 0 B, pass --smem-dynamic N for the real number"
                )
            if k_dims is None:
                md = meta_dims.get(name) or {}
                if md.get("reqntid"):
                    k_dims = md["reqntid"]
                    notes.append(f"block shape {k_dims} from binary metadata (.reqntid)")
                elif md.get("maxntid"):
                    k_dims = md["maxntid"]
                    notes.append(
                        f"block shape assumed {k_dims} from __launch_bounds__ "
                        "(.maxntid upper bound) — pass --threads to override"
                    )
                if k_dims:
                    k_threads = k_dims[0] * k_dims[1] * k_dims[2]
            if k_dims:
                depths = cfg.get(name).loop_depth if name in cfg else {}
                acc = analyze_accesses(func, k_dims, depths)
                fcfg = cfg.get(name)
                loops = loop_report(func, fcfg, acc["accesses"])
                for row in loops:
                    row["bound"] = classify(row["est_arithmetic_intensity"],
                                            peak_tflops, peak_gbs)
                k["roofline"] = loops[:6]
                k["access"] = {
                    key: acc[key] for key in (
                        "block_dims", "analyzed_count", "unanalyzed_count",
                        "unanalyzed_by_reason", "block_invariant_read_bytes",
                        "worst_bank_conflict_ways",
                        "conflicted_shared_accesses", "uncoalesced_global_accesses",
                    )
                }
                k["access"]["by_site"] = acc["by_site"][:30]
                if k_dims[0] % 32:
                    notes.append(
                        f"blockDim.x={k_dims[0]} is not a multiple of 32 — "
                        "access analysis models warp 0 only; other warps may differ"
                    )

        if spec and k_threads and r and r.reg is not None:
            occ = compute(spec, r.reg, k_threads, smem_static=smem_static,
                          smem_dynamic=smem_dynamic or 0,
                          carveout_kb=carveout_kb)
            d = occ.to_dict()
            d["cliffs"] = find_cliffs(spec, occ)
            d["register_reallocation"] = realloc
            d["smem_headroom_bytes"] = smem_headroom(spec, occ)
            k["occupancy"] = d

        kernels.append(k)

    return {
        "label": unit.label,
        "arch": arch,
        "cubin_sha256": hashlib.sha256(data).hexdigest(),
        "kernels": kernels,
    }


def _cache_key(unit: CubinUnit, tc: Toolchain, kw: dict) -> str:
    """Cache key: cubin bytes + analyzer version + toolchain versions +
    analysis parameters."""
    h = hashlib.sha256()
    h.update(unit.cubin.read_bytes())
    h.update(__version__.encode())
    try:
        h.update(tc.versions().encode())  # tool versions, not paths
    except Exception:
        h.update(b"unknown-toolchain")
    h.update(json.dumps(kw, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _analyze_unit_cached(unit, tc, use_cache: bool, **kw) -> dict:
    if not use_cache:
        return analyze_unit(unit, tc, **kw)
    from .toolchain import cache_dir
    cdir = cache_dir() / "reports"
    key = _cache_key(unit, tc, kw)
    path = cdir / f"{key}.json"
    if path.exists():
        try:
            doc = json.loads(path.read_text())
            doc["from_cache"] = True
            doc["label"] = unit.label  # same bytes, possibly different filename
            return doc
        except (json.JSONDecodeError, OSError):
            pass  # corrupt cache entry: recompute
    doc = analyze_unit(unit, tc, **kw)
    try:
        cdir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc))
        tmp.rename(path)
    except OSError:
        pass  # cache is best-effort
    return doc


def _analyze_units(units, tc, use_cache: bool = True, **kw) -> list[dict]:
    """Analyze units in parallel — the work is subprocess-bound (nvdisasm),
    so threads give near-linear speedup on multi-cubin artifacts. Results are
    cached on disk keyed by content + config + versions."""
    if len(units) == 1:
        return [_analyze_unit_cached(units[0], tc, use_cache, **kw)]
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=min(8, len(units))) as ex:
        return list(ex.map(
            lambda u: _analyze_unit_cached(u, tc, use_cache, **kw), units))


def build_report(
    path: str | Path,
    tc: Toolchain,
    threads: Optional[str] = None,  # "256" or "32,8[,1]" block shape
    carveout_kb: Optional[int] = None,
    kernel_re: Optional[str] = None,
    arch: Optional[str] = None,
    level: str = "full",
    fast: bool = False,
    smem_dynamic: Optional[int] = None,
    use_cache: bool = True,
    peak_tflops: Optional[float] = None,
    peak_gbs: Optional[float] = None,
) -> dict:
    import tempfile

    if kernel_re:
        re.compile(kernel_re)  # validate once, fail fast with re.error
    block_dims, total_threads = parse_block_dims(threads)
    p = Path(path)
    workdir_ctx = tempfile.TemporaryDirectory(prefix="cuxray_")
    units = ingest(p, tc, workdir=Path(workdir_ctx.name), arch=arch)
    if arch and len(units) > 1:
        # For multi-cubin artifacts, --arch acts as a unit filter (sm_90a
        # matches sm_90 requests and vice versa; the ELF header carries no
        # 'a' suffix).
        want = arch.rstrip("af")
        units = [u for u in units
                 if (elf.sm_arch(u.cubin.read_bytes()) or "").rstrip("af") == want
                 or want in u.label]
        if not units:
            raise IngestError(f"no cubins matching --arch {arch} in {p}")
    try:
        return {
            "schema": SCHEMA_VERSION,
            "cuxray_version": __version__,
            "artifact": {"path": str(p), "sha256": _sha256(p) if p.is_file() else None},
            "toolchain": tc.describe(),
            "units": _analyze_units(
                units, tc, use_cache=use_cache,
                threads=total_threads, carveout_kb=carveout_kb,
                kernel_re=kernel_re, level=level, fast=fast,
                smem_dynamic=smem_dynamic, block_dims=block_dims,
                peak_tflops=peak_tflops, peak_gbs=peak_gbs,
            ),
        }
    finally:
        workdir_ctx.cleanup()
