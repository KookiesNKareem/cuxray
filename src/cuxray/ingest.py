"""Artifact ingestion: everything becomes a list of cubins to analyze.

Accepted inputs:
  - raw .cubin              (ELF with e_machine == EM_CUDA (190))
  - host ELF (.so/.o/exe)   → cuobjdump -xelf all, iterate embedded cubins
  - directory               → recursive *.cubin walk (Triton cache layout)
  - .ptx                    → compiled with ptxas (--gpu-name from .target)
"""

from __future__ import annotations

import re
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .toolchain import Toolchain, ToolchainError

EM_CUDA = 190


class IngestError(RuntimeError):
    pass


@dataclass
class CubinUnit:
    cubin: Path
    label: str                      # human-facing name for this unit
    source: Path                    # the artifact the user pointed at
    arch: Optional[str] = None      # filled from nvdisasm .target during analysis


def _elf_machine(path: Path) -> Optional[int]:
    with open(path, "rb") as f:
        head = f.read(20)
    if len(head) < 20 or head[:4] != b"\x7fELF":
        return None
    return struct.unpack_from("<H", head, 18)[0]


_PTX_TARGET = re.compile(r"^\s*\.target\s+(sm_\w+)", re.M)


def _compile_ptx(path: Path, tc: Toolchain, workdir: Path, arch: Optional[str]) -> Path:
    text = path.read_text(errors="replace")
    if arch is None:
        m = _PTX_TARGET.search(text)
        if not m:
            raise IngestError(f"{path}: no .target in PTX; pass --arch")
        arch = m.group(1)
    out = workdir / (path.stem + f".{arch}.cubin")
    tc.run("ptxas", ["--gpu-name", arch, "--generate-line-info", "-o", str(out), str(path)])
    return out


def _extract_host_elf(path: Path, tc: Toolchain, workdir: Path) -> list[Path]:
    # cuobjdump writes extracted cubins into the CWD
    dest = Path(tempfile.mkdtemp(prefix="xelf_", dir=workdir))
    tc.run("cuobjdump", ["-xelf", "all", str(path.resolve())], cwd=dest)
    cubins = sorted(dest.glob("*.cubin"))
    if not cubins:
        raise IngestError(f"{path}: no embedded cubins found (cuobjdump -xelf)")
    return cubins


def ingest(
    path: str | Path,
    tc: Toolchain,
    workdir: Optional[Path] = None,
    arch: Optional[str] = None,
) -> list[CubinUnit]:
    p = Path(path)
    if not p.exists():
        raise IngestError(f"no such file or directory: {p}")
    workdir = workdir or Path(tempfile.mkdtemp(prefix="cuxray_"))
    workdir.mkdir(parents=True, exist_ok=True)

    if p.is_dir():
        cubins = sorted(p.rglob("*.cubin"))
        if not cubins:
            raise IngestError(f"{p}: no *.cubin files found in directory tree")
        return [
            CubinUnit(cubin=c, label=str(c.relative_to(p)), source=p)
            for c in cubins
        ]

    suffix = p.suffix.lower()
    if suffix == ".ptx":
        out = _compile_ptx(p, tc, workdir, arch)
        return [CubinUnit(cubin=out, label=p.name, source=p)]

    machine = _elf_machine(p)
    if machine == EM_CUDA:
        return [CubinUnit(cubin=p, label=p.name, source=p)]
    if machine is not None:
        cubins = _extract_host_elf(p, tc, workdir)
        return [CubinUnit(cubin=c, label=c.name, source=p) for c in cubins]

    if p.read_bytes()[:4] == b"\xcf\xfa\xed\xfe":
        raise IngestError(f"{p}: Mach-O binary — CUDA kernels only live in Linux/Windows artifacts")
    # Last resort: maybe it's a raw fatbin; cuobjdump can often extract those too
    try:
        cubins = _extract_host_elf(p, tc, workdir)
        return [CubinUnit(cubin=c, label=c.name, source=p) for c in cubins]
    except (ToolchainError, IngestError):
        raise IngestError(
            f"{p}: unrecognized artifact (expected cubin, host ELF, directory, or .ptx)"
        )
