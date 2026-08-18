"""Artifact ingestion: everything becomes a list of cubins to analyze.

Accepted inputs:
  - raw .cubin              (ELF with e_machine == EM_CUDA (190))
  - host ELF (.so/.o/exe)   → cuobjdump -xelf all, iterate embedded cubins
  - directory               → recursive *.cubin walk (Triton cache layout)
  - .ptx                    → compiled with ptxas (--gpu-name from .target)
  - .whl/.zip/.pt2          → scan archived cubins and host ELF files
"""

from __future__ import annotations

import re
import struct
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .toolchain import Toolchain, ToolchainError

EM_CUDA = 190
_ARCHIVE_SUFFIXES = {".whl", ".zip", ".pt2"}
_MAX_ARCHIVE_MEMBERS = 50_000
_MAX_ARCHIVE_MEMBER_BYTES = 2 * 1024**3
_MAX_ARCHIVE_TOTAL_BYTES = 4 * 1024**3
_MAX_ARCHIVE_COMPRESSION_RATIO = 200


class IngestError(RuntimeError):
    pass


class _NoCubinsError(IngestError):
    pass


@dataclass
class CubinUnit:
    cubin: Path
    label: str                      # human-facing name for this unit
    source: Path                    # the artifact the user pointed at
    arch: Optional[str] = None      # filled from nvdisasm .target during analysis


def _elf_machine_bytes(head: bytes) -> Optional[int]:
    if len(head) < 20 or head[:4] != b"\x7fELF":
        return None
    return struct.unpack_from("<H", head, 18)[0]


def _elf_machine(path: Path) -> Optional[int]:
    with open(path, "rb") as f:
        return _elf_machine_bytes(f.read(20))


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
    dest = Path(tempfile.mkdtemp(prefix="xelf_", dir=workdir))
    try:
        tc.run("cuobjdump", ["-xelf", "all", str(path.resolve())], cwd=dest)
    except ToolchainError as exc:
        if "does not contain device code" in str(exc).lower():
            raise _NoCubinsError(f"{path}: no embedded cubins found") from exc
        raise
    cubins = sorted(dest.glob("*.cubin"))
    if not cubins:
        raise _NoCubinsError(f"{path}: no embedded cubins found (cuobjdump -xelf)")
    return cubins


def _validate_archive_member(path: Path, info: zipfile.ZipInfo) -> None:
    if info.file_size > _MAX_ARCHIVE_MEMBER_BYTES:
        raise IngestError(
            f"{path}: archive member {info.filename} exceeds the "
            f"{_MAX_ARCHIVE_MEMBER_BYTES}-byte extraction limit"
        )
    ratio = info.file_size / max(info.compress_size, 1)
    if ratio > _MAX_ARCHIVE_COMPRESSION_RATIO:
        raise IngestError(
            f"{path}: archive member {info.filename} exceeds the "
            f"{_MAX_ARCHIVE_COMPRESSION_RATIO}:1 compression-ratio limit"
        )


def _extract_archive_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    extracted: Path,
    path: Path,
    remaining: int,
) -> int:
    written = 0
    try:
        with archive.open(info) as source, extracted.open("wb") as dest:
            while chunk := source.read(min(1024 * 1024, remaining - written + 1)):
                written += len(chunk)
                if written > remaining or written > _MAX_ARCHIVE_MEMBER_BYTES:
                    raise IngestError(
                        f"{path}: extracted archive data exceeds the configured limit"
                    )
                dest.write(chunk)
    except IngestError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise IngestError(f"{path}: cannot extract archive member {info.filename}") from exc
    return written


def _ingest_archive(
    path: Path,
    tc: Toolchain,
    workdir: Path,
) -> list[CubinUnit]:
    root = workdir / "archive"
    root.mkdir(exist_ok=True)
    units = []
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise IngestError(f"{path}: invalid ZIP-based archive") from exc

    with archive:
        members = sorted(archive.infolist(), key=lambda info: info.filename)
        if len(members) > _MAX_ARCHIVE_MEMBERS:
            raise IngestError(
                f"{path}: archive has {len(members)} members; limit is {_MAX_ARCHIVE_MEMBERS}"
            )
        extracted_bytes = 0
        for index, info in enumerate(members):
            if info.is_dir():
                continue
            try:
                with archive.open(info) as member:
                    machine = _elf_machine_bytes(member.read(20))
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise IngestError(f"{path}: cannot read archive member {info.filename}") from exc
            if machine is None:
                continue

            _validate_archive_member(path, info)
            if extracted_bytes + info.file_size > _MAX_ARCHIVE_TOTAL_BYTES:
                raise IngestError(
                    f"{path}: CUDA-bearing archive members exceed the "
                    f"{_MAX_ARCHIVE_TOTAL_BYTES}-byte extraction limit"
                )

            member_dir = root / str(index)
            member_dir.mkdir(exist_ok=True)
            extracted = member_dir / "artifact"
            extracted_bytes += _extract_archive_member(
                archive,
                info,
                extracted,
                path,
                _MAX_ARCHIVE_TOTAL_BYTES - extracted_bytes,
            )

            label = info.filename.replace("\\", "/")
            if machine == EM_CUDA:
                units.append(CubinUnit(cubin=extracted, label=label, source=path))
                continue
            try:
                cubins = _extract_host_elf(extracted, tc, member_dir)
            except _NoCubinsError:
                continue
            units.extend(
                CubinUnit(cubin=cubin, label=f"{label}!{cubin.name}", source=path)
                for cubin in cubins
            )

    if not units:
        raise IngestError(f"{path}: no CUDA cubins found in archive")
    return units


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
    if suffix in _ARCHIVE_SUFFIXES:
        return _ingest_archive(p, tc, workdir)
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
