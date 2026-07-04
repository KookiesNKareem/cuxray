"""Locate or fetch the CUDA binary utilities cuxray drives.

Resolution order:
  1. $CUXRAY_TOOLCHAIN (a directory containing the binaries)
  2. $CUDA_HOME/bin (or $CUDA_PATH)
  3. anything on $PATH
  4. cached auto-fetch from NVIDIA's official redistributable archive
     (https://developer.download.nvidia.com/compute/cuda/redist/), pinned
     version + sha256 verification, cached under ~/.cache/cuxray/.

The binaries are Linux-only (x86_64 / aarch64). On macOS/Windows we fail with
a pointer at the container/CI path rather than pretending.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REDIST_BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
# Manifest pin; component versions come from the manifest. Overridable for
# CI matrix testing across toolkit versions.
REDIST_VERSION = os.environ.get("CUXRAY_REDIST_VERSION", "13.3.1")
COMPONENTS = ("cuda_nvdisasm", "cuda_cuobjdump", "cuda_nvcc")  # nvcc archive supplies ptxas
TOOLS = ("nvdisasm", "cuobjdump", "ptxas")

EULA_NOTE = (
    "cuxray fetched NVIDIA CUDA binary utilities from NVIDIA's redistributable\n"
    "archive. They are licensed under the NVIDIA CUDA Toolkit EULA:\n"
    "https://docs.nvidia.com/cuda/eula/index.html"
)


class ToolchainError(RuntimeError):
    pass


@dataclass
class Toolchain:
    nvdisasm: Path
    cuobjdump: Path
    ptxas: Path
    origin: str  # "env" | "cuda_home" | "path" | "fetched"

    def run(self, tool: str, args: list[str], cwd: Optional[Path] = None) -> str:
        exe = getattr(self, tool)
        proc = subprocess.run(
            [str(exe), *args],
            capture_output=True, text=True, cwd=cwd,
        )
        if proc.returncode != 0:
            raise ToolchainError(
                f"{tool} {' '.join(args)} failed (exit {proc.returncode}):\n{proc.stderr.strip()}"
            )
        return proc.stdout

    def describe(self) -> dict:
        out = {"origin": self.origin}
        for t in TOOLS:
            exe = getattr(self, t)
            try:
                ver = subprocess.run([str(exe), "--version"], capture_output=True,
                                     text=True).stdout.strip().splitlines()
                out[t] = {"path": str(exe), "version": ver[-1] if ver else "?"}
            except OSError as e:
                out[t] = {"path": str(exe), "version": f"error: {e}"}
        return out


def cache_dir() -> Path:
    root = os.environ.get("CUXRAY_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")), "cuxray"
    )
    return Path(root)


def _plat_tag() -> str:
    if sys.platform != "linux":
        raise ToolchainError(
            f"NVIDIA publishes no CUDA binary utilities for {sys.platform}; "
            "run cuxray inside a Linux container or CI runner "
            "(everything is CPU-only — no GPU needed)."
        )
    machine = platform.machine()
    if machine == "x86_64":
        return "linux-x86_64"
    if machine in ("aarch64", "arm64"):
        return "linux-sbsa"
    raise ToolchainError(f"unsupported machine architecture: {machine}")


def _from_dir(d: Path, origin: str) -> Optional[Toolchain]:
    paths = {t: d / t for t in TOOLS}
    if all(p.is_file() and os.access(p, os.X_OK) for p in paths.values()):
        return Toolchain(**{k: v for k, v in paths.items()}, origin=origin)
    return None


def _from_path_env() -> Optional[Toolchain]:
    found = {t: shutil.which(t) for t in TOOLS}
    if all(found.values()):
        return Toolchain(
            nvdisasm=Path(found["nvdisasm"]),
            cuobjdump=Path(found["cuobjdump"]),
            ptxas=Path(found["ptxas"]),
            origin="path",
        )
    return None


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)


def _fetch(quiet: bool = False) -> Toolchain:
    plat = _plat_tag()
    root = cache_dir() / "toolchain" / REDIST_VERSION / plat
    bin_dir = root / "bin"
    tc = _from_dir(bin_dir, "fetched")
    if tc:
        return tc

    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        _download(f"{REDIST_BASE}redistrib_{REDIST_VERSION}.json", manifest_path)
    manifest = json.loads(manifest_path.read_text())

    bin_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as td:
        for comp in COMPONENTS:
            try:
                entry = manifest[comp][plat]
                entry["relative_path"], entry["sha256"]
            except (KeyError, TypeError):
                raise ToolchainError(
                    f"redist manifest has no {comp}/{plat} entry — toolkit "
                    f"version {REDIST_VERSION} may be unsupported or the "
                    "manifest schema changed; pin CUXRAY_REDIST_VERSION to a "
                    "known-good version (e.g. 13.3.1)"
                )
            url = REDIST_BASE + entry["relative_path"]
            archive = Path(td) / Path(entry["relative_path"]).name
            if not quiet:
                print(f"cuxray: fetching {comp} {manifest[comp]['version']} ({plat})...",
                      file=sys.stderr)
            _download(url, archive)
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            if digest != entry["sha256"]:
                raise ToolchainError(f"sha256 mismatch for {archive.name}")
            with tarfile.open(archive, mode="r:xz") as tar:
                for member in tar.getmembers():
                    base = os.path.basename(member.name)
                    if member.isfile() and base in TOOLS and "/bin/" in member.name:
                        member.name = base
                        try:
                            tar.extract(member, bin_dir, filter="data")
                        except TypeError:  # Python < 3.10.7 lacks filter=
                            tar.extract(member, bin_dir)
                        (bin_dir / base).chmod(0o755)
    (root / "EULA_NOTICE.txt").write_text(EULA_NOTE + "\n")
    if not quiet:
        print(f"cuxray: toolchain cached in {bin_dir}\n{EULA_NOTE}", file=sys.stderr)

    tc = _from_dir(bin_dir, "fetched")
    if not tc:
        missing = [t for t in TOOLS if not (bin_dir / t).exists()]
        raise ToolchainError(f"fetch completed but tools missing: {missing}")
    return tc


def resolve(allow_fetch: bool = True, quiet: bool = False) -> Toolchain:
    env_dir = os.environ.get("CUXRAY_TOOLCHAIN")
    if env_dir:
        tc = _from_dir(Path(env_dir), "env")
        if tc:
            return tc
        raise ToolchainError(f"$CUXRAY_TOOLCHAIN={env_dir} lacks {'/'.join(TOOLS)}")
    for var in ("CUDA_HOME", "CUDA_PATH"):
        home = os.environ.get(var)
        if home:
            tc = _from_dir(Path(home) / "bin", "cuda_home")
            if tc:
                return tc
    tc = _from_path_env()
    if tc:
        return tc
    if allow_fetch:
        return _fetch(quiet=quiet)
    raise ToolchainError("no CUDA binary utilities found and fetching disabled")
