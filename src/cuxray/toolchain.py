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
# ptxas (from the cuda_nvcc archive) is only needed for .ptx inputs — cubin
# and ELF analysis fetches just the two small disassembly tools (~5 MB vs ~35 MB).
CORE_COMPONENTS = ("cuda_nvdisasm", "cuda_cuobjdump")
PTXAS_COMPONENT = "cuda_nvcc"
CORE_TOOLS = ("nvdisasm", "cuobjdump")
TOOLS = ("nvdisasm", "cuobjdump", "ptxas")

EULA_NOTE = (
    "cuxray fetched NVIDIA CUDA binary utilities from NVIDIA's redistributable\n"
    "archive. They are licensed under the NVIDIA CUDA Toolkit EULA:\n"
    "https://docs.nvidia.com/cuda/eula/index.html"
)


class ToolchainError(RuntimeError):
    pass


def _decode_arch_error(stderr: str) -> Optional[str]:
    """The arch name from an 'nvdisasm fatal: Cannot decode architecture
    'SM100'' failure, or None if this is a different error. Signals a
    too-old nvdisasm (the arch postdates that toolkit)."""
    if "Cannot decode architecture" not in stderr:
        return None
    import re
    m = re.search(r"architecture '([^']+)'", stderr)
    return m.group(1) if m else "the target architecture"


@dataclass
class Toolchain:
    nvdisasm: Path
    cuobjdump: Path
    ptxas: Optional[Path]  # only needed for .ptx inputs
    origin: str  # "env" | "cuda_home" | "path" | "fetched"

    def run(self, tool: str, args: list[str], cwd: Optional[Path] = None) -> str:
        exe = getattr(self, tool)
        if exe is None:
            raise ToolchainError(
                f"{tool} is not available — it is only fetched for PTX inputs; "
                "run `cuxray doctor --fetch` to prefetch it, or install a CUDA toolkit"
            )
        proc = subprocess.run(
            [str(exe), *args],
            capture_output=True, text=True, cwd=cwd,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            arch = _decode_arch_error(stderr)
            if arch is not None:
                # The resolved nvdisasm predates this GPU. If it is a system
                # tool (PATH/CUDA_HOME/env), self-heal by falling back to the
                # pinned toolchain (tracks a recent CUDA) and retry once.
                if self.origin != "fetched" and not os.environ.get("CUXRAY_NO_FETCH"):
                    try:
                        fetched = _fetch(quiet=True, need_ptxas=self.ptxas is not None)
                    except ToolchainError:
                        fetched = None
                    if fetched is not None:
                        self.nvdisasm = fetched.nvdisasm
                        self.cuobjdump = fetched.cuobjdump
                        self.ptxas = fetched.ptxas or self.ptxas
                        self.origin = "fetched"
                        self._version_cache = None
                        return self.run(tool, args, cwd)
                raise ToolchainError(
                    f"nvdisasm at {exe} cannot decode {arch} — it predates that "
                    f"GPU (origin: {self.origin}). cuxray pins CUDA {REDIST_VERSION}, "
                    "which is newer; unset CUXRAY_TOOLCHAIN/CUDA_HOME and remove any "
                    "old CUDA from PATH so cuxray uses its pinned toolchain, or set "
                    f"CUXRAY_REDIST_VERSION to a toolkit that supports {arch}."
                )
            raise ToolchainError(
                f"{tool} {' '.join(args)} failed (exit {proc.returncode}):\n{stderr}"
            )
        return proc.stdout

    _version_cache: Optional[dict] = None

    def versions(self) -> str:
        """Stable version fingerprint for cache keys (computed once)."""
        if self._version_cache is None:
            object.__setattr__(self, "_version_cache", self.describe())
        d = self._version_cache
        return "|".join(str((d.get(t) or {}).get("version", "?")) for t in TOOLS)

    def describe(self) -> dict:
        out = {"origin": self.origin}
        for t in TOOLS:
            exe = getattr(self, t)
            if exe is None:
                out[t] = None
                continue
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


def _ok(p: Path) -> bool:
    return p.is_file() and os.access(p, os.X_OK)


def _from_dir(d: Path, origin: str, need_ptxas: bool = False) -> Optional[Toolchain]:
    core = {t: d / t for t in CORE_TOOLS}
    if not all(_ok(p) for p in core.values()):
        return None
    ptxas = d / "ptxas"
    if need_ptxas and not _ok(ptxas):
        return None
    return Toolchain(nvdisasm=core["nvdisasm"], cuobjdump=core["cuobjdump"],
                     ptxas=ptxas if _ok(ptxas) else None, origin=origin)


def _from_path_env(need_ptxas: bool = False) -> Optional[Toolchain]:
    found = {t: shutil.which(t) for t in CORE_TOOLS}
    if not all(found.values()):
        return None
    ptxas = shutil.which("ptxas")
    if need_ptxas and not ptxas:
        return None
    return Toolchain(
        nvdisasm=Path(found["nvdisasm"]),
        cuobjdump=Path(found["cuobjdump"]),
        ptxas=Path(ptxas) if ptxas else None,
        origin="path",
    )


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)


def _fetch(quiet: bool = False, need_ptxas: bool = False) -> Toolchain:
    plat = _plat_tag()
    root = cache_dir() / "toolchain" / REDIST_VERSION / plat
    bin_dir = root / "bin"
    tc = _from_dir(bin_dir, "fetched", need_ptxas)
    if tc:
        return tc
    components = list(CORE_COMPONENTS) + ([PTXAS_COMPONENT] if need_ptxas else [])

    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        _download(f"{REDIST_BASE}redistrib_{REDIST_VERSION}.json", manifest_path)
    manifest = json.loads(manifest_path.read_text())

    bin_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as td:
        for comp in components:
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

    tc = _from_dir(bin_dir, "fetched", need_ptxas)
    if not tc:
        missing = [t for t in TOOLS if not (bin_dir / t).exists()]
        raise ToolchainError(f"fetch completed but tools missing: {missing}")
    return tc


def resolve(allow_fetch: bool = True, quiet: bool = False,
            need_ptxas: bool = False) -> Toolchain:
    if os.environ.get("CUXRAY_NO_FETCH"):
        allow_fetch = False
    env_dir = os.environ.get("CUXRAY_TOOLCHAIN")
    if env_dir:
        tc = _from_dir(Path(env_dir), "env", need_ptxas)
        if tc:
            return tc
        raise ToolchainError(f"$CUXRAY_TOOLCHAIN={env_dir} lacks {'/'.join(TOOLS)}")
    for var in ("CUDA_HOME", "CUDA_PATH"):
        home = os.environ.get(var)
        if home:
            tc = _from_dir(Path(home) / "bin", "cuda_home", need_ptxas)
            if tc:
                return tc
    tc = _from_path_env(need_ptxas)
    if tc:
        return tc
    if allow_fetch:
        return _fetch(quiet=quiet, need_ptxas=need_ptxas)
    raise ToolchainError(
        "no CUDA binary utilities found and fetching disabled "
        "(CUXRAY_NO_FETCH is set or --offline)"
    )
