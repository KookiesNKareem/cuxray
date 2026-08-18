"""
Transparent Linux-container backend for macOS.
We need it since NVIDIA's
``nvdisasm``, ``cuobjdump``, and ``ptxas`` binaries are Linux-only.
"""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TextIO

from . import __version__

OFFICIAL_IMAGE = f"ghcr.io/kookiesnkareem/cuxray:{__version__}"
OFFICIAL_COMPILER_IMAGE = f"ghcr.io/kookiesnkareem/cuxray:{__version__}-nvcc"
COLIMA_PROFILE = "cuxray"
COLIMA_DOCKER_CONTEXT = f"colima-{COLIMA_PROFILE}"
_FORWARDED_ENV = (
    "CUXRAY_NO_FETCH",
    "CUXRAY_REDIST_VERSION",
    "CUXRAY_REDIST_VERSION_LEGACY",
    "NO_COLOR",
    "TERM",
)


class MacRuntimeError(RuntimeError):
    """The local Linux-container helper could not be prepared."""


def cache_dir() -> Path:
    override = os.environ.get("CUXRAY_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / "Library" / "Caches" / "cuxray").resolve()


def image_name(compiler: bool = False) -> str:
    if compiler:
        return os.environ.get(
            "CUXRAY_COMPILER_IMAGE",
            os.environ.get("CUXRAY_CONTAINER_IMAGE", OFFICIAL_COMPILER_IMAGE),
        )
    return os.environ.get("CUXRAY_CONTAINER_IMAGE", OFFICIAL_IMAGE)


def _quiet_ok(argv: list[str]) -> bool:
    try:
        return subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
    except OSError:
        return False


def _confirm(prompt: str, stream: TextIO = sys.stderr) -> bool:
    auto = os.environ.get("CUXRAY_CONTAINER_AUTO_START", "").lower()
    if auto in {"1", "true", "yes"}:
        return True
    if auto in {"0", "false", "no"} or not sys.stdin.isatty():
        return False
    stream.write(prompt)
    stream.flush()
    answer = sys.stdin.readline().strip().lower()
    return answer in {"", "y", "yes"}


def ensure_runtime(stream: TextIO = sys.stderr) -> str:
    """Return a ready Docker CLI path, starting Colima with permission."""
    docker = shutil.which("docker")
    if docker and _quiet_ok([docker, "info"]):
        return docker

    colima = shutil.which("colima")
    if docker and colima:
        # Reuse cuxray's isolated profile without changing the user's active
        # Docker context or their default Colima VM.
        if _quiet_ok([docker, "--context", COLIMA_DOCKER_CONTEXT, "info"]):
            os.environ["DOCKER_CONTEXT"] = COLIMA_DOCKER_CONTEXT
            return docker
        if not _confirm(
            "cuxray needs a lightweight Linux helper because NVIDIA's "
            "analysis tools are unavailable for macOS.\n"
            "Start it with Colima now? [Y/n] ",
            stream,
        ):
            raise MacRuntimeError(
                "Linux helper is not running; rerun the command and answer yes, "
                "or run `colima --profile cuxray start`"
            )
        stream.write("cuxray: starting the Linux helper (first start may take a moment)...\n")
        stream.flush()
        guest_arch = "aarch64" if platform.machine() in {"arm64", "aarch64"} else "x86_64"
        proc = subprocess.run(
            [
                colima,
                "--profile",
                COLIMA_PROFILE,
                "start",
                "--arch",
                guest_arch,
                "--cpus",
                "2",
                "--memory",
                "2",
                "--disk",
                "20",
                "--activate=false",
            ],
            check=False,
        )
        os.environ["DOCKER_CONTEXT"] = COLIMA_DOCKER_CONTEXT
        if proc.returncode == 0 and _quiet_ok([docker, "info"]):
            return docker
        raise MacRuntimeError("Colima did not start a working Docker runtime")

    if not docker:
        raise MacRuntimeError(
            "Mac support needs a container runtime. Install the lightweight "
            "runtime once with `brew install colima docker`, then rerun cuxray. "
            "Docker Desktop also works."
        )
    raise MacRuntimeError(
        "Docker is installed but its runtime is not running. Open Docker Desktop, "
        "or install/start Colima with `brew install colima && colima start`."
    )


def _bootstrap_image(docker: str, stream: TextIO = sys.stderr) -> str:
    """Build an exact-version local image when GHCR is not yet reachable."""
    local = f"cuxray-local:{__version__}"
    if _quiet_ok([docker, "image", "inspect", local]):
        return local

    stream.write(
        f"cuxray: the release image is unavailable; building {local} from PyPI once...\n"
    )
    stream.flush()
    dockerfile = f"""\
FROM python:3.12-slim
RUN apt-get update \\
 && apt-get install -y --no-install-recommends binutils ca-certificates \\
 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir cuxray=={__version__}
ENV CUXRAY_CACHE=/cuxray-cache
WORKDIR /work
ENTRYPOINT [\"cuxray\"]
"""
    proc = subprocess.run(
        [docker, "build", "-t", local, "-"],
        input=dockerfile,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise MacRuntimeError(
            f"could not pull {OFFICIAL_IMAGE} or build the exact-version fallback"
        )
    return local


def ensure_image(
    docker: str,
    stream: TextIO = sys.stderr,
    compiler: bool = False,
) -> str:
    image = image_name(compiler)
    if _quiet_ok([docker, "image", "inspect", image]):
        return image

    if compiler and not _confirm(
        "cuxray tune needs the optional CUDA compiler helper.\n"
        "Download it now? [Y/n] ",
        stream,
    ):
        raise MacRuntimeError(
            "optional compiler helper is not installed; rerun and answer yes, "
            "or run tune on Linux with a CUDA toolkit"
        )

    stream.write(f"cuxray: downloading {image} (one-time setup)...\n")
    stream.flush()
    pull = subprocess.run(
        [docker, "pull", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if pull.returncode == 0:
        return image
    if not compiler and image == OFFICIAL_IMAGE:
        return _bootstrap_image(docker, stream)
    detail = (pull.stderr or "").strip().splitlines()
    reason = detail[-1] if detail else "pull failed"
    if compiler:
        raise MacRuntimeError(
            "could not download the optional compiler helper; run tune on Linux "
            f"with a CUDA toolkit instead ({reason})"
        )
    raise MacRuntimeError(f"could not pull {image}: {reason}")


def container_command(
    docker: str,
    image: str,
    args: list[str],
    cwd: Path | None = None,
    cache: Path | None = None,
) -> list[str]:
    """Construct the path-preserving, host-user container invocation."""
    cwd = (cwd or Path.cwd()).resolve()
    cache = (cache or cache_dir()).resolve()
    command = [docker, "run", "--rm", "--init"]
    if sys.stdin.isatty():
        command.append("-i")
    if sys.stdout.isatty() and sys.stderr.isatty():
        command.append("-t")
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
    command.extend(
        [
            "--mount",
            f"type=bind,source={cwd},target={cwd}",
            "--mount",
            f"type=bind,source={cache},target=/cuxray-cache",
            "--workdir",
            str(cwd),
            "--env",
            "CUXRAY_CACHE=/cuxray-cache",
            "--env",
            "CUXRAY_CONTAINER_ACTIVE=1",
        ]
    )
    for name in _FORWARDED_ENV:
        if name in os.environ:
            command.extend(["--env", f"{name}={os.environ[name]}"])
    command.extend([image, *args])
    return command


def run_in_container(
    args: list[str],
    stream: TextIO = sys.stderr,
    compiler: bool = False,
) -> int:
    try:
        docker = ensure_runtime(stream)
        image = ensure_image(docker, stream, compiler=compiler)
        cache = cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        command = container_command(docker, image, args, cache=cache)
        if os.environ.get("CUXRAY_CONTAINER_DEBUG"):
            stream.write("cuxray: " + shlex.join(command) + "\n")
            stream.flush()
        return subprocess.run(command, check=False).returncode
    except MacRuntimeError as exc:
        stream.write(f"cuxray: {exc}\n")
        return 2
    except KeyboardInterrupt:
        return 130


def doctor(stream: TextIO = sys.stdout) -> int:
    docker = shutil.which("docker")
    colima = shutil.which("colima")
    ready = bool(docker and _quiet_ok([docker, "info"]))
    cuxray_profile_ready = bool(
        docker
        and not ready
        and _quiet_ok([docker, "--context", COLIMA_DOCKER_CONTEXT, "info"])
    )
    if cuxray_profile_ready:
        os.environ["DOCKER_CONTEXT"] = COLIMA_DOCKER_CONTEXT
        ready = True
    image = image_name()
    compiler_image = image_name(compiler=True)
    present = bool(docker and ready and _quiet_ok([docker, "image", "inspect", image]))
    compiler_present = bool(
        docker and ready and _quiet_ok([docker, "image", "inspect", compiler_image])
    )

    stream.write(f"platform: darwin / {platform.machine()}\n")
    stream.write(f"docker CLI: {'yes' if docker else 'no'}\n")
    stream.write(f"container runtime: {'ready' if ready else 'not running'}\n")
    stream.write(f"colima: {'installed' if colima else 'not installed'}\n")
    stream.write(f"image: {image} ({'ready' if present else 'downloaded on first run'})\n")
    stream.write(
        f"compiler image: {compiler_image} "
        f"({'ready' if compiler_present else 'downloaded on first tune'})\n"
    )
    stream.write(f"cache: {cache_dir()}\n")
    if ready:
        return 0
    if docker and colima:
        stream.write("next: rerun an analysis command and approve the setup prompt\n")
    elif docker:
        stream.write("next: open Docker Desktop, or install Colima with `brew install colima`\n")
    else:
        stream.write("next: run `brew install colima docker`\n")
    return 2
