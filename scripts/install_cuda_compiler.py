from __future__ import annotations

import hashlib
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REDIST_BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
COMPONENTS = (
    "cuda_nvcc",
    "libnvvm",
    "cuda_crt",
    "cuda_cudart",
    "cccl",
    "cuda_nvdisasm",
    "cuda_cuobjdump",
)

PINNED_COMPONENT_VERSIONS = {
    "13.3.1": {
        "linux-x86_64": {
            "cuda_nvcc": (
                "13.3.73",
                "2ff9f9954060794a1c5134a933ccb45bec723d866b2629dadfe4a1a313f21068",
            ),
            "libnvvm": (
                "13.3.73",
                "206b1ab4979c09b5c32f8bf907c42bc9e16cd7454cf6036f524c45a58d060f93",
            ),
            "cuda_crt": (
                "13.3.73",
                "1251aa9d668c607a103489cd2250773701e83a313e355578044622cf36713a9d",
            ),
            "cuda_cudart": (
                "13.3.29",
                "1e59c4888267d27ba1a9bd0f3669a6439db1334a96e754cd9013c7c73e18dc9d",
            ),
            "cccl": (
                "13.3.3.4.1",
                "26957cede74f9341174ecaf0372f3f886e7c46ceccb98d6dc775fe2b68d19268",
            ),
            "cuda_nvdisasm": (
                "13.3.73",
                "38b2eca90c0ffa2ce815bd392077f68b7fe94992ea0ec0f8e85df30ca01fc4e9",
            ),
            "cuda_cuobjdump": (
                "13.3.73",
                "2f3cd496b9183ca2de9e17dbcd7f992eac3cd4475052b151605d9e8445736b94",
            ),
        },
        "linux-sbsa": {
            "cuda_nvcc": (
                "13.3.73",
                "87044b338bf1cb062512c4bc790dd24d1c61b043119b5743f583128a954474a8",
            ),
            "libnvvm": (
                "13.3.73",
                "4bdb28ca53b714ae48887921f462a913b6806dd847978f8eba40434c85d1016f",
            ),
            "cuda_crt": (
                "13.3.73",
                "d28e00e9455fac2c2defd6459a2dbcf118d13eb950646074290bc0d29e32a239",
            ),
            "cuda_cudart": (
                "13.3.29",
                "0cdd73d11885062daf3aa98ad4d7b8bd84f89b398be11f7054edea9ed31f597d",
            ),
            "cccl": (
                "13.3.3.4.1",
                "c0dd608d18ff7014c5fe0d3b2c7d1a6b9b855ec6fcad1f379cc3315610d22635",
            ),
            "cuda_nvdisasm": (
                "13.3.73",
                "9e707ad0bd95c5525dca32f542ecb26aa6d0537540a5744339a987f7e6512ade",
            ),
            "cuda_cuobjdump": (
                "13.3.73",
                "dfc2b954b540df857a586353d52744f89568e8ca366ef5f2f8890f83da365813",
            ),
        },
    }
}


def platform_tag() -> str:
    machine = platform.machine()
    if machine == "x86_64":
        return "linux-x86_64"
    if machine in {"aarch64", "arm64"}:
        return "linux-sbsa"
    raise RuntimeError(f"unsupported compiler-helper architecture: {machine}")


def download(url: str, path: Path) -> str:
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as source, path.open("wb") as dest:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            dest.write(chunk)
    return digest.hexdigest()


def component_entries(version: str, tag: str) -> dict[str, tuple[str, str]]:
    try:
        versions = PINNED_COMPONENT_VERSIONS[version][tag]
    except KeyError as exc:
        raise RuntimeError(
            f"no independently verified compiler pins for CUDA {version} on {tag}"
        ) from exc
    if set(versions) != set(COMPONENTS):
        raise RuntimeError(f"incomplete compiler pins for CUDA {version} on {tag}")
    return {
        name: (f"{name}/{tag}/{name}-{tag}-{component_version}-archive.tar.xz", sha256)
        for name, (component_version, sha256) in versions.items()
    }


def install_component(
    name: str,
    relative_path: str,
    expected_sha256: str,
    destination: Path,
) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        archive = temporary_path / "component.tar.xz"
        digest = download(REDIST_BASE + relative_path, archive)
        if digest != expected_sha256:
            raise RuntimeError(f"sha256 mismatch for {name}")

        unpacked = temporary_path / "unpacked"
        unpacked.mkdir()
        with tarfile.open(archive, "r:xz") as tar:
            tar.extractall(unpacked, filter="data")
        roots = list(unpacked.iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise RuntimeError(f"unexpected archive layout for {name}")

        licenses = destination / "licenses"
        licenses.mkdir(parents=True, exist_ok=True)
        for license_file in roots[0].glob("LICENSE*"):
            shutil.copy2(license_file, licenses / f"{name}-{license_file.name}")
            license_file.unlink()
        shutil.copytree(roots[0], destination, dirs_exist_ok=True, symlinks=True)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: install_cuda_compiler.py CUDA_VERSION DESTINATION")
    version = sys.argv[1]
    destination = Path(sys.argv[2])
    tag = platform_tag()
    entries = component_entries(version, tag)
    for component, (relative_path, expected_sha256) in entries.items():
        print(f"installing pinned {component} ({tag})")
        install_component(component, relative_path, expected_sha256, destination)


if __name__ == "__main__":
    main()
