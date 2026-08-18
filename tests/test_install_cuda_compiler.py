from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load_installer():
    path = Path(__file__).parents[1] / "scripts" / "install_cuda_compiler.py"
    spec = importlib.util.spec_from_file_location("install_cuda_compiler", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compiler_pins_cover_both_platforms():
    installer = load_installer()

    for tag in ("linux-x86_64", "linux-sbsa"):
        entries = installer.component_entries("13.3.1", tag)
        assert set(entries) == set(installer.COMPONENTS)
        assert all(len(sha256) == 64 for _, sha256 in entries.values())


def test_unpinned_compiler_version_is_rejected():
    installer = load_installer()

    with pytest.raises(RuntimeError, match="no independently verified compiler pins"):
        installer.component_entries("99.0", "linux-x86_64")


def test_component_checksum_mismatch_is_rejected(tmp_path, monkeypatch):
    installer = load_installer()
    monkeypatch.setattr(installer, "download", lambda url, path: "wrong")

    with pytest.raises(RuntimeError, match="sha256 mismatch for cuda_nvcc"):
        installer.install_component(
            "cuda_nvcc",
            "cuda_nvcc/archive.tar.xz",
            "expected",
            tmp_path,
        )
