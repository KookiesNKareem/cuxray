"""elf.py and toolchain.py unit tests — pure Python, committed fixtures only."""

import os
import sys
from pathlib import Path

import pytest

from cuxray.parse import elf
from cuxray import toolchain

BIN = Path(__file__).parent / "fixtures" / "bin"


class TestElf:
    def test_machine_detects_cubin(self):
        data = (BIN / "saxpy.sm_90.cubin").read_bytes()
        assert elf.machine(data) == elf.EM_CUDA

    def test_machine_rejects_garbage(self):
        assert elf.machine(b"\x7fELF") is None      # too short
        assert elf.machine(b"MZ" + b"\0" * 100) is None

    def test_sm_arch(self):
        assert elf.sm_arch((BIN / "saxpy.sm_90.cubin").read_bytes()) == "sm_90"
        assert elf.sm_arch((BIN / "saxpy.sm_120a.cubin").read_bytes()) == "sm_120"

    def test_functions(self):
        data = (BIN / "launch_bounds.sm_90.cubin").read_bytes()
        names = [n for _, n in elf.functions(data)]
        assert any("bounded" in n for n in names)
        assert any("plain" in n for n in names)
        # symbol indices must be valid ints usable with nvdisasm -fun
        assert all(isinstance(i, int) and i > 0 for i, _ in elf.functions(data))

    def test_launch_dims_absent_when_no_bounds(self):
        dims = elf.launch_dims((BIN / "saxpy.sm_90.cubin").read_bytes())
        saxpy = next(v for k, v in dims.items() if "saxpy" in k)
        assert saxpy["reqntid"] is None


class TestToolchain:
    def test_plat_tag_darwin_raises(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        with pytest.raises(toolchain.ToolchainError, match="Linux container"):
            toolchain._plat_tag()

    def test_from_dir_ptxas_optional_unless_needed(self, tmp_path):
        for t in ("nvdisasm", "cuobjdump"):  # core tools only
            p = tmp_path / t
            p.write_text("#!/bin/sh\n")
            p.chmod(0o755)
        tc = toolchain._from_dir(tmp_path, "env")
        assert tc is not None and tc.ptxas is None  # cubin analysis works
        assert toolchain._from_dir(tmp_path, "env", need_ptxas=True) is None
        p = tmp_path / "ptxas"
        p.write_text("#!/bin/sh\n")
        p.chmod(0o755)
        tc = toolchain._from_dir(tmp_path, "env", need_ptxas=True)
        assert tc is not None and tc.ptxas is not None

    def test_run_without_ptxas_raises_helpfully(self, tmp_path):
        for t in ("nvdisasm", "cuobjdump"):
            p = tmp_path / t
            p.write_text("#!/bin/sh\n")
            p.chmod(0o755)
        tc = toolchain._from_dir(tmp_path, "env")
        with pytest.raises(toolchain.ToolchainError, match="doctor --fetch"):
            tc.run("ptxas", ["--version"])

    def test_no_fetch_env_disables_fetch(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CUXRAY_NO_FETCH", "1")
        monkeypatch.delenv("CUXRAY_TOOLCHAIN", raising=False)
        monkeypatch.delenv("CUDA_HOME", raising=False)
        monkeypatch.delenv("CUDA_PATH", raising=False)
        monkeypatch.setattr(toolchain.shutil, "which", lambda _: None)
        with pytest.raises(toolchain.ToolchainError, match="fetching disabled"):
            toolchain.resolve()

    def test_resolution_prefers_env_dir(self, tmp_path, monkeypatch):
        for t in toolchain.TOOLS:
            p = tmp_path / t
            p.write_text("#!/bin/sh\n")
            p.chmod(0o755)
        monkeypatch.setenv("CUXRAY_TOOLCHAIN", str(tmp_path))
        tc = toolchain.resolve(allow_fetch=False)
        assert tc.origin == "env"

    def test_env_dir_missing_tools_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CUXRAY_TOOLCHAIN", str(tmp_path))
        with pytest.raises(toolchain.ToolchainError, match="CUXRAY_TOOLCHAIN"):
            toolchain.resolve(allow_fetch=False)

    def test_no_toolchain_no_fetch_raises(self, monkeypatch):
        monkeypatch.delenv("CUXRAY_TOOLCHAIN", raising=False)
        monkeypatch.delenv("CUDA_HOME", raising=False)
        monkeypatch.delenv("CUDA_PATH", raising=False)
        monkeypatch.setattr(toolchain.shutil, "which", lambda _: None)
        with pytest.raises(toolchain.ToolchainError, match="fetching disabled"):
            toolchain.resolve(allow_fetch=False)
