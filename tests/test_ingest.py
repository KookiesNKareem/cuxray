import struct
import zipfile
from pathlib import Path

import pytest

import cuxray.ingest as ingest_module
from cuxray.ingest import EM_CUDA, IngestError, ingest
from cuxray.toolchain import ToolchainError


def elf(machine: int) -> bytes:
    data = bytearray(64)
    data[:4] = b"\x7fELF"
    struct.pack_into("<H", data, 18, machine)
    return bytes(data)


def make_archive(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


class FakeToolchain:
    def __init__(self):
        self.calls = []

    def run(self, tool, args, cwd=None):
        self.calls.append((tool, args, cwd))
        if tool == "cuobjdump":
            (cwd / "embedded.sm_90.cubin").write_bytes(elf(EM_CUDA))
        return ""


@pytest.mark.parametrize("suffix", [".whl", ".zip", ".pt2"])
def test_archive_ingests_raw_cubin(suffix, tmp_path):
    archive = tmp_path / f"artifact{suffix}"
    member = "data/aotinductor/model/a.cubin" if suffix == ".pt2" else "kernels/a.cubin"
    make_archive(archive, {member: elf(EM_CUDA), "metadata.json": b"{}"})

    units = ingest(archive, FakeToolchain(), workdir=tmp_path / "work")

    assert [unit.label for unit in units] == [member]
    assert units[0].source == archive
    assert units[0].cubin.read_bytes() == elf(EM_CUDA)


def test_wheel_extracts_cubins_from_host_elf(tmp_path):
    archive = tmp_path / "extension.whl"
    make_archive(archive, {"package/extension.so": elf(62), "package/data.txt": b"x"})
    toolchain = FakeToolchain()

    units = ingest(archive, toolchain, workdir=tmp_path / "work")

    assert [unit.label for unit in units] == ["package/extension.so!embedded.sm_90.cubin"]
    assert toolchain.calls[0][0] == "cuobjdump"


def test_archive_member_cannot_escape_workdir(tmp_path):
    archive = tmp_path / "unsafe.zip"
    make_archive(archive, {"../../kernel.cubin": elf(EM_CUDA)})
    workdir = tmp_path / "work"

    unit = ingest(archive, FakeToolchain(), workdir=workdir)[0]

    assert unit.cubin.resolve().is_relative_to(workdir.resolve())
    assert not (tmp_path / "kernel.cubin").exists()


def test_archive_without_cuda_is_actionable(tmp_path):
    archive = tmp_path / "empty.pt2"
    make_archive(archive, {"metadata.json": b"{}"})

    with pytest.raises(IngestError, match="no CUDA cubins found in archive"):
        ingest(archive, FakeToolchain(), workdir=tmp_path / "work")


def test_archive_skips_host_elf_without_device_code(tmp_path):
    archive = tmp_path / "extension.whl"
    make_archive(archive, {"package/extension.so": elf(62)})

    class NoDeviceCodeToolchain(FakeToolchain):
        def run(self, tool, args, cwd=None):
            raise ToolchainError("cuobjdump failed: does not contain device code")

    with pytest.raises(IngestError, match="no CUDA cubins found in archive"):
        ingest(archive, NoDeviceCodeToolchain(), workdir=tmp_path / "work")


def test_archive_propagates_cuobjdump_failure(tmp_path):
    archive = tmp_path / "extension.whl"
    make_archive(archive, {"package/extension.so": elf(62)})

    class BrokenToolchain(FakeToolchain):
        def run(self, tool, args, cwd=None):
            raise ToolchainError("cuobjdump crashed")

    with pytest.raises(ToolchainError, match="cuobjdump crashed"):
        ingest(archive, BrokenToolchain(), workdir=tmp_path / "work")


def test_archive_member_count_is_limited(tmp_path, monkeypatch):
    archive = tmp_path / "many.zip"
    make_archive(archive, {"a": b"x", "b": b"y"})
    monkeypatch.setattr(ingest_module, "_MAX_ARCHIVE_MEMBERS", 1)

    with pytest.raises(IngestError, match="archive has 2 members; limit is 1"):
        ingest(archive, FakeToolchain(), workdir=tmp_path / "work")


def test_archive_member_size_is_limited(tmp_path, monkeypatch):
    archive = tmp_path / "large.zip"
    make_archive(archive, {"kernel.cubin": elf(EM_CUDA)})
    monkeypatch.setattr(ingest_module, "_MAX_ARCHIVE_MEMBER_BYTES", 32)

    with pytest.raises(IngestError, match="32-byte extraction limit"):
        ingest(archive, FakeToolchain(), workdir=tmp_path / "work")


def test_archive_total_size_is_limited(tmp_path, monkeypatch):
    archive = tmp_path / "large.zip"
    make_archive(
        archive,
        {"a.cubin": elf(EM_CUDA), "b.cubin": elf(EM_CUDA)},
    )
    monkeypatch.setattr(ingest_module, "_MAX_ARCHIVE_TOTAL_BYTES", 100)

    with pytest.raises(IngestError, match="100-byte extraction limit"):
        ingest(archive, FakeToolchain(), workdir=tmp_path / "work")


def test_archive_compression_ratio_is_limited(tmp_path, monkeypatch):
    archive = tmp_path / "compressed.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("kernel.cubin", elf(EM_CUDA) + bytes(4096))
    monkeypatch.setattr(ingest_module, "_MAX_ARCHIVE_COMPRESSION_RATIO", 2)

    with pytest.raises(IngestError, match="2:1 compression-ratio limit"):
        ingest(archive, FakeToolchain(), workdir=tmp_path / "work")


def test_invalid_archive_is_actionable(tmp_path):
    archive = tmp_path / "broken.whl"
    archive.write_bytes(b"not a wheel")

    with pytest.raises(IngestError, match="invalid ZIP-based archive"):
        ingest(archive, FakeToolchain(), workdir=tmp_path / "work")
