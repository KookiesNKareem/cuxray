"""macOS entry/delegation tests; no container runtime required."""

from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path

import pytest

from cuxray import entry, macos


def test_command_name_skips_global_flags():
    assert entry.command_name(["--debug", "report", "k.cubin"]) == "report"
    assert entry.command_name(["--version"]) is None


def test_delegates_only_toolchain_commands_on_macos():
    assert entry.should_delegate(["report", "k.cubin"], platform="darwin")
    assert entry.should_delegate(["triton", "cache"], platform="darwin")
    assert not entry.should_delegate(["report", "--help"], platform="darwin")
    assert not entry.should_delegate(["occupancy"], platform="darwin")
    assert not entry.should_delegate(["doctor"], platform="darwin")
    assert not entry.should_delegate(["report", "k.cubin"], platform="linux")
    assert not entry.should_delegate(["definitely-not-a-command"], platform="darwin")


def test_cache_dir_uses_macos_location(monkeypatch, tmp_path):
    monkeypatch.delenv("CUXRAY_CACHE", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert macos.cache_dir() == tmp_path / "Library" / "Caches" / "cuxray"


def test_cache_dir_override(monkeypatch, tmp_path):
    cache = tmp_path / "custom"
    monkeypatch.setenv("CUXRAY_CACHE", str(cache))
    assert macos.cache_dir() == cache


def test_container_command_preserves_paths_and_user(monkeypatch, tmp_path):
    work = tmp_path / "project with spaces"
    cache = tmp_path / "cache"
    work.mkdir()
    cache.mkdir()
    monkeypatch.setattr(macos.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(macos.sys.stdout, "isatty", lambda: False)
    monkeypatch.setattr(macos.sys.stderr, "isatty", lambda: False)

    command = macos.container_command(
        "/opt/homebrew/bin/docker",
        "cuxray:test",
        ["report", "kernel.cubin", "--json"],
        cwd=work,
        cache=cache,
    )

    assert command[:4] == ["/opt/homebrew/bin/docker", "run", "--rm", "--init"]
    assert f"type=bind,source={work},target={work}" in command
    assert f"type=bind,source={cache},target=/cuxray-cache" in command
    assert command[-4:] == ["cuxray:test", "report", "kernel.cubin", "--json"]
    if hasattr(os, "getuid"):
        assert f"{os.getuid()}:{os.getgid()}" in command


def test_ready_docker_is_used_without_colima(monkeypatch):
    monkeypatch.setattr(macos.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(macos, "_quiet_ok", lambda argv: argv[-1] == "info")
    assert macos.ensure_runtime(io.StringIO()) == "/bin/docker"


def test_colima_starts_after_noninteractive_opt_in(monkeypatch):
    monkeypatch.setenv("CUXRAY_CONTAINER_AUTO_START", "1")
    monkeypatch.setenv("DOCKER_CONTEXT", "before-test")
    monkeypatch.setattr(macos.shutil, "which", lambda name: f"/bin/{name}")
    checks = iter([False, False, True])
    monkeypatch.setattr(macos, "_quiet_ok", lambda argv: next(checks))
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(
        macos.subprocess,
        "run",
        run,
    )
    assert macos.ensure_runtime(io.StringIO()) == "/bin/docker"
    assert calls[0][:4] == ["/bin/colima", "--profile", "cuxray", "start"]
    assert calls[0][calls[0].index("--arch") + 1] in {"aarch64", "x86_64"}
    assert calls[0][calls[0].index("--disk") + 1] == "20"
    assert os.environ["DOCKER_CONTEXT"] == "colima-cuxray"


def test_missing_runtime_has_one_command_fix(monkeypatch):
    monkeypatch.setattr(macos.shutil, "which", lambda name: None)
    with pytest.raises(macos.MacRuntimeError, match="brew install colima docker"):
        macos.ensure_runtime(io.StringIO())


def test_custom_image_pull_failure_is_actionable(monkeypatch):
    monkeypatch.setenv("CUXRAY_CONTAINER_IMAGE", "example.invalid/cuxray:test")
    monkeypatch.setattr(macos, "_quiet_ok", lambda argv: False)
    monkeypatch.setattr(
        macos.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 1, stderr="not found\n"),
    )
    with pytest.raises(macos.MacRuntimeError, match="not found"):
        macos.ensure_image("docker", io.StringIO())


def test_run_in_container_returns_runtime_error(monkeypatch):
    def fail(stream):
        raise macos.MacRuntimeError("not ready")

    monkeypatch.setattr(macos, "ensure_runtime", fail)
    output = io.StringIO()
    assert macos.run_in_container(["report", "x"], output) == 2
    assert "not ready" in output.getvalue()
