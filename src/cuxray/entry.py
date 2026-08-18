"""
Platform-aware console entry point.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

_NATIVE_COMMANDS = {"occupancy", "roofline", "schema"}


def command_name(args: Sequence[str]) -> str | None:
    """Return Click's command token (the only global option is flag-only)."""
    for arg in args:
        if not arg.startswith("-"):
            return arg
    return None


def should_delegate(args: Sequence[str], platform: str | None = None) -> bool:
    """Whether this invocation needs the Linux toolchain container."""
    platform = platform or sys.platform
    if platform != "darwin" or "--help" in args:
        return False
    command = command_name(args)
    if command is None or command in _NATIVE_COMMANDS or command == "doctor":
        return False

    # Leave misspelled/unknown commands to Click so help and errors remain
    # immediate and do not start a container.
    from .cli import main as cli_main

    return command in cli_main.commands


def main() -> None:
    args = sys.argv[1:]
    if sys.platform == "darwin":
        from . import macos

        command = command_name(args)
        if command == "doctor" and "--help" not in args:
            if "--fetch" in args:
                raise SystemExit(macos.run_in_container(args))
            raise SystemExit(macos.doctor())
        if should_delegate(args):
            raise SystemExit(macos.run_in_container(args))

    from .cli import main as cli_main

    cli_main(prog_name="cuxray")
