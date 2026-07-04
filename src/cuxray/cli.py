"""cuxray command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console

from . import __version__
from .archspec import lookup
from .diffgate import GateSyntaxError, diff_reports, eval_gate, parse_gate
from .occupancy import compute, find_cliffs, sweep_block_sizes
from .render import render_diff, render_ls, render_report
from .report import build_report
from .toolchain import ToolchainError, resolve

console = Console()
err = Console(stderr=True)


def _toolchain():
    try:
        return resolve()
    except ToolchainError as e:
        err.print(f"[red]toolchain error:[/] {e}")
        sys.exit(2)


def _report_or_die(path, **kw):
    try:
        return build_report(path, _toolchain(), **kw)
    except Exception as e:
        err.print(f"[red]error:[/] {e}")
        sys.exit(2)


@click.group()
@click.version_option(__version__)
def main():
    """Hardware-free static analyzer for CUDA kernel binaries."""


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="emit JSON")
def ls(path, as_json):
    """List kernels and headline resources in an artifact."""
    doc = _report_or_die(path)
    if as_json:
        click.echo(json.dumps(doc, indent=2))
    else:
        render_ls(doc, console)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--threads", type=int, default=None, help="block size for occupancy analysis")
@click.option("--carveout", type=int, default=None, help="smem carveout KB (default: max)")
@click.option("--kernel", "kernel_re", default=None, help="regex filter on kernel names")
@click.option("--arch", default=None, help="architecture for .ptx input (e.g. sm_120a)")
@click.option("--json", "as_json", is_flag=True, help="emit JSON")
def report(path, threads, carveout, kernel_re, arch, as_json):
    """Full static report: resources, pressure, spills, occupancy."""
    doc = _report_or_die(path, threads=threads, carveout_kb=carveout,
                         kernel_re=kernel_re, arch=arch)
    if as_json:
        click.echo(json.dumps(doc, indent=2))
    else:
        render_report(doc, console)
        if not threads:
            console.print("\n[dim]tip: pass --threads N for occupancy + cliff analysis[/]")


@main.command()
@click.option("--arch", required=True, help="e.g. sm_120, sm_90")
@click.option("--regs", type=int, required=True)
@click.option("--threads", type=int, required=True)
@click.option("--smem", type=int, default=0, help="static+dynamic smem bytes")
@click.option("--carveout", type=int, default=None, help="carveout KB")
@click.option("--sweep", is_flag=True, help="sweep block sizes")
@click.option("--json", "as_json", is_flag=True)
def occupancy(arch, regs, threads, smem, carveout, sweep, as_json):
    """Pure occupancy what-if — no binary needed."""
    try:
        spec = lookup(arch)
    except KeyError as e:
        err.print(f"[red]{e}[/]")
        sys.exit(2)
    occ = compute(spec, regs, threads, smem_static=smem, carveout_kb=carveout)
    d = occ.to_dict()
    d["cliffs"] = find_cliffs(spec, occ)
    if sweep:
        d["sweep"] = [
            {"threads": o.threads_per_block, "occupancy_pct": o.occupancy_pct,
             "blocks_per_sm": o.blocks_per_sm}
            for o in sweep_block_sizes(spec, regs, smem_static=smem, carveout_kb=carveout)
        ]
    if as_json:
        click.echo(json.dumps(d, indent=2))
        return
    console.print(
        f"[bold]{spec.sm}[/] ({spec.name}) — {regs} regs, {threads} threads, "
        f"{smem} B smem"
    )
    console.print(
        f"  {occ.blocks_per_sm} blocks/SM · {occ.active_warps}/{occ.max_warps} warps · "
        f"[bold]{occ.occupancy_pct}%[/] — limiter: [bold]{occ.limiter}[/]"
    )
    console.print(f"  limits: {occ.limits}")
    for c in d["cliffs"]:
        console.print(
            f"  cliff ({c['kind']}): {c['resource']} → {c['at']} ({c['delta']:+}) "
            f"gives {c['blocks_per_sm']} blocks/SM ({c['occupancy_pct']}%)"
        )
    for n in occ.notes:
        console.print(f"  [dim]{n}[/]")
    if sweep:
        for row in d["sweep"]:
            bar = "█" * int(row["occupancy_pct"] / 5)
            console.print(f"  {row['threads']:>5} thr {row['occupancy_pct']:>6}% {bar}")


@main.command()
@click.argument("old", type=click.Path(exists=True))
@click.argument("new", type=click.Path(exists=True))
@click.option("--threads", type=int, default=None)
@click.option("--kernel", "kernel_re", default=None)
@click.option("--json", "as_json", is_flag=True)
def diff(old, new, threads, kernel_re, as_json):
    """Compare two artifacts kernel-by-kernel."""
    tc = _toolchain()
    do = build_report(old, tc, threads=threads, kernel_re=kernel_re)
    dn = build_report(new, tc, threads=threads, kernel_re=kernel_re)
    d = diff_reports(do, dn, kernel_re)
    if as_json:
        click.echo(json.dumps(d, indent=2))
    else:
        render_diff(d, console)
    sys.exit(0)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.argument("expr")
@click.option("--kernel", "kernel_re", default=None)
@click.option("--json", "as_json", is_flag=True)
def gate(path, expr, kernel_re, as_json):
    """CI gate: exit 1 if EXPR is violated (e.g. "spill_instrs==0, regs<=168")."""
    try:
        clauses = parse_gate(expr)
    except GateSyntaxError as e:
        err.print(f"[red]gate syntax:[/] {e}")
        sys.exit(2)
    doc = _report_or_die(path, kernel_re=kernel_re)
    violations = eval_gate(doc, clauses)
    if as_json:
        click.echo(json.dumps({"violations": violations, "passed": not violations}, indent=2))
    else:
        if violations:
            for v in violations:
                console.print(f"[red]✗[/] {v['kernel']} ({v['unit']}): {v['reason']}")
            console.print(f"[red]GATE FAILED[/] — {len(violations)} violation(s)")
        else:
            console.print(f"[green]✓ gate passed[/] ({', '.join(str(c) for c in clauses)})")
    sys.exit(1 if violations else 0)


@main.command()
def mcp():
    """Run the MCP server (stdio) exposing cuxray to agents."""
    try:
        from .mcp_server import run
    except ImportError:
        err.print("[red]MCP extra not installed:[/] pip install 'cuxray[mcp]'")
        sys.exit(2)
    run()


if __name__ == "__main__":
    main()
