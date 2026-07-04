"""cuxray command-line interface."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import click
from rich.console import Console

from . import __version__
from .archspec import lookup
from .diffgate import (GateError, GateSyntaxError, diff_reports, eval_budget,
                       eval_gate, parse_gate)
from .ingest import IngestError
from .occupancy import compute, find_cliffs, sweep_block_sizes
from .render import render_diff, render_ls, render_report
from .sarif import gate_to_sarif
from .report import build_report, parse_block_dims
from .toolchain import ToolchainError, resolve

console = Console()
err = Console(stderr=True)

_DEBUG = False


def _toolchain(need_ptxas: bool = False):
    try:
        return resolve(need_ptxas=need_ptxas)
    except ToolchainError as e:
        if _DEBUG:
            raise
        err.print(f"[red]toolchain error:[/] {e}")
        sys.exit(2)


def _report_or_die(path, **kw):
    try:
        need_ptxas = str(path).endswith(".ptx")
        return build_report(path, _toolchain(need_ptxas), **kw)
    except Exception as e:
        if _DEBUG:
            raise
        if isinstance(e, IngestError):
            err.print(f"[red]cannot ingest:[/] {e}")
        elif isinstance(e, ToolchainError):
            err.print(f"[red]toolchain error:[/] {e}")
        elif isinstance(e, re.error):
            err.print(f"[red]invalid --kernel regex:[/] {e}")
        elif isinstance(e, ValueError):
            err.print(f"[red]error:[/] {e}")
        else:
            err.print(f"[red]internal error ({type(e).__name__}):[/] {e} "
                      "— rerun with --debug for a traceback")
        sys.exit(2)


def _emit(doc: dict, as_json: bool, output, human_render) -> None:
    if output:
        Path(output).write_text(json.dumps(doc, indent=2))
        console.print(f"[dim]wrote {output}[/]")
        return
    if as_json:
        click.echo(json.dumps(doc, indent=2))
    else:
        human_render()


@click.group()
@click.version_option(__version__)
@click.option("--debug", is_flag=True, help="re-raise errors with tracebacks")
def main(debug):
    """Hardware-free static analyzer for CUDA kernel binaries."""
    global _DEBUG
    _DEBUG = debug


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--kernel", "kernel_re", default=None, help="regex filter on kernel names")
@click.option("--json", "as_json", is_flag=True, help="emit JSON")
@click.option("--no-cache", is_flag=True, help="bypass the on-disk analysis cache")
@click.option("--output", "-o", default=None, help="write JSON to a file")
def ls(path, kernel_re, as_json, output, no_cache):
    """List kernels and headline resources in an artifact (fast: no disassembly)."""
    doc = _report_or_die(path, level="resources", kernel_re=kernel_re, use_cache=not no_cache)
    _emit(doc, as_json, output, lambda: render_ls(doc, console))


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--threads", type=str, default=None,
              help="block shape for occupancy + access analysis: '256' or '32,8[,1]' "
                   "(inferred from .reqntid/__launch_bounds__ metadata when omitted)")
@click.option("--carveout", type=int, default=None, help="smem carveout KB (default: max)")
@click.option("--kernel", "kernel_re", default=None, help="regex filter on kernel names")
@click.option("--arch", default=None, help="architecture for .ptx input (e.g. sm_120a)")
@click.option("--fast", is_flag=True, help="skip liveness analysis (no pressure curve)")
@click.option("--smem-dynamic", "smem_dynamic", type=click.IntRange(min=0), default=None,
              help="dynamic shared memory bytes passed at launch (not recorded "
                   "in the binary; applies to all matched kernels)")
@click.option("--verbose", is_flag=True, help="show toolchain provenance")
@click.option("--json", "as_json", is_flag=True, help="emit JSON")
@click.option("--no-cache", is_flag=True, help="bypass the on-disk analysis cache")
@click.option("--output", "-o", default=None, help="write JSON to a file")
def report(path, threads, carveout, kernel_re, arch, fast, smem_dynamic,
           verbose, as_json, output, no_cache):
    """Full static report: resources, pressure, spills, occupancy, access patterns."""
    doc = _report_or_die(path, threads=threads, carveout_kb=carveout,
                         kernel_re=kernel_re, arch=arch, fast=fast,
                         smem_dynamic=smem_dynamic, use_cache=not no_cache)

    def human():
        render_report(doc, console)
        if verbose:
            tc = doc.get("toolchain", {})
            console.print(f"\n[dim]toolchain ({tc.get('origin', '?')}):[/]")
            for t in ("nvdisasm", "cuobjdump", "ptxas"):
                info = tc.get(t, {})
                console.print(f"  [dim]{t}: {info.get('version', '?')}[/]")
        has_occ = any(k.get("occupancy") for u in doc["units"] for k in u["kernels"])
        if not threads and not has_occ:
            console.print("\n[dim]tip: pass --threads N (or 'X,Y') for occupancy, "
                          "cliff, and access analysis[/]")

    _emit(doc, as_json, output, human)


@main.command()
@click.option("--arch", required=True, help="e.g. sm_120, sm_90")
@click.option("--regs", type=click.IntRange(min=0), required=True)
@click.option("--threads", type=str, required=True,
              help="block shape: '256' or '32,8[,1]'")
@click.option("--smem", type=click.IntRange(min=0), default=0, help="static+dynamic smem bytes")
@click.option("--carveout", type=click.IntRange(min=0), default=None, help="carveout KB")
@click.option("--sweep", is_flag=True, help="sweep block sizes")
@click.option("--json", "as_json", is_flag=True)
@click.option("--output", "-o", default=None, help="write JSON to a file")
def occupancy(arch, regs, threads, smem, carveout, sweep, as_json, output):
    """Pure occupancy what-if — no binary needed."""
    try:
        spec = lookup(arch)
        _, total = parse_block_dims(threads)
    except (KeyError, ValueError) as e:
        err.print(f"[red]{e}[/]")
        sys.exit(2)
    occ = compute(spec, regs, total, smem_static=smem, carveout_kb=carveout)
    d = occ.to_dict()
    d["cliffs"] = find_cliffs(spec, occ)
    if sweep:
        d["sweep"] = [
            {"threads": o.threads_per_block, "occupancy_pct": o.occupancy_pct,
             "blocks_per_sm": o.blocks_per_sm}
            for o in sweep_block_sizes(spec, regs, smem_static=smem, carveout_kb=carveout)
        ]

    def human():
        console.print(
            f"[bold]{spec.sm}[/] ({spec.name}) — {regs} regs, {total} threads, "
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

    _emit(d, as_json, output, human)


@main.command()
@click.option("--fetch", is_flag=True, help="prefetch the full toolchain (incl. ptxas)")
def doctor(fetch):
    """Show toolchain resolution, cache state, and environment health."""
    import os
    import platform as _platform
    from .toolchain import cache_dir
    console.print(f"platform: {sys.platform} / {_platform.machine()}")
    if os.environ.get("CUXRAY_NO_FETCH"):
        console.print("[yellow]CUXRAY_NO_FETCH is set — auto-fetch disabled[/]")
    try:
        tc = resolve(need_ptxas=fetch, allow_fetch=fetch or None or True)
        d = tc.describe()
        console.print(f"toolchain origin: [bold]{d['origin']}[/]")
        for t in ("nvdisasm", "cuobjdump", "ptxas"):
            info = d.get(t)
            if info and "error" not in str(info.get("version")):
                console.print(f"  [green]✓[/] {t}: {info['version']}  [dim]{info['path']}[/]")
            else:
                console.print(f"  [yellow]-[/] {t}: not available"
                              " (fetched on demand for .ptx inputs)" if t == "ptxas" else "")
    except ToolchainError as e:
        console.print(f"[red]✗ toolchain: {e}[/]")
        sys.exit(2)
    cdir = cache_dir()
    reports = cdir / "reports"
    n = len(list(reports.glob("*.json"))) if reports.exists() else 0
    console.print(f"cache: {cdir}  [dim]({n} cached analysis result(s))[/]")


@main.command()
def schema():
    """Print the JSON Schema for report documents (cuxray.schema/1)."""
    from importlib.resources import files
    click.echo(files("cuxray.schema").joinpath("cuxray.schema.1.json").read_text())


@main.command()
@click.argument("old", type=click.Path(exists=True))
@click.argument("new", type=click.Path(exists=True))
@click.option("--threads", type=str, default=None)
@click.option("--kernel", "kernel_re", default=None)
@click.option("--fail-on-change", is_flag=True,
              help="exit 1 if any metric changed (or kernels added/removed)")
@click.option("--fail-on-regression", is_flag=True,
              help="exit 1 if any metric moved in the bad direction")
@click.option("--json", "as_json", is_flag=True)
@click.option("--no-cache", is_flag=True, help="bypass the on-disk analysis cache")
@click.option("--output", "-o", default=None, help="write JSON to a file")
def diff(old, new, threads, kernel_re, fail_on_change, fail_on_regression,
         as_json, output, no_cache):
    """Compare two artifacts kernel-by-kernel."""
    do = _report_or_die(old, threads=threads, kernel_re=kernel_re, use_cache=not no_cache)
    dn = _report_or_die(new, threads=threads, kernel_re=kernel_re, use_cache=not no_cache)
    try:
        d = diff_reports(do, dn, kernel_re)
    except ValueError as e:
        err.print(f"[red]{e}[/]")
        sys.exit(2)
    _emit(d, as_json, output, lambda: render_diff(d, console))
    if fail_on_regression and d["regressions"]:
        err.print(f"[red]{d['regressions']} kernel(s) regressed[/]")
        sys.exit(1)
    if fail_on_change and (d["changed"] or d["added"] or d["removed"]):
        err.print("[red]changes detected[/]")
        sys.exit(1)
    sys.exit(0)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.argument("expr", required=False, default=None)
@click.option("--budget", "budget_file", type=click.Path(exists=True), default=None,
              help="JSON budget file with per-kernel gates: "
                   '{"default": "...", "kernels": [{"match": re, "gate": "..."}], "threads": "256"}')
@click.option("--kernel", "kernel_re", default=None)
@click.option("--threads", type=str, default=None,
              help="block shape, required for bank_ways/uncoalesced metrics")
@click.option("--json", "as_json", is_flag=True)
@click.option("--sarif", default=None, help="write SARIF 2.1.0 to a file (GitHub code scanning)")
@click.option("--source-root", default=None,
              help="strip this prefix from source paths in SARIF for repo-relative URIs")
@click.option("--no-cache", is_flag=True, help="bypass the on-disk analysis cache")
@click.option("--output", "-o", default=None, help="write JSON to a file")
def gate(path, expr, budget_file, kernel_re, threads, as_json, output, sarif, source_root, no_cache):
    """CI gate: exit 1 if EXPR is violated
    (e.g. "spill_instrs==0, regs<=168, bank_ways<=2")."""
    if bool(expr) == bool(budget_file):
        err.print("[red]pass exactly one of EXPR or --budget FILE[/]")
        sys.exit(2)
    budget = None
    try:
        if budget_file:
            budget = json.loads(Path(budget_file).read_text())
            threads = threads or budget.get("threads")
            clauses = None
        else:
            clauses = parse_gate(expr)
    except (GateSyntaxError, json.JSONDecodeError) as e:
        err.print(f"[red]gate syntax:[/] {e}")
        sys.exit(2)
    doc = _report_or_die(path, kernel_re=kernel_re, threads=threads, use_cache=not no_cache)
    try:
        if budget is not None:
            violations, clauses = eval_budget(doc, budget)
        else:
            violations = eval_gate(doc, clauses)
    except (GateError, GateSyntaxError) as e:
        err.print(f"[red]gate error:[/] {e}")
        sys.exit(2)
    result = {"violations": violations, "passed": not violations}
    if sarif:
        Path(sarif).write_text(json.dumps(
            gate_to_sarif(path, clauses, violations, source_root), indent=2))
        console.print(f"[dim]wrote {sarif}[/]")

    def human():
        if violations:
            for v in violations:
                console.print(f"[red]✗[/] {v['kernel']} ({v['unit']}): {v['reason']}")
            console.print(f"[red]GATE FAILED[/] — {len(violations)} violation(s)")
        else:
            console.print(f"[green]✓ gate passed[/] ({', '.join(str(c) for c in clauses)})")

    _emit(result, as_json, output, human)
    sys.exit(1 if violations else 0)


if __name__ == "__main__":
    main()
