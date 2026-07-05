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
@click.option("--peak-tflops", type=float, default=None,
              help="device peak TFLOP/s for roofline bound classification (estimate)")
@click.option("--peak-gbs", type=float, default=None,
              help="device peak memory bandwidth GB/s for roofline classification")
@click.option("--grid", "grid_str", default=None,
              help="launch grid 'X[,Y[,Z]]' — adds worst-case (L2-cold) traffic "
                   "amplification of block-invariant reads across the grid")
@click.option("--json", "as_json", is_flag=True, help="emit JSON")
@click.option("--no-cache", is_flag=True, help="bypass the on-disk analysis cache")
@click.option("--output", "-o", default=None, help="write JSON to a file")
def report(path, threads, carveout, kernel_re, arch, fast, smem_dynamic,
           verbose, peak_tflops, peak_gbs, grid_str, as_json, output, no_cache):
    """Full static report: resources, pressure, spills, occupancy, access
    patterns, per-loop roofline estimates."""
    grid_dims = None
    if grid_str:
        try:
            parts = [int(x) for x in grid_str.split(",")]
            if not (1 <= len(parts) <= 3 and all(x >= 1 for x in parts)):
                raise ValueError(grid_str)
        except ValueError:
            err.print(f"[red]invalid --grid:[/] {grid_str!r} (want X[,Y[,Z]])")
            sys.exit(2)
        while len(parts) < 3:
            parts.append(1)
        grid_dims = tuple(parts)
    doc = _report_or_die(path, threads=threads, carveout_kb=carveout,
                         kernel_re=kernel_re, arch=arch, fast=fast,
                         smem_dynamic=smem_dynamic, use_cache=not no_cache,
                         peak_tflops=peak_tflops, peak_gbs=peak_gbs,
                         grid_dims=grid_dims)

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
@click.argument("path", type=click.Path(exists=True))
@click.option("--threads", type=str, required=True,
              help="block shape: '256' or '32,8[,1]'")
@click.option("--kernel", "kernel_re", default=None, help="regex filter on kernel names")
@click.option("--arch", default=None, help="architecture for .ptx input")
@click.option("--json", "as_json", is_flag=True)
@click.option("--output", "-o", default=None, help="write JSON to a file")
def solve(path, threads, kernel_re, arch, as_json, output):
    """Derive an XOR swizzle making ALL shared accesses conflict-free.

    Searches CUTLASS-style Swizzle<B,M,S> transforms and returns only
    layouts verified clean for every shared access in each kernel."""
    from .analyze.access import analyze_accesses
    from .analyze.solver import patterns_from_accesses, solve as solve_layout
    from .parse import cfgdot, sass
    from .report import parse_block_dims
    from .ingest import ingest

    need_ptxas = str(path).endswith(".ptx")
    tc = _toolchain(need_ptxas)
    try:
        dims, _total = parse_block_dims(threads)
        units = ingest(Path(path), tc, arch=arch)
    except Exception as e:
        err.print(f"[red]error:[/] {e}")
        sys.exit(2)

    pat = re.compile(kernel_re) if kernel_re else None
    results = []
    for unit in units:
        cubin = str(unit.cubin)
        dis = sass.parse_gi(tc.run("nvdisasm", ["-c", "-gi", cubin]))
        try:
            cfg = cfgdot.parse(tc.run("nvdisasm", ["-cfg", cubin]))
        except Exception:
            cfg = {}
        for name, func in dis.functions.items():
            if pat and not pat.search(name):
                continue
            depths = cfg[name].loop_depth if name in cfg else {}
            acc = analyze_accesses(func, dims, depths, keep_vecs=True)
            patterns = patterns_from_accesses(acc["accesses"])
            conflicted = [p for p in patterns if p.ways_before > 1]
            if not conflicted:
                continue
            sols = solve_layout(patterns)
            results.append({
                "unit": unit.label, "kernel": name,
                "conflicted_sites": len(conflicted),
                "shared_accesses": len(patterns),
                "solutions": [{
                    "cutlass": sol.cutlass, "cute_type": sol.cute_type,
                    "formula": sol.formula,
                    "b": sol.b, "m": sol.m, "s": sol.s,
                    "cuda_snippet": sol.cuda_snippet(),
                    "per_pattern": sol.per_pattern,
                } for sol in sols],
            })

    doc = {"schema": "cuxray.solve/1", "results": results}

    def human():
        if not results:
            console.print("[green]no conflicted shared accesses found — "
                          "nothing to solve[/]")
            return
        for r in results:
            console.print(f"\n[bold]{r['kernel'][:100]}[/]  "
                          f"[dim]({r['unit']})[/]")
            console.print(f"  {r['conflicted_sites']} conflicted of "
                          f"{r['shared_accesses']} shared accesses")
            if not r["solutions"]:
                console.print("  [yellow]no single swizzle cleans every access "
                              "— consider restructuring or padding[/]")
                continue
            best = r["solutions"][0]
            console.print(f"  [green]solution:[/] [bold]{best['cutlass']}[/]  "
                          f"(zero smem cost, verified on all accesses)")
            console.print(f"    apply to byte offsets: {best['formula']}")
            worst = max(best["per_pattern"], key=lambda pp: pp["before"])
            console.print(f"    e.g. {worst['label']}: "
                          f"{worst['before']}-way → clean")
            if len(r["solutions"]) > 1:
                alts = ", ".join(s2["cutlass"] for s2 in r["solutions"][1:])
                console.print(f"    [dim]alternatives: {alts}[/]")
            console.print("")
            for line in best["cuda_snippet"].split("\n"):
                console.print(f"    [dim]{line}[/]")

    _emit(doc, as_json, output, human)
    sys.exit(0)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--threads", type=str, required=True,
              help="block shape: '256' or '32,8[,1]'")
@click.option("--kernel", "kernel_re", default=None, help="regex filter on kernel names")
@click.option("--addr", "addr_hex", default=None,
              help="explain one instruction (hex address); default: every unanalyzed access")
@click.option("--arch", default=None, help="architecture for .ptx input")
@click.option("--json", "as_json", is_flag=True)
@click.option("--output", "-o", default=None, help="write JSON to a file")
def why(path, threads, kernel_re, addr_hex, arch, as_json, output):
    """Explain why an access is (or is not) analyzable.

    Backward-slices the address computation, printing each defining
    instruction with its abstract lane-value and marking the first
    instruction where precision is lost."""
    from .analyze.whytrace import why_kernel
    from .parse import cfgdot, sass
    from .report import parse_block_dims
    from .ingest import ingest

    tc = _toolchain(str(path).endswith(".ptx"))
    try:
        dims, _total = parse_block_dims(threads)
        units = ingest(Path(path), tc, arch=arch)
    except Exception as e:
        err.print(f"[red]error:[/] {e}")
        sys.exit(2)

    target = int(addr_hex, 16) if addr_hex else None
    pat = re.compile(kernel_re) if kernel_re else None
    results = []
    for unit in units:
        cubin = str(unit.cubin)
        dis = sass.parse_gi(tc.run("nvdisasm", ["-c", "-gi", cubin]))
        try:
            cfg = cfgdot.parse(tc.run("nvdisasm", ["-cfg", cubin]))
        except Exception:
            cfg = {}
        for name, func in dis.functions.items():
            if pat and not pat.search(name):
                continue
            depths = cfg[name].loop_depth if name in cfg else {}
            slices = why_kernel(func, dims, target_addr=target, loop_depth=depths)
            if slices:
                results.append({"unit": unit.label, "kernel": name, "slices": slices})

    doc = {"schema": "cuxray.why/1", "results": results}

    def human():
        if not results:
            console.print("[green]every access analyzable — nothing to explain[/]")
            return
        for r in results:
            console.print(f"\n[bold]{r['kernel'][:100]}[/]  [dim]({r['unit']})[/]")
            for s in r["slices"]:
                t = s["target"]
                loc = f"  [{t['file'].rsplit('/', 1)[-1]}:{t['line']}]" if t.get("line") else ""
                console.print(f"  [yellow]{hex(t['addr'])}[/] {t['opcode']} "
                              f"[{t['mem']}]{loc}"
                              + (f"  — {s['reason']}" if s.get("reason") else ""))
                console.print(f"    address = {s['address_value']}")
                for row in reversed(s["chain"]):
                    mark = "  [red]◀ precision lost here[/]" if row.get("degrades_here") else ""
                    console.print(f"    {hex(row['addr']):>8} {row['opcode']:<16} "
                                  f"{row['operands'][:60]:<62} {row['value'][:44]}{mark}")
                if s["unresolved_inputs"]:
                    console.print(f"    [dim]inputs not sliced: "
                                  f"{', '.join(s['unresolved_inputs'])}[/]")

    _emit(doc, as_json, output, human)
    sys.exit(0)


@main.command()
@click.argument("src", type=click.Path(exists=True))
@click.option("--arch", required=True, help="e.g. sm_120a")
@click.option("--define", "-D", "defines", multiple=True,
              help="K=v1,v2,... — swept as a Cartesian product")
@click.option("--flag", "flags", multiple=True, help="extra nvcc flag (repeatable)")
@click.option("--threads", type=str, default=None, help="block shape for occupancy/access columns")
@click.option("--smem-dynamic", "smem_dynamic", type=click.IntRange(min=0), default=0)
@click.option("--json", "as_json", is_flag=True)
@click.option("--output", "-o", default=None, help="write JSON to a file")
def tune(src, arch, defines, flags, threads, smem_dynamic, as_json, output):
    """Compile a CUDA source across a matrix of -D defines and rank the
    variants statically (regs, spills, occupancy, bank conflicts)."""
    from .report import parse_block_dims
    from .tunematrix import sweep_matrix

    dmap = {}
    for d in defines:
        if "=" not in d:
            err.print(f"[red]--define needs K=v1,v2 form:[/] {d}")
            sys.exit(2)
        k, vals = d.split("=", 1)
        dmap[k] = vals.split(",")
    tc = _toolchain()
    try:
        dims, total = parse_block_dims(threads)
        doc = sweep_matrix(src, tc, arch, dmap, list(flags),
                           block_dims=dims, threads=total,
                           smem_dynamic=smem_dynamic)
    except Exception as e:
        if _DEBUG:
            raise
        err.print(f"[red]error:[/] {e}")
        sys.exit(2)

    def human():
        from rich.table import Table
        t = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        keys = sorted(dmap)
        for col in (*keys, "regs", "smem", "spill B", "hot spills",
                    "bank ways", "uncoal", "blocks/SM", "occ %", ""):
            t.add_column(col, justify="right" if col else "left")
        for v in doc["variants"]:
            if "error" in v:
                t.add_row(*[str(v["config"].get(k, "")) for k in keys],
                          "[red]compile error[/]", "", "", "", "", "", "", "", "")
                continue
            ks = v["kernels"]
            regs = max((k["regs"] or 0) for k in ks)
            smem = max(k["smem"] for k in ks)
            spills = sum(k["spill_bytes"] for k in ks)
            hot = sum(k["hot_spills"] for k in ks)
            ways = max((k.get("bank_ways") or 1) for k in ks)
            unc = sum((k.get("uncoalesced") or 0) for k in ks)
            blocks = min((k.get("blocks_per_sm", 0) or 0) for k in ks)
            occ = min((k.get("occupancy_pct", 0) or 0) for k in ks)
            t.add_row(*[str(v["config"].get(k, "")) for k in keys],
                      str(regs), str(smem), str(spills), str(hot),
                      str(ways), str(unc), str(blocks), str(occ),
                      "[green]● pareto[/]" if v.get("pareto") else "")
        console.print(t)
        if doc["failed"]:
            console.print(f"[yellow]{doc['failed']} variant(s) failed to compile[/]")

    _emit(doc, as_json, output, human)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--kernel", "kernel_re", default=None, help="regex filter on kernel names")
@click.option("--arch", default=None, help="architecture for .ptx input")
@click.option("--threads", type=str, default=None,
              help="block shape — adds bytes/iter + stall cycles per 512 B")
@click.option("--json", "as_json", is_flag=True)
@click.option("--output", "-o", default=None, help="write JSON to a file")
def sched(path, kernel_re, arch, threads, as_json, output):
    """Per-loop cycle estimates from the compiler's embedded schedule
    (sm_80-sm_90a; encodings for other architectures are unverified).

    With --threads, loops also get global bytes/iter and stall cycles per
    512 B streamed — comparable across kernels of different widths."""
    from .analyze.schedule import loop_schedule
    from .parse import cfgdot, ctrl, sass
    from .ingest import ingest

    tc = _toolchain(str(path).endswith(".ptx"))
    try:
        units = ingest(Path(path), tc, arch=arch)
    except Exception as e:
        err.print(f"[red]error:[/] {e}")
        sys.exit(2)
    pat = re.compile(kernel_re) if kernel_re else None

    results = []
    for unit in units:
        cubin = str(unit.cubin)
        dis = sass.parse_gi(tc.run("nvdisasm", ["-c", "-gi", cubin]))
        from .parse import elf as _elf
        arch_eff = dis.target or _elf.sm_arch(unit.cubin.read_bytes())
        if not ctrl.arch_supported(arch_eff):
            err.print(f"[yellow]{unit.label}: control-bit encoding for "
                      f"{arch_eff or 'unknown arch'} is unverified — skipping "
                      "(supported: sm_80-sm_90a)[/]")
            continue
        controls = ctrl.parse_sass_controls(
            tc.run("cuobjdump", ["-sass", cubin]))
        try:
            cfg = cfgdot.parse(tc.run("nvdisasm", ["-cfg", cubin]))
        except Exception:
            cfg = {}
        dims = None
        if threads:
            from .report import parse_block_dims
            dims, _total = parse_block_dims(threads)
        for name, func in dis.functions.items():
            if pat and not pat.search(name):
                continue
            bpi = None
            if dims is not None:
                from .analyze.access import analyze_accesses
                from .analyze.roofline import loop_report
                fcfg = cfg.get(name)
                depths = fcfg.loop_depth if fcfg else {}
                acc = analyze_accesses(func, dims, depths)
                bpi = {r["header"]: r["est_global_bytes_per_warp_iter"]
                       for r in loop_report(func, fcfg, acc["accesses"])
                       if r.get("est_global_bytes_per_warp_iter")}
            rows = loop_schedule(func, cfg.get(name), controls.get(name, {}), bpi)
            if rows:
                results.append({"unit": unit.label, "kernel": name,
                                "arch": arch_eff, "loops": rows})

    doc = {"schema": "cuxray.sched/1", "results": results}

    def human():
        if not results:
            console.print("[dim]no loops with schedule data[/]")
            return
        for r in results:
            console.print(f"\n[bold]{r['kernel'][:100]}[/]  [dim]{r['arch']}[/]")
            for lp in r["loops"][:4]:
                span = (f"lines {lp['line_span'][0]}-{lp['line_span'][1]}"
                        if lp["line_span"] else lp["header"])
                console.print(
                    f"  [magenta]est.[/] loop {span} (depth {lp['loop_depth']}): "
                    f"[bold]{lp['est_issue_stall_cycles_per_iter']}[/] issue+stall "
                    f"cycles/iter · {lp['scoreboard_waits_per_iter']} scoreboard "
                    f"wait(s) not included"
                )
                if lp.get("est_stall_cycles_per_512B") is not None:
                    console.print(
                        f"      [bold]{lp['est_stall_cycles_per_512B']}[/] stall "
                        f"cycles per 512 B streamed "
                        f"({lp['est_global_bytes_per_warp_iter']} B/iter)")
                if lp["top_stall_lines"]:
                    for t in lp["top_stall_lines"][:3]:
                        loc = f"{(t['file'] or '?').rsplit('/', 1)[-1]}:{t['line']}"
                        console.print(f"      {loc}: {t['est_stall_cycles']} cycles")
                else:  # no lineinfo in the binary — attribute by opcode
                    for t in lp.get("top_stall_opcodes", [])[:4]:
                        console.print(f"      {t['opcode']:<10} x{t['count']}: "
                                      f"{t['est_stall_cycles']} cycles")

    _emit(doc, as_json, output, human)


@main.command(name="tune-regs")
@click.argument("ptx", type=click.Path(exists=True))
@click.option("--arch", default=None, help="e.g. sm_120a (default: PTX .target)")
@click.option("--threads", type=str, default=None,
              help="block shape for occupancy columns")
@click.option("--caps", default=None,
              help="comma-separated maxrregcount ladder (default: 24..255)")
@click.option("--smem-dynamic", "smem_dynamic", type=click.IntRange(min=0), default=0)
@click.option("--json", "as_json", is_flag=True)
@click.option("--output", "-o", default=None, help="write JSON to a file")
def tune_regs(ptx, arch, threads, caps, smem_dynamic, as_json, output):
    """Map the -maxrregcount frontier: recompile at a ladder of register caps
    (no GPU) and report actual regs, spills, and occupancy for each."""
    from .report import parse_block_dims
    from .tune import DEFAULT_CAPS, sweep_regcaps

    tc = _toolchain(need_ptxas=True)
    try:
        _dims, total = parse_block_dims(threads)
        cap_list = (tuple(int(c) for c in caps.split(",")) if caps else DEFAULT_CAPS)
        doc = sweep_regcaps(ptx, tc, arch=arch, caps=cap_list,
                            threads=total, smem_dynamic=smem_dynamic)
    except Exception as e:
        if _DEBUG:
            raise
        err.print(f"[red]error:[/] {e}")
        sys.exit(2)

    def human():
        from rich.table import Table
        for k in doc["kernels"]:
            console.print(f"\n[bold]{k['kernel'][:100]}[/]  [dim]{doc['arch']}[/]")
            t = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
            for col in ("cap", "regs", "spill bytes", "spill instrs", "top spill line",
                        "blocks/SM", "occupancy", ""):
                t.add_column(col, justify="right" if col != "" else "left")
            for r in k["rows"]:
                occ = f"{r.get('occupancy_pct', '')}%" if r.get("occupancy_pct") is not None else "-"
                blocks = str(r.get("blocks_per_sm", "-"))
                mark = "[green]● pareto[/]" if r.get("pareto") else ""
                spill_col = f"[red]{r['spill_bytes']}[/]" if r["spill_bytes"] else "0"
                t.add_row(str(r["cap"] or "none"), str(r["regs"]), spill_col,
                          str(r["spill_instrs"]),
                          str(r["spill_top_line"] or "-"), blocks, occ, mark)
            console.print(t)
        if not threads:
            console.print("[dim]tip: pass --threads for occupancy columns and "
                          "Pareto marking[/]")

    _emit(doc, as_json, output, human)


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
