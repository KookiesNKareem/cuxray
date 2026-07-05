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
@click.argument("path", type=click.Path(exists=True))
@click.option("--threads", type=str, default=None,
              help="block shape: '256' or '32,8[,1]'")
@click.option("--grid", "grid_str", default=None, help="launch grid 'X[,Y[,Z]]'")
@click.option("--smem-dynamic", "smem_dynamic", type=click.IntRange(min=0), default=None,
              help="dynamic shared memory bytes at launch")
@click.option("--carveout", type=int, default=None, help="smem carveout KB")
@click.option("--kernel", "kernel_re", default=None, help="regex filter on kernel names")
@click.option("--arch", default=None, help="architecture for .ptx input")
@click.option("--peak-tflops", type=float, default=None)
@click.option("--peak-gbs", type=float, default=None)
@click.option("--profile", "profile_file", type=click.Path(exists=True), default=None,
              help="JSON {kernel_regex: runtime_fraction} — weight findings by "
                   "each kernel's share of end-to-end time (impact ranking)")
@click.option("--json", "as_json", is_flag=True)
@click.option("--no-cache", is_flag=True, help="bypass the on-disk analysis cache")
@click.option("--output", "-o", default=None, help="write JSON to a file")
def advise(path, threads, grid_str, smem_dynamic, carveout, kernel_re, arch,
           peak_tflops, peak_gbs, profile_file, as_json, output, no_cache):
    """Ranked, confidence-tagged design actions synthesized from the full
    analysis (spills, occupancy cliffs, bank/coalescing, grid traffic,
    per-byte schedule, tensor-core datapath ceiling)."""
    from .advise import advise as make_actions

    profile = _load_profile(profile_file)
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
                         kernel_re=kernel_re, arch=arch,
                         smem_dynamic=smem_dynamic, use_cache=not no_cache,
                         peak_tflops=peak_tflops, peak_gbs=peak_gbs,
                         grid_dims=grid_dims)
    results = []
    for unit in doc["units"]:
        for k in unit["kernels"]:
            w = _profile_weight(profile, k) if profile else 1.0
            actions = make_actions(k, arch=unit.get("arch"), weight=w)
            results.append({"unit": unit["label"], "kernel": k["name"],
                            "demangled": k.get("demangled"),
                            "weight": w, "actions": actions})
    out = {"schema": "cuxray.advise/1", "results": results}

    _COLOR = {"high": "red", "medium": "yellow", "low": "dim"}

    def human():
        any_action = False
        for r in results:
            if not r["actions"]:
                continue
            any_action = True
            name = r.get("demangled") or r["kernel"]
            wtag = (f"  [dim]· {r['weight']:.0%} of runtime[/]"
                    if profile else "")
            console.print(f"\n[bold]{name[:100]}[/]  [dim]({r['unit']})[/]{wtag}")
            for i, a in enumerate(r["actions"], 1):
                sev = _COLOR.get(a["severity"], "white")
                imp = f" · impact {a['impact']:g}" if a.get("impact") else ""
                console.print(f"  [{sev}]{i}. {a['title']}[/]  "
                              f"[dim]· {a['confidence']} confidence{imp}[/]")
                console.print(f"     {a['detail']}")
                for e in a.get("evidence", []):
                    console.print(f"     [dim]evidence: {e}[/]")
        if not any_action:
            console.print("[green]no design actions — kernel looks clean "
                          "at this launch config[/]")
            if not threads:
                console.print("[dim]tip: pass --threads for occupancy + access "
                              "actions[/]")

    _emit(out, as_json, output, human)
    sys.exit(0)


def _load_profile(path):
    """{kernel_regex: runtime_fraction} for --profile weighting, or None."""
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    return [(re.compile(k), float(v)) for k, v in data.items()]


def _profile_weight(profile, kernel: dict) -> float:
    name = kernel.get("demangled") or kernel.get("name", "")
    for pat, w in profile:
        if pat.search(name) or pat.search(kernel.get("name", "")):
            return w
    return 0.0  # kernels absent from the profile contribute nothing


def _kernel_summary(k: dict) -> dict:
    """Flat headline metrics for compare/survey."""
    r = k.get("resources") or {}
    occ = k.get("occupancy") or {}
    acc = k.get("access") or {}
    sp = k.get("spills") or {}
    hot = next((lp for lp in (k.get("roofline") or [])
                if lp.get("loop_depth", 0) >= 1), None)
    return {
        "regs": r.get("regs"),
        "occupancy_pct": occ.get("occupancy_pct"),
        "limiter": occ.get("limiter"),
        "spill_bytes": (sp.get("store_bytes", 0) or 0) + (sp.get("load_bytes", 0) or 0),
        "conflicted_shared": acc.get("conflicted_shared_accesses"),
        "uncoalesced_global": acc.get("uncoalesced_global_accesses"),
        "hot_loop_ai": hot.get("est_arithmetic_intensity") if hot else None,
        "hot_loop_bytes_iter": hot.get("est_global_bytes_per_warp_iter") if hot else None,
        "tensor_crossover": (hot.get("tensor_crossover") or {}).get(
            "tensor_speedup_ceiling") if hot else None,
    }


@main.command()
@click.option("--bytes", "dram_bytes", type=float, required=True,
              help="total DRAM traffic for the launch (bytes) — from problem dims")
@click.option("--macs", type=float, default=0, help="multiply-accumulates (compute-bound)")
@click.option("--precision", default="int8", help="int8/fp16/fp32 for the MAC peak")
@click.option("--datapath", default="simt-int", help="simt-int/simt-fp/tensor")
@click.option("--sms", type=int, required=True, help="device SM count")
@click.option("--clock", type=float, required=True, help="sustained clock GHz")
@click.option("--cc", default=None, help="compute capability e.g. sm_80 (for the MAC peak)")
@click.option("--peak-gbs", type=float, required=True, help="measured achievable DRAM GB/s")
@click.option("--calibrate", type=click.Path(exists=True), default=None,
              help="OPTIONAL empirical fit: JSON [[ideal_us, measured_us], ...] of "
                   "the SAME kernel family on THIS device — adds an ESTIMATE line")
@click.option("--json", "as_json", is_flag=True)
@click.option("--output", "-o", default=None)
def roofline(dram_bytes, macs, precision, datapath, sms, clock, cc, peak_gbs,
             calibrate, as_json, output):
    """Roofline floor for a launch: the fastest this work can run on this
    device, and whether it is memory- or compute-bound. A FLOOR, not a
    prediction — a real kernel runs slower by its efficiency. With
    --calibrate (measurements of the same family on this device) it also
    prints a fitted wall-clock ESTIMATE, clearly labelled as empirical."""
    from .analyze.perfmodel import (Calibration, Device, Work, fit, ideal_us,
                                    predict)
    from .archspec import lookup

    dev_cc = None
    if cc:
        try:
            dev_cc = lookup(cc).cc
        except (KeyError, ValueError):
            pass
    dev = Device(sms=sms, clock_ghz=clock, achievable_gbs=peak_gbs, cc=dev_cc)
    work = Work(dram_bytes=dram_bytes, macs=int(macs), precision=precision,
                datapath=datapath)
    base = ideal_us(work, dev)
    out = {"schema": "cuxray.roofline/1",
           "t_ideal_us": round(base["t_ideal_us"], 3),
           "t_mem_ideal_us": round(base["t_mem_ideal_us"], 3),
           "t_compute_ideal_us": round(base["t_compute_ideal_us"], 3),
           "bound": base["bound"]}
    estimate = None
    if calibrate:
        samples = [(float(a), float(b)) for a, b in
                   json.loads(Path(calibrate).read_text())]
        calib = fit(samples)
        estimate = predict(work, dev, calib)
        out["estimate_us"] = estimate["us"]
        out["calibration"] = {"e_sat": calib.e_sat, "t_fixed_us": calib.t_fixed_us,
                              "n_samples": len(samples)}

    def human():
        console.print(f"[bold]roofline floor: {out['t_ideal_us']} µs[/]  "
                      f"[dim]({out['bound']}-bound)[/]")
        console.print(f"  memory {out['t_mem_ideal_us']} µs · "
                      f"compute {out['t_compute_ideal_us']} µs")
        console.print("[dim]a lower bound — a real kernel runs slower by its "
                      "efficiency[/]")
        if estimate:
            console.print(f"\n  [yellow]empirical estimate: ~{estimate['us']} µs[/] "
                          f"[dim](fitted e_sat={out['calibration']['e_sat']}, "
                          f"t_fixed={out['calibration']['t_fixed_us']}µs from "
                          f"{out['calibration']['n_samples']} measurements of this "
                          "family — NOT a static result; verify on hardware)[/]")

    _emit(out, as_json, output, human)
    sys.exit(0)


@main.command()
@click.argument("old", type=click.Path(exists=True))
@click.argument("new", type=click.Path(exists=True))
@click.option("--threads", type=str, default=None, help="block shape for both")
@click.option("--kernel", "kernel_re", default=None, help="regex filter")
@click.option("--arch", default=None)
@click.option("--json", "as_json", is_flag=True)
@click.option("--no-cache", is_flag=True)
@click.option("--output", "-o", default=None)
def compare(old, new, threads, kernel_re, arch, as_json, output, no_cache):
    """Side-by-side headline metrics of two cubins (regs, occupancy, spills,
    conflicts, roofline, tensor-core ceiling) — the "am I actually winning?"
    view for an optimization."""
    da = _report_or_die(old, threads=threads, kernel_re=kernel_re, arch=arch,
                        use_cache=not no_cache)
    db = _report_or_die(new, threads=threads, kernel_re=kernel_re, arch=arch,
                        use_cache=not no_cache)

    def flat(doc):
        return {k["name"]: (_kernel_summary(k), u["label"])
                for u in doc["units"] for k in u["kernels"]}
    fa, fb = flat(da), flat(db)
    names = [n for n in fa if n in fb] or (list(fa)[:1] + list(fb)[:1])
    pairs = []
    for n in fa:
        # pair by exact name, else positionally when each side has one kernel
        if n in fb:
            pairs.append((n, fa[n][0], fb[n][0]))
    if not pairs and len(fa) == 1 and len(fb) == 1:
        (na, (sa, _)), (nb, (sb, _)) = list(fa.items())[0], list(fb.items())[0]
        pairs.append((f"{na[:30]} vs {nb[:30]}", sa, sb))

    out = {"schema": "cuxray.compare/1",
           "pairs": [{"kernel": n, "old": a, "new": b} for n, a, b in pairs]}
    _METRICS = [("regs", "regs", False), ("occupancy_pct", "occupancy %", True),
                ("spill_bytes", "spill B", False),
                ("conflicted_shared", "bank-conflict acc", False),
                ("uncoalesced_global", "uncoalesced acc", False),
                ("hot_loop_ai", "hot-loop AI", None),
                ("hot_loop_bytes_iter", "hot-loop B/iter", False),
                ("tensor_crossover", "tensor ceiling x", None)]

    def human():
        if not pairs:
            console.print("[yellow]no comparable kernels (name mismatch; pass "
                          "--kernel to align)[/]")
            return
        from rich.table import Table
        for n, a, b in pairs:
            console.print(f"\n[bold]{n[:90]}[/]")
            t = Table(box=None, header_style="dim", padding=(0, 2))
            t.add_column("metric"); t.add_column("old"); t.add_column("new"); t.add_column("Δ")
            for key, label, higher_better in _METRICS:
                va, vb = a.get(key), b.get(key)
                if va is None and vb is None:
                    continue
                delta = ""
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    d = vb - va
                    if d and higher_better is not None:
                        good = (d > 0) == higher_better
                        delta = f"[{'green' if good else 'red'}]{d:+g}[/]"
                    elif d:
                        delta = f"{d:+g}"
                t.add_row(label, str(va), str(vb), delta)
            console.print(t)

    _emit(out, as_json, output, human)
    sys.exit(0)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--threads", type=str, default=None, help="block shape for occupancy/access")
@click.option("--arch", default=None)
@click.option("--profile", "profile_file", type=click.Path(exists=True), default=None,
              help="JSON {kernel_regex: runtime_fraction} — rank by whole-program impact")
@click.option("--top", type=int, default=15, help="show the N highest-impact kernels")
@click.option("--json", "as_json", is_flag=True)
@click.option("--no-cache", is_flag=True)
@click.option("--output", "-o", default=None)
def survey(path, threads, arch, profile_file, top, as_json, output, no_cache):
    """Sweep every kernel in an artifact and rank them by total recoverable
    impact — the corpus-scan worklist for finding where a win might live."""
    from .advise import advise as make_actions

    profile = _load_profile(profile_file)
    doc = _report_or_die(path, threads=threads, arch=arch, use_cache=not no_cache)
    rows = []
    for unit in doc["units"]:
        for k in unit["kernels"]:
            w = _profile_weight(profile, k) if profile else 1.0
            actions = make_actions(k, arch=unit.get("arch"), weight=w)
            total = round(sum(a.get("impact", 0) for a in actions), 2)
            if actions:
                rows.append({
                    "unit": unit["label"], "kernel": k["name"],
                    "demangled": k.get("demangled"), "weight": w,
                    "total_impact": total, "n_actions": len(actions),
                    "top_action": actions[0]["title"],
                    "actions": actions,
                })
    rows.sort(key=lambda r: -r["total_impact"])
    out = {"schema": "cuxray.survey/1", "ranked": rows}

    def human():
        if not rows:
            console.print("[green]no actionable findings across the artifact[/]")
            return
        from rich.table import Table
        t = Table(box=None, header_style="bold", padding=(0, 2))
        t.add_column("#"); t.add_column("impact", justify="right")
        if profile:
            t.add_column("runtime", justify="right")
        t.add_column("kernel"); t.add_column("top action")
        for i, r in enumerate(rows[:top], 1):
            name = (r.get("demangled") or r["kernel"])
            name = name[:54] + "…" if len(name) > 55 else name
            cells = [str(i), f"{r['total_impact']:g}"]
            if profile:
                cells.append(f"{r['weight']:.0%}")
            cells += [name, r["top_action"]]
            t.add_row(*cells)
        console.print(t)
        if len(rows) > top:
            console.print(f"[dim](+{len(rows) - top} more; --top to show more, "
                          "`cuxray advise --kernel <name>` to drill in)[/]")

    _emit(out, as_json, output, human)
    sys.exit(0)


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
    from .analyze.solver import patterns_from_accesses, solve_grouped
    from .parse import cfgdot, sass
    from .report import parse_block_dims
    from .ingest import ingest

    def _sol_json(sol):
        return {"cutlass": sol.cutlass, "cute_type": sol.cute_type,
                "formula": sol.formula, "b": sol.b, "m": sol.m, "s": sol.s,
                "cuda_snippet": sol.cuda_snippet(),
                "per_pattern": sol.per_pattern}

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
            grouped = solve_grouped(patterns)
            results.append({
                "unit": unit.label, "kernel": name,
                "conflicted_sites": len(conflicted),
                "shared_accesses": len(patterns),
                "global": [_sol_json(s) for s in (grouped["global"] or [])],
                "groups": [{
                    "sites": grp["sites"],
                    "solutions": [_sol_json(s) for s in grp["solutions"]],
                } for grp in grouped["groups"]],
                "unsolved_sites": grouped["unsolved"],
            })

    doc = {"schema": "cuxray.solve/1", "results": results}

    def _print_group(grp, scope):
        sols = grp["solutions"]
        best = sols[0]
        console.print(f"  [green]{scope}:[/] [bold]{best['cutlass']}[/]  "
                      f"(zero smem cost, verified)")
        console.print(f"    apply to byte offsets: {best['formula']}")
        worst = max(best["per_pattern"], key=lambda pp: pp["before"])
        console.print(f"    e.g. {worst['label']}: {worst['before']}-way → clean")
        if len(sols) > 1:
            alts = ", ".join(s2["cutlass"] for s2 in sols[1:])
            console.print(f"    [dim]alternatives: {alts}[/]")

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
            if r["global"]:
                _print_group({"solutions": r["global"]},
                             "solution (all accesses)")
                console.print("")
                for line in r["global"][0]["cuda_snippet"].split("\n"):
                    console.print(f"    [dim]{line}[/]")
            elif r["groups"]:
                console.print("  [yellow]no single swizzle cleans every "
                              "access; per-tile layouts (apply each to its "
                              "own shared region):[/]")
                for grp in r["groups"]:
                    console.print(f"  [cyan]tile[/] {', '.join(grp['sites'])}")
                    _print_group(grp, "swizzle")
                if r["unsolved_sites"]:
                    console.print(f"  [red]no swizzle found for:[/] "
                                  f"{', '.join(r['unsolved_sites'])} "
                                  "— consider padding or restructuring")
            else:
                console.print("  [yellow]no swizzle cleans these accesses — "
                              "consider restructuring or padding[/]")

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
@click.option("--allow-unverified-arch", is_flag=True,
              help="decode control bits on architectures whose encoding "
                   "cuxray has not validated against hardware (results marked "
                   "unverified)")
@click.option("--json", "as_json", is_flag=True)
@click.option("--output", "-o", default=None, help="write JSON to a file")
def sched(path, kernel_re, arch, threads, allow_unverified_arch, as_json, output):
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
        verified = ctrl.arch_supported(arch_eff)
        if not verified:
            if not allow_unverified_arch:
                err.print(f"[yellow]{unit.label}: control-bit encoding for "
                          f"{arch_eff or 'unknown arch'} is unverified — "
                          "skipping (supported: sm_80-sm_90a; pass "
                          "--allow-unverified-arch to decode anyway)[/]")
                continue
            err.print(f"[yellow]{unit.label}: decoding {arch_eff} control bits "
                      "with the sm_80-sm_90a model — UNVERIFIED, cycle "
                      "estimates may be wrong[/]")
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
                                "arch": arch_eff, "loops": rows,
                                "control_bits_verified": verified})

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
@click.option("--commands", is_flag=True,
              help="print the schema for advise/sched/solve/why/tune --json "
                   "instead of the report schema")
def schema(commands):
    """Print the JSON Schema for cuxray JSON output."""
    from importlib.resources import files
    fname = ("cuxray.commands.1.json" if commands
             else "cuxray.schema.1.json")
    click.echo(files("cuxray.schema").joinpath(fname).read_text())


@main.command()
@click.argument("old", type=click.Path(exists=True))
@click.argument("new", type=click.Path(exists=True))
@click.option("--threads", type=str, default=None)
@click.option("--smem-dynamic", "smem_dynamic", type=click.IntRange(min=0), default=None,
              help="dynamic shared memory bytes at launch")
@click.option("--carveout", type=int, default=None, help="smem carveout KB")
@click.option("--kernel", "kernel_re", default=None)
@click.option("--fail-on-change", is_flag=True,
              help="exit 1 if any metric changed (or kernels added/removed)")
@click.option("--fail-on-regression", is_flag=True,
              help="exit 1 if any metric moved in the bad direction")
@click.option("--json", "as_json", is_flag=True)
@click.option("--no-cache", is_flag=True, help="bypass the on-disk analysis cache")
@click.option("--output", "-o", default=None, help="write JSON to a file")
def diff(old, new, threads, smem_dynamic, carveout, kernel_re, fail_on_change,
         fail_on_regression, as_json, output, no_cache):
    """Compare two artifacts kernel-by-kernel."""
    do = _report_or_die(old, threads=threads, kernel_re=kernel_re,
                        smem_dynamic=smem_dynamic, carveout_kb=carveout,
                        use_cache=not no_cache)
    dn = _report_or_die(new, threads=threads, kernel_re=kernel_re,
                        smem_dynamic=smem_dynamic, carveout_kb=carveout,
                        use_cache=not no_cache)
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
@click.option("--smem-dynamic", "smem_dynamic", type=click.IntRange(min=0), default=None,
              help="dynamic shared memory bytes at launch (occupancy parity with report)")
@click.option("--carveout", type=int, default=None, help="smem carveout KB")
@click.option("--json", "as_json", is_flag=True)
@click.option("--sarif", default=None, help="write SARIF 2.1.0 to a file (GitHub code scanning)")
@click.option("--source-root", default=None,
              help="strip this prefix from source paths in SARIF for repo-relative URIs")
@click.option("--no-cache", is_flag=True, help="bypass the on-disk analysis cache")
@click.option("--output", "-o", default=None, help="write JSON to a file")
def gate(path, expr, budget_file, kernel_re, threads, smem_dynamic, carveout,
         as_json, output, sarif, source_root, no_cache):
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
    doc = _report_or_die(path, kernel_re=kernel_re, threads=threads,
                         smem_dynamic=smem_dynamic, carveout_kb=carveout,
                         use_cache=not no_cache)
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
