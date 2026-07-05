"""Terminal rendering of report/diff documents (rich)."""

from __future__ import annotations

from rich.console import Console

from .diffgate import HIGHER_BETTER
from rich.table import Table
from rich.text import Text


def _fmt_bytes(n) -> str:
    if n is None:
        return "-"
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024} KB"
    return f"{n} B"


def render_report(doc: dict, console: Console) -> None:
    shown = 0
    for unit in doc["units"]:
        if not unit["kernels"]:
            continue
        shown += 1
        console.print(f"\n[bold cyan]{unit['label']}[/]  [dim]{unit['arch'] or '?'}[/]")
        for k in unit["kernels"]:
            _render_kernel(k, console)
    skipped = len(doc["units"]) - shown
    if skipped:
        console.print(f"\n[dim]({skipped} unit(s) with no matching kernels not shown)[/]")


def _shorten(name: str, limit: int = 140) -> str:
    if len(name) <= limit:
        return name
    # Template monsters (CUTLASS): keep the head, elide the middle
    return name[: limit - 22] + " …[" + str(len(name)) + " chars]"


def _render_kernel(k: dict, console: Console) -> None:
    r = k["resources"]
    name = k["demangled"] if k["demangled"] != k["name"] else k["name"]
    console.print(f"\n  [bold]{_shorten(name)}[/]")
    parts = [f"regs [bold]{r['regs']}[/]"]
    if r.get("smem_static"):
        parts.append(f"smem {_fmt_bytes(r['smem_static'])}")
    if r.get("stack_frame"):
        parts.append(f"stack [yellow]{r['stack_frame']} B[/]")
    console.print("    " + " · ".join(parts))

    sp = k.get("spills")
    if sp and (sp["store_instructions"] or sp["load_instructions"]):
        console.print(
            f"    [red]spills:[/] {sp['store_instructions']} stores "
            f"({_fmt_bytes(sp['store_bytes'])}) / {sp['load_instructions']} loads "
            f"({_fmt_bytes(sp['load_bytes'])})"
        )
        t = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
        t.add_column("location")
        t.add_column("stores")
        t.add_column("loads")
        t.add_column("loop depth")
        rows = sp["by_line"]
        for row in rows[:8]:
            loc = f"{(row['file'] or '?').rsplit('/', 1)[-1]}:{row['line']}" if row["line"] else "?"
            depth = row["loop_depth"]
            mark = "🔥" * depth if depth else ""
            t.add_row(loc, str(row["stores"]), str(row["loads"]), f"{depth} {mark}")
        if len(rows) > 8:
            t.add_row(f"[dim](+{len(rows) - 8} more locations)[/]", "", "", "")
        console.print(t)
    elif sp is not None:
        console.print("    [green]no spills[/]")

    pr = k.get("pressure") or {}
    if pr.get("available"):
        p = pr["peak"]
        loc = f"{(p['file'] or '?').rsplit('/', 1)[-1]}:{p['line']}" if p["line"] else f"addr {p['addr']:#x}"
        console.print(f"    peak pressure: [bold]{p['live_gpr']}[/] live GPRs at {loc}")

    occ = k.get("occupancy")
    if occ:
        pct = occ["occupancy_pct"]
        color = "green" if pct >= 50 else ("yellow" if pct >= 25 else "red")
        console.print(
            f"    occupancy @{occ['threads_per_block']} thr: [{color}]{pct}%[/] "
            f"({occ['blocks_per_sm']} blocks/SM, {occ['active_warps']}/{occ['max_warps']} warps) "
            f"— limiter: [bold]{occ['limiter']}[/]"
        )
        for c in occ.get("cliffs", []):
            if c["kind"] == "gain":
                console.print(
                    f"      [cyan]cliff:[/] {c['resource']} → {c['at']} "
                    f"({c['delta']:+}) gives {c['blocks_per_sm']} blocks/SM "
                    f"({c['occupancy_pct']}%)"
                )
    acc = k.get("access")
    if acc:
        bad = [s for s in acc["by_site"]
               if s["verdict"] in ("conflict", "uncoalesced")]
        if bad:
            t = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
            t.add_column("location")
            t.add_column("issue")
            t.add_column("count")
            t.add_column("loop depth")
            for s in bad[:8]:
                loc = f"{(s['file'] or '?').rsplit('/', 1)[-1]}:{s['line']}" if s["line"] else "?"
                if s["verdict"] == "conflict":
                    issue = f"[red]{s['conflict_ways']}-way bank conflict[/]"
                    if s.get("stride") is not None:
                        issue += f" (stride {s['stride']} B)"
                else:
                    issue = f"[yellow]uncoalesced ({s['efficiency_pct']}% efficiency)[/]"
                depth = s["loop_depth"]
                t.add_row(loc, issue, str(s["count"]), f"{depth} {'🔥' * depth}")
            if len(bad) > 8:
                t.add_row(f"[dim](+{len(bad) - 8} more sites)[/]", "", "", "")
            console.print(f"    [red]access issues:[/] {acc['conflicted_shared_accesses']} "
                          f"conflicted shared · {acc['uncoalesced_global_accesses']} uncoalesced global")
            console.print(t)
            headroom = (k.get("occupancy") or {}).get("smem_headroom_bytes")
            for s in bad[:3]:
                for fix in s.get("fixes") or []:
                    desc = fix["description"] if isinstance(fix, dict) else fix
                    console.print(f"      [cyan]fix:[/] {desc}")
                    if (isinstance(fix, dict) and fix.get("kind") == "pad"
                            and headroom is not None):
                        console.print(
                            f"        [dim]smem headroom before occupancy drops: "
                            f"{headroom} B/block[/]"
                        )
        else:
            console.print(
                f"    [green]access patterns clean[/] "
                f"({acc['analyzed_count']} analyzed)"
            )
        bib = acc.get("block_invariant_read_bytes") or 0
        if bib >= 1024:
            console.print(
                f"    [magenta]est.[/] {bib} B/block of global reads are "
                f"block-invariant — every block re-fetches the same data "
                f"(grid-level traffic = {bib} B × gridDim; amortize with more "
                f"work per block)"
            )
        if acc["unanalyzed_count"]:
            top = max(acc["unanalyzed_by_reason"].items(), key=lambda kv: kv[1])
            console.print(
                f"    [dim]{acc['unanalyzed_count']} access(es) not analyzable "
                f"(mostly: {top[0]})[/]"
            )
    loops = k.get("roofline") or []
    hot = [r for r in loops if r["loop_depth"] >= 1][:3]
    for r in hot:
        span = f"lines {r['line_span'][0]}-{r['line_span'][1]}" if r["line_span"] else r["header"]
        parts = [f"[bold]loop {span}[/] (depth {r['loop_depth']})"]
        if r["est_flops_per_warp_iter"]:
            parts.append(f"{r['est_flops_per_warp_iter']} FLOP/warp-iter")
        if r["est_global_bytes_per_warp_iter"]:
            parts.append(f"≥{r['est_global_bytes_per_warp_iter']} B global/iter")
        if r["est_arithmetic_intensity"] is not None:
            parts.append(f"AI {r['est_arithmetic_intensity']}")
        console.print("    [magenta]est.[/] " + " · ".join(parts))
        details = []
        if r["est_smem_replay_factor"] and r["est_smem_replay_factor"] > 1.05:
            details.append(f"smem replay ×{r['est_smem_replay_factor']}")
        if r["est_traffic_inflation"] and r["est_traffic_inflation"] > 1.05:
            details.append(f"global traffic ×{r['est_traffic_inflation']} vs coalesced")
        b = r.get("bound")
        if b:
            details.append(f"{b['bound']}-bound (ridge {b['ridge_flop_per_byte']} FLOP/B)")
        if details:
            console.print(f"      [magenta]est.[/] {' · '.join(details)}")
    for n in k.get("notes", []):
        console.print(f"    [dim]note: {n}[/]")


def render_ls(doc: dict, console: Console) -> None:
    t = Table(show_header=True, header_style="bold")
    t.add_column("unit")
    t.add_column("arch")
    t.add_column("kernel")
    t.add_column("regs", justify="right")
    t.add_column("smem", justify="right")
    t.add_column("stack", justify="right")
    for unit in doc["units"]:
        for k in unit["kernels"]:
            r = k["resources"]
            t.add_row(
                unit["label"], unit["arch"] or "?", _shorten(k["demangled"], 90),
                str(r["regs"]), _fmt_bytes(r["smem_static"]), _fmt_bytes(r["stack_frame"]),
            )
    console.print(t)


def render_diff(diff: dict, console: Console) -> None:
    if not diff["kernels"]:
        console.print("[dim]no kernels in common[/]")
        return
    t = Table(show_header=True, header_style="bold")
    t.add_column("kernel")
    t.add_column("metric")
    t.add_column("old", justify="right")
    t.add_column("new", justify="right")
    t.add_column("Δ", justify="right")
    for kd in diff["kernels"]:
        for m in kd["changes"]:
            delta = m["delta"]
            if delta is None or delta == 0:
                style = "dim"
            elif m["metric"] in HIGHER_BETTER:
                style = "green" if delta > 0 else "red"
            else:
                style = "red" if delta > 0 else "green"
            t.add_row(
                kd["demangled"], m["metric"], str(m["old"]), str(m["new"]),
                Text(f"{delta:+}" if isinstance(delta, (int, float)) else str(delta), style=style),
            )
        if not kd["changes"]:
            t.add_row(kd["demangled"], "[dim]unchanged[/]", "", "", "")
    console.print(t)
    for kd in diff["kernels"]:
        for sc in kd.get("spill_site_changes", []):
            loc = f"{(sc['file'] or '?').rsplit('/', 1)[-1]}:{sc['line']}"
            o, n = sc["old"], sc["new"]
            if o and not n:
                console.print(f"  [green]spill site removed:[/] {loc} ({o['instrs']} instrs, depth {o['loop_depth']})")
            elif n and not o:
                col = "red" if n["loop_depth"] >= 1 else "yellow"
                console.print(f"  [{col}]spill site added:[/] {loc} ({n['instrs']} instrs, depth {n['loop_depth']})")
            else:
                console.print(f"  spill site changed: {loc} {o['instrs']}→{n['instrs']} instrs, depth {o['loop_depth']}→{n['loop_depth']}")
    for name, status in (("added", diff["added"]), ("removed", diff["removed"])):
        if status:
            console.print(f"[yellow]{name}:[/] {', '.join(status)}")
