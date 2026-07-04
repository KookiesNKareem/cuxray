"""Terminal rendering of report/diff documents (rich)."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.text import Text


def _fmt_bytes(n) -> str:
    if n is None:
        return "-"
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024} KB"
    return f"{n} B"


def render_report(doc: dict, console: Console) -> None:
    for unit in doc["units"]:
        console.print(f"\n[bold cyan]{unit['label']}[/]  [dim]{unit['arch'] or '?'}[/]")
        for k in unit["kernels"]:
            _render_kernel(k, console)


def _render_kernel(k: dict, console: Console) -> None:
    r = k["resources"]
    name = k["demangled"] if k["demangled"] != k["name"] else k["name"]
    console.print(f"\n  [bold]{name}[/]")
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
        for row in sp["by_line"][:8]:
            loc = f"{(row['file'] or '?').rsplit('/', 1)[-1]}:{row['line']}" if row["line"] else "?"
            depth = row["loop_depth"]
            mark = "🔥" * depth if depth else ""
            t.add_row(loc, str(row["stores"]), str(row["loads"]), f"{depth} {mark}")
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
                unit["label"], unit["arch"] or "?", k["demangled"],
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
            style = "red" if (delta or 0) > 0 else "green"
            t.add_row(
                kd["demangled"], m["metric"], str(m["old"]), str(m["new"]),
                Text(f"{delta:+}" if isinstance(delta, (int, float)) else str(delta), style=style),
            )
        if not kd["changes"]:
            t.add_row(kd["demangled"], "[dim]unchanged[/]", "", "", "")
    console.print(t)
    for name, status in (("added", diff["added"]), ("removed", diff["removed"])):
        if status:
            console.print(f"[yellow]{name}:[/] {', '.join(status)}")
