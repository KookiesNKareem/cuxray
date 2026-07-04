"""Diff two report documents; evaluate CI gate expressions.

Gate DSL (comma-separated clauses, all must hold for every matched kernel):

    regs<=168, spill_instrs==0, stack==0, smem<=99328,
    pressure_peak<=200, occupancy(threads=256)>=25

No eval(): a tiny regex parser over a fixed metric vocabulary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .archspec import lookup
from .occupancy import compute

_OPS = {
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
}

_CLAUSE = re.compile(
    r"^\s*(?P<metric>\w+)(?:\(\s*threads\s*=\s*(?P<threads>\d+)\s*\))?"
    r"\s*(?P<op><=|>=|==|!=|<|>)\s*(?P<value>-?\d+(?:\.\d+)?)\s*$"
)

METRICS = (
    "regs", "stack", "smem", "spill_instrs", "spill_stores", "spill_loads",
    "spill_bytes", "pressure_peak", "occupancy", "bank_ways", "uncoalesced",
    "unanalyzed_accesses",
)


@dataclass
class Clause:
    metric: str
    op: str
    value: float
    threads: Optional[int] = None

    def __str__(self) -> str:
        m = f"{self.metric}(threads={self.threads})" if self.threads else self.metric
        v = int(self.value) if self.value == int(self.value) else self.value
        return f"{m}{self.op}{v}"


class GateSyntaxError(ValueError):
    pass


def parse_gate(expr: str) -> list[Clause]:
    clauses = []
    for part in expr.split(","):
        if not part.strip():
            continue
        m = _CLAUSE.match(part)
        if not m:
            raise GateSyntaxError(f"cannot parse gate clause: {part.strip()!r}")
        metric = m.group("metric")
        if metric not in METRICS:
            raise GateSyntaxError(f"unknown metric {metric!r} (known: {', '.join(METRICS)})")
        threads = int(m.group("threads")) if m.group("threads") else None
        if metric == "occupancy" and threads is None:
            raise GateSyntaxError("occupancy gate needs threads: occupancy(threads=256)>=25")
        clauses.append(Clause(metric, m.group("op"), float(m.group("value")), threads))
    if not clauses:
        raise GateSyntaxError("empty gate expression")
    return clauses


def _metric_value(kernel: dict, arch: Optional[str], clause: Clause):
    r = kernel["resources"]
    sp = kernel.get("spills") or {}
    if clause.metric == "regs":
        return r.get("regs")
    if clause.metric == "stack":
        return r.get("stack_frame")
    if clause.metric == "smem":
        return r.get("smem_static")
    if clause.metric == "spill_stores":
        return sp.get("store_instructions")
    if clause.metric == "spill_loads":
        return sp.get("load_instructions")
    if clause.metric == "spill_instrs":
        if sp:
            return (sp.get("store_instructions") or 0) + (sp.get("load_instructions") or 0)
        return None
    if clause.metric == "spill_bytes":
        if sp:
            return (sp.get("store_bytes") or 0) + (sp.get("load_bytes") or 0)
        return None
    if clause.metric == "pressure_peak":
        pr = kernel.get("pressure") or {}
        return pr.get("peak", {}).get("live_gpr") if pr.get("available") else None
    if clause.metric in ("bank_ways", "uncoalesced", "unanalyzed_accesses"):
        acc = kernel.get("access")
        if not acc:
            return None  # needs report/gate --threads
        return {
            "bank_ways": acc.get("worst_bank_conflict_ways"),
            "uncoalesced": acc.get("uncoalesced_global_accesses"),
            "unanalyzed_accesses": acc.get("unanalyzed_count"),
        }[clause.metric]
    if clause.metric == "occupancy":
        if not arch or r.get("regs") is None:
            return None
        occ = compute(lookup(arch), r["regs"], clause.threads,
                      smem_static=r.get("smem_static") or 0)
        return occ.occupancy_pct
    return None


def eval_gate(doc: dict, clauses: list[Clause]) -> list[dict]:
    violations = []
    for unit in doc["units"]:
        for k in unit["kernels"]:
            for c in clauses:
                val = _metric_value(k, unit.get("arch"), c)
                if val is None:
                    violations.append({
                        "kernel": k["demangled"], "unit": unit["label"],
                        "clause": str(c), "value": None,
                        "reason": f"metric {c.metric} unavailable",
                    })
                    continue
                if not _OPS[c.op](val, c.value):
                    violations.append({
                        "kernel": k["demangled"], "unit": unit["label"],
                        "clause": str(c), "value": val,
                        "reason": f"{c.metric}={val} violates {c}",
                    })
    return violations


_DIFF_METRICS = (
    ("regs", ("resources", "regs")),
    ("stack_frame", ("resources", "stack_frame")),
    ("smem_static", ("resources", "smem_static")),
    ("spill_store_instrs", ("spills", "store_instructions")),
    ("spill_load_instrs", ("spills", "load_instructions")),
    ("spill_bytes_total", None),  # computed
    ("pressure_peak", None),      # computed
    ("occupancy_pct", None),      # computed
    ("bank_conflict_ways", ("access", "worst_bank_conflict_ways")),
    ("uncoalesced_accesses", ("access", "uncoalesced_global_accesses")),
)


def _get(k: dict, name: str):
    if name == "spill_bytes_total":
        sp = k.get("spills") or {}
        if not sp:
            return None
        return (sp.get("store_bytes") or 0) + (sp.get("load_bytes") or 0)
    if name == "pressure_peak":
        pr = k.get("pressure") or {}
        return pr.get("peak", {}).get("live_gpr") if pr.get("available") else None
    if name == "occupancy_pct":
        occ = k.get("occupancy")
        return occ.get("occupancy_pct") if occ else None
    for metric, path in _DIFF_METRICS:
        if metric == name and path:
            obj = k.get(path[0]) or {}
            return obj.get(path[1])
    return None


def diff_reports(old: dict, new: dict, kernel_re: Optional[str] = None) -> dict:
    pat = re.compile(kernel_re) if kernel_re else None

    def flatten(doc):
        out = {}
        for unit in doc["units"]:
            for k in unit["kernels"]:
                if pat and not pat.search(k["name"]):
                    continue
                out[(unit.get("arch"), k["name"])] = (unit, k)
        return out

    o, n = flatten(old), flatten(new)
    kernels = []
    for key in sorted(o.keys() & n.keys()):
        _, ko = o[key]
        _, kn = n[key]
        changes = []
        for metric, _ in _DIFF_METRICS:
            a, b = _get(ko, metric), _get(kn, metric)
            if a is None and b is None:
                continue
            if a != b:
                delta = None
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    delta = round(b - a, 2)
                changes.append({"metric": metric, "old": a, "new": b, "delta": delta})
        kernels.append({
            "name": key[1], "arch": key[0],
            "demangled": kn["demangled"], "changes": changes,
        })
    return {
        "schema": "cuxray.diff/1",
        "kernels": kernels,
        "added": sorted(k[1] for k in n.keys() - o.keys()),
        "removed": sorted(k[1] for k in o.keys() - n.keys()),
        "changed": sum(1 for k in kernels if k["changes"]),
    }
