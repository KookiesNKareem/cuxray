"""SARIF 2.1.0 output for gate results — GitHub code scanning ingests this
natively, so gate violations appear as PR annotations."""

from __future__ import annotations

from . import __version__

_RULE_DESC = {
    "regs": "Register usage exceeds the gated budget",
    "stack": "Stack frame (local memory) exceeds the gated budget",
    "smem": "Static shared memory exceeds the gated budget",
    "spill_instrs": "Register spill instructions present",
    "spill_stores": "Register spill stores present",
    "spill_loads": "Register spill loads present",
    "spill_bytes": "Register spill bytes exceed the gated budget",
    "pressure_peak": "Peak live-register pressure exceeds the gated budget",
    "occupancy": "Theoretical occupancy below the gated floor",
    "bank_ways": "Shared-memory bank conflict multiplicity exceeds the gate",
    "uncoalesced": "Uncoalesced global accesses exceed the gate",
    "unanalyzed_accesses": "Unanalyzable accesses exceed the gate",
}


def _relativize(path: str, source_root: str | None) -> str:
    if source_root and path.startswith(source_root):
        path = path[len(source_root):].lstrip("/")
    return path


def gate_to_sarif(artifact_path: str, clauses, violations: list[dict],
                  source_root: str | None = None) -> dict:
    rules_by_id: dict[str, dict] = {}
    for c in clauses:
        rules_by_id.setdefault(f"cuxray/{c.metric}", {
            "id": f"cuxray/{c.metric}",
            "shortDescription": {"text": _RULE_DESC.get(c.metric, c.metric)},
            "help": {"text": f"gate clause: {c}"},
        })
    rules = list(rules_by_id.values())
    results = []
    for v in violations:
        clause_metric = v["clause"].split("<")[0].split(">")[0].split("=")[0].split("!")[0].strip()
        clause_metric = clause_metric.split("(")[0]
        uri = _relativize(v.get("file") or artifact_path, source_root)
        loc = {
            "physicalLocation": {
                "artifactLocation": {"uri": uri},
            }
        }
        if v.get("line"):
            loc["physicalLocation"]["region"] = {"startLine": v["line"]}
        results.append({
            "ruleId": f"cuxray/{clause_metric}",
            "level": "error",
            "message": {"text": f"{v['kernel']}: {v['reason']} (unit {v['unit']})"},
            "locations": [loc],
        })
    return {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {
                "name": "cuxray",
                "informationUri": "https://github.com/KookiesNKareem/cuxray",
                "version": __version__,
                "rules": rules,
            }},
            "results": results,
        }],
    }
