"""Parser for `nvdisasm -cfg` Graphviz output → per-block loop depth.

Format (tests/fixtures/recorded/nvdisasm_cfg.*): one cluster per function,
nodes named by the block's leading label, edges like

    "_Z6spillyPKfPfii":exit0:e -> ".L_x_0":entry:n [style=solid];
    ".L_x_2":exit0:e -> ".L_x_2":entry:n [style=solid];   <- loop back edge

Loop depth is computed via DFS back-edge detection + natural-loop membership
(standard dominator-free approximation: a back edge is an edge to a node
currently on the DFS stack; the natural loop is every node that can reach the
back edge's source without passing through the header). Good enough for
"is this spill inside a loop, and how nested" — not a general decompiler.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CLUSTER = re.compile(r'subgraph\s+"cluster_([^"]+)"')
_EDGE = re.compile(r'"([^"]+)"(?::\S+)?\s*->\s*"([^"]+)"')


@dataclass
class FunctionCFG:
    name: str
    nodes: set[str] = field(default_factory=set)
    edges: list[tuple[str, str]] = field(default_factory=list)
    loop_depth: dict[str, int] = field(default_factory=dict)
    loops: dict[str, set[str]] = field(default_factory=dict)  # header -> members


def parse(text: str) -> dict[str, FunctionCFG]:
    funcs: dict[str, FunctionCFG] = {}
    cur: FunctionCFG | None = None
    for line in text.splitlines():
        m = _CLUSTER.search(line)
        if m:
            cur = FunctionCFG(name=m.group(1))
            funcs[cur.name] = cur
            continue
        if cur is None:
            continue
        m = _EDGE.search(line)
        if m:
            a, b = m.group(1), m.group(2)
            cur.nodes.update((a, b))
            cur.edges.append((a, b))
    for f in funcs.values():
        f.loop_depth, f.loops = _loop_depths(f)
    return funcs


def _loop_depths(cfg: FunctionCFG) -> tuple[dict[str, int], dict[str, set[str]]]:
    succ: dict[str, list[str]] = {n: [] for n in cfg.nodes}
    pred: dict[str, list[str]] = {n: [] for n in cfg.nodes}
    for a, b in cfg.edges:
        succ[a].append(b)
        pred[b].append(a)

    entry = cfg.name if cfg.name in cfg.nodes else (next(iter(cfg.nodes)) if cfg.nodes else None)
    if entry is None:
        return {}, {}

    # Iterative DFS with an explicit stack-set to find back edges
    back_edges: list[tuple[str, str]] = []
    color: dict[str, int] = {}  # 0 unseen / 1 on stack / 2 done
    stack: list[tuple[str, int]] = [(entry, 0)]
    color[entry] = 1
    while stack:
        node, i = stack[-1]
        if i < len(succ[node]):
            stack[-1] = (node, i + 1)
            nxt = succ[node][i]
            c = color.get(nxt, 0)
            if c == 1:
                back_edges.append((node, nxt))
            elif c == 0:
                color[nxt] = 1
                stack.append((nxt, 0))
        else:
            color[node] = 2
            stack.pop()

    # A loop may have several back edges to one header (multiple `continue`
    # paths); union their natural loops so each header counts once toward
    # nesting depth.
    by_header: dict[str, set[str]] = {}
    for tail, header in back_edges:
        members = by_header.setdefault(header, {header})
        work = [tail]
        members.add(tail)
        while work:
            n = work.pop()
            if n == header:
                continue  # do not walk past the loop header
            for p in pred[n]:
                if p not in members:
                    members.add(p)
                    work.append(p)
    depth = {n: 0 for n in cfg.nodes}
    for members in by_header.values():
        for n in members:
            depth[n] += 1
    return depth, by_header
