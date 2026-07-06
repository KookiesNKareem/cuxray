"""Ingest a Triton / TorchInductor kernel cache.

Triton writes, next to every compiled ``<name>.cubin``, a ``<name>.json``
metadata sidecar carrying the launch facts the cubin does not expose:
``shared`` (dynamic shared-memory bytes) and ``num_warps`` (block size =
num_warps * warp_size). Analyzing the cubin alone must assume 0 B of shared
memory and guess the block shape; pairing each cubin with its metadata makes
occupancy exact and every finding legible by kernel name.

Layout (Triton cache dir, one hashed subdir per kernel)::

    <hash>/triton_red_fused_native_layer_norm_0.cubin
    <hash>/triton_red_fused_native_layer_norm_0.json   <- metadata sidecar
    <hash>/__grp__triton_....json                       <- group index (ignored)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# TorchInductor names encode the fusion kind: triton_<kind>_fused_<ops>_<n>.
# Best-effort cosmetic tag only; unknown prefixes pass through unchanged.
_KIND = {"poi": "pointwise", "red": "reduction", "per": "persistent-reduction",
         "tem": "template", "mm": "matmul", "bmm": "batched-matmul",
         "for": "foreach"}


@dataclass
class TritonKernel:
    cubin: Path
    name: str
    shared: Optional[int]        # dynamic shared-memory bytes (metadata "shared")
    num_warps: Optional[int]
    warp_size: int
    arch: Optional[str]          # e.g. "sm80", from metadata when present

    @property
    def threads(self) -> Optional[int]:
        return self.num_warps * self.warp_size if self.num_warps else None


def kind(name: str) -> Optional[str]:
    """The TorchInductor fusion kind (pointwise/reduction/...), best-effort
    from the name. Cosmetic only: an unrecognized prefix returns None."""
    m = re.match(r"triton_([a-z]+)", name)
    return _KIND.get(m.group(1)) if m else None


def code_line(path: Optional[str], lineno: Optional[int]) -> Optional[str]:
    """The source line the cubin's own debug info points at, if the file is on
    disk. This is the standard debugger source-attribution path (the file is
    named by the cubin's DWARF, extracted by nvdisasm), NOT any Triton or
    Inductor internal format, so it survives version churn and simply returns
    None when the file is absent."""
    if not path or not lineno:
        return None
    try:
        p = Path(path)
        if not p.is_file():
            return None
        with p.open() as f:
            for i, line in enumerate(f, 1):
                if i == lineno:
                    return line.strip() or None
    except OSError:
        return None
    return None


def _load_meta(cubin: Path) -> dict:
    meta = cubin.with_suffix(".json")
    if not meta.is_file():
        return {}
    try:
        return json.loads(meta.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _arch_str(meta: dict) -> Optional[str]:
    a = meta.get("arch")
    if isinstance(a, str):
        return a if a.startswith("sm") else f"sm{a}"
    if isinstance(a, int):
        return f"sm{a}"
    tgt = meta.get("target")
    if isinstance(tgt, dict) and tgt.get("arch") is not None:
        return f"sm{tgt['arch']}"
    return None


def discover(root: Path) -> list["TritonKernel"]:
    """Every Triton kernel under ``root`` (a cache directory or a single
    ``.cubin``), each paired with its metadata sidecar when present. The
    ``__grp__*.json`` group index files are not cubins and are skipped."""
    root = Path(root)
    cubins = [root] if root.suffix == ".cubin" else sorted(root.rglob("*.cubin"))
    out: list[TritonKernel] = []
    for c in cubins:
        meta = _load_meta(c)
        out.append(TritonKernel(
            cubin=c,
            name=meta.get("name") or c.stem,
            shared=meta.get("shared"),
            num_warps=meta.get("num_warps"),
            warp_size=int(meta.get("warp_size") or 32),
            arch=_arch_str(meta),
        ))
    return out
