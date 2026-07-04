#!/usr/bin/env python3
"""Differential sweep: cuxray spill-byte accounting vs ptxas -v.

Generates kernels across dtypes (different spill access widths), accumulator
sizes, and register caps; compiles each; asserts that spill bytes computed
from SASS opcode widths equal ptxas's reported spill stores/loads exactly.

Needs nvcc+nvdisasm on PATH. Run inside the dev container or CI e2e.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cuxray.analyze.spillmap import spill_map          # noqa: E402
from cuxray.parse import ptxasv, sass                  # noqa: E402

TEMPLATE = """
{include}
typedef {dtype} T;
#define ACC {acc}

__global__ void sweep_kernel(const T* __restrict__ x, T* __restrict__ out,
                             int n, int iters) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    T acc[ACC];
    for (int j = 0; j < ACC; ++j) acc[j] = x[(i + j) % n];
    for (int it = 0; it < iters; ++it) {{
        for (int j = 0; j < ACC; ++j) {{
            acc[j] = {op};
        }}
    }}
    T s = acc[0];
    for (int j = 1; j < ACC; ++j) {{ {reduce} }}
    out[i] = s;
}}
"""

CASES = []
for dtype, include, op, reduce in [
    ("float", "", "acc[j] * 1.0009765625f + acc[(j + 1) % ACC]", "s += acc[j];"),
    ("double", "", "acc[j] * 1.0009765625 + acc[(j + 1) % ACC]", "s += acc[j];"),
    ("float4", "",
     "make_float4(acc[j].x + acc[(j+1)%ACC].x, acc[j].y * 1.5f, acc[j].z, acc[j].w)",
     "s.x += acc[j].x; s.y += acc[j].w;"),
    ("short", "", "(short)(acc[j] + acc[(j + 1) % ACC])", "s += acc[j];"),
    ("unsigned char", "", "(unsigned char)(acc[j] ^ acc[(j + 1) % ACC])", "s ^= acc[j];"),
]:
    for acc in (24, 48):
        for cap in (32, 64):
            CASES.append((dtype, include, op, reduce, acc, cap))


def run(arch: str) -> int:
    failures = 0
    tested = 0
    with tempfile.TemporaryDirectory() as td:
        for dtype, include, op, reduce, acc, cap in CASES:
            src = Path(td) / "k.cu"
            cub = Path(td) / "k.cubin"
            src.write_text(TEMPLATE.format(dtype=dtype, include=include, op=op,
                                           reduce=reduce, acc=acc))
            r = subprocess.run(
                ["nvcc", "-cubin", f"-arch={arch}", "-lineinfo",
                 "-maxrregcount", str(cap), "--resource-usage",
                 "-o", str(cub), str(src)],
                capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  SKIP (nvcc fail) {dtype} acc={acc} cap={cap}")
                continue
            pk = list(ptxasv.parse(r.stderr).values())[0]
            gi = subprocess.run(["nvdisasm", "-c", "-gi", str(cub)],
                                capture_output=True, text=True, check=True).stdout
            dis = sass.parse_gi(gi)
            func = list(dis.functions.values())[0]
            sm = spill_map(func, {})
            ok = (sm["store_bytes"] == pk.spill_stores
                  and sm["load_bytes"] == pk.spill_loads)
            tested += 1
            status = "ok  " if ok else "FAIL"
            print(f"  {status} {dtype:14s} acc={acc:<3d} cap={cap:<3d} "
                  f"ptxas={pk.spill_stores}/{pk.spill_loads} "
                  f"cuxray={sm['store_bytes']}/{sm['load_bytes']} "
                  f"({sm['store_instructions']}+{sm['load_instructions']} instrs)")
            if not ok:
                failures += 1
    print(f"{arch}: {tested - failures}/{tested} exact matches")
    return failures


if __name__ == "__main__":
    total = 0
    for arch in sys.argv[1:] or ["sm_90", "sm_120a"]:
        print(f"== {arch} ==")
        total += run(arch)
    sys.exit(1 if total else 0)
