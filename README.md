# cuxray

[![PyPI](https://img.shields.io/pypi/v/cuxray.svg)](https://pypi.org/project/cuxray/)
[![CI](https://github.com/KookiesNKareem/cuxray/actions/workflows/ci.yml/badge.svg)](https://github.com/KookiesNKareem/cuxray/actions)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Static analysis and optimization for CUDA kernel binaries — register
pressure, spills, occupancy, bank conflicts — **without a GPU**.

cuxray reads the decisions the compiler froze into your cubin and combines
them with NVIDIA's published architecture tables. Because those models are
exact, it can go beyond reporting problems: it searches for fixes (shared-
memory swizzles, register caps) and verifies them before suggesting them.
Everything it reports is ground truth; anything unknowable is reported as
unknowable, with the reason.

```console
$ pip install cuxray
$ cuxray report kernel.cubin --threads 256

  moe_gemm(float const*, float*, int)
    regs 168 · smem 8 KB
    spills: 24 stores (96 B) / 24 loads (96 B)
  location         stores   loads   loop depth
  moe_gemm.cu:145  24       24      1 🔥
    peak pressure: 166 live GPRs at moe_gemm.cu:142
    occupancy @256 thr: 16.7% (1 blocks/SM) — limiter: registers
      cliff: registers → 128 (-40) gives 2 blocks/SM (33.3%)
```

## Features

**Analyze**

- Spill maps: every spill instruction is attributed to a source line and
  weighted by loop depth, instead of one aggregate byte count per kernel.
- Occupancy with limiters and cliffs: blocks/SM, which resource binds, and
  the nearest boundary (e.g. "8 fewer registers gains a block per SM").
- Bank-conflict and coalescing analysis computed from per-lane address
  tracking in the SASS, including XOR-swizzled and `ldmatrix` layouts.
- Register-pressure curves from `nvdisasm` life ranges, mapped to source.

**Optimize**

- `cuxray solve` searches CUTLASS-style swizzles and returns only layouts
  it has verified conflict-free for every shared access in the kernel.
- `cuxray tune-regs` recompiles across `-maxrregcount` values and marks the
  Pareto-optimal occupancy/spill trade-offs.
- Conflict reports include verified fix suggestions with their shared-memory
  cost and occupancy impact.

**Integrate**

- CI gating with exit codes, per-kernel budget files, build-to-build `diff`
  with regression detection, SARIF output for PR annotations, and a
  reusable GitHub Action.
- Stable, versioned JSON from every command; results cached on disk.
- No setup: runs on any Linux machine, fetching pinned, sha256-verified
  NVIDIA binary utilities on first use. Works on binaries you didn't build
  — a vLLM wheel, a Triton cache, a `.so` from PyPI.

## Usage

**Inspect** — resources, spills, pressure, occupancy, access patterns
(`--threads` is inferred from `.reqntid`/`__launch_bounds__` when present):

```console
cuxray report kernels.so --kernel "moe.*" --threads 256
cuxray ls kernels.so                      # fast listing, no disassembly
```

**Fix bank conflicts** — derive a verified swizzle for the whole kernel:

```console
$ cuxray solve kernel.cubin --threads 32
  solution: Swizzle<3,4,3>  (zero smem cost, verified on all accesses)
    apply to byte offsets: addr ^ ((addr >> 3) & 0x70)
    e.g. ldsm_async.cu:37 LDSM: 8-way → clean
```

**Tune register caps** — map the whole trade-off in seconds of CPU:

```console
$ cuxray tune-regs kernel.ptx --threads 256
   cap   regs   spill bytes   blocks/SM   occupancy
    40     40           424           6      100.0%   ● pareto
    48     48             0           5       83.3%   ● pareto
  none     56             0           4       66.7%
```

**Gate regressions in CI** — exit 1 on violation, per-kernel budgets,
SARIF annotations:

```console
cuxray gate kernels.so "spill_instrs==0, regs<=168, bank_ways<=2" --threads 256
cuxray gate kernels.so --budget budgets.json --sarif out.sarif
cuxray diff old.so new.so --fail-on-regression
```

```yaml
- uses: KookiesNKareem/cuxray@main
  with: { path: build/kernels.so, gate: "spill_instrs==0", threads: "256" }
```

**What-if** — no binary needed:

```console
cuxray occupancy --arch sm_120 --regs 168 --threads 256 --sweep
```

Every command takes `--json` / `-o` (schema: `cuxray schema`). `cuxray
doctor` shows toolchain and cache state.

## Inputs

| Input | Handling |
|---|---|
| `.cubin` | analyzed directly |
| host ELF (`.so`, `.o`, executable) | embedded cubins extracted and analyzed |
| directory | recursive `*.cubin` walk (Triton caches) |
| `.ptx` | assembled with `ptxas` |

Architectures: compute capability 7.5–12.x (Turing → Blackwell, including
`a`-variants like `sm_120a`).

## How it works

`nvdisasm` life ranges give per-instruction register liveness with source
mapping; `cuobjdump` provides per-kernel resources from any cubin; a lane-
value dataflow over the SASS recovers how each memory address varies across
the 32 lanes of a warp; occupancy is a port of NVIDIA's `cuda_occupancy.h`.
GPUs execute SASS in order with no register renaming, so the binary is a
complete record of what the hardware will do — reading it is not simulation.

## Validation

- Occupancy: 4,344/4,344 configs match NVIDIA's `cuda_occupancy.h` across
  all supported architectures; 54/54 match the CUDA runtime on real hardware.
- Spill bytes: byte-exact against `ptxas -v` across dtypes and architectures.
- Bank verdicts: hardware-timed (flagged kernel 12× slower than its clean
  twin; swizzled and padded twins within 2%); `solve` re-derives the
  canonical CUTLASS `Swizzle<3,4,3>` for 128-byte fp16 tiles.
- Robustness: ~161k production kernels (vLLM, PyTorch wheels) analyzed with
  zero crashes and zero false-positive conflict flags.

## Limitations

- Static facts only: cache behavior, achieved bandwidth, and data-dependent
  addressing need a profiler; cuxray reports those accesses as unanalyzable
  rather than guessing.
- Block shape and dynamic shared memory are launch parameters — pass
  `--threads` / `--smem-dynamic` when the binary carries no metadata (cuxray
  warns when this matters).
- Warp-specialized register reallocation (`setmaxnreg`) makes static
  occupancy pessimistic; detected and flagged.
- Linux only. Compile with `-lineinfo` for source attribution.

Roadmap: scheduling/stall analysis from the compiler's embedded control bits.

## License

Apache-2.0. Not affiliated with NVIDIA; CUDA binary utilities are downloaded
from NVIDIA's redistributable archive under the
[CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html).
