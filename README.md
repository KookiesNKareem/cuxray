# cuxray

[![PyPI](https://img.shields.io/pypi/v/cuxray.svg)](https://pypi.org/project/cuxray/)
[![CI](https://github.com/KookiesNKareem/cuxray/actions/workflows/ci.yml/badge.svg)](https://github.com/KookiesNKareem/cuxray/actions)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Static analysis and optimization for CUDA kernel binaries (register
pressure, spills, occupancy, bank conflicts) **without a GPU**.

cuxray reads the decisions the compiler froze into your cubin and combines
them with NVIDIA's published architecture tables. Because those models are
exact, it can go beyond reporting problems: it searches for fixes (shared-
memory swizzles, register caps, tile configurations) and verifies them
before suggesting them. Measurements are ground truth; performance
estimates (roofline, cycle counts) are explicitly labeled `est.` and
validated against hardware; anything unknowable is reported as unknowable,
with the reason.

Point it at any cubin and get a ranked, confidence-tagged list of exactly what's slow and how to fix it. No GPU is touched.

```console
$ pip install cuxray
$ cuxray advise w4a8_gemv.sm_80.cubin --threads 256

  gemv_w4a8<__half, 1, 1, 1>(uint4 const*, signed char const*, ...)  (w4a8_gemv.sm_80.cubin)
    1. cut registers to 40 (-4)  · high confidence · impact 50
       unlocks 6 blocks/SM (62.5% → 75.0%); current limiter is registers
       evidence: occupancy model (validated vs cuda_occupancy.h + runtime API)
    2. SIMT datapath caps this loop — tensor cores scale past it  · medium confidence · impact 40
       MACs run on the SIMT int8 datapath (dp4a/FMA). On sm_80 the tensor cores do ~8x more
       int8 MACs/clock. This is fine while memory-bound (low arithmetic-per-byte / batch 1),
       but as arithmetic-per-byte grows the loop becomes SIMT-compute-bound and a tensor-core
       implementation would be up to ~8x faster. Measure across your batch sizes to find the
       crossover.
       evidence: static op-mix + per-arch MAC-rate model (approximate)
```

Every finding is a fact frozen by the compiler into the binary. For bank conflicts it goes
further and *synthesizes a verified fix* (`cuxray solve`, below).

## Features

**Analyze**

- Spill maps: every spill instruction is attributed to a source line and
  weighted by loop depth, instead of one aggregate byte count per kernel.
- Occupancy with limiters and cliffs: blocks/SM, which resource binds, and
  the nearest boundary (e.g. "8 fewer registers gains a block per SM").
- Bank-conflict and coalescing analysis computed from per-lane address
  tracking in the SASS, including XOR-swizzled and `ldmatrix` layouts.
- Register-pressure curves from `nvdisasm` life ranges, mapped to source.
- Per-loop roofline estimates: FLOPs and memory traffic per warp-iteration,
  arithmetic intensity, shared-memory replay and traffic-inflation factors.
- Per-loop cycle estimates (`cuxray sched`) from the compiler's embedded
  instruction schedule — validated within 1% of measured hardware cycles on
  deterministic loops (sm_80–sm_90a).
- Datapath crossover: flags a loop whose MACs run on the SIMT lanes (dp4a /
  FMA) and reports how much tensor-core headroom the arithmetic leaves — the
  ceiling a memory-bound kernel hides until batch grows.

**Optimize**

- `cuxray advise` ranks every finding by impact-weighted severity into one
  action list; `cuxray survey` does it across a whole library, heaviest
  kernels first, so you fix what moves the needle.
- `cuxray solve` searches CUTLASS-style swizzles and returns only layouts
  it has verified conflict-free for every shared access in the kernel.
- `cuxray tune-regs` recompiles across `-maxrregcount` values and marks the
  Pareto-optimal occupancy/spill trade-offs; `cuxray tune` sweeps whole
  `-D` define matrices (tile shapes, stage counts) and ranks every variant
  statically — autotuning's search space, cut before a GPU is involved.
- Conflict reports include verified fix suggestions with their shared-memory
  cost and occupancy impact.

## Quick start

```console
pip install cuxray                            # CPU-only — no CUDA, no GPU
cuxray advise mykernels.so --threads 256      # ranked fixes for every kernel
cuxray solve mykernels.so --threads 256       # verified swizzle for any bank conflict
cuxray gate mykernels.so "spill_instrs==0"    # exit 1 in CI on a regression
```

Point it at anything holding cubins — a `.cubin`, a host `.so`, a directory of
Triton caches, a `.ptx`, even a wheel you `pip download`ed. On first run it
fetches pinned, sha256-verified NVIDIA binary utilities; there is nothing else
to install and no GPU is ever touched.

## Usage

**Inspect** — resources, spills, pressure, occupancy, access patterns
(`--threads` is inferred from `.reqntid`/`__launch_bounds__` when present):

```console
cuxray report kernels.so --kernel "moe.*" --threads 256
cuxray ls kernels.so                      # fast listing, no disassembly
```

**Prioritize across a library** — `advise` (above) ranks one kernel; these
scale it out and track fixes across builds:

```console
cuxray survey kernels.so --threads 256    # rank every kernel by fixable impact
cuxray compare old.so new.so              # did the fix land? per-kernel A/B
cuxray why kernel.cubin --line 145        # dataflow slice: where an address came from
```

**Fix bank conflicts** — derive a verified swizzle for the whole kernel:

```console
$ cuxray solve bank_conflict.cubin --threads 256
  _Z12col_conflictPKfPfi
    224 conflicted of 256 shared accesses
  solution (all accesses): Swizzle<5,2,5>  (zero smem cost, verified)
    apply to byte offsets: addr ^ ((addr >> 5) & 0x7c)
    e.g. bank_conflict.cu:19 LDS: 32-way → clean
    // + a ready-to-paste __device__ swizzle() and the cute::Swizzle<5,2,5> layout
```

**Tune register caps** — map the whole trade-off in seconds of CPU:

```console
$ cuxray tune-regs spill.ptx --threads 256
   cap    regs    spill bytes    spill instrs    top spill line    blocks/SM    occupancy
    24      24           1860             463                14            8       100.0%
    32      32           1172             291                14            8       100.0%    ● pareto
    40      40            468             115                10            6        75.0%    ● pareto
    48      48              0               0                 -            5        62.5%    ● pareto
    64      64              0               0                 -            4        50.0%
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

**Estimate cycles** — the compiler's own schedule, summed per loop:

```console
$ cuxray sched spill.cubin        # a register-capped build that spills
  est. loop lines 12-14 (depth 1): 466 issue+stall cycles/iter
      spill.cu:14: 374 cycles     # the spill traffic IS the stall cost
      spill.cu:13: 85 cycles
```

**Bound what's possible** — the roofline floor for a launch (a true lower
bound, no free parameters) and which resource binds:

```console
$ cuxray roofline --bytes 134217728 --sms 108 --clock 1.41 --peak-gbs 1400 --cc sm_80
  roofline floor: 95.87 µs  (memory-bound)
    memory 95.87 µs · compute 0.0 µs
  a lower bound — a real kernel runs slower by its efficiency
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
- Cycle estimates: control-bit decode confirmed via scoreboard pairing and
  known FP32 latencies; loop estimate within 0.7% of `clock64()`-measured
  cycles on hardware.
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
