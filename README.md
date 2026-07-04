# cuxray

**Hardware-free static analyzer for CUDA kernel binaries.** Point it at a
`.cubin`, a compiled `.so`, or a Triton cache and get source-attributed
register pressure, spill locations, and occupancy analysis — plus diffs
between builds and a CI gate. **No GPU required, ever.**

Everything cuxray reports is either read out of the binary (decisions the
compiler already froze into the SASS) or computed from NVIDIA's published
architecture tables. It never estimates.

```console
$ pip install cuxray
$ cuxray report kernel.cubin --threads 256
$ cuxray diff old.so new.so --kernel "moe.*"
$ cuxray gate build/kernels.so "spill_instrs==0, regs<=168"
```

Runs on any Linux machine — laptops, CI runners, containers. If no CUDA
toolkit is installed, cuxray fetches pinned, sha256-verified binary utilities
(`nvdisasm`, `cuobjdump`, `ptxas`) from NVIDIA's official redistributable
archive on first use (x86_64 and aarch64).

## Why

- **Spills are silent.** `ptxas -v` tells you "548 bytes spill stores" per
  kernel — not which loop, not which variable. cuxray tells you the source
  line, and whether the spill sits in your inner loop or the prologue.
- **Occupancy cliffs are invisible.** Register allocation is quantized; one
  extra register can cost a whole block per SM. cuxray names the binding
  limiter and the nearest cliff.
- **Profiling needs privileged GPUs.** Rented and containerized GPUs often
  can't run Nsight Compute at all (`ERR_NVGPU_CTRPERM`). cuxray's entire
  analysis needs no GPU — it works where you compile, not where you run.
- **Agents need structured feedback.** Every command has `--json` with a
  stable, versioned schema, and `gate` communicates through exit codes —
  script it, CI it, or hand it to a coding agent as a documented CLI.

## What it reports

`cuxray report kernel.cubin --threads 256`:

```text
spill.sm_120a.cubin  sm_120a

  spilly(float const*, float*, int, int)
    regs 32 · stack 208 B
    spills: 135 stores (548 B) / 137 loads (556 B)
  location       stores    loads    loop depth
  spill.cu:14    57        59       1 🔥
  spill.cu:13    22        22       1 🔥
  spill.cu:10    56        45       0
  spill.cu:18    0         11       0
    peak pressure: 30 live GPRs at spill.cu:10
    occupancy @256 thr: 100.0% (6 blocks/SM, 48/48 warps) — limiter: warps
```

- **Registers** — per-kernel usage and the per-source-line *pressure curve*
  (which line holds the most live registers), from `nvdisasm` life ranges.
- **Spills** — every `STL`/`LDL` mapped to a source line and weighted by loop
  depth from the control-flow graph. Spill *bytes* are computed from SASS
  access widths and reproduce `ptxas -v`'s byte counts exactly (pinned by a
  test), so they work on binaries you didn't compile.
- **Occupancy** — a faithful port of NVIDIA's `cuda_occupancy.h` algorithm:
  blocks/SM, the binding limiter, and cliff detection.

Compile with `-lineinfo` (free — debug metadata only, no codegen impact) to
get source attribution; without it cuxray reports SASS addresses.

### Diff two builds

Recompile the kernel above with `-maxrregcount 48` instead of 32:

```text
$ cuxray diff spill32.cubin spill48.cubin --threads 256
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━┳━━━━━━━┓
┃ kernel                                 ┃ metric             ┃   old ┃  new ┃     Δ ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━╇━━━━━━━┩
│ spilly(float const*, float*, int, int) │ regs               │    32 │   48 │   +16 │
│ spilly(float const*, float*, int, int) │ stack_frame        │   208 │    0 │  -208 │
│ spilly(float const*, float*, int, int) │ spill_store_instrs │   135 │    0 │  -135 │
│ spilly(float const*, float*, int, int) │ spill_load_instrs  │   137 │    0 │  -137 │
│ spilly(float const*, float*, int, int) │ spill_bytes_total  │  1104 │    0 │ -1104 │
│ spilly(float const*, float*, int, int) │ pressure_peak      │    30 │   46 │   +16 │
│ spilly(float const*, float*, int, int) │ occupancy_pct      │ 100.0 │ 83.3 │ -16.7 │
└────────────────────────────────────────┴────────────────────┴───────┴──────┴───────┘
```

The whole register/spill/occupancy trade-off in one table: +16 registers
eliminated every spill, at the cost of 16.7 points of occupancy.

### Gate resource regressions in CI

```console
$ cuxray gate kernels.so "spill_instrs==0, regs<=168, occupancy(threads=256)>=25"
✗ moe_gemm (kernels.so): spill_instrs=24 violates spill_instrs==0
GATE FAILED — 1 violation(s)   (exit code 1)
```

Metrics: `regs`, `stack`, `smem`, `spill_instrs`, `spill_stores`,
`spill_loads`, `spill_bytes`, `pressure_peak`, `occupancy(threads=N)`.

### Occupancy what-if — no binary needed

```text
$ cuxray occupancy --arch sm_120 --regs 168 --threads 256
sm_120 (Blackwell (RTX 50 / RTX PRO)) — 168 regs, 256 threads, 0 B smem
  1 blocks/SM · 8/48 warps · 16.7% — limiter: registers
  limits: {'warps': 6, 'blocks': 24, 'registers': 1, 'shared_memory': 100}
  cliff (gain): registers → 128 (-40) gives 2 blocks/SM (33.3%)
```

Add `--sweep` for a block-size sweep, `--smem N` for shared memory.

## Inputs

| Input | Handling |
|---|---|
| `kernel.cubin` | analyzed directly |
| host ELF (`.so`, `.o`, executable) | embedded cubins extracted via `cuobjdump`, all analyzed |
| directory | recursive `*.cubin` walk (Triton cache layout) |
| `kernel.ptx` | assembled with `ptxas` (arch from `.target` or `--arch`) |

Supported architectures: compute capability 7.5 through 12.x (Turing,
Ampere, Ada, Hopper, Blackwell — including `a`-variant cubins like
`sm_120a`).

## How it works

cuxray drives three battle-tested NVIDIA tools and joins their output:

- `cuobjdump --dump-resource-usage` — registers, stack, shared memory per
  kernel, from any cubin;
- `nvdisasm -plr` — per-instruction register life ranges (the pressure
  curve), joined by instruction address with `-gi` source-line mapping;
- `nvdisasm -cfg` — control-flow graph, for loop-depth weighting of spills;
- occupancy is computed from the algorithm in NVIDIA's `cuda_occupancy.h`
  plus the capacity tables in the CUDA Programming Guide.

None of these require a GPU. The GPU executes SASS in order, with no
register renaming — so the binary is a complete record of what the hardware
will do, and reading it is not a simulation.

## Limitations (v0.1)

- Structural facts only: cache behavior, achieved bandwidth, and
  data-dependent divergence need a profiler. cuxray is the predict half of
  predict → measure → explain.
- Block size is a runtime choice — pass `--threads` for occupancy analysis.
- Kernel names are demangled via `c++filt` when available.
- Linux only (NVIDIA publishes no macOS/Windows binary utilities; on a Mac,
  run cuxray in any Linux container — no GPU passthrough needed).

## Roadmap

- **Layer B** — static shared-memory bank-conflict and global coalescing
  analysis (affine + XOR-swizzle address reasoning, honest "can't analyze"
  on data-dependent indices).
- **Layer C** — control-bit/scheduling analysis: static stall estimates and
  critical-path cycles from the compiler's own embedded schedule.

## License

Apache-2.0. Not affiliated with NVIDIA. CUDA binary utilities are downloaded
from NVIDIA's redistributable archive under the [CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html).
