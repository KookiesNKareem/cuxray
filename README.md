# cuxray

[![PyPI](https://img.shields.io/pypi/v/cuxray.svg)](https://pypi.org/project/cuxray/)
[![CI](https://github.com/KookiesNKareem/cuxray/actions/workflows/ci.yml/badge.svg)](https://github.com/KookiesNKareem/cuxray/actions)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Static analysis and optimization for CUDA kernel binaries (register pressure,
spills, occupancy, bank conflicts) **without a GPU**.

cuxray reads what the compiler froze into your cubin and combines it with
NVIDIA's exact architecture tables, so it goes past *reporting* problems to
*synthesizing and verifying* fixes (swizzles, register caps, tile configs).
Measured facts are ground truth; estimates are labeled `est.` and validated
against hardware; anything unknowable is reported as such, with the reason.

Point it at any cubin and get a ranked, confidence-tagged list of what's slow
and how to fix it, with no GPU touched:

```console
$ pip install cuxray
$ cuxray advise w4a8_gemv.sm_80.cubin --threads 256

  gemv_w4a8<__half, 1, 1, 1>(uint4 const*, signed char const*, ...)  (w4a8_gemv.sm_80.cubin)
    1. cut registers to 40 (-4)  · high confidence · impact 50
       unlocks 6 blocks/SM (62.5% → 75.0%); current limiter is registers
       evidence: occupancy model (validated vs cuda_occupancy.h + runtime API)
    2. SIMT datapath caps this loop: tensor cores scale past it  · medium confidence · impact 40
       MACs run on the SIMT int8 datapath (dp4a/FMA). On sm_80 the tensor cores do ~8x more
       int8 MACs/clock. This is fine while memory-bound (low arithmetic-per-byte / batch 1),
       but as arithmetic-per-byte grows the loop becomes SIMT-compute-bound and a tensor-core
       implementation would be up to ~8x faster. Measure across your batch sizes to find the
       crossover.
       evidence: static op-mix + per-arch MAC-rate model (approximate)
```

## Quick start

```console
pip install cuxray                            # CPU-only (no CUDA, no GPU)
cuxray advise mykernels.so --threads 256      # ranked fixes for every kernel
cuxray solve mykernels.so --threads 256       # verified swizzle for any bank conflict
cuxray gate mykernels.so "spill_instrs==0"    # exit 1 in CI on a regression
```

Point it at anything holding cubins: a `.cubin`, a host `.so`, a directory of
Triton caches, a `.ptx`, even a wheel you `pip download`ed. On first run it
fetches pinned, sha256-verified NVIDIA binary utilities; nothing else to install.

## What it does

| command | what you get |
|---|---|
| `advise` · `survey` | ranked, impact-weighted fixes for one kernel · for a whole library |
| `report` · `ls` | spills, register-pressure curve, occupancy + cliffs, access patterns · fast listing |
| `solve` | a verified conflict-free swizzle (`Swizzle<B,M,S>` plus ready-to-paste CUDA) |
| `tune-regs` · `tune` | Pareto occupancy/spill frontier over `-maxrregcount` · over `-D` tile matrices |
| `sched` | per-loop issue+stall cycle estimate from the compiler's own schedule |
| `roofline` | the memory/compute floor for a launch and which resource binds |
| `why` | dataflow slice: where a divergent or uncoalesced address came from |
| `compare` · `diff` | per-kernel A/B across two builds · CI regression detection |
| `gate` | CI exit codes, per-kernel budgets, SARIF annotations, a reusable Action |
| `occupancy` | what-if occupancy sweeps, no binary needed |

Every command takes `--json` / `-o` (schema: `cuxray schema`); `cuxray doctor`
shows toolchain and cache state. The datapath-crossover check (SIMT dp4a/FMA vs
tensor-core headroom) is surfaced by `advise` and `report`.

`solve` is the one thing nothing else does: it finds a bank conflict and hands
back the proven fix, with no GPU:

```console
$ cuxray solve bank_conflict.cubin --threads 256
  _Z12col_conflictPKfPfi
    224 conflicted of 256 shared accesses
  solution (all accesses): Swizzle<5,2,5>  (zero smem cost, verified)
    apply to byte offsets: addr ^ ((addr >> 5) & 0x7c)
    e.g. bank_conflict.cu:19 LDS: 32-way → clean
    // + a ready-to-paste __device__ swizzle() and the cute::Swizzle<5,2,5> layout
```

## Validation

- Occupancy: 4,344/4,344 configs match NVIDIA's `cuda_occupancy.h`; 54/54 match
  the CUDA runtime on real hardware.
- Spill bytes byte-exact vs `ptxas -v`; cycle estimate within 0.7% of
  `clock64()`; `solve` re-derives the canonical CUTLASS `Swizzle<3,4,3>` and its
  fix is hardware-timed within 2% of the padded twin.
- ~161k production kernels (vLLM, PyTorch wheels) analyzed with zero crashes and
  zero false-positive conflict flags.

## Notes

- Inputs: `.cubin`, host ELF (`.so`/`.o`/exe, cubins extracted), directories
  (Triton caches), `.ptx`. Compute capability 7.5–12.x (Turing → Blackwell,
  incl. `sm_120a`).
- Static facts only: cache behavior and achieved bandwidth need a profiler;
  those accesses are reported as unanalyzable, not guessed.
- Pass `--threads` / `--smem-dynamic` when the binary carries no launch metadata
  (cuxray warns when it matters). Linux only; build with `-lineinfo` for source
  attribution.

## License

Apache-2.0. Not affiliated with NVIDIA; CUDA binary utilities are downloaded
from NVIDIA's redistributable archive under the
[CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html).
