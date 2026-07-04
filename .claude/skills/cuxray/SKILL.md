---
name: cuxray
description: Static analysis of CUDA kernel binaries with no GPU — register pressure, spills, occupancy, and build-to-build diffs. Use when optimizing CUDA/Triton/CUTLASS kernels, investigating register pressure or spills, checking occupancy impact of a change, or gating kernel resource regressions in CI.
---

# cuxray — hardware-free CUDA kernel analysis

cuxray reads facts the compiler froze into a kernel binary. Nothing it
reports is an estimate, and it never needs a GPU.

## Core loop for kernel optimization

1. Build the kernel (`nvcc -cubin -lineinfo ...`, or find the cubin in a
   Triton cache / compiled `.so`).
2. `cuxray report kernel.cubin --threads <block size> --json`
   - `resources`: regs, stack (nonzero stack = local memory, often spills)
   - `spills.by_line`: STL/LDL locations, **loop_depth ≥ 1 = hot, fix these**
   - `pressure.peak`: which source line holds the most live registers
   - `occupancy`: limiter + cliffs ("8 fewer regs → +1 block/SM")
3. Edit the kernel, rebuild, then:
   `cuxray diff old.cubin new.cubin --threads N --json`
   — shows exactly what the edit cost/saved (regs, spill bytes, occupancy).
4. Gate the result: `cuxray gate new.cubin "spill_instrs==0, regs<=168"`
   (exit 1 on violation — CI-ready).

## What-if without any binary

`cuxray occupancy --arch sm_120 --regs 168 --threads 256 --smem 8192 --sweep`
answers "is this tile config occupancy-viable" before writing code.

## Reading results

- Spills in loop_depth 0 (prologue) are usually harmless; loop_depth ≥ 1
  spills convert registers into DRAM traffic every iteration — top priority.
- Low occupancy is NOT automatically bad (big-tile GEMMs run at 25%
  deliberately); it matters when the kernel is latency-bound. Report the
  limiter, don't moralize.
- `occupancy.cliffs` gives the nearest actionable boundary — mention it when
  regs are within ~8 of a cliff.
- Compile with `-lineinfo` (free, no perf impact) or attribution falls back
  to SASS addresses.

## Requirements

Linux only (any machine, containers fine). First run auto-fetches NVIDIA
binary utilities (~35 MB) from NVIDIA's redist archive into ~/.cache/cuxray.
