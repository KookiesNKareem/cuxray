# cuxray — hardware-free static analyzer for CUDA kernel binaries

**Goal:** `pip install cuxray` → point it at any cubin / `.so` / Triton cache /
`.cu` and get source-attributed register pressure, spills, and occupancy —
plus diffs between builds and a CI gate — with **no GPU, no CUDA toolkit, on
any Linux machine (incl. CI runners and containers on a Mac)**. CLI + JSON +
MCP server from day 1. Name/PyPI/GitHub verified free 2026-07-04.

**Positioning (from verified landscape research, 2026-07-04):** nothing wraps
`nvdisasm -plr -gi` (per-instruction liveness + source attribution); spill
tracking in CI is still "grep ptxas -v" (rapidsai/cuml#1658 open since 2020);
NCU's analyses are profile-gated and GUI-first; the LLM-kernel wave needs
cheap pre-execution feedback. Layer A ships v0.1 because it is 100% ground
truth (compiler's own ledger + published spec tables — nothing estimated).
Layer B (static bank-conflict/coalescing, affine+XOR dataflow) = v0.2.
Layer C (control-bits/stall decode, "llvm-mca for SASS") = v0.3, gated on
sm_120 decode validation against NCU on our box.

## Decisions (locked)

- **Name:** cuxray. **License:** Apache-2.0. **Language:** Python 3.10+
  (ecosystem, MCP, agents; parsing perf is fine — stream, don't slurp).
- **v0.1 = Layer A only.** Consumer: well-documented CLI + `--json` (MCP + skill removed 2026-07-04 — agents drive the CLI).
- **Repo:** github.com/<kareem>/cuxray (public at v0.1 tag, not before).

## v0.1 CLI surface

```
cuxray ls <artifact>                          # kernels found, arch, sizes
cuxray report <artifact> [--kernel RE] [--threads N] [--json] [--sass]
cuxray occupancy <artifact> --threads N [--what-if regs=-8] [--sweep] [--json]
cuxray diff <old> <new> [--kernel RE] [--json]
cuxray gate <artifact> "spills==0, regs<=168, occupancy(threads=256)>=25"
```

`<artifact>` = .cubin | ELF (exe/.so/torch ext → cuobjdump extract, iterate,
`--kernel` filter, demangle) | Triton cache dir (walk metadata+cubins) |
.cu/.ptx (compile via bundled ptxas; .cu needs nvcc → document "compile
yourself or give me PTX/cubin"; PTX path is first-class since ptxas is pip-able).

## Report contents (per kernel)

1. **Resources:** regs, static/dyn smem, spill store/load bytes, stack,
   `__launch_bounds__` if present. Sources: `.nv.info` metadata in the cubin
   (works on binaries we didn't build) cross-checked vs `ptxas -v` when we
   compiled. Disagreement = bug, assert loudly.
2. **Pressure curve:** `nvdisasm -plr` → live-reg count per instruction →
   peak + top contributing source lines (`-gi`). Degrade gracefully without
   -lineinfo (report SASS addrs + print the one flag to add).
3. **Spill map:** STL/LDL scan → file:line, **weighted by loop depth** from
   `nvdisasm -cfg` back-edge detection (prologue spill = info; inner-loop
   spill = red).
4. **Occupancy:** per-arch tables (sm_75→sm_121: regfile 64K, warp/block
   slots, smem configs, reg alloc granularity — from CUDA occupancy
   calculator data / programming guide; validate against NCU's
   `ncu_occupancy` module in a dev-time test, NOT a runtime dep). Limiter
   naming + **cliff detection** ("8 regs from 37%"). Block size is runtime →
   `--threads`, read launch_bounds, or sweep table.
5. **Meta:** toolchain versions, artifact sha256, schema version — reports
   must be reproducible artifacts.

**JSON schema:** versioned (`cuxray.schema/1`), one doc per artifact, kernels
keyed by mangled name (demangled alongside). The diff engine and MCP return
the same shapes. Schema is the product — design review it before M4.

## Toolchain acquisition (the "no CUDA install" trick)

Resolution order: `$CUXRAY_TOOLCHAIN` → `$CUDA_HOME/bin` → PATH → conda env →
**auto-fetch from NVIDIA redist** (`developer.download.nvidia.com/compute/cuda/redist/`,
`cuda_nvdisasm` + `cuda_cuobjdump` + `cuda_nvcc`(ptxas), linux-x86_64 +
linux-sbsa/aarch64, pinned versions + sha256, cached in `~/.cache/cuxray/`,
EULA notice on first fetch). Verified available through CUDA 13.3; conda-forge
mirror as fallback. No macOS binaries exist → macOS host = clear error
pointing at container/CI path.

## Testing / CI (GPU-free by construction)

- **Fixture corpus** compiled in CI (ubuntu-latest + fetched toolchain):
  saxpy, tiled matmul, forced-spill kernel (`-maxrregcount 32`), launch_bounds
  kernel, multi-kernel fatbin (sm_90 + sm_120a), a Triton cache fixture
  (recorded, not generated — no torch dep in CI), CUTLASS-sized SASS fixture
  (recorded) for parser streaming/perf.
- Parser unit tests run off **recorded** nvdisasm/ptxas outputs (no network);
  golden-snapshot reports; occupancy vs hand-computed + `ncu_occupancy`
  cross-check job. Matrix: CUDA 12.9 + 13.3 toolchains (nvdisasm output
  format drift is our #1 supply-chain risk — snapshot both).

## Milestones

- **M0** scaffold: pyproject, toolchain fetcher, `cuxray ls` end-to-end on a
  cubin. (First real code; everything downstream hangs off ingestion.)
- **M1** resources: .nv.info + ptxas -v → report + JSON + terminal renderer.
- **M2** liveness/spills: -plr/-gi/-cfg parsers, pressure curve, loop-weighted
  spill map. (The hard/novel parsing; budget half the total effort here.)
- **M3** occupancy: arch tables, limiter, what-if, cliffs, sweep.
- **M4** diff + gate expression parser (schema freeze first).
- **M5** MCP server + skill file.
- **M6** dogfood + launch: run on the MXFP4 sm_120a kernel build and a vLLM
  `.so`; README with REAL outputs from that run; PyPI 0.1.0; post to GPU MODE.

Estimate: M0–M6 ≈ 1.5–2 weeks of focused work.

## Dev environment

- Mac has **no Docker/colima** (checked 2026-07-04) → install colima; arm64
  Linux container + linux-sbsa toolchain = full local loop, no GPU needed.
- GPU box (RTX PRO 6000, sm_120a) DOWN since 2026-07-03 — NOT a blocker for
  v0.1 (only needed for Layer C validation later + dogfood artifacts; the
  MXFP4 cubins can be built in-container too).
- Godbolt API (nvcc133, sm_120a) verified working for quick fixture SASS.

## Explicitly out of v0.1

Layer B (access patterns), Layer C (control bits), Windows, AMD/rocm,
`.cu` compilation (nvcc), TMA-specific anything, any perf *estimation* —
v0.1 never prints a number that isn't ground truth.

## Risks

- **nvdisasm output format drift** across CUDA versions → recorded-fixture
  matrix, tolerant parsers, loud version warnings.
- **NVIDIA ships this** (nsight-python trajectory) → our moat is B/C + CI/MCP
  ergonomics; ship fast, be the default before they notice.
- **-lineinfo absence** in the wild → the degraded report must still be
  obviously useful (SASS-addr pressure curve + resource diff).
