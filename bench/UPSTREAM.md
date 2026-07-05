# Upstreaming notes

Two tracks for getting the campaign results into public tools. Status and
findings live here; the campaign itself is in CAMPAIGN.md.

## Track 1: llama.cpp (Kareem authors; cuxray assists)

llama.cpp bans predominantly AI-generated PRs and AI-written PR text, and
requires the submitter to explain every line. Division of labor: cuxray
produces findings + receipts, Kareem writes and defends the change.

### Finding A — streaming weight loads: measured, too small, dropped
`__ldcs` on q4_0 weight-side loads (the v6 trick) measured on their own
`test-backend-ops perf` at 4 shapes: 0.2-0.8% — within noise. Their mmvq
already keeps activations L2-resident. Not PR-worthy.

### Finding B — q4_K decode overhead: open, best candidate
llama.cpp's own harness (A5000, m=4096 n=1 k=14336):
q4_0 = 51.28 us, q4_K = 53.13 us (+3.6%) at identical 4.5 bpw traffic.
cuxray on the sm_86 `mul_mat_vec_q<type,1,false,false>` instantiations:

| | q4_0 | q4_K |
|---|---|---|
| regs / occupancy | 56 / 75% (reg-limited) | 39 / 100% |
| hot-loop instructions | 95 | 80 |
| issue+stall cycles per iter | 187 | 143 |
| global bytes per warp-iter | 2560 | 1536 |
| **stall cycles per byte** | **0.073** | **0.093 (+27%)** |
| scoreboard waits per 256 B | 1.6 | 1.0 |

No bank conflicts, traffic inflation 1.0 for both. The +27% issue work per
byte points at the per-call superblock scale decode (6-bit packed scales).
Hypothesis: hoist/vectorize scale unpacking so it amortizes across the
vec_dot calls that share a superblock. Verify: patch, `test-backend-ops`
perf (bs 1..8 for regressions) + correctness mode, perplexity check per
their contributing rules.

Cubins: box `/root/ggmlx/libggml-cuda.37.sm_86.cubin` (extracted from
`build/bin/libggml-cuda.so`); analysis scripts `/root/ggmlx/*.py`.

### Tool gap found while analyzing their kernels
Most weight-load addresses in mmvq are unanalyzable: 64-bit address pairs
built with `LEA.HI.X` carry chains (.X carry-in currently poisons the
dataflow). Tracking register pairs through the carry would unlock
coalescing verdicts on most real-world kernels' global loads.

## Track 2: vLLM batch-1 GPTQ GEMV (full assistance OK)

vLLM's `gptq_marlin` backend calls Marlin unconditionally; batch-1 GPTQ
decode on Ampere = Marlin, which our v6 beats by 15-24% (see CAMPAIGN.md).

Integration shape: a new `MPLinearKernel` (mixed-precision kernels dir),
NOT a patch to gptq_marlin — Marlin owns its weight tiling, so a
coexisting kernel needs its own load-time repack (precedent:
`ExllamaLinearKernel`) plus a prefill fallback (dequant + cublas, same
precedent) since our GEMV covers batch<=~8 only.

Work list: vLLM master checkout; kernel generalization (arbitrary K/M,
zeros/act-order gating in `can_implement` — start sym g=128 only, bf16
activation staging with fp16-convert in the smem pass); measure the
batch crossover vs Marlin (expect ~8); position opt-in with data first,
since default-on regresses prefill vs Marlin.
