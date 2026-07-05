# W4A16 decode GEMV campaign (sm_86)

Goal: beat the best published batch-1 int4 GEMV kernels (exllamav2,
llama.cpp, Marlin) at Llama shapes on RTX A5000, with design decisions
driven by cuxray. Scoreboard: fraction of measured achievable DRAM
bandwidth (681 GB/s on a 256 MB stream; theoretical 768).

## Status: 4096x4096, g=128

| version | change (cuxray finding that drove it) | us | GB/s | % peak |
|---|---|---|---|---|
| v0 | scalar thread-per-row baseline | 177.9 | 49 | 7% |
| v1 | warp/row, uint4 loads, x in smem | 28.0 | 310 | 46% |
| v2 | half2 magic dequant (`sched`: 257/281 cycles in I2F chain) | 18.0 | 481 | 71% |
| v3 | xs swizzle (`solve`: Swizzle<2,4,3>) + dual acc (precision gate) | 17.5 | 494 | 73% |
| v4 | rows-per-warp + block sweep (x re-staging amortization) RPW=2 B=512 | 15.7 | 551 | 81% |

| v5 | independent FMA chains per u-slice (sched: chain stalls) | 15.9 | 543 | 80% — no gain: stalls weren't binding |
| v5b | no-smem, x from L1 | 24.8 | 350 | 51% — L1 can't serve 64 B/lane walks; staging vindicated |
| v6 | `__ldcs` streaming weight loads (block-invariant x reads flagged by report; weights are single-use, keep L2 for x) | 15.5 | 558 | 82% |
| v7 | int8 activations + `dp4a` (exact integer dot; llama.cpp's precision class). Swizzle<1,4,3> from `solve` on the int8 x tile; report verifies it clean post-apply | 17.3 incl. x-quant / 15.0 gemv | — | 83-85% |

Big shapes (v6/v7k, RPW=2 B=512): 11008x4096 = 35.8/34.5 us (95.5/**98.9%**);
4096x11008 = 36.5/35.8 us (93.1/94.7%); 4096x14336 = 46.4/45.3 us
(95.2/**97.6%**). The 4096x4096 gap is launch/tail overhead at 15 us scale,
not kernel deficiency (RPW/BLOCK sweep confirmed no better point).

Correctness gate: max rel err < 0.03 vs fp64 reference (scale-aware), with
the reference reading the same fp16-rounded x the kernels see. v7 gates
against a quantized-x fp64 reference (0.009-0.013), which isolates kernel
arithmetic; the int8-activation policy itself adds ~0.21-0.25 noise vs
fp16 x on this synthetic data — identical to llama.cpp's q4_0*q8_1 class.

## Final head-to-head (us, batch-1, CUDA-graph or native harness, A5000)

| M x K | **v6 (ours)** | v7 incl. x-quant | llama.cpp q4_0 | llama.cpp q4_K | exllamav2 gptq4 g128 | Marlin g128 |
|---|---|---|---|---|---|---|
| 4096 x 4096 | **15.5** | 17.3 | 17.9 | 18.5 | 19.6 | 21.9 |
| 11008 x 4096 | **35.8** | 36.6 | 40.5 | 43.3 | 43.9 | 46.9 |
| 4096 x 11008 | **36.5** | 38.0 | 41.1 | 42.3 | 42.4 | 42.9 |
| 4096 x 14336 | **46.4** | 47.6 | 51.3 | 53.1 | 52.1 | 54.5 |

v6 wins every shape against every baseline (10-24%). Context on format
bytes: q4_0 is 4.5 bpw (fp16 scale per 32) vs our 4.125 (per 128), so
~8% of the q4_0 margin is format; exllamav2/Marlin GPTQ g=128 is 4.16 bpw
— those margins (11-24%) are kernel efficiency, same traffic. At 95-99%
of measured achievable DRAM bandwidth there is <5% left on the table for
any kernel of this format class on this GPU.

Baseline provenance: llama.cpp test-backend-ops perf (own harness, build
w/ CUDA 12.9, campaign shapes added); exllamav2 0.3.2 wheel
(gemm_half_q_half, gemv path cross-validated against its own
dequant+cublas to 0.043); Marlin via vllm 0.6.1 gptq_marlin_gemm
(uint4b8 sym g=128, dequant-reference max_rel 0.009-0.015; best of runs
taken for all baselines).

**2026-07-05 CORRECTION — current Marlin.** The Marlin column above is
vLLM 0.6.1's build. vLLM 0.19's production path (apply_gptq_marlin_linear,
same GPU, interleaved 7-trial medians, tight IQRs) is far faster at
batch 1 and flat across batch: 17.4 / 39.4 / 38.3 / 47.8 us at the four
shapes. Against it, our GPTQ-format kernel (bench/gemv_vllm.cu, folded
epilogue) measures 15.6 / 37.4 / 37.5 / 47.5 us at batch 1 — +10% / +5% /
+2% / tie — and ties-to-loses at batch >= 2. The llama.cpp and exllamav2
comparisons are unaffected (their stacks cannot use Marlin). Full matrix
in UPSTREAM.md.

Tool development driven by this campaign:
- block-invariant global read detection (blockIdx taint) — automates the
  x re-staging discovery (v1) and drove `__ldcs` on weights (v6).
- alignment-tracked MIXED lane values (`U % 2^a == 0` on the unknown
  uniform part): XOR-swizzled indices now trace through SHR/AND/XOR and
  LOP3 (via LUT decomposition into fused binary ops), so `report`
  verifies the swizzle `solve` prescribed — found on v7, where the
  swizzled LDS.128s were previously "not analyzable".
- SHF.R.*.HI funnel-shift form (source in the high word) was unmodeled.
- blockDim read from the constant bank (c[0x0][0x0..8]) now resolves to
  the caller-supplied block shape instead of an unknown uniform.
- coalescing efficiency could exceed 100% for partial-broadcast reads;
  ideal sectors now use the warp's unique byte footprint.

## Next

- Kernel: fuse x-quant into gemv_v7 (removes the ~2 us extra launch);
  cp.async double-buffering remains unexplored.
- Grid-level traffic accounting in cuxray roofline (per-block bytes x
  blocks — the x re-staging cost was found by hand).
- Port the campaign kernel to a vLLM-compatible op as the upstreaming
  vehicle; sweep batch 2-8 where Marlin starts to win.
