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

Correctness gate: max rel err < 0.03 vs fp64 reference (scale-aware).

## Next

- v5: cp.async double-buffered weight loads; fold to float per chunk
  (sched: fold = 64 cycles/iter, dequant chains 207).
- Grid-level traffic accounting in cuxray roofline (per-block bytes x
  blocks — the x re-staging cost was found by hand).
- External baselines compiled on-box: exllamav2 q4 gemv, llama.cpp
  mul_mat_vec_q, Marlin at its best batch; fp16 cuBLAS reference.
- Shapes 11008x4096 and 4096x11008; final head-to-head table.
