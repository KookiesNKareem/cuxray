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

### Finding B — q4_K decode overhead: measured, dropped
The fused-scale-load fix (one 128-bit load for dm+scales replacing the
16-bit loads) measured on their harness: 52.75 us at 4096x14336 (-0.7%)
but +2% at 11008x4096 and 4096x11008. A wash — dropped, patch reverted.
Verdict after two experiments (this + __ldcs): mmvq is at a robust
plateau; single-site micro-diffs don't clear a PR bar. The 3.6% q4_K
gap would need an iteration-mapping restructure (amortize superblock
scale decode across calls) — high regression risk, poor first-PR fit.
Next candidate search: systematic cuxray sweep over the other ggml-cuda
kernels (mul_mat_id/MoE, FA, quantize_q8_1) using the new grid-traffic
and per-byte sched views.

### (superseded) original q4_K analysis
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

### Tool gaps found while analyzing their kernels (partially fixed)
Sliced the actual failing chains (tests/tools-style scripts on box:
/root/{slice,origin,prol}.py). Root causes were NOT LEA.HI.X:
- predicated FIRST writes joined against uninitialized garbage, poisoning
  both halves of `@P/@!P` pairs — fixed (strong update on first write);
- `LEA.HI` (signed-div idiom `x + (x>>31)`) unmodeled — fixed exactly;
- `SHF.R.*.HI` source-in-high-word form — fixed earlier the same day.
Remaining barrier (documented, next session): a control-flow merge in the
warp-uniform MoE channel/fastdiv math where one predicated path loses
uniformity; it dominates mmvq's remaining unanalyzed count (30 of 36 in
q4_0). All fixes unit-tested (130 pass); v7 analysis unaffected.

### Fork ready for the PR
github.com/KookiesNKareem/llama.cpp, branch `cuda-mmvq-q4k-scales`
(created at upstream master 7a63fde). Kareem authors the change there;
the box clone /root/llama.cpp has the build + bench setup but carries a
local test-shapes patch in tests/test-backend-ops.cpp — do NOT include
that in the PR branch.

## Track 3 (2026-07-05, A100 session): THE PR TARGET — W4A8 on Ampere

vLLM has NO W4A8 kernel below Hopper: CutlassW4A8LinearKernel and Machete
both gate on compute capability 90 ("CUTLASS W4A8 requires compute
capability of 90 (Hopper)"); QQQ was removed. On every A100/A30 and all
RTX 30/40-series, CompressedTensorsW4A8Int checkpoints have no native
executor. Meanwhile, measured on A100-SXM4-80GB (achievable 1682 GB/s):

| batch-1 us | 4096^2 | 11008x4096 | 4096x14336 |
|---|---|---|---|
| Marlin W4A16 (vLLM 0.24) | 15.6 | 21.8 | 23.2 |
| our dp4a int8-act kernel (v7k) | 9.5 | 13.0 | 17.9 |

v7k saturates the A100 (99.9% of achievable at 14336) where Marlin sits
at 61-77%; margins +23-40%. The fp16-activation path is NOT the vehicle
on A100 — cuxray sched explains why statically: v6 needs 196 stall
cycles per 512 B streamed = 106% of an SM's issue capacity at A100's
per-SM bandwidth (impossible -> measured 59% cap), v7 needs 63.7 = 34%
(fits -> saturates). Same numbers predict the A5000 behavior. Static
cross-arch saturation prediction — flagship cuxray capability.

Plan: port the dp4a math to CompressedTensorsW4A8Int semantics (s4
weights -> offset-binary in repack, PER-TOKEN dynamic activation scales
via vLLM's existing ops.scaled_int8_quant — their infra, their numerics
contract, so no accuracy-policy objection), integrate as an
AmpereW4A8LinearKernel in the mixed-precision registry (precedent:
AllSpark/Conch/Humming are third-party kernels in that exact list),
dequant+mm prefill fallback, tests, PR. This is a capability-gap PR
("enables W4A8 on Ampere, saturates HBM"), not a marginal-perf PR.

## Track 2 verdict (2026-07-05, definitive matrix): current Marlin is
## at the ceiling too — vLLM PR premise substantially weakened

vLLM 0.19's Marlin (their production linear path, interleaved 7-trial
medians, IQRs within ~1%) at batch 1: 17.4 / 39.4 / 38.3 / 47.8 us at
4096^2, 11008x4096, 4096x11008, 4096x14336 — 2-3x faster than the 0.6.1
build the campaign originally benchmarked, flat across batch 1-8. Ours
(GPTQ format, folded epilogue): 15.6 / 37.4 / 37.5 / 47.5 at batch 1 =
+10% / +5% / +2% / tie; batch >= 2: Marlin ties or wins everywhere but
4096^2. A batch-1-only kernel at +0-10% is a weak vLLM PR; park unless
multi-GPU tuning (4090/A100) widens the margin. What survives: the
llama.cpp/exllamav2 wins (their stacks can't use Marlin), the native
4.125 bpw format kernel (15.5 us at 4096^2), the methodology, and the
tooling story. Recommended vehicle: standalone repo + writeup framed as
"decode GEMVs at the bandwidth ceiling, and the static tooling that
built them" — not a SOTA claim.

## Track 2 (historical): vLLM GPTQ GEMV — kernel built and measured (bench/gemv_vllm.cu)

Status 2026-07-05 (A5000, GPTQ uint4b8 sym g=128, CUDA-graph us):

| shape / batch | NC=1 | NC=2 | NC=4 | NC=8 | Marlin bs1 | bs4 | bs8 |
|---|---|---|---|---|---|---|---|
| 4096x4096 fp16 | 16.9 | 18.5 | 25.6 | 76 | 21.9-39.9* | 24.9-26.1 | 24.0 |
| 4096x14336 fp16 | 48.8 | 58.7 | 85.9 | 215 | 54.5-73.5* | 59.5 | 58.1 |
| 4096x4096 bf16 | 17.5 | — | 35.9 | — | | | |

*Marlin shows 1.5-1.8x run variance on this box (clock lock unsupported);
ranges span best/worst observed. Crossover: we win batch 1-2 clearly,
roughly tie at 4, Marlin wins 8+. Dispatch boundary ~4.

Built & validated: GPTQ consumption via one-time host repack (interleaved
nibble order); fp16 (max_rel 0.018-0.024 vs fp64 GPTQ ref with
dtype-rounded scales/x) and bf16 (exact 0.0000: fp32 accumulation + bf16
magic dequant 0x4300/136); arbitrary M (tested 4000); split-K (correct;
+1% at K=14336, slower at 4096^2); per-config rows-per-warp dispatch
(fp16 NC<=4: 2 rows; NC=8 and bf16 NC>=4: 1 row — register pressure
measured via cuxray: 96 regs -> 33% occupancy).

Tool-driven fixes: cuxray report diagnosed the NC=4 collapse (registers,
no spills), and the interleaved-x-tile bank catastrophe (128*NC-byte
lane strides -> all lanes bank 0) was fixed with per-NC
alignment-preserving XOR swizzles (53.8 -> 25.6 us at NC=4).

Remaining before PR: dequant+cublas prefill fallback; NC=8 register diet
(or cede to Marlin at 8); fold output conversion into the epilogue
(~1.5 us of the NC=1 gap vs campaign v6); multi-arch tune (needs
4090/A100 rental); vLLM master checkout + MPLinearKernel integration.

### Original integration plan

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
