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

## Track 3 progress (same day): W4A8 kernel built, both GPUs measured

bench/gemv_vllm.cu now has the full CompressedTensorsW4A8Int path: raw-s4
repack (signed high-nibble dp4a — bytes are 16*s4, /16 folded into weight
scales, which deletes the offset-correction sum subsystem entirely),
2-phase per-token dynamic int8 quant (stand-in for vLLM's
ops.scaled_int8_quant), batch 1-8 (interleaved swizzled int8 tiles),
split-K, pointer-marched inner loop (cuxray-guided: 147.7 -> ~65 stall
cyc/512B). Correct at 0.0098-0.0122 everywhere on both GPUs.

Batch-1 vs current Marlin-W4A16 fallback (incl. stand-in quant):
- A100:  15.7/18.5/20.0/24.3 vs 15.6/21.8/19.5/23.2 -> tie/+18%/-2%/-5%
- A5000: 18.5/-/-/49.4 vs 17.4/-/-/47.8 (dp4a) BUT the fp16-math kernel
  wins there: 15.6/47.5 -> +10%/+1%
Stand-in quant costs ~2-3 us/call (3 launches); the vLLM integration
uses their optimized quant op, which should close most of the A100
14336 deficit (v7-equivalence bound: 22.0 us incl quant, beats 23.2).

PR DESIGN INSIGHT (cuxray-derived): the optimal math flips per SKU —
fp16 dequant math saturates sm_86 but is issue-capacity-impossible on
sm_80 (196 stall-cyc/512B = 106% of an SM's issue slots at A100's
per-SM bandwidth demand); dp4a (65) fits both but loses to fp16 on
sm_86. AmpereW4A8LinearKernel should dispatch internally per capability:
sm_80 -> dp4a, sm_86/89 -> fp16-dequant math. Static analysis receipts
go in the PR description.

Remaining for the PR: vLLM-tree integration (kernel in csrc, registry
entry, their quant op, their tests), real W4A8 checkpoint end-to-end,
prefill fallback, 4090 (sm_89) datapoint.

## Track 3 original finding (2026-07-05, A100 session): W4A8 on Ampere

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


## Kernel hunt round 2 (2026-07-05): cuxray sweep of llama.cpp CUDA backend

Ran `cuxray advise` across the built libggml-cuda.so kernels (sm_86) to find
a PR-worthy inefficiency in younger/hot code. cuxray surfaced real findings;
none cleared the bar, and understanding *why* is the useful result:

- **topk_moe_cuda<256>** (DeepSeek-scale MoE routing): 46 spill stores /
  30 loads, 496 B/thread local traffic. BUT regs=31, occupancy=100%
  (limiter=warps) — the spill is ptxas's own occupancy-preserving choice,
  not a hard pressure wall. And MoE *routing* is negligible next to the
  expert *matmuls* it precedes, so even a 2x routing speedup is ~0
  end-to-end. Not worth a PR.
- **mul_mat_vec_f (bf16)**: 6 uncoalesced global accesses (44% efficiency)
  but only ×1.17 overall traffic inflation, partial coverage — a minor,
  likely-known layout tradeoff.
- **flash_attn_tile**: 70 8-way bank conflicts + a register cliff
  (33%→50% at -21 regs). Real, but FA has a standing optimization army and
  the conflicts are plausibly an intentional padding tradeoff; not a fight
  worth picking.

Verdict: mature llama.cpp CUDA kernels are near-optimal (this echoes the
earlier mmvq dry run). cuxray *works* — it found the issues and, crucially,
gave the numbers to reject each — but the PR-worthy opportunity is a
capability gap, not a marginal win in tuned code. That gap is vLLM W4A8
on Ampere (branches banked, Track 3). The `advise` verdicts here are the
tool doing exactly its job: telling you where NOT to spend effort.


## DEFINITIVE head-to-head (2026-07-05, A100-80GB driver 580, vLLM 0.24 + #38066)

The measurement that survives full rigor: both W4A8-INT paths driven
END-TO-END through vLLM, identical weights, both using vLLM's own
ops.scaled_int8_quant, CUDA-graph timed as one unit. Ours = the exact
Dp4aW4A8LinearKernel.apply_weights forward. Marlin = vLLM's real
CompressedTensorsW4A8Int scheme (create_weights -> process -> apply).

| N x K | ours | Marlin (#38066 path) | speedup | out agreement |
|---|---|---|---|---|
| 4096x4096   | 11.5 us | 21.8 | 1.89x | 0.9% |
| 11008x4096  | 19.1 us | 28.6 | 1.50x | 1.4% |
| 4096x11008  | 23.3 us | 32.1 | 1.37x | 1.0% |
| 4096x14336  | 31.1 us | 38.1 | 1.23x | 1.0% |

Output agreement = ours vs Marlin on identical weights: ~1%, i.e. our
kernel produces the answer vLLM already trusts, 1.2-1.9x faster. A5000
(consumer Ampere) earlier showed 7-11% via the same method; the win is
LARGER on the datacenter A100. Standalone .cu numbers (15.7us at 4096^2)
UNDERSTATED this because they used a slower 2-phase quant; the shipped
kernel uses vLLM's scaled_int8_quant and hits 11.5us.

This resolves the whole vLLM track: the dp4a kernel is a materially
better implementation of the exact path the open (stalled, unreviewed)
PR #38066 enables — not a duplicate. Positioning: land as a follow-on /
alternative backend once #38066 establishes act_type=int8 selection, or
engage #38066 with these numbers.
Earlier 'inf' correctness readings were a broken hand-rolled harness;
driven through vLLM's real scheme, correctness is clean (~1%).
