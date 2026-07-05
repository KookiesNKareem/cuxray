// vLLM port candidate: W4A16 GPTQ (uint4b8 sym, g=128) decode GEMM for
// batch 1-8 on Ampere. Consumes GPTQ tensors via a one-time repack (the
// pattern every vLLM mixed-precision kernel uses), fp16 or bf16
// activations, arbitrary M, optional split-K for latency-bound shapes.
//
// Build: nvcc -arch=sm_86 -O3 -lineinfo -o gemv_vllm gemv_vllm.cu
// Run:   ./gemv_vllm M K [NC]
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#ifndef RPW
#define RPW 2
#endif
#ifndef BLOCK
#define BLOCK 512
#endif
#ifndef TK
#define TK 2048            // x tile elements staged in smem per iteration
#endif
#define G 128              // GPTQ group size

// Swizzle<2,4,3> on byte offsets (from cuxray solve): conflict-free half2
// reads of the x tile at the v6 access pattern.
__device__ __forceinline__ unsigned sw243(unsigned byteoff) {
    return byteoff ^ ((byteoff >> 3) & 0x30u);
}

// Interleaved-tile swizzle: lanes stride 16 pair-blocks of NC*4 bytes, all
// landing on bank 0; XOR lane-entropy bits into the bank bits while keeping
// the vector alignment (low log2(NC*4) bits) intact.
template <int NC>
__device__ __forceinline__ unsigned swI(unsigned off) {
    if (NC == 2) return off ^ ((off >> 4) & 0x78u);
    if (NC == 4) return off ^ ((off >> 4) & 0x70u);
    if (NC == 8) return off ^ ((off >> 4) & 0x60u);  // keep 32B blocks intact
    return off;
}

// fp16 / bf16 magic dequant traits. Packing a nibble into the mantissa of
// (1<<mbits)-scaled 1.0 gives base+n exactly; subtracting (base+8) yields
// n-8 with one instruction pair per lane pair.
template <typename T> struct dq_traits;
template <> struct dq_traits<half> {
    using T2 = half2;
    static constexpr unsigned magic = 0x64006400u;   // fp16 1024.0 per half
    static __device__ __forceinline__ T2 bias() {    // 1032 = 1024 + 8
        return __halves2half2(__ushort_as_half(0x6408), __ushort_as_half(0x6408));
    }
    static __device__ __forceinline__ T2 zero() {
        return __halves2half2(__ushort_as_half(0), __ushort_as_half(0));
    }
    static __device__ __forceinline__ T2 sub(T2 a, T2 b) { return __hsub2(a, b); }
    static __device__ __forceinline__ float tof(half v) { return __half2float(v); }
    static constexpr bool heavy_acc = false;
    using ACC = half2;   // fp16 chains of 8 are fine; fold per u
    static __device__ __forceinline__ ACC zacc() { return zero(); }
    static __device__ __forceinline__ void fma_acc(T2 w, T2 x, ACC& a) {
        a = __hfma2(w, x, a);
    }
    static __device__ __forceinline__ float fold(ACC a) {
        const float2 f = __half22float2(a);
        return f.x + f.y;
    }
};
template <> struct dq_traits<__nv_bfloat16> {
    using T2 = __nv_bfloat162;
    static constexpr unsigned magic = 0x43004300u;   // bf16 128.0 per bf16
    static __device__ __forceinline__ T2 bias() {    // 136 = 128 + 8
        return __halves2bfloat162(__ushort_as_bfloat16(0x4308),
                                  __ushort_as_bfloat16(0x4308));
    }
    static __device__ __forceinline__ T2 zero() {
        return __halves2bfloat162(__ushort_as_bfloat16(0), __ushort_as_bfloat16(0));
    }
    static __device__ __forceinline__ T2 sub(T2 a, T2 b) { return __hsub2(a, b); }
    static __device__ __forceinline__ float tof(__nv_bfloat16 v) { return __bfloat162float(v); }
    static constexpr bool heavy_acc = true;   // float2 doubles register cost
    using ACC = float2;  // 7 mantissa bits cannot carry an 8-term chain
    static __device__ __forceinline__ ACC zacc() { return make_float2(0.f, 0.f); }
    static __device__ __forceinline__ void fma_acc(T2 w, T2 x, ACC& a) {
        const float2 wf = __bfloat1622float2(w), xf = __bfloat1622float2(x);
        a.x = fmaf(wf.x, xf.x, a.x);
        a.y = fmaf(wf.y, xf.y, a.y);
    }
    static __device__ __forceinline__ float fold(ACC a) { return a.x + a.y; }
};

// ---------------------------------------------------------------------------
// Batched decode GEMM: y[nc][m] = sum_k w[m][k] * x[nc][k].
// One warp owns RPW rows; NC columns share each dequantized weight pair.
// x is staged tile-by-tile in swizzled smem (NC * TK * 2 bytes).
// SPLITS > 1 splits K across blockIdx.y into float partials (atomicAdd).
template <typename T, int NC, int SPLITS, int R>
__global__ void __launch_bounds__(BLOCK)
gemv_w4(const uint4* __restrict__ w, const T* __restrict__ x,
        const T* __restrict__ scales, float* __restrict__ ypartial,
        T* __restrict__ y, int M, int K, int ldx, int tk) {
    using TR = dq_traits<T>;
    using T2 = typename TR::T2;
    extern __shared__ char smem[];   // NC swizzled x tiles, tk*2 B each

    const int warps = blockDim.x / 32;
    const int warp  = threadIdx.x / 32;
    const int lane  = threadIdx.x & 31;
    const int m0    = (blockIdx.x * warps + warp) * R;
    // per-split range in whole tiles; trailing splits may idle
    const int ntiles = (K + tk - 1) / tk;
    const int tiles_per = (ntiles + SPLITS - 1) / SPLITS;
    const int kbeg = blockIdx.y * tiles_per * tk;
    const int kend = min(K, kbeg + tiles_per * tk);

    float acc[R][NC];
    #pragma unroll
    for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int nc = 0; nc < NC; ++nc) acc[r][nc] = 0.f;

    const int chunks_row = K / 32;   // uint4 per row overall

    for (int kt = kbeg; kt < kend; kt += tk) {
        const int tlen = min(tk, kend - kt);
        // cooperative staging. NC==1: swizzled tile (Swizzle<2,4,3> from
        // cuxray solve). NC>1: column-interleaved pairs [pair][nc] so one
        // vector load serves every column at a given k.
        for (int i = threadIdx.x; i < NC * tlen / 2; i += blockDim.x) {
            int nc, pp, off;
            if (NC == 1) {
                nc = 0; pp = i;
                off = (int)sw243((unsigned)(4 * pp));
            } else {
                pp = i / NC; nc = i % NC;
                off = (int)swI<NC>((unsigned)(pp * NC * 4)) + nc * 4;
            }
            const T2 v = *(const T2*)(x + (size_t)nc * ldx + kt + 2 * pp);
            *(T2*)(smem + off) = v;
        }
        __syncthreads();

        if (m0 < M) {
            const int c0 = kt / 32, c1 = (kt + tlen) / 32;
            for (int c = c0 + lane; c < c1; c += 32) {
                const int k0 = c * 32;
                uint4 q[R];
                #pragma unroll
                for (int r = 0; r < R; ++r)
                    if (m0 + r < M)
                        q[r] = __ldcs(&w[(size_t)(m0 + r) * chunks_row + c]);
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (m0 + r >= M) break;
                    const float s = TR::tof(scales[(size_t)(m0 + r) * (K / G) + k0 / G]);
                    const unsigned qw[4] = {q[r].x, q[r].y, q[r].z, q[r].w};
                    typename TR::ACC h[NC];
                    #pragma unroll
                    for (int nc = 0; nc < NC; ++nc) h[nc] = TR::zacc();
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const int pair0 = (k0 - kt) / 2 + u * 4;
                        #pragma unroll
                        for (int t = 0; t < 4; ++t) {
                            const unsigned b = ((qw[u] >> (4 * t)) & 0x000F000Fu) | TR::magic;
                            const T2 wpair = TR::sub(*(const T2*)&b, TR::bias());
                            T2 xp[NC];
                            if (NC == 1) {
                                xp[0] = *(const T2*)(smem +
                                        sw243((unsigned)(pair0 + t) * 4));
                            } else {
                                const char* base = smem +
                                    swI<NC>((unsigned)(pair0 + t) * NC * 4);
                                #pragma unroll
                                for (int nc = 0; nc < NC; ++nc)
                                    xp[nc] = *(const T2*)(base + nc * 4);
                            }
                            #pragma unroll
                            for (int nc = 0; nc < NC; ++nc)
                                TR::fma_acc(wpair, xp[nc], h[nc]);
                        }
                        // fold each 8-term chain into the float accumulator
                        #pragma unroll
                        for (int nc = 0; nc < NC; ++nc) {
                            acc[r][nc] += s * TR::fold(h[nc]);
                            h[nc] = TR::zacc();
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < R; ++r) {
        if (m0 + r >= M) break;
        #pragma unroll
        for (int nc = 0; nc < NC; ++nc) {
            float a = acc[r][nc];
            #pragma unroll
            for (int off = 16; off; off >>= 1)
                a += __shfl_down_sync(0xffffffff, a, off);
            if (lane == 0) {
                if (SPLITS == 1) y[(size_t)nc * M + m0 + r] = (T)a;
                else atomicAdd(&ypartial[(size_t)nc * M + m0 + r], a);
            }
        }
    }
}

template <typename T>
__global__ void to_out(const float* __restrict__ y32, T* __restrict__ y,
                       int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) y[i] = (T)y32[i];
}

// ---------------------------------------------------------------------------
// Host: GPTQ generation, repack, fp64 reference, graph timing.

// GPTQ: qweight int32[k/8][n], nibble i of word (u, j) = quant of
// w[8u + i][j], LSB-first. Sym 4-bit: value = (q - 8) * scale.
static void make_gptq(unsigned* qweight, float* scales, int k, int n) {
    for (long i = 0; i < (long)k / 8 * n; ++i)
        qweight[i] = (unsigned)rand() ^ ((unsigned)rand() << 16);
    for (long i = 0; i < (long)k / G * n; ++i)
        scales[i] = (rand() % 900 + 100) / 1000.f;
}

// Repack to the kernel layout: row-major [n][k/8] u32 with the interleaved
// nibble order (slot j = k0 + (j<4 ? 2j : 2(j-4)+1)) the magic dequant wants.
static void repack_gptq(const unsigned* qweight, unsigned* out, int k, int n) {
    for (int j = 0; j < n; ++j)
        for (int u = 0; u < k / 8; ++u) {
            const unsigned src = qweight[(size_t)u * n + j];  // seq nibbles
            unsigned dst = 0;
            for (int slot = 0; slot < 8; ++slot) {
                const int korig = (slot < 4) ? 2 * slot : 2 * (slot - 4) + 1;
                dst |= ((src >> (4 * korig)) & 0xF) << (4 * slot);
            }
            out[(size_t)j * (k / 8) + u] = dst;
        }
}

static void ref_gemv_gptq(const unsigned* qweight, const float* scales,
                          const float* x, double* y, int k, int n, int nc,
                          int ldx) {
    for (int col = 0; col < nc; ++col)
        for (int j = 0; j < n; ++j) {
            double accd = 0;
            for (int i = 0; i < k; ++i) {
                const int q = (qweight[(size_t)(i / 8) * n + j] >> (4 * (i % 8))) & 0xF;
                accd += (double)(q - 8) * scales[(size_t)(i / G) * n + j]
                        * x[(size_t)col * ldx + i];
            }
            y[(size_t)col * n + j] = accd;
        }
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), \
            __FILE__, __LINE__); exit(1); } } while (0)

cudaStream_t g_cap_stream = 0;

static float bench_us(void (*fn)(void*), void* ud, int iters = 200) {
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaGraph_t graph; cudaGraphExec_t exec;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    g_cap_stream = s;
    fn(ud);
    CK(cudaStreamEndCapture(s, &graph));
    g_cap_stream = 0;
    CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int i = 0; i < 20; ++i) CK(cudaGraphLaunch(exec, s));
    CK(cudaStreamSynchronize(s));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, s));
    for (int i = 0; i < iters; ++i) CK(cudaGraphLaunch(exec, s));
    CK(cudaEventRecord(e1, s));
    CK(cudaStreamSynchronize(s));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(s);
    return ms * 1000.f / iters;
}

static double host_tof(half v) { return __half2float(v); }
static double host_tof(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T, int NC, int SPLITS = 1>
struct Launch {
    // measured on sm_86: register pressure flips the win to 1 row/warp at
    // NC=8 (fp16) and NC>=4 (bf16's float2 accumulators)
    static constexpr int R =
        (NC >= 8 || (dq_traits<T>::heavy_acc && NC >= 4)) ? 1 : RPW;
    const uint4* w; const T* x; const T* sc; float* y32; T* y;
    int M, K, ldx;
    static void run(void* p) {
        auto* a = (Launch*)p;
        const int rows_per_block = (BLOCK / 32) * R;
        dim3 grid((a->M + rows_per_block - 1) / rows_per_block, SPLITS);
        // whole-K tile when it fits (v6-equivalent, one sync); else stream
        int tk = ((size_t)NC * a->K * 2 <= 96 * 1024 && SPLITS == 1) ? a->K : TK;
        size_t smem = (size_t)NC * tk * 2;
        cudaFuncSetAttribute(gemv_w4<T, NC, SPLITS, R>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             99 * 1024);
        if (SPLITS > 1)
            cudaMemsetAsync(a->y32, 0, sizeof(float) * NC * a->M, g_cap_stream);
        gemv_w4<T, NC, SPLITS, R><<<grid, BLOCK, smem, g_cap_stream>>>(
            a->w, a->x, a->sc, a->y32, a->y, a->M, a->K, a->ldx, tk);
        if (SPLITS > 1)   // split-K: reduce float partials to the out dtype
            to_out<T><<<(NC * a->M + 255) / 256, 256, 0, g_cap_stream>>>(
                a->y32, a->y, NC * a->M);
    }
};

static bool g_fast = false;

template <typename T, int NC, int SPLITS = 1>
static void run_case(const char* tag, const uint4* dw, const T* dx,
                     const T* dsc, float* dy32, T* dy, int M, int K, int ldx,
                     const double* ref, double rms) {
    Launch<T, NC, SPLITS> a{dw, dx, dsc, dy32, dy, M, K, ldx};
    if (g_fast) {
        const float us = bench_us(Launch<T, NC, SPLITS>::run, &a);
        const double bytes = (double)M * K / 2 + (double)M * (K / G) * 2
                           + (double)NC * K * 2 + (double)NC * M * 2;
        printf("%-22s NC=%d  %8.1f us  %6.0f GB/s  max_rel -1 SKIP\n",
               tag, NC, us, bytes / us / 1e3);
        return;
    }
    Launch<T, NC, SPLITS>::run(&a);   // once, eagerly, for the check
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    T* hy = (T*)malloc(sizeof(T) * NC * M);
    CK(cudaMemcpy(hy, dy, sizeof(T) * NC * M, cudaMemcpyDeviceToHost));
    double maxrel = 0;
    for (long i = 0; i < (long)NC * M; ++i) {
        const double got = host_tof(hy[i]);
        const double want = ref[i];
        maxrel = fmax(maxrel, fabs(got - want) / fmax(fabs(want), 0.05 * rms));
    }
    // output rounding is part of got now: bf16 stores carry 2^-8 ulps
    const double gate = sizeof(T) == 2 && dq_traits<T>::heavy_acc ? 0.10 : 0.03;
    const float us = bench_us(Launch<T, NC, SPLITS>::run, &a);
    const double bytes = (double)M * K / 2 + (double)M * (K / G) * 2
                       + (double)NC * K * 2 + (double)NC * M * 2;
    printf("%-22s NC=%d  %8.1f us  %6.0f GB/s  max_rel %.4f %s\n",
           tag, NC, us, bytes / us / 1e3, maxrel, maxrel < gate ? "OK" : "FAIL");
    free(hy);
}

int main(int argc, char** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 4096;
    const int K = argc > 2 ? atoi(argv[2]) : 4096;
    g_fast = argc > 3 && strcmp(argv[3], "fast") == 0;
    srand(42);

    unsigned* hq = (unsigned*)malloc((size_t)K / 8 * M * 4);
    float* hs = (float*)malloc((size_t)K / G * M * 4);
    make_gptq(hq, hs, K, M);
    unsigned* hw = (unsigned*)malloc((size_t)M * K / 8 * 4);
    repack_gptq(hq, hw, K, M);

    const int NCMAX = 8, ldx = K;
    float* hx = (float*)malloc(sizeof(float) * NCMAX * K);
    for (long i = 0; i < (long)NCMAX * K; ++i)
        hx[i] = (rand() % 1000 - 500) / 500.f;

    // device tensors (scales transposed to [M][K/G] row-major)
    uint4* dw; cudaMalloc(&dw, (size_t)M * K / 2);
    cudaMemcpy(dw, hw, (size_t)M * K / 2, cudaMemcpyHostToDevice);
    half* hsc16 = (half*)malloc((size_t)M * (K / G) * 2);
    __nv_bfloat16* hscbf = (__nv_bfloat16*)malloc((size_t)M * (K / G) * 2);
    for (int j = 0; j < M; ++j)
        for (int g2 = 0; g2 < K / G; ++g2) {
            const float v = hs[(size_t)g2 * M + j];
            hsc16[(size_t)j * (K / G) + g2] = __float2half(v);
            hscbf[(size_t)j * (K / G) + g2] = __float2bfloat16(v);
        }
    half *dsc16, *dx16, *dy16;
    __nv_bfloat16 *dscbf, *dxbf, *dybf;
    float* dy32;
    cudaMalloc(&dsc16, (size_t)M * (K / G) * 2);
    cudaMalloc(&dscbf, (size_t)M * (K / G) * 2);
    cudaMalloc(&dx16, sizeof(half) * NCMAX * K);
    cudaMalloc(&dxbf, sizeof(__nv_bfloat16) * NCMAX * K);
    cudaMalloc(&dy16, sizeof(half) * NCMAX * M);
    cudaMalloc(&dybf, sizeof(__nv_bfloat16) * NCMAX * M);
    cudaMalloc(&dy32, sizeof(float) * NCMAX * M);
    cudaMemcpy(dsc16, hsc16, (size_t)M * (K / G) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dscbf, hscbf, (size_t)M * (K / G) * 2, cudaMemcpyHostToDevice);
    half* hx16 = (half*)malloc(sizeof(half) * NCMAX * K);
    __nv_bfloat16* hxbf = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * NCMAX * K);
    for (long i = 0; i < (long)NCMAX * K; ++i) {
        hx16[i] = __float2half(hx[i]);
        hxbf[i] = __float2bfloat16(hx[i]);
        hx[i] = __half2float(hx16[i]);   // reference sees rounded x (fp16 run)
    }
    cudaMemcpy(dx16, hx16, sizeof(half) * NCMAX * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dxbf, hxbf, sizeof(__nv_bfloat16) * NCMAX * K, cudaMemcpyHostToDevice);

    // the kernel reads dtype-rounded scales; the reference must too
    float* hs16 = (float*)malloc((size_t)K / G * M * 4);
    float* hsbf = (float*)malloc((size_t)K / G * M * 4);
    for (long i = 0; i < (long)K / G * M; ++i) {
        hs16[i] = __half2float(__float2half(hs[i]));
        hsbf[i] = __bfloat162float(__float2bfloat16(hs[i]));
    }
    double* ref = (double*)malloc(sizeof(double) * NCMAX * M);
    if (!g_fast) ref_gemv_gptq(hq, hs16, hx, ref, K, M, NCMAX, ldx);
    double rms = 0;
    for (int j = 0; j < M; ++j) rms += ref[j] * ref[j];
    rms = sqrt(rms / M);

    printf("GPTQ uint4b8 g=128, M=%d K=%d (repacked once on host)\n", M, K);
    run_case<half, 1>("fp16", dw, dx16, dsc16, dy32, dy16, M, K, ldx, ref, rms);
    run_case<half, 2>("fp16", dw, dx16, dsc16, dy32, dy16, M, K, ldx, ref, rms);
    run_case<half, 4>("fp16", dw, dx16, dsc16, dy32, dy16, M, K, ldx, ref, rms);
    run_case<half, 8>("fp16", dw, dx16, dsc16, dy32, dy16, M, K, ldx, ref, rms);
    run_case<half, 1, 4>("fp16 split-K x4", dw, dx16, dsc16, dy32, dy16, M, K, ldx, ref, rms);

    // bf16 reference must see bf16-rounded x and scales
    for (long i = 0; i < (long)NCMAX * K; ++i)
        hx[i] = __bfloat162float(hxbf[i]);
    if (!g_fast) ref_gemv_gptq(hq, hsbf, hx, ref, K, M, NCMAX, ldx);
    run_case<__nv_bfloat16, 1>("bf16", dw, dxbf, dscbf, dy32, dybf, M, K, ldx, ref, rms);
    run_case<__nv_bfloat16, 4>("bf16", dw, dxbf, dscbf, dy32, dybf, M, K, ldx, ref, rms);
    return 0;
}
