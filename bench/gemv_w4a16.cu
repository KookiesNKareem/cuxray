// W4A16 decode GEMV campaign harness (sm_86 target).
// y[M] = sum_k scale[m][k/G] * (q[m][k] - 8) * x[k], 4-bit packed weights.
// Scoreboard: achieved GB/s of the weight stream vs measured device peak.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>

#define G 128  // quant group size

// ---------------- achievable-bandwidth probe ----------------
__global__ void bw_probe(const uint4* __restrict__ in, uint4* __restrict__ out,
                         size_t n16) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    uint4 acc = {0, 0, 0, 0};
    for (size_t k = i; k < n16; k += stride) {
        uint4 v = in[k];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if (acc.x == 0xDEADBEEF) out[i % 1024] = acc;  // defeat DCE, ~never taken
}

// ---------------- v0: one thread per row, scalar ----------------
__global__ void gemv_v0(const unsigned* __restrict__ w, const half* __restrict__ x,
                        const half* __restrict__ scales, half* __restrict__ y,
                        int M, int K) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    float acc = 0.f;
    for (int k = 0; k < K; k += 8) {
        unsigned q = w[(size_t)m * (K / 8) + k / 8];
        float s = __half2float(scales[(size_t)m * (K / G) + k / G]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int qv = (int)((q >> (4 * j)) & 0xF) - 8;
            acc += s * qv * __half2float(x[k + j]);
        }
    }
    y[m] = __float2half(acc);
}

// ---------------- v1: warp per row, uint4 loads, x in smem ----------------
// Weight layout: row-major, K packed 8 nibbles/u32, read as uint4 (32 vals).
// Chunk c (= iter*32 + lane) covers k in [c*32, c*32+32) — one scale group
// slice per chunk (G=128 = 4 chunks/group).
__global__ void gemv_v1(const uint4* __restrict__ w, const half* __restrict__ x,
                        const half* __restrict__ scales, half* __restrict__ y,
                        int M, int K) {
    extern __shared__ half xs[];
    for (int k = threadIdx.x; k < K; k += blockDim.x) xs[k] = x[k];
    __syncthreads();

    int warps = blockDim.x / 32;
    int m = blockIdx.x * warps + (threadIdx.x / 32);
    if (m >= M) return;
    int lane = threadIdx.x & 31;

    const uint4* row = w + (size_t)m * (K / 32);
    int chunks = K / 32;
    float acc = 0.f;
    for (int c = lane; c < chunks; c += 32) {
        uint4 q = row[c];
        int k0 = c * 32;
        float s = __half2float(scales[(size_t)m * (K / G) + k0 / G]);
        const unsigned qw[4] = {q.x, q.y, q.z, q.w};
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int qv = (int)((qw[u] >> (4 * j)) & 0xF) - 8;
                acc += s * qv * __half2float(xs[k0 + u * 8 + j]);
            }
        }
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) y[m] = __float2half(acc);
}

// ---------------- v2: magic-constant half2 dequant, interleaved pack ------
// Pack order per u32 (8 weights k0..k0+7): nibble slot j holds
// q[k0 + (j<4 ? 2j : 2(j-4)+1)], so masking (w >> 4s) & 0x000F000F yields
// half2 pairs (k0+2s, k0+2s+1) matching contiguous x half2 reads.
// Nibble n in a half's low mantissa with exponent bits 0x6400 = 1024+n,
// so dq = (h - 1032) gives n-8 with no I2F.
__global__ void gemv_v2(const uint4* __restrict__ w, const half* __restrict__ x,
                        const half* __restrict__ scales, half* __restrict__ y,
                        int M, int K) {
    extern __shared__ half xs[];
    for (int k = threadIdx.x * 8; k < K; k += blockDim.x * 8)
        *(uint4*)(xs + k) = *(const uint4*)(x + k);
    __syncthreads();

    int warps = blockDim.x / 32;
    int m = blockIdx.x * warps + (threadIdx.x / 32);
    if (m >= M) return;
    int lane = threadIdx.x & 31;

    const uint4* row = w + (size_t)m * (K / 32);
    int chunks = K / 32;
    const half2 magic = __float2half2_rn(0.f);  // placeholder, set below
    const half2 sub = __halves2half2(__ushort_as_half(0x6408), __ushort_as_half(0x6408)); // 1032.0
    float acc = 0.f;
    for (int c = lane; c < chunks; c += 32) {
        uint4 q = row[c];
        int k0 = c * 32;
        float s = __half2float(scales[(size_t)m * (K / G) + k0 / G]);
        const unsigned qw[4] = {q.x, q.y, q.z, q.w};
        float chunk = 0.f;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const half2* x2 = (const half2*)(xs + k0 + u * 8);
            half2 hacc = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                unsigned bits = ((qw[u] >> (4 * t)) & 0x000F000Fu) | 0x64006400u;
                half2 dq = __hsub2(*(const half2*)&bits, sub);
                hacc = __hfma2(dq, x2[t], hacc);
            }
            chunk += __half2float(__low2half(hacc)) + __half2float(__high2half(hacc));
        }
        acc += s * chunk;
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) y[m] = __float2half(acc);
}

// ---------------- v3: + xs swizzle (cuxray solve: Swizzle<2,4,3>) and dual
// accumulators (halves the fp16 rounding chain, adds ILP) -----------------
__device__ __forceinline__ unsigned sw243(unsigned byteoff) {
    return byteoff ^ ((byteoff >> 3) & 0x30u);  // flips 16B-granule bits only
}

__global__ void gemv_v3(const uint4* __restrict__ w, const half* __restrict__ x,
                        const half* __restrict__ scales, half* __restrict__ y,
                        int M, int K) {
    extern __shared__ half xs[];
    char* xb = (char*)xs;
    for (int k = threadIdx.x * 8; k < K; k += blockDim.x * 8)
        *(uint4*)(xb + sw243(k * 2)) = *(const uint4*)(x + k);
    __syncthreads();

    int warps = blockDim.x / 32;
    int m = blockIdx.x * warps + (threadIdx.x / 32);
    if (m >= M) return;
    int lane = threadIdx.x & 31;

    const uint4* row = w + (size_t)m * (K / 32);
    int chunks = K / 32;
    const half2 sub = __halves2half2(__ushort_as_half(0x6408), __ushort_as_half(0x6408));
    float acc = 0.f;
    for (int c = lane; c < chunks; c += 32) {
        uint4 q = row[c];
        int k0 = c * 32;
        float s = __half2float(scales[(size_t)m * (K / G) + k0 / G]);
        const unsigned qw[4] = {q.x, q.y, q.z, q.w};
        float chunk = 0.f;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const char* xu = xb + sw243((unsigned)(k0 + u * 8) * 2);  // 16B block
            half2 h0 = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));
            half2 h1 = h0;
            #pragma unroll
            for (int t = 0; t < 4; t += 2) {
                unsigned b0 = ((qw[u] >> (4 * t)) & 0x000F000Fu) | 0x64006400u;
                unsigned b1 = ((qw[u] >> (4 * t + 4)) & 0x000F000Fu) | 0x64006400u;
                h0 = __hfma2(__hsub2(*(const half2*)&b0, sub),
                             *(const half2*)(xu + 4 * t), h0);
                h1 = __hfma2(__hsub2(*(const half2*)&b1, sub),
                             *(const half2*)(xu + 4 * t + 4), h1);
            }
            half2 hs = __hadd2(h0, h1);
            chunk += __half2float(__low2half(hs)) + __half2float(__high2half(hs));
        }
        acc += s * chunk;
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) y[m] = __float2half(acc);
}

// ---------------- v4: rows-per-warp amortization (sweep via cuxray tune) --
#ifndef RPW
#define RPW 4     // rows per warp, processed with interleaved loads for MLP
#endif
#ifndef BLOCK
#define BLOCK 256
#endif
__global__ void gemv_v4(const uint4* __restrict__ w, const half* __restrict__ x,
                        const half* __restrict__ scales, half* __restrict__ y,
                        int M, int K) {
    extern __shared__ half xs[];
    char* xb = (char*)xs;
    for (int k = threadIdx.x * 8; k < K; k += blockDim.x * 8)
        *(uint4*)(xb + sw243(k * 2)) = *(const uint4*)(x + k);
    __syncthreads();

    int warps = blockDim.x / 32;
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int m0 = (blockIdx.x * warps + warp) * RPW;
    if (m0 >= M) return;
    int chunks = K / 32;
    const half2 sub = __halves2half2(__ushort_as_half(0x6408), __ushort_as_half(0x6408));

    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; ++r) acc[r] = 0.f;

    for (int c = lane; c < chunks; c += 32) {
        int k0 = c * 32;
        // interleaved row loads: RPW independent 16B loads in flight
        uint4 q[RPW];
        #pragma unroll
        for (int r = 0; r < RPW; ++r)
            q[r] = w[(size_t)(m0 + r) * chunks + c];
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            float s = __half2float(scales[(size_t)(m0 + r) * (K / G) + k0 / G]);
            const unsigned qw[4] = {q[r].x, q[r].y, q[r].z, q[r].w};
            float chunk = 0.f;
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const char* xu = xb + sw243((unsigned)(k0 + u * 8) * 2);
                half2 h0 = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));
                half2 h1 = h0;
                #pragma unroll
                for (int t = 0; t < 4; t += 2) {
                    unsigned b0 = ((qw[u] >> (4 * t)) & 0x000F000Fu) | 0x64006400u;
                    unsigned b1 = ((qw[u] >> (4 * t + 4)) & 0x000F000Fu) | 0x64006400u;
                    h0 = __hfma2(__hsub2(*(const half2*)&b0, sub),
                                 *(const half2*)(xu + 4 * t), h0);
                    h1 = __hfma2(__hsub2(*(const half2*)&b1, sub),
                                 *(const half2*)(xu + 4 * t + 4), h1);
                }
                half2 hs = __hadd2(h0, h1);
                chunk += __half2float(__low2half(hs)) + __half2float(__high2half(hs));
            }
            acc[r] += s * chunk;
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float a = acc[r];
        #pragma unroll
        for (int off = 16; off; off >>= 1)
            a += __shfl_down_sync(0xffffffff, a, off);
        if (lane == 0) y[m0 + r] = __float2half(a);
    }
}

// ---------------- v5: independent accumulator chains per u-slice ---------
// sched on v4: dequant FMA chains stall ~4-5 cy per dependent op. Give the
// scheduler 8 independent chains per row (h[u][2]) and fold once per chunk.
__global__ void gemv_v5(const uint4* __restrict__ w, const half* __restrict__ x,
                        const half* __restrict__ scales, half* __restrict__ y,
                        int M, int K) {
    extern __shared__ half xs[];
    char* xb = (char*)xs;
    for (int k = threadIdx.x * 8; k < K; k += blockDim.x * 8)
        *(uint4*)(xb + sw243(k * 2)) = *(const uint4*)(x + k);
    __syncthreads();

    int warps = blockDim.x / 32;
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int m0 = (blockIdx.x * warps + warp) * RPW;
    if (m0 >= M) return;
    int chunks = K / 32;
    const half2 sub = __halves2half2(__ushort_as_half(0x6408), __ushort_as_half(0x6408));
    const half2 hz = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));

    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; ++r) acc[r] = 0.f;

    for (int c = lane; c < chunks; c += 32) {
        int k0 = c * 32;
        uint4 q[RPW];
        #pragma unroll
        for (int r = 0; r < RPW; ++r)
            q[r] = w[(size_t)(m0 + r) * chunks + c];
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            float s = __half2float(scales[(size_t)(m0 + r) * (K / G) + k0 / G]);
            const unsigned qw[4] = {q[r].x, q[r].y, q[r].z, q[r].w};
            half2 h[4] = {hz, hz, hz, hz};   // 4 independent chains (chain len 4)
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const char* xu = xb + sw243((unsigned)(k0 + u * 8) * 2);
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    unsigned b = ((qw[u] >> (4 * t)) & 0x000F000Fu) | 0x64006400u;
                    h[u] = __hfma2(__hsub2(*(const half2*)&b, sub),
                                   *(const half2*)(xu + 4 * t), h[u]);
                }
            }
            float chunk = 0.f;                 // fold per-8-term fp16 sums
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                float2 f = __half22float2(h[u]);
                chunk += f.x + f.y;
            }
            acc[r] += s * chunk;
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float a = acc[r];
        #pragma unroll
        for (int off = 16; off; off >>= 1)
            a += __shfl_down_sync(0xffffffff, a, off);
        if (lane == 0) y[m0 + r] = __float2half(a);
    }
}

// v5b: no shared memory — x served from L1/L2 (8 KB, hot after first blocks)
__global__ void gemv_v5b(const uint4* __restrict__ w, const half* __restrict__ x,
                         const half* __restrict__ scales, half* __restrict__ y,
                         int M, int K) {
    int warps = blockDim.x / 32;
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int m0 = (blockIdx.x * warps + warp) * RPW;
    if (m0 >= M) return;
    int chunks = K / 32;
    const half2 sub = __halves2half2(__ushort_as_half(0x6408), __ushort_as_half(0x6408));
    const half2 hz = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));

    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; ++r) acc[r] = 0.f;

    for (int c = lane; c < chunks; c += 32) {
        int k0 = c * 32;
        uint4 q[RPW];
        #pragma unroll
        for (int r = 0; r < RPW; ++r)
            q[r] = w[(size_t)(m0 + r) * chunks + c];
        half2 xv[16];
        #pragma unroll
        for (int t = 0; t < 16; ++t)
            xv[t] = __ldg((const half2*)(x + k0) + t);
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            float s = __half2float(scales[(size_t)(m0 + r) * (K / G) + k0 / G]);
            const unsigned qw[4] = {q[r].x, q[r].y, q[r].z, q[r].w};
            half2 h[4] = {hz, hz, hz, hz};
            #pragma unroll
            for (int u = 0; u < 4; ++u)
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    unsigned b = ((qw[u] >> (4 * t)) & 0x000F000Fu) | 0x64006400u;
                    h[u] = __hfma2(__hsub2(*(const half2*)&b, sub),
                                   xv[u * 4 + t], h[u]);
                }
            float chunk = 0.f;                 // fold per-8-term fp16 sums
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                float2 f = __half22float2(h[u]);
                chunk += f.x + f.y;
            }
            acc[r] += s * chunk;
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float a = acc[r];
        #pragma unroll
        for (int off = 16; off; off >>= 1)
            a += __shfl_down_sync(0xffffffff, a, off);
        if (lane == 0) y[m0 + r] = __float2half(a);
    }
}

// v6: v5 + streaming (evict-first) weight loads — weights are
// single-use; preserve L2 for x and scales.
__global__ void gemv_v6(const uint4* __restrict__ w, const half* __restrict__ x,
                        const half* __restrict__ scales, half* __restrict__ y,
                        int M, int K) {
    extern __shared__ half xs[];
    char* xb = (char*)xs;
    for (int k = threadIdx.x * 8; k < K; k += blockDim.x * 8)
        *(uint4*)(xb + sw243(k * 2)) = *(const uint4*)(x + k);
    __syncthreads();

    int warps = blockDim.x / 32;
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int m0 = (blockIdx.x * warps + warp) * RPW;
    if (m0 >= M) return;
    int chunks = K / 32;
    const half2 sub = __halves2half2(__ushort_as_half(0x6408), __ushort_as_half(0x6408));
    const half2 hz = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));

    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; ++r) acc[r] = 0.f;

    for (int c = lane; c < chunks; c += 32) {
        int k0 = c * 32;
        uint4 q[RPW];
        #pragma unroll
        for (int r = 0; r < RPW; ++r)
            q[r] = __ldcs(&w[(size_t)(m0 + r) * chunks + c]);
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            float s = __half2float(scales[(size_t)(m0 + r) * (K / G) + k0 / G]);
            const unsigned qw[4] = {q[r].x, q[r].y, q[r].z, q[r].w};
            half2 h[4] = {hz, hz, hz, hz};   // 4 independent chains (chain len 4)
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const char* xu = xb + sw243((unsigned)(k0 + u * 8) * 2);
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    unsigned b = ((qw[u] >> (4 * t)) & 0x000F000Fu) | 0x64006400u;
                    h[u] = __hfma2(__hsub2(*(const half2*)&b, sub),
                                   *(const half2*)(xu + 4 * t), h[u]);
                }
            }
            float chunk = 0.f;                 // fold per-8-term fp16 sums
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                float2 f = __half22float2(h[u]);
                chunk += f.x + f.y;
            }
            acc[r] += s * chunk;
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float a = acc[r];
        #pragma unroll
        for (int off = 16; off; off >>= 1)
            a += __shfl_down_sync(0xffffffff, a, off);
        if (lane == 0) y[m0 + r] = __float2half(a);
    }
}


// ---------------- v7: int8 activations + dp4a (exact integer dot) --------
// x is quantized once per call to q8 with one fp scale per 128-group plus a
// per-32-chunk int sum (for the -8 offset fold). Weights repacked so a u32
// holds 8 weights: lo nibbles = k0..k0+3 (bytes), hi nibbles = k0+4..k0+7.
// dot = sw*sx*(dp4a(vlo,x0..3) + dp4a(vhi,x4..7) - 8*sum(x_chunk)).
__global__ void quant_x_q8(const half* __restrict__ x, signed char* __restrict__ xq,
                           float* __restrict__ xscale, int* __restrict__ xsum,
                           int K) {
    int g = blockIdx.x;              // one block per 128-group
    int k0 = g * G;
    __shared__ float wmax[4];
    __shared__ float fabsmax;
    float m = 0.f;
    for (int i = threadIdx.x; i < G; i += blockDim.x)
        m = fmaxf(m, fabsf(__half2float(x[k0 + i])));
    #pragma unroll
    for (int off = 16; off; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
    if ((threadIdx.x & 31) == 0) wmax[threadIdx.x / 32] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float mm = fmaxf(fmaxf(wmax[0], wmax[1]), fmaxf(wmax[2], wmax[3]));
        fabsmax = mm; xscale[g] = mm / 127.f;
    }
    __syncthreads();
    float inv = fabsmax > 0.f ? 127.f / fabsmax : 0.f;
    for (int i = threadIdx.x; i < G; i += blockDim.x) {
        int q = __float2int_rn(__half2float(x[k0 + i]) * inv);
        xq[k0 + i] = (signed char)q;
    }
    __syncthreads();
    // per-32-chunk sums (4 chunks per group)
    if (threadIdx.x < 4) {
        int sum = 0;
        for (int i = 0; i < 32; ++i) sum += xq[k0 + threadIdx.x * 32 + i];
        xsum[g * 4 + threadIdx.x] = sum;
    }
}

__global__ void __launch_bounds__(BLOCK, 1536 / BLOCK)
gemv_v7(const uint4* __restrict__ w, const signed char* __restrict__ xq,
                        const float* __restrict__ xscale, const int* __restrict__ xsum,
                        const half* __restrict__ scales, half* __restrict__ y,
                        int M, int K) {
    extern __shared__ int xsi[];               // q8 x + chunk sums in smem
    int* xs32 = xsi;                            // K/4 ints of x bytes
    int* xsum_s = xsi + K / 4;                  // K/32 chunk sums
    // Swizzle<1,4,3> from cuxray solve: byte^((byte>>3)&0x10); int form below
    for (int i = threadIdx.x; i < K / 4; i += blockDim.x)
        xs32[i ^ ((i >> 3) & 0x4)] = ((const int*)xq)[i];
    for (int i = threadIdx.x; i < K / 32; i += blockDim.x)
        xsum_s[i] = xsum[i];
    __syncthreads();

    int warps = blockDim.x / 32;
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int m0 = (blockIdx.x * warps + warp) * RPW;
    if (m0 >= M) return;
    int chunks = K / 32;

    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; ++r) acc[r] = 0.f;

    for (int c = lane; c < chunks; c += 32) {
        int k0 = c * 32;
        float sx = xscale[k0 / G];
        int sumx = xsum_s[c];
        uint4 q[RPW];
        #pragma unroll
        for (int r = 0; r < RPW; ++r)
            q[r] = __ldcs(&w[(size_t)(m0 + r) * chunks + c]);
        const int4* xv = (const int4*)xs32;
        int va = (k0 >> 4), vb = va + 1;             // int4-index swizzle form
        int4 xa = xv[va ^ ((va >> 3) & 1)];
        int4 xb = xv[vb ^ ((vb >> 3) & 1)];
        const int xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            float sw = __half2float(scales[(size_t)(m0 + r) * (K / G) + k0 / G]);
            const unsigned qw[4] = {q[r].x, q[r].y, q[r].z, q[r].w};
            int isum = 0;
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                unsigned lo = qw[u] & 0x0F0F0F0Fu;
                unsigned hi = (qw[u] >> 4) & 0x0F0F0F0Fu;
                isum = __dp4a((int)lo, xw[2 * u], isum);
                isum = __dp4a((int)hi, xw[2 * u + 1], isum);
            }
            acc[r] += sw * sx * (float)(isum - 8 * sumx);
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float a = acc[r];
        #pragma unroll
        for (int off = 16; off; off >>= 1)
            a += __shfl_down_sync(0xffffffff, a, off);
        if (lane == 0) y[m0 + r] = __float2half(a);
    }
}

// ---------------- host ----------------
static void ref_gemv(const unsigned* w, const float* x, const float* s,
                     float* y, int M, int K) {
    for (int m = 0; m < M; ++m) {
        double acc = 0;
        for (int k = 0; k < K; k += 8) {
            unsigned q = w[(size_t)m * (K / 8) + k / 8];
            for (int j = 0; j < 8; ++j) {
                int qv = (int)((q >> (4 * j)) & 0xF) - 8;
                acc += (double)s[(size_t)m * (K / G) + k / G] * qv * x[k + j];
            }
        }
        y[m] = (float)acc;
    }
}

template <typename F>
static float bench_us(F launch, int iters = 200) {
    // CUDA-graph timing: capture the launch once, replay N times.
    // Matches llama.cpp test-backend-ops methodology (no launch gaps).
    cudaStream_t st; cudaStreamCreate(&st);
    cudaGraph_t g; cudaGraphExec_t ge;
    launch(); cudaDeviceSynchronize();
    cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
    launch();
    cudaStreamEndCapture(st, &g);
    if (cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0) != cudaSuccess) {
        // fallback: plain launches
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) launch();
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        return ms * 1000.f / iters;
    }
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 3; ++i) cudaGraphLaunch(ge, st);
    cudaStreamSynchronize(st);
    cudaEventRecord(a, st);
    for (int i = 0; i < iters; ++i) cudaGraphLaunch(ge, st);
    cudaEventRecord(b, st); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(st);
    return ms * 1000.f / iters;
}

int main(int argc, char** argv) {
    int M = argc > 1 ? atoi(argv[1]) : 4096;
    int K = argc > 2 ? atoi(argv[2]) : 4096;
    size_t wbytes = (size_t)M * K / 2;
    size_t sbytes = (size_t)M * (K / G) * sizeof(half);
    double stream_bytes = (double)wbytes + sbytes + K * 2 + M * 2;

    unsigned *hw = (unsigned*)malloc(wbytes);
    float *hx = (float*)malloc(K * 4), *hs = (float*)malloc(sbytes * 2), *hy = (float*)malloc(M * 4);
    srand(42);
    for (size_t i = 0; i < wbytes / 4; ++i) hw[i] = rand() ^ (rand() << 16);
    for (int i = 0; i < K; ++i) hx[i] = (rand() % 1000 - 500) / 500.f;
    for (size_t i = 0; i < (size_t)M * (K / G); ++i) hs[i] = (rand() % 900 + 100) / 1000.f;

    unsigned* dw; half *dx, *ds, *dy;
    cudaMalloc(&dw, wbytes); cudaMalloc(&dx, K * 2);
    cudaMalloc(&ds, sbytes); cudaMalloc(&dy, M * 2);
    cudaMemcpy(dw, hw, wbytes, cudaMemcpyHostToDevice);
    half* tmp = (half*)malloc((size_t)M * (K / G) * 2);
    for (size_t i = 0; i < (size_t)M * (K / G); ++i) tmp[i] = __float2half(hs[i]);
    cudaMemcpy(ds, tmp, sbytes, cudaMemcpyHostToDevice);
    half* tx = (half*)malloc(K * 2);
    for (int i = 0; i < K; ++i) tx[i] = __float2half(hx[i]);
    cudaMemcpy(dx, tx, K * 2, cudaMemcpyHostToDevice);
    // reference must see the fp16-rounded x the kernels actually read
    for (int i = 0; i < K; ++i) hx[i] = __half2float(tx[i]);

    // correctness (small subset check against fp64 reference)
    float* refy = (float*)malloc(M * 4);
    ref_gemv(hw, hx, hs, refy, M, K);

    double rms = 0;
    for (int m = 0; m < M; ++m) rms += refy[m] * refy[m];
    rms = sqrt(rms / M);
    auto check = [&](const char* name) {
        half* hy2 = (half*)malloc(M * 2);
        cudaMemcpy(hy2, dy, M * 2, cudaMemcpyDeviceToHost);
        double maxrel = 0;
        for (int m = 0; m < M; ++m) {
            double got = __half2float(hy2[m]);
            double want = refy[m];
            // scale-aware: cancellation makes tiny |want| meaningless
            double rel = fabs(got - want) / fmax(fabs(want), 0.05 * rms);
            if (rel > maxrel) maxrel = rel;
        }
        printf("  %-8s max_rel_err %.4f %s\n", name, maxrel,
               maxrel < 0.03 ? "OK" : "FAIL");
        free(hy2);
    };

    // achievable peak: large streaming read (DRAM-resident, not L2-warm)
    size_t probe_bytes = 256ull << 20;
    uint4* probe_in; cudaMalloc(&probe_in, probe_bytes);
    cudaMemset(probe_in, 1, probe_bytes);
    uint4* probe_out; cudaMalloc(&probe_out, 1024 * 16);
    float us_bw = bench_us([&] {
        bw_probe<<<432, 256>>>(probe_in, probe_out, probe_bytes / 16); }, 20);
    double peak = probe_bytes / us_bw / 1e3;  // GB/s
    printf("achievable read BW: %.0f GB/s (256 MB stream)\n", peak);
    cudaFree(probe_in);

    float us0 = bench_us([&] {
        gemv_v0<<<(M + 255) / 256, 256>>>(dw, dx, ds, dy, M, K); });
    check("v0");
    float us1 = bench_us([&] {
        gemv_v1<<<(M + 7) / 8, 256, K * 2>>>((uint4*)dw, dx, ds, dy, M, K); });
    check("v1");

    // v2: repack nibbles into the interleaved order
    unsigned* hw2 = (unsigned*)malloc(wbytes);
    for (size_t u = 0; u < wbytes / 4; ++u) {
        unsigned src = hw[u], dst = 0;
        for (int j = 0; j < 8; ++j) {
            int korig = (j < 4) ? 2 * j : 2 * (j - 4) + 1;
            unsigned nib = (src >> (4 * korig)) & 0xF;
            dst |= nib << (4 * j);
        }
        hw2[u] = dst;
    }
    unsigned* dw2; cudaMalloc(&dw2, wbytes);
    cudaMemcpy(dw2, hw2, wbytes, cudaMemcpyHostToDevice);
    float us2 = bench_us([&] {
        gemv_v2<<<(M + 7) / 8, 256, K * 2>>>((uint4*)dw2, dx, ds, dy, M, K); });
    check("v2");
    float us3 = bench_us([&] {
        gemv_v3<<<(M + 7) / 8, 256, K * 2>>>((uint4*)dw2, dx, ds, dy, M, K); });
    check("v3");
    int rows_per_block = (BLOCK / 32) * RPW;
    float us4 = bench_us([&] {
        gemv_v4<<<(M + rows_per_block - 1) / rows_per_block, BLOCK, K * 2>>>(
            (uint4*)dw2, dx, ds, dy, M, K); });
    check("v4");
    float us5 = bench_us([&] {
        gemv_v5<<<(M + rows_per_block - 1) / rows_per_block, BLOCK, K * 2>>>(
            (uint4*)dw2, dx, ds, dy, M, K); });
    check("v5");
    float us5b = bench_us([&] {
        gemv_v5b<<<(M + rows_per_block - 1) / rows_per_block, BLOCK>>>(
            (uint4*)dw2, dx, ds, dy, M, K); });
    check("v5b");
    float us6 = bench_us([&] {
        gemv_v6<<<(M + rows_per_block - 1) / rows_per_block, BLOCK, K * 2>>>(
            (uint4*)dw2, dx, ds, dy, M, K); });
    check("v6");

    // v7: repack for byte-nibble order (lo nibbles k0..k0+3, hi k0+4..k0+7)
    unsigned* hw3 = (unsigned*)malloc(wbytes);
    for (size_t u = 0; u < wbytes / 4; ++u) {
        unsigned src = hw[u], dst = 0;
        for (int j = 0; j < 8; ++j) {
            unsigned nib = (src >> (4 * j)) & 0xF;
            // j<4 -> byte j lo nibble; j>=4 -> byte (j-4) hi nibble
            if (j < 4) dst |= nib << (8 * j);
            else       dst |= nib << (8 * (j - 4) + 4);
        }
        hw3[u] = dst;
    }
    unsigned* dw3; cudaMalloc(&dw3, wbytes);
    cudaMemcpy(dw3, hw3, wbytes, cudaMemcpyHostToDevice);
    signed char* dxq; float* dxs; int* dxsum;
    cudaMalloc(&dxq, K); cudaMalloc(&dxs, (K / G) * 4); cudaMalloc(&dxsum, (K / 32) * 4);
    size_t v7smem = K + (K / 32) * 4;
    // v7 reference: same int8 x-quantization policy, fp64 arithmetic.
    // Checks kernel correctness; the policy's own noise vs fp16 x is printed
    // separately (llama.cpp q4_0*q8_1 shares this policy).
    float* hxq = (float*)malloc(K * 4);
    for (int g = 0; g < K / G; ++g) {
        float m = 0.f;
        for (int i = 0; i < G; ++i) m = fmaxf(m, fabsf(hx[g * G + i]));
        float sx = m / 127.f, inv = m > 0.f ? 127.f / m : 0.f;
        for (int i = 0; i < G; ++i)
            hxq[g * G + i] = sx * (float)(int)rintf(hx[g * G + i] * inv);
    }
    float* refy7 = (float*)malloc(M * 4);
    ref_gemv(hw, hxq, hs, refy7, M, K);
    double policy = 0;
    for (int m = 0; m < M; ++m)
        policy = fmax(policy, fabs(refy7[m] - refy[m]) / fmax(fabs(refy[m]), 0.05 * rms));
    printf("  x-quant policy noise vs fp16 x: %.4f (shared with q4_0*q8_1 class)\n", policy);
    float* refy_save = refy;
    float us7 = bench_us([&] {
        quant_x_q8<<<K / G, 128>>>(dx, dxq, dxs, dxsum, K);
        gemv_v7<<<(M + rows_per_block - 1) / rows_per_block, BLOCK, v7smem>>>(
            (uint4*)dw3, dxq, dxs, dxsum, ds, dy, M, K); });
    refy = refy7; check("v7"); refy = refy_save;
    float us7k = bench_us([&] {
        gemv_v7<<<(M + rows_per_block - 1) / rows_per_block, BLOCK, v7smem>>>(
            (uint4*)dw3, dxq, dxs, dxsum, ds, dy, M, K); });

    printf("shape %dx%d  stream %.1f MB\n", M, K, stream_bytes / 1e6);
    printf("v0: %7.1f us  %6.0f GB/s  (%4.1f%% of achievable)\n",
           us0, stream_bytes / us0 / 1e3, 100.0 * stream_bytes / us0 / 1e3 / peak);
    printf("v1: %7.1f us  %6.0f GB/s  (%4.1f%% of achievable)\n",
           us1, stream_bytes / us1 / 1e3, 100.0 * stream_bytes / us1 / 1e3 / peak);
    printf("v2: %7.1f us  %6.0f GB/s  (%4.1f%% of achievable)\n",
           us2, stream_bytes / us2 / 1e3, 100.0 * stream_bytes / us2 / 1e3 / peak);
    printf("v3: %7.1f us  %6.0f GB/s  (%4.1f%% of achievable)\n",
           us3, stream_bytes / us3 / 1e3, 100.0 * stream_bytes / us3 / 1e3 / peak);
    printf("v4(RPW=%d,B=%d): %.1f us  %6.0f GB/s  (%4.1f%% of achievable)\n",
           RPW, BLOCK, us4, stream_bytes / us4 / 1e3, 100.0 * stream_bytes / us4 / 1e3 / peak);
    printf("v5:  %7.1f us  %6.0f GB/s  (%4.1f%% of achievable)\n",
           us5, stream_bytes / us5 / 1e3, 100.0 * stream_bytes / us5 / 1e3 / peak);
    printf("v5b: %7.1f us  %6.0f GB/s  (%4.1f%% of achievable)\n",
           us5b, stream_bytes / us5b / 1e3, 100.0 * stream_bytes / us5b / 1e3 / peak);
    printf("v6:  %7.1f us  %6.0f GB/s  (%4.1f%% of achievable)\n",
           us6, stream_bytes / us6 / 1e3, 100.0 * stream_bytes / us6 / 1e3 / peak);
    printf("v7:  %7.1f us  %6.0f GB/s  (%4.1f%% of achievable)  [incl. x-quant]\n",
           us7, stream_bytes / us7 / 1e3, 100.0 * stream_bytes / us7 / 1e3 / peak);
    printf("v7k: %7.1f us  %6.0f GB/s  (%4.1f%% of achievable)  [gemv only]\n",
           us7k, stream_bytes / us7k / 1e3, 100.0 * stream_bytes / us7k / 1e3 / peak);
    return 0;
}
