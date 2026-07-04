// ldmatrix + cp.async fixtures (Layer B v0.2.1). One-warp block.
// Ground truth (hand-derived, hardware-timed in tests/tools/validate_ldsm.cu):
//   ldsm_row_major : 128 B row pitch → LDSM 8-way conflict per phase group
//   ldsm_padded    : 144 B row pitch → LDSM conflict-free
//   cp.async fill  : contiguous 16 B per lane → global coalesced, shared clean
#include <cstdint>

#define ROWS 32

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

template <int PITCH_HALVES>
__device__ __forceinline__ unsigned ldsm_loop(const uint4* g, int tx, int iters) {
    __shared__ __align__(16) uint16_t tile[ROWS][PITCH_HALVES];
    // fill smem via cp.async: each lane copies one contiguous 16 B chunk
    uint32_t dst = smem_addr(&tile[0][0]) + tx * 16;
    const char* src = reinterpret_cast<const char*>(g) + tx * 16;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();

    unsigned r0, r1, r2, r3, acc = 0;
    for (int it = 0; it < iters; ++it) {
        uint32_t a = smem_addr(&tile[tx][0]);  // lane tx supplies row tx
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(a));
        acc ^= r0 ^ r1 ^ r2 ^ r3;
    }
    return acc;
}

__global__ void ldsm_row_major(const uint4* g, unsigned* out, int iters) {
    out[threadIdx.x] = ldsm_loop<64>(g, threadIdx.x, iters);   // 128 B pitch
}

__global__ void ldsm_padded(const uint4* g, unsigned* out, int iters) {
    out[threadIdx.x] = ldsm_loop<72>(g, threadIdx.x, iters);   // 144 B pitch
}
