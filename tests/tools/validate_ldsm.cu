#include <cstdio>
#include <cstdint>
#define ROWS 32
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
template <int PITCH_HALVES>
__device__ __forceinline__ unsigned ldsm_loop(const uint4* g, int tx, int iters) {
    __shared__ __align__(16) uint16_t tile[ROWS][PITCH_HALVES];
    uint32_t dst = smem_addr(&tile[0][0]) + tx * 16;
    const char* src = reinterpret_cast<const char*>(g) + tx * 16;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    unsigned r0, r1, r2, r3, acc = 0;
    for (int it = 0; it < iters; ++it) {
        uint32_t a = smem_addr(&tile[tx][0]);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
        acc ^= r0 ^ r1 ^ r2 ^ r3;
    }
    return acc;
}
__global__ void ldsm_row_major(const uint4* g, unsigned* out, int iters) {
    out[threadIdx.x] = ldsm_loop<64>(g, threadIdx.x, iters);
}
__global__ void ldsm_padded(const uint4* g, unsigned* out, int iters) {
    out[threadIdx.x] = ldsm_loop<72>(g, threadIdx.x, iters);
}
template <typename K>
static float bench(const char* n, K k, const uint4* g, unsigned* o, int iters) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    k<<<1, 32>>>(g, o, iters); cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < 10; ++r) k<<<1, 32>>>(g, o, iters);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    printf("%-16s %8.3f ms\n", n, ms / 10);
    return ms / 10;
}
int main() {
    uint4* g; unsigned* o;
    cudaMalloc(&g, 4096); cudaMalloc(&o, 256); cudaMemset(g, 1, 4096);
    float rm = bench("ldsm_row_major", ldsm_row_major, g, o, 400000);
    float pd = bench("ldsm_padded", ldsm_padded, g, o, 400000);
    printf("row_major/padded: %.2fx (cuxray predicts 8-way vs clean)\n", rm / pd);
    return 0;
}
