// Hardware validation for Layer B verdicts: time the fixture access patterns.
// If cuxray's static verdicts are right, col_conflict runs ~an order of
// magnitude slower per iteration than padded_clean/xor_swizzle, which run at
// the same speed as each other; strided_global runs far slower than a
// coalesced twin. No profiler needed — conflicts show up in wall clock.
//
// Build: nvcc -arch=sm_86 -O3 -o validate_banks validate_banks.cu -cudart static
#include <cstdio>

#define N 32

__global__ void col_conflict(const float* x, float* y, int iters) {
    __shared__ float t[N][N];
    int tx = threadIdx.x;
    for (int j = 0; j < N; ++j) t[j][tx] = x[j * N + tx];
    __syncthreads();
    volatile float (*vt)[N] = t;
    float s = 0.f;
    for (int it = 0; it < iters; ++it)
        for (int j = 0; j < N; ++j) s += vt[tx][j];
    y[tx] = s;
}

__global__ void padded_clean(const float* x, float* y, int iters) {
    __shared__ float t[N][N + 1];
    int tx = threadIdx.x;
    for (int j = 0; j < N; ++j) t[j][tx] = x[j * N + tx];
    __syncthreads();
    volatile float (*vt)[N + 1] = t;
    float s = 0.f;
    for (int it = 0; it < iters; ++it)
        for (int j = 0; j < N; ++j) s += vt[tx][j];
    y[tx] = s;
}

__global__ void xor_swizzle(const float* x, float* y, int iters) {
    __shared__ float t[N][N];
    int tx = threadIdx.x;
    for (int j = 0; j < N; ++j) t[j][tx ^ (j & (N - 1))] = x[j * N + tx];
    __syncthreads();
    volatile float (*vt)[N] = t;
    float s = 0.f;
    for (int it = 0; it < iters; ++it)
        for (int j = 0; j < N; ++j) s += vt[tx][j ^ (tx & (N - 1))];
    y[tx] = s;
}

__global__ void strided_global(const float* x, float* y, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0.f;
    for (int it = 0; it < iters; ++it)
        s += x[i * 32 + (it & 1023)];
    y[i] = s;
}

__global__ void coalesced_global(const float* x, float* y, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0.f;
    for (int it = 0; it < iters; ++it)
        s += x[i + (it & 1023) * 32];
    y[i] = s;
}

template <typename K>
static float bench(const char* name, K kernel, const float* x, float* y,
                   int blocks, int threads, int iters) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    kernel<<<blocks, threads>>>(x, y, iters);  // warmup
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < 10; ++r) kernel<<<blocks, threads>>>(x, y, iters);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    std::printf("%-18s %8.3f ms\n", name, ms / 10);
    return ms / 10;
}

int main() {
    float *x, *y;
    cudaMalloc(&x, (32 * 32 * 40 + 2048) * sizeof(float));
    cudaMalloc(&y, 32 * 40 * sizeof(float));
    cudaMemset(x, 0, (32 * 32 * 40 + 2048) * sizeof(float));
    const int iters = 200000;

    float col = bench("col_conflict", col_conflict, x, y, 1, 32, iters);
    float pad = bench("padded_clean", padded_clean, x, y, 1, 32, iters);
    float xr = bench("xor_swizzle", xor_swizzle, x, y, 1, 32, iters);
    float sg = bench("strided_global", strided_global, x, y, 8, 32, iters / 50);
    float cg = bench("coalesced_global", coalesced_global, x, y, 8, 32, iters / 50);

    std::printf("\ncol/padded ratio:   %.1fx (predicted ~32-way conflict)\n", col / pad);
    std::printf("xor/padded ratio:   %.2fx (predicted clean == clean)\n", xr / pad);
    std::printf("strided/coalesced:  %.1fx (predicted 32 vs 4-5 sectors)\n", sg / cg);
    return 0;
}
