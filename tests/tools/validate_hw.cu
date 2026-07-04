// Real-hardware occupancy validation: ask the CUDA runtime what it will
// actually co-schedule, print rows for comparison against cuxray's engine.
//
// Build: nvcc -arch=<native sm> -o validate_hw validate_hw.cu -cudart static
// Output rows: name numRegs staticSmem threads dynSmem apiBlocks

#include <cstdio>

__global__ void k_light(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 2.f * x[i];
}

__global__ void __launch_bounds__(256, 6) k_bounded(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
#pragma unroll
    for (int j = 0; j < 32; ++j) v = fmaf(v, 1.25f, 0.5f * j);
    y[i] = v;
}

__global__ void k_heavy(const double* x, double* y, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double acc[24];
    for (int j = 0; j < 24; ++j) acc[j] = x[(i + j) % n];
    for (int it = 0; it < iters; ++it)
        for (int j = 0; j < 24; ++j) acc[j] = acc[j] * 1.0009765625 + acc[(j + 1) % 24];
    double s = 0;
    for (int j = 0; j < 24; ++j) s += acc[j];
    y[i] = s;
}

__global__ void k_smem(const float* x, float* y, int n) {
    extern __shared__ float buf[];
    __shared__ float fixed[1024];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    fixed[threadIdx.x % 1024] = x[i % n];
    buf[threadIdx.x] = fixed[(threadIdx.x * 7) % 1024];
    __syncthreads();
    y[i % n] = buf[(threadIdx.x + 1) % blockDim.x];
}

template <typename F>
static void probe(const char* name, F* fn, bool optin_smem) {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)fn);
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    size_t max_dyn = prop.sharedMemPerBlockOptin - attr.sharedSizeBytes;
    if (optin_smem)
        cudaFuncSetAttribute((const void*)fn,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)max_dyn);
    for (int threads = 64; threads <= 1024; threads += 192) {
        size_t dyn_cases[3] = {0, 16 * 1024, optin_smem ? max_dyn : (size_t)0};
        for (int d = 0; d < (optin_smem ? 3 : 2); ++d) {
            int blocks = -1;
            cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks, (const void*)fn, threads, dyn_cases[d]);
            std::printf("%s %d %zu %d %zu %d %s\n", name, attr.numRegs,
                        attr.sharedSizeBytes, threads, dyn_cases[d],
                        e == cudaSuccess ? blocks : -1, cudaGetErrorString(e));
        }
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("# device %s sm_%d%d smemSM=%zu smemOptin=%zu regsSM=%d thrSM=%d\n",
                prop.name, prop.major, prop.minor,
                prop.sharedMemPerMultiprocessor, prop.sharedMemPerBlockOptin,
                prop.regsPerMultiprocessor, prop.maxThreadsPerMultiProcessor);
    probe("k_light", k_light, false);
    probe("k_bounded", k_bounded, false);
    probe("k_heavy", k_heavy, false);
    probe("k_smem", k_smem, true);
    return 0;
}
