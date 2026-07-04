// __launch_bounds__ fixture: caps regs via minBlocksPerMultiprocessor and
// records maxThreadsPerBlock in the binary metadata.
__global__ void __launch_bounds__(256, 2)
bounded(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    #pragma unroll
    for (int k = 0; k < 16; ++k) v = fmaf(v, 1.25f, 0.5f * k);
    y[i] = v;
}

// Second kernel in the same cubin: multi-kernel parsing fixture.
__global__ void plain(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 2.0f * x[i];
}
