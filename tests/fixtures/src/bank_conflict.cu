// Layer B ground-truth fixtures. Compile for a 32-thread (one-warp) block.
// Expected verdicts (hand-derived, validated on hardware in B4):
//   col_conflict   : shared loads 32-way conflicted (classic column access)
//   padded_clean   : same pattern, [33]-padded rows → conflict-free
//   xor_swizzle    : same column pattern, XOR-swizzled → conflict-free
//   broadcast_read : all lanes read the same word → broadcast, clean
//   strided_global : global loads at 128 B lane stride → fully uncoalesced

#define N 32

__global__ void col_conflict(const float* x, float* y, int iters) {
    __shared__ float t[N][N];
    int tx = threadIdx.x;
    for (int j = 0; j < N; ++j) t[j][tx] = x[j * N + tx];
    __syncthreads();
    volatile float (*vt)[N] = t;
    float s = 0.f;
    for (int it = 0; it < iters; ++it)
        for (int j = 0; j < N; ++j) s += vt[tx][j];     // lane stride 128 B
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
        for (int j = 0; j < N; ++j) s += vt[tx][j];     // lane stride 132 B
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

__global__ void broadcast_read(const float* x, float* y, int iters) {
    __shared__ float t[N];
    int tx = threadIdx.x;
    t[tx] = x[tx];
    __syncthreads();
    float s = 0.f;
    for (int it = 0; it < iters; ++it)
        for (int j = 0; j < N; ++j) s += t[j];          // all lanes same word
    y[tx] = s;
}

__global__ void strided_global(const float* x, float* y, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0.f;
    for (int it = 0; it < iters; ++it)
        s += x[i * 32 + it];                            // 128 B lane stride
    y[i] = s;
}
