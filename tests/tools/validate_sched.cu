#include <cstdio>
__global__ void chain_loop(float* y, float a, int iters, long long* cyc) {
    float v = a;
    long long t0 = clock64();
    for (int it = 0; it < iters; ++it) {
        v = v * a + 1.f; v = v * a + 2.f; v = v * a + 3.f; v = v * a + 4.f;
        v = v * a + 5.f; v = v * a + 6.f; v = v * a + 7.f; v = v * a + 8.f;
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) *cyc = t1 - t0;
    y[threadIdx.x] = v;
}
int main() {
    float* y; long long* cyc;
    cudaMalloc(&y, 128); cudaMalloc(&cyc, 8);
    const int iters = 1000000;
    chain_loop<<<1, 32>>>(y, 1.0001f, iters, cyc);  // warmup
    chain_loop<<<1, 32>>>(y, 1.0001f, iters, cyc);
    long long c; cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost);
    printf("measured cycles/iter: %.2f\n", (double)c / iters);
    return 0;
}
