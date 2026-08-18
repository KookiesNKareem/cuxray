// Deliberate spill demo.
// Compiled with -maxrregcount 32.
// Line numbers match the bundled cubin.
#define ACC 40

__global__ void spilly(const float* __restrict__ x, float* __restrict__ out,
                       int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float acc[ACC];
    for (int j = 0; j < ACC; ++j) acc[j] = x[(i + j) % n];

    for (int it = 0; it < iters; ++it) {
        for (int j = 0; j < ACC; ++j) {
            acc[j] = acc[j] * 1.0009765625f + acc[(j + 1) % ACC];
        }
    }
    float s = 0.f;
    for (int j = 0; j < ACC; ++j) s += acc[j];
    out[i] = s;
}
