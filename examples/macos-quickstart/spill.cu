// Deliberate register-spill fixture. A per-thread accumulator array too big
// for the register budget, kept live across a loop; compiled with
// -maxrregcount 32 to force STL/LDL inside the hot loop.
#define ACC 40

__global__ void spilly(const float* __restrict__ x, float* __restrict__ out,
                       int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float acc[ACC];
    for (int j = 0; j < ACC; ++j) acc[j] = x[(i + j) % n];

    for (int it = 0; it < iters; ++it) {          // hot loop: spills live here
        for (int j = 0; j < ACC; ++j) {
            acc[j] = acc[j] * 1.0009765625f + acc[(j + 1) % ACC];
        }
    }
    float s = 0.f;
    for (int j = 0; j < ACC; ++j) s += acc[j];
    out[i] = s;
}
