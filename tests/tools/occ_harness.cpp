// Differential-test harness: drives NVIDIA's reference occupancy
// implementation (cuda_occupancy.h — pure host code, no GPU) so cuxray's
// Python port can be validated against it exactly.
//
// Build:  g++ -O2 -I <cudart-include-dir> -o occ_harness occ_harness.cpp
// Input (stdin), one config per line:
//   major minor maxThrSM smemSM smemBlkOptin reserved threads regs smemStatic smemDyn
// Output, one line per input:
//   err activeBlocks limRegs limSmem limWarps limBlocks allocRegs allocSmem

#include <cstdio>
#include <cstring>

#include "cuda_occupancy.h"

int main() {
    int major, minor, maxThrSM, threads, regs;
    long smemSM, smemBlkOptin, reserved, smemStatic, smemDyn;
    while (std::scanf("%d %d %d %ld %ld %ld %d %d %ld %ld",
                      &major, &minor, &maxThrSM, &smemSM, &smemBlkOptin,
                      &reserved, &threads, &regs, &smemStatic, &smemDyn) == 10) {
        cudaOccDeviceProp p;
        std::memset(&p, 0, sizeof(p));
        p.computeMajor = major;
        p.computeMinor = minor;
        p.maxThreadsPerBlock = 1024;
        p.maxThreadsPerMultiprocessor = maxThrSM;
        p.regsPerBlock = 65536;
        p.regsPerMultiprocessor = 65536;
        p.warpSize = 32;
        p.sharedMemPerBlock = 48 * 1024;
        p.sharedMemPerMultiprocessor = (size_t)smemSM;
        p.numSms = 1;
        p.sharedMemPerBlockOptin = (size_t)smemBlkOptin;
        p.reservedSharedMemPerBlock = (size_t)reserved;

        cudaOccFuncAttributes a;
        a.maxThreadsPerBlock = 1024;
        a.numRegs = regs;
        a.sharedSizeBytes = (size_t)smemStatic;
        a.partitionedGCConfig = PARTITIONED_GC_OFF;
        a.shmemLimitConfig = FUNC_SHMEM_LIMIT_OPTIN;
        a.maxDynamicSharedSizeBytes = (size_t)smemBlkOptin;
        a.numBlockBarriers = 1;

        cudaOccDeviceState s;  // defaults: no cache/carveout preference

        cudaOccResult r;
        std::memset(&r, 0, sizeof(r));
        cudaOccError e = cudaOccMaxActiveBlocksPerMultiprocessor(
            &r, &p, &a, &s, threads, (size_t)smemDyn);
        std::printf("%d %d %d %d %d %d %d %ld\n",
                    (int)e, r.activeBlocksPerMultiprocessor, r.blockLimitRegs,
                    r.blockLimitSharedMem, r.blockLimitWarps, r.blockLimitBlocks,
                    r.allocatedRegistersPerBlock,
                    (long)r.allocatedSharedMemPerBlock);
    }
    return 0;
}
