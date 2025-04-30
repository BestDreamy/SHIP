#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda.h>
#include <iostream>

#define WORLD_SIZE 8

#include <cuda.h>
#include <vector>
#include <cstdint>
#include "../include/buffer.h"
#include "../all2all/intranode.h"

using namespace ship;

void testDispatch(
    // cudaStream_t stream,
    unsigned rank,
    unsigned world_size,
    uint32_t localTokens = 4,
    uint32_t hiddenDim = 16,
    uint32_t numExperts = 8,
    uint32_t expertsPerToken = 2
) {
    std::vector<uint32_t> tokens_h(localTokens * hiddenDim, rank); // All elements initialized to rank
    std::vector<uint32_t> indices_h(localTokens * expertsPerToken, 0);
    uint32_t numLocalExperts = ceil_div(numExperts, world_size);
    for (int i = 0; i < localTokens; i ++) {
        // For each token, assign it to other rank
        for (int j = 0; j < expertsPerToken; j ++) {
            indices_h[i * expertsPerToken + j] = (++ numLocalExperts) % numExperts;
        }
    }

    // Device buffers
    DeviceBuffer<uint32_t> tokens_d(tokens_h);
    DeviceBuffer<uint32_t> indices_d(indices_h);

    const uint32_t hiddenDimBytes = hiddenDim * sizeof(T);

    AllToAllIntranode allToAllIntranode(
        rank,
        world_size,
        localTokens,
        hiddenDim,
        numExperts,
        expertsPerToken,
        // localTokens * world_size, // maxNumTokens
        ceil_div(numExperts, world_size) // numLocalExperts
    );

    allToAllIntranode.dispatch(
        Stride1D<uint32_t>(tokens_d, hiddenDim),
        Stride2D<uint32_t>(indices_d, 1, expertsPerToken)
    );

}

int main(int argc, char **argv) {
    nvshmem_init();

    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    int device = my_pe % 8;
    cudaSetDevice(device);

    testDispatch(my_pe, n_pes);

    nvshmem_finalize();

    return 0;
}