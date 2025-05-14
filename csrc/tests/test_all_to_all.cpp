#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda.h>

#define WORLD_SIZE 8

#include <cuda.h>
#include <cstdint>
#include "../include/buffer.h"
#include "../all2all/intranode.h"
#include <assert.h>
#include "../include/api.h"

using namespace ship;

void testDispatch(
    // cudaStream_t stream,
    unsigned rank,
    unsigned world_size,
    uint32_t localTokens = 4,
    uint32_t hiddenDim = 16,
    uint32_t numExperts = 8,
    uint32_t expertsPerToken = 3,
    uint32_t maxNumTokens = 10
) {
    std::vector<uint32_t> tokens_h(localTokens * hiddenDim, rank); // All elements initialized to rank
    std::vector<uint32_t> indices_h(localTokens * expertsPerToken, 0);
    uint32_t numLocalExperts = numExperts / world_size;
    assert(numExperts % world_size == 0);

    if (rank == 0) {
        std::cout << "Total ranks: " << world_size << "\n";
        std::cout << "Each rank will transfer tokens num: " << localTokens << "\n";
        std::cout << "Each rank have experts num: " << numLocalExperts << "\n";
        std::cout << "Each token have experts num: " << expertsPerToken << "\n";
    }
    for (int i = 0; i < localTokens; i ++) {
        // For each token, assign it to other rank
        for (int j = 0; j < expertsPerToken; j ++) {
            // indices_h[i * expertsPerToken + j] = (++ numLocalExperts) % numExperts;
            indices_h[i * expertsPerToken + j] = j;
        }
    }
    for (int i = 0; i < localTokens; i ++) {
        for (int j = 1; j < expertsPerToken; j ++) {
            Assert(indices_h[i * expertsPerToken] != indices_h[i * expertsPerToken + j], "The same token should not be assigned to the same expert");
        }
    }
    if (rank == 0) {
        printf("Rank 0: \n");
        print_transmit_information(tokens_h, indices_h, localTokens, hiddenDim, expertsPerToken);
    }


    // Device buffers
    DeviceBuffer<uint32_t> tokens_d(tokens_h);
    DeviceBuffer<uint32_t> indices_d(indices_h);

    const uint32_t hiddenDimBytes = hiddenDim * sizeof(uint32_t);

    AllToAllIntraNode allToAllIntranode(
        rank,
        world_size,
        localTokens,
        hiddenDim,
        hiddenDimBytes,
        numExperts,
        expertsPerToken,
        maxNumTokens
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