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
#include <fstream>

using namespace ship;

void testDispatch(
    // cudaStream_t stream,
    unsigned rank,
    unsigned world_size,
    uint32_t localTokens = 4,
    uint32_t hiddenDim = 16,
    uint32_t numExperts = 8,
    uint32_t expertsPerToken = 2,
    uint32_t maxNumTokens = 10
) {
    Assert(numExperts / world_size == expertsPerToken, "Just for test, rank[i] and rank[i+1] buterfly transfer the same token");
    std::vector<uint32_t> tokens_h(localTokens * hiddenDim, rank); // All elements initialized to rank
    std::vector<uint32_t> indices_h(localTokens * expertsPerToken, 0);
    uint32_t numLocalExperts = numExperts / world_size;
    assert(numExperts % world_size == 0);

    std::ofstream logFile("rank_" + std::to_string(rank) + ".log");
    logFile << "Total ranks: " << world_size << "\n";
    logFile << "Each rank will transfer tokens num: " << localTokens << "\n";
    logFile << "Each rank have experts num: " << numLocalExperts << "\n";
    logFile << "Each token have experts num: " << expertsPerToken << "\n";

    for (int i = 0; i < localTokens; i ++) {
        // For each token, assign it to other rank
        for (int j = 0; j < expertsPerToken; j ++) {
            // indices_h[i * expertsPerToken + j] = j;
            indices_h[i * expertsPerToken + j] = (rank ^ 0x1) * numLocalExperts + j;
        }
    }
    for (int i = 0; i < localTokens; i ++) {
        for (int j = 1; j < expertsPerToken; j ++) {
            Assert(indices_h[i * expertsPerToken] != indices_h[i * expertsPerToken + j], "The same token should not be assigned to the same expert");
        }
    }
        
    print_transmit_information(tokens_h, indices_h, localTokens, hiddenDim, expertsPerToken, rank, logFile);


    // Device buffers
    DeviceBuffer<uint32_t> tokens_d(tokens_h);
    DeviceBuffer<uint32_t> indices_d(indices_h);

    const uint32_t hiddenDimBytes = hiddenDim * sizeof(tokens_d[0]);

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

    logFile << "\n\n\n--------------Dispatch start----------------\n\n\n";

    allToAllIntranode.dispatch(
        Stride1D<uint32_t>(tokens_d, hiddenDim),
        Stride2D<uint32_t>(indices_d, 1, expertsPerToken),
        logFile
    );

}

int main(int argc, char **argv) {
    nvshmem_init();

    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    int deviceId = my_pe % 8;
    cudaSetDevice(deviceId);

    testDispatch(my_pe, n_pes);

    nvshmem_finalize();

    return 0;
}