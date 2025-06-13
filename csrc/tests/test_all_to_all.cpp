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

//#ifdef NVSHMEM_MPI_SUPPORT
#include "mpi.h"
// #endif

using namespace ship;

void testDispatch(
    // cudaStream_t stream,
    unsigned rank,
    unsigned world_size,
    uint32_t numLocalTokens = 4,
    uint32_t hiddenDim = 16,
    uint32_t numExperts = 8,
    uint32_t topk = 2,
    uint32_t maxNumTokens = 10
) {
    Assert(numExperts / world_size == topk, "Just for test, rank[i] and rank[i+1] buterfly transfer the same token");
    std::vector<uint32_t> tokens_h(numLocalTokens * hiddenDim, rank + 10); // All elements initialized to rank
    std::vector<uint32_t> indices_h(numLocalTokens * topk, 0);
    uint32_t numLocalExperts = numExperts / world_size;
    assert(numExperts % world_size == 0);

    std::ofstream logFile("rank_" + std::to_string(rank) + ".log");
    logFile << "Total ranks: " << world_size << "\n";
    logFile << "Each rank will transfer tokens num: " << numLocalTokens << "\n";
    logFile << "Each rank have experts num: " << numLocalExperts << "\n";
    logFile << "Each token have experts num: " << topk << "\n";

    for (int i = 0; i < numLocalTokens; i ++) {
        // For each token, assign it to other rank
        for (int j = 0; j < topk; j ++) {
            // indices_h[i * topk + j] = j;
            indices_h[i * topk + j] = (rank ^ 0x1) * numLocalExperts + j;
        }
    }
    for (int i = 0; i < numLocalTokens; i ++) {
        for (int j = 1; j < topk; j ++) {
            Assert(indices_h[i * topk] != indices_h[i * topk + j], "The same token should not be assigned to the same expert");
        }
    }
        
    print_transmit_information(tokens_h, indices_h, numLocalTokens, hiddenDim, topk, rank, logFile);


    // Device buffers
    // DeviceBuffer<uint32_t> tokens_d(tokens_h);
    uint32_t *tokens_d = (uint32_t*)nvshmem_malloc(sizeof(uint32_t) * numLocalTokens * hiddenDim);
    cudaMemcpy(tokens_d, tokens_h.data(), sizeof(uint32_t) * numLocalTokens * hiddenDim, cudaMemcpyHostToDevice);
    DeviceBuffer<uint32_t> indices_d(indices_h);

    const uint32_t hiddenDimBytes = hiddenDim * sizeof(tokens_h[0]);

    AllToAllIntraNode allToAllIntranode(
        rank,
        world_size,
        numLocalTokens,
        hiddenDim,
        hiddenDimBytes,
        numExperts,
        topk,
        maxNumTokens
    );

    allToAllIntranode.dispatch(
        Stride1D<uint32_t>(tokens_d, hiddenDim),
        Stride2D<uint32_t>(indices_d, 1, topk),
        logFile
    );

}

#define MPICHECK(cmd)                                                                              \
  do {                                                                                             \
    int e = cmd;                                                                                   \
    if (e != MPI_SUCCESS) {                                                                        \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);                             \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

int main(int argc, char **argv) {
    int rank, world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();
    assert(my_pe == rank);
    assert(n_pes == world_size);

    int deviceId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(deviceId);

    testDispatch(my_pe, n_pes);

    nvshmem_barrier_all();
    printf("Rank %d: Dispatch test completed.\n", my_pe);
    nvshmem_finalize();
    MPICHECK(MPI_Finalize());

    /*
    // nvshmrun -np 4 ./test_all_to_all
    nvshmem_init();
    
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();
    
    int deviceId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(deviceId);
    
    testDispatch(my_pe, n_pes);
    
    nvshmem_barrier_all();
    nvshmem_finalize();
    */
    return 0;
}