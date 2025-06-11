#ifndef SHIP_INTRANODE_H 
#define SHIP_INTRANODE_H
#include <cstdint>
#include "../include/api.h"
#include "../include/buffer.h"
#include <nvshmem.h>
#include <cuda_runtime.h>
#include <../include/api.h>
#include <fstream>

namespace {
    template <typename T> T *mallocZeroBuffer(size_t size) {
      T *ptr;
      cudaMalloc(&ptr, size * sizeof(T));
      cudaMemset(ptr, 0, size * sizeof(T));
      return ptr;
    }
} // namespace

namespace ship {
    enum {
        SEND,
        RECV,
    };

    // For each rank
    struct AllToAllIntraNode {
        uint32_t rank;
        uint32_t world_size;
        uint32_t localTokens;
        uint32_t hiddenDim;
        uint32_t hiddenDimBytes; // The number of bytes for each token
        uint32_t numExperts;      // For the whole world
        uint32_t expertsPerToken; // The number of experts per token.
        uint32_t maxNumTokens;    // Each rank be allowed to receive maxNumTokens tokens
        uint32_t numLocalExperts;

        AllToAllIntraNode(
            uint32_t rank = 0,
            uint32_t world_size = 4,
            uint32_t localTokens = 4,
            uint32_t hiddenDim = 512,
            uint32_t hiddenDimBytes = 4 * 512, // Each token's bytes
            uint32_t numExperts = 8,
            uint32_t expertsPerToken = 3,
            uint32_t maxNumTokens = 10
        ): rank(rank),
        world_size(world_size),
        localTokens(localTokens),
        hiddenDim(hiddenDim),
        hiddenDimBytes(hiddenDimBytes),
        numExperts(numExperts),
        expertsPerToken(expertsPerToken),
        maxNumTokens(maxNumTokens)
        {
            Assert(numExperts % world_size == 0, "numExperts should be divisible by world_size");
            numLocalExperts = numExperts / world_size;

            numTokensBuffer = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * world_size * numLocalExperts);
            Assert(numTokensBuffer != nullptr, "Failed to allocate numTokensBuffer");
            cudaMemset(numTokensBuffer, 0, sizeof(uint64_t) * numLocalExperts * world_size);

            numDispatchRecvBuffer = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * world_size * numLocalExperts);
            Assert(numDispatchRecvBuffer != nullptr, "Failed to allocate numDispatchRecvBuffer");
            cudaMemset(numDispatchRecvBuffer, 0, sizeof(uint64_t) * numLocalExperts * world_size);

            uint32_t perTokenBytes = hiddenDimBytes;
            xDispatchOut = (std::byte *)nvshmem_malloc(world_size * numLocalExperts * maxNumTokens * perTokenBytes);
            Assert(xDispatchOut != nullptr, "Failed to allocate xDispatchOut");
        }
        
        // Storage the number of tokens for each local expert
        // Each rank will receive its tokens to the local experts
        // 64bit type for nvshmemx_signal_op
        uint64_t *numTokensBuffer = nullptr;

        // Size is similar to numTokensBuffer
        // Each rank will receive its tokens from the local experts
        uint64_t *numDispatchRecvBuffer = nullptr;

        std::byte *xDispatchOut = nullptr;

        void dispatch(
            const Stride1D<uint32_t> &tokens_d,
	        const Stride2D<uint32_t> &indices_d,
            std::ofstream &logFile
        );
    };
}

#endif // SHIP_uint32_tRANODE_H