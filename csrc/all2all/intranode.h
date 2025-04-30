#ifndef SHIP_INTRANODE_H 
#define SHIP_INTRANODE_H
#include <cstdint>
#include "../include/api.h"
#include "../include/buffer.h"
#include <nvshmem.h>
#include <cuda_runtime.h>

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
        uint32_t numExperts; // For the whole world
        uint32_t expertsPerToken; // The number of experts per token.
        // uint32_t maxNumTokens; // For the local rank
        uint32_t numLocalExperts;

        AllToAllIntraNode(
            uint32_t rank = 0,
            uint32_t world_size = 4,
            uint32_t localTokens = 1,
            uint32_t hiddenDim = 512,
            uint32_t numExperts = 8,
            uint32_t expertsPerToken = 3
            // uint32_t maxNumTokens = localTokens,
        ): rank(rank),
        world_size(world_size),
        localTokens(localTokens),
        hiddenDim(hiddenDim),
        numExperts(numExperts),
        expertsPerToken(expertsPerToken)
        // maxNumTokens(maxNumTokens)
        {
            numLocalExperts = ceil_div(numExperts, world_size);
            numTokensBuffer = (uint32_t *)nvshmem_malloc(sizeof(uint32_t) * numLocalExperts * world_size);
            cudaMemset(numTokensBuffer, 0, sizeof(uint32_t) * numLocalExperts * world_size);
        }
        
        // Storage the number of tokens for each local expert
        // Each rank will transfer its tokens to the local experts
        uint32_t *numTokensBuffer = nullptr;

        void dispatch(
            Stride1D<uint32_t> &tokens_d,
	        Stride2D<uint32_t> &indices_d
        );
    };

}

#endif // SHIP_uint32_tRANODE_H