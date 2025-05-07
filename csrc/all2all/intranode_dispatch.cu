#include "intranode.h"
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include "../include/cuda_utils.h"

using namespace ship;

template <bool isSend, bool isRecv>
__global__ void dispatchKernel (
	const uint32_t &rank,
	const uint32_t &localTokens,
	const uint32_t &numLocalExperts,
	const uint32_t &expertsPerToken,
	const uint32_t &worldSize,
	const uint32_t &numExperts,
	uint64_t *numTokensBuffer,

	uint32_t *tokens,
	uint32_t tokenElemStride,
	uint32_t *indices,
	uint32_t indexElemStride,
	uint32_t indexRowStride
) {
	const unsigned WARP_SIZE = 32;
	const unsigned NUM_WARPS = blockDim.x / WARP_SIZE;
	const unsigned blockId = blockIdx.x;
	const unsigned warpId = threadIdx.x / WARP_SIZE;
	const unsigned laneId = threadIdx.x % WARP_SIZE;

	if constexpr (isSend) {
		if (warpId == NUM_WARPS - 1) {
			for (int dstExpert = blockId; dstExpert < numExperts; dstExpert += gridDim.x) {
				const uint32_t dstRank = dstExpert / numLocalExperts;
				const uint32_t dstLocalExpert = dstExpert % numLocalExperts;

				unsigned dstExpertCount = 0;
				for (int i = laneId; i < localTokens * expertsPerToken; i += WARP_SIZE) {
					unsigned expert = __ldg(indices + i);
					if (expert == dstLocalExpert) {
						dstExpertCount++;
					}
				}

				if (laneId == 0) {
					unsigned dstExpertCountSum = device::warp_sum(dstExpertCount);
					nvshmemx_signal_op(
						numTokensBuffer + dstRank,
						dstExpertCountSum,
						NVSHMEM_SIGNAL_SET,
						dstRank
					);
				}
			}
		}
	}
}

void AllToAllIntraNode::dispatch (
	const Stride1D<uint32_t> &tokens_d,
	const Stride2D<uint32_t> &indices_d
) {
	constexpr unsigned NUM_WRAPS = 10;
	constexpr unsigned numThreadsperBlock = 32 * NUM_WRAPS;
	const unsigned numBlocks = std::min((uint32_t)132, numExperts);

	dim3 dimGrid(numBlocks);
	dim3 dimBlock(numThreadsperBlock);

	void *args[] = {
		const_cast<uint32_t*>(&rank),
		const_cast<uint32_t*>(&localTokens),
		const_cast<uint32_t*>(&numLocalExperts),
		const_cast<uint32_t*>(&expertsPerToken),
		const_cast<uint32_t*>(&world_size),
		const_cast<uint32_t*>(&numExperts),
		&numTokensBuffer,
		const_cast<uint32_t**>(&tokens_d.data),
		const_cast<size_t*>(&tokens_d.strideElem),
		const_cast<uint32_t**>(&indices_d.data),
		const_cast<size_t*>(&indices_d.strideElem),
		const_cast<size_t*>(&indices_d.strideRow)
	};

	cudaLaunchCooperativeKernel(
        (void *)&dispatchKernel<true, true>,
        dimGrid,
        dimBlock,
        args
    );

	// printf("[Kernel Args]\n"
	// 	"rank=%u\n"
	// 	"localTokens=%u\n" 
	// 	"numLocalExperts=%u\n"
	// 	"expertsPerToken=%u\n"
	// 	"worldSize=%u\n"
	// 	"numExperts=%u\n",
	// 	rank, localTokens, numLocalExperts, expertsPerToken, world_size, numExperts);

}