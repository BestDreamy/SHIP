#include "intranode.h"
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include "../include/cuda_utils.h"

using namespace ship;

template <bool isSend, bool isRecv>
__global__ void dispatchKernel (
	uint32_t rank,
	uint32_t localTokens,
	uint32_t numLocalExperts,
	uint32_t expertsPerToken,
	uint32_t worldSize,
	uint32_t numExperts,
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
		// Dispatch tokens to the local experts.
		// warp9 counts the number of tokens assigned to each expert.
		if (warpId == NUM_WARPS - 1) {
			for (int dstExpert = blockId; dstExpert < numExperts; dstExpert += gridDim.x) {
				const uint32_t dstRank = dstExpert / numLocalExperts;
				const uint32_t dstLocalExpert = dstExpert % numLocalExperts;

				unsigned dstExpertCount = 0;
				for (int i = laneId; i < localTokens * expertsPerToken; i += WARP_SIZE) {
					unsigned expert = __ldg(indices + i);
					if (expert == dstExpert) {
						dstExpertCount++;
					}
				}

				unsigned dstExpertCountSum = device::warp_sum(dstExpertCount);
				if (laneId == 0) {
					// printf("Dispatch dstRank %d, dstExpert %d, dstExpertCount %d\n", dstRank, dstLocalExpert, dstExpertCountSum);
					nvshmemx_signal_op(
						// numTokensBuffer + dstLocalExpert + dstRank * numLocalExperts,
						numTokensBuffer + dstLocalExpert * worldSize + rank,
						dstExpertCountSum + 1,
						NVSHMEM_SIGNAL_SET,
						dstRank
					);
				}
			}
		// warp0-8 are responsible for sending tokens to the local experts.
		} /* else {
			const unsigned numGroupWarps = NUM_WARPS - 1;
			const unsigned numGroupThreads = numGroupWarps * WARP_SIZE;
			for (unsigned i = 0; i < localTokens; i++) {
				// If the token is assigned to this block, handle it.
				if (i % gridDim.x == blockIdx.x) {
	
					// Send the token to the other ranks, one send per warp.
					for (unsigned j = warpId; j < expertsPerToken; j += numGroupWarps) {
						const uint32_t dstExpert = __ldg(indices + i * expertsPerToken + j);
						const uint32_t dstRank = dstExpert / numLocalExperts;
						const uint32_t dstLocalExpert = dstExpert % numLocalExperts;
	
						const uint32_t index = tokenIndex[dstExpert];
						const uint32_t group = dstLocalExpert * numDPGroups + dpGroup;
						const unsigned loc = group * maxNumTokens + index;
	
						std::byte *destPointer = xBufferOut + loc * tokenStride;
						nvshmemx_putmem_signal_nbi_warp(
							destPointer,
							xInPtr,
							tokenStride,
							&numRecvBuffer[group],
							1,
							NVSHMEM_SIGNAL_ADD,
							dstRank
						);
					}
				}
		
				// Replicate the token count calculation across all blocks.
				if (warpId == 0 && laneId < numExpertsPerToken) {
					uint32_t dstExpert = __ldg(&indices[i * numExpertsPerToken + laneId]);
					tokenIndex[dstExpert]++;
				}
			}
		}
		*/
	}

	if constexpr (isRecv) {
		for (int i = blockId * blockDim.x + threadIdx.x; i < numExperts; i += gridDim.x * blockDim.x) {
			const uint32_t srcRank = i / numLocalExperts;
			const uint32_t srcLocalExpert = i % numLocalExperts;
			// const uint32_t dstLocalExpert = dstExpert % numLocalExperts;

			// Wait for the token count to be set.
			nvshmem_uint64_wait_until(
				numTokensBuffer + srcLocalExpert * worldSize + srcRank,
				NVSHMEM_CMP_NE,
			    0
			);
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
	cudaDeviceSynchronize();

	uint64_t *numTokensBuffer_h = new uint64_t[numLocalExperts * world_size];
	cudaMemcpy(
		numTokensBuffer_h,
		numTokensBuffer,
		numLocalExperts * world_size * sizeof(uint64_t),
		cudaMemcpyDeviceToHost
	);
	for (int i = 0; i < numLocalExperts * world_size; i++) {
		printf("numTokensBuffer_h[%d] = %lu\n", i, numTokensBuffer_h[i]);
	}
	delete[] numTokensBuffer_h;
}