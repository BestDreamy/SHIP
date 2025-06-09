#include "intranode.h"
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include "../include/cuda_utils.h"
#include <cooperative_groups.h>
#include <sys/types.h>

using namespace ship;

namespace cg = cooperative_groups;

__device__ float clockRate_d;

template <bool isSend, bool isRecv>
__global__ void dispatchKernel (
	uint32_t rank,
	uint32_t localTokens,
	uint32_t hiddenDim,
	uint32_t hiddenDimBytes,
	uint32_t numLocalExperts,
	uint32_t expertsPerToken,
	uint32_t worldSize,
	uint32_t numExperts,
	uint32_t maxNumTokens,

	uint64_t *numTokensBuffer,
	uint64_t *numRecvBuffer,
	std::byte *xDispatchOut,

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

	const unsigned tokenDim = hiddenDimBytes;

	if constexpr (isSend) {
		// tokenIndex[i] identifies the number of tokens have assigned to the local expert i just in this rank.
		extern __shared__ uint32_t tokenIndex[];
		for (uint32_t i = threadIdx.x; i < numExperts; i += blockDim.x) {
			tokenIndex[i] = 0;
		}
		__syncthreads();


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
						// numTokenBuffer[i][j]: rank(i) transfer dstExpertCountSum tokens to dstRank.
						// Because numTokenBuffer's size is worldSize * numLocalExperts,
						// only the whole row of numTokenBuffer[rank][numLocalExperts] can be chosen when this PE=rank.
						numTokensBuffer + dstLocalExpert + rank * numLocalExperts,
						// numTokensBuffer + dstLocalExpert * worldSize + rank,
						dstExpertCountSum + 1,
						NVSHMEM_SIGNAL_SET,
						dstRank
					);
				}
			}
		// warp0-8 are responsible for sending tokens to the local experts.
		} else {
			const unsigned numGroupWarps = NUM_WARPS - 1;
			const unsigned numGroupThreads = numGroupWarps * WARP_SIZE;
			for (unsigned i = 0; i < localTokens; i++) {
				// For each token, in order to be saw by all blocks,
				// though only one block transmit this token, each block should count it.
				if (warpId == 0 && laneId < expertsPerToken) {
					uint32_t dstExpert = __ldg(&indices[i * expertsPerToken + laneId]);
					tokenIndex[dstExpert]++;
				}

				// Synchronize the warps within this warp group.
          		asm volatile("bar.sync 1, %0;" ::"r"(numGroupThreads));

				// If the token is assigned to this block, handle it. (Each block handles one token)
 				if (i % gridDim.x == blockIdx.x) {
					 for (unsigned j = warpId; j < expertsPerToken; j += numGroupWarps) {
						// Each warp in block transmit the token to one expert.
						const uint32_t dstExpert = __ldg(indices + i * expertsPerToken + j);
						const uint32_t dstRank = dstExpert / numLocalExperts;
						const uint32_t dstLocalExpert = dstExpert % numLocalExperts;
	
						const uint32_t index = tokenIndex[dstExpert] - 1;

						nvshmemx_putmem_signal_nbi_warp(
							// xDispatchOut[i][j][k]: rank(i) transfer token to expert(j) which in dstRank.
							// Because each expert which in dstRank could receive maxNumTokens tokens,
							// only the whole row of xDispatchOut[rank][dstLocalExpert][tokenIndex] can be chosen when this PE=rank.
							xDispatchOut + ((dstLocalExpert + rank * numLocalExperts) * maxNumTokens + index) * tokenDim,
							(std::byte *)tokens + i * tokenDim,
							tokenDim, // num bytes
							// numRecvBuffer same as numTokenBuffer.
							numRecvBuffer + dstLocalExpert + rank * numLocalExperts,
							1,
							NVSHMEM_SIGNAL_ADD,
							dstRank
						);
					}
				}

			} 
		}

		if (isRecv) cg::this_grid().sync();
	}

	if constexpr (isRecv) {
		for (int i = blockId * blockDim.x + threadIdx.x; i < numExperts; i += gridDim.x * blockDim.x) {
			const uint32_t srcRank = i / numLocalExperts;
			const uint32_t srcLocalExpert = i % numLocalExperts;
			// const uint32_t dstLocalExpert = dstExpert % numLocalExperts;

			// Wait for the token count to be set.
			nvshmem_uint64_wait_until(
				numTokensBuffer + i,
				NVSHMEM_CMP_NE,
			    0
			);
			uint64_t numTokens = numTokensBuffer[i] - 1;
			nvshmem_uint64_wait_until(
				numRecvBuffer + i, 
				NVSHMEM_CMP_EQ, 
				numTokens
			);

			// Clean the buffers.
			// numTokensBuffer[i] = 0;
			// numRecvBuffer[i] = 0;
		}
		cg::this_grid().sync();
	}
}

void AllToAllIntraNode::dispatch (
	const Stride1D<uint32_t> &tokens_d,
	const Stride2D<uint32_t> &indices_d,
	std::ofstream &logFile
) {
	constexpr unsigned NUM_WRAPS = 10;
	constexpr unsigned numThreadsperBlock = 32 * NUM_WRAPS;
	const unsigned numBlocks = std::min((uint32_t)132, numExperts);

	dim3 dimGrid(numBlocks);
	dim3 dimBlock(numThreadsperBlock);

	void *args[] = {
		const_cast<uint32_t*>(&rank),
		const_cast<uint32_t*>(&localTokens),
		const_cast<uint32_t*>(&hiddenDim),
		const_cast<uint32_t*>(&hiddenDimBytes),
		const_cast<uint32_t*>(&numLocalExperts),
		const_cast<uint32_t*>(&expertsPerToken),
		const_cast<uint32_t*>(&world_size),
		const_cast<uint32_t*>(&numExperts),
		const_cast<uint32_t*>(&maxNumTokens),
		&numTokensBuffer,
		&numDispatchRecvBuffer,
		&xDispatchOut,
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
        args,
		sizeof(uint32_t) * numExperts
    );
	cudaDeviceSynchronize();

	uint64_t *numTokensBuffer_h = new uint64_t[numLocalExperts * world_size];
	cudaMemcpy(
		numTokensBuffer_h,
		numTokensBuffer,
		numLocalExperts * world_size * sizeof(uint64_t),
		cudaMemcpyDeviceToHost
	);
	for (int i = 0; i < world_size; i++) {
		for (int j = 0; j < numLocalExperts; j++) {
			if (numTokensBuffer_h[i * numLocalExperts + j] == 1) continue;

			int idxExpert = rank * numLocalExperts + j;
			logFile << "Expert " << idxExpert << ": reveive " << numTokensBuffer_h[i * numLocalExperts + j] - 1 << " tokens.\n";
		}
	}
	delete[] numTokensBuffer_h;

	std::byte *xDispatchOut_h = new std::byte[world_size * numLocalExperts * maxNumTokens * hiddenDimBytes];
	cudaMemcpy(
		xDispatchOut_h,
		xDispatchOut,
		world_size * numLocalExperts * maxNumTokens * hiddenDimBytes * sizeof(std::byte),
		cudaMemcpyDeviceToHost
	);
	for (int i = 0; i < world_size; i++) {
		for (int j = 0; j < numLocalExperts; j++) {
			for (int k = 0; k < maxNumTokens; k++) {
				int idxExpert = rank * numLocalExperts + j;
				logFile << "Expert " << idxExpert << " reveive token from rank " << i << ": ";
				for (int w = 0; w < hiddenDimBytes; w += 4) {
					uint32_t val = *((uint32_t*)(xDispatchOut_h + i * numLocalExperts * maxNumTokens * hiddenDimBytes + j * maxNumTokens * hiddenDimBytes + k * hiddenDimBytes + w));
					logFile << val << " ";
				}
				logFile << "\n";
			}
		}
	}
	delete[] xDispatchOut_h;
}