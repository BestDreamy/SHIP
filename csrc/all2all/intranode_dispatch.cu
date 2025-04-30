#include "intranode.h"
#include <cstdint>

using namespace ship;

void AllToAllIntraNode::dispatch (
	const Stride1D<uint32_t> &tokens_d,
	const Strided2D<uint32_t> &indices_d
) {
	constexpr unsigned NUM_WRAPS = 10;
	constexpr unsigned numThreadsperBlock = 32 * NUM_WRAPS;
	constexpr unsigned numBlocks = 132;

	dim3 dimGrid(numBlocks);
	dim3 dimBlock(numThreadsperBlock);

	void *args[] = {
		&numTokensBuffer,
		tokens_d.data,
      	tokens_d.strideElem,
		indices_d.data,
      	indices_d.strideElem,
		indices_d.strideRow
	};

	dispatchKernel<>
}