#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda.h>
#include <iostream>

#define WORLD_SIZE 8

#include <cuda.h>
#include <vector>
#include <cstdint>
#include "../include/buffer.h"

using namespace ship;

template <typename T>
void testDispatch(
    // cudaStream_t stream,
    unsigned rank,
    unsigned world_size,
    size_t numTokens = 4,
    size_t hiddenDim = 16,
    size_t numExperts = 8,
    size_t expertsPerToken = 3
) {
  std::vector<T> inputData(numTokens * hiddenDim, rank); // All elements initialized to rank
  std::vector<uint32_t> indices(numTokens * expertsPerToken, 0); // All routed to expert 0

  // Device buffers
  DeviceBuffer<T> inputDevice(inputData);
  DeviceBuffer<uint32_t> indicesDevice(indices);
  DeviceBuffer<T> outputDevice(numExperts * numTokens * hiddenDim);

  const size_t hiddenDimBytes = hiddenDim * sizeof(T);

  // Initialize the kernel
  Kernel kernel(
      numTokens,
      numExperts,
      expertsPerToken,
      0, // epRank
      1, // epSize
      1, // dpSize
      hiddenDim,
      hiddenDimBytes,
      hiddenDimBytes
  );

  // Dispatch
  kernel.dispatch(
      Strided2D<std::byte>(outputDevice, hiddenDimBytes, hiddenDimBytes * numTokens),
      Strided1D<std::byte>(inputDevice, hiddenDimBytes),
      Strided2D<uint32_t>(indicesDevice, 1, expertsPerToken),
      numTokens,
      nullptr,
      SplitMode::NONE,
      stream
  );

  // Synchronize the stream
  CUDACHECK(cudaStreamSynchronize(stream));

  // Copy the output back to host
  HostBuffer<T> outputHost(outputDevice);

  // Print the output for verification
  std::cout << "Dispatch output:" << std::endl;
  for (size_t i = 0; i < numExperts; ++i) {
    std::cout << "Expert " << i << ": ";
    for (size_t j = 0; j < hiddenDim; ++j) {
      std::cout << outputHost[i * hiddenDim + j] << " ";
    }
    std::cout << std::endl;
  }
}

int main(int argc, char **argv) {
    nvshmem_init();

    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    int device = my_pe % 8;
    cudaSetDevice(device);

    testDispatch<int32_t>(unsigned int rank, unsigned int world_size)

    nvshmem_finalize();

    return 0;
}