#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

namespace pplx {
    namespace device {
    __forceinline__ __device__ unsigned warp_sum(unsigned value) {
      value += __shfl_xor_sync(0xffffffff, value, 16);
      value += __shfl_xor_sync(0xffffffff, value, 8);
      value += __shfl_xor_sync(0xffffffff, value, 4);
      value += __shfl_xor_sync(0xffffffff, value, 2);
      value += __shfl_xor_sync(0xffffffff, value, 1);
      return value;
    }
    
    template <typename T> __device__ T ceil_div(T x, T y) { return (x + y - 1) / y; }
    
    template <typename T> __device__ T round_up(T x, T y) { return ceil_div<T>(x, y) * y; }
    
    } // namespace device
} // namespace pplx

#endif