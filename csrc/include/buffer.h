#ifndef SHIP_BUFFER_H
#define SHIP_BUFFER_H

#include <cerrno>
#include <cstdint>
#include <sys/types.h>
#include <vector>
#include <cuda_runtime.h>

namespace ship {

// Host Buffer class
template <typename T> struct HostBuffer {
    uint32_t size;
    T *data = nullptr;

    HostBuffer(const std::vector<T> &a) {
        size = a.size();
        cudaMallocHost(&data, size * sizeof(T));
        cudaMemcpy(data, a.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    }

    const T *get() const { return data; }
    T *get() { return data; }
};

// Device Buffer class
template <typename T> struct DeviceBuffer {
    uint32_t size;
    T *data = nullptr;

    DeviceBuffer(const std::vector<T> &a) {
        size = a.size();
        cudaMalloc(&data, size * sizeof(T));
        cudaMemcpy(data, a.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    }

    const T *get() const { return data; }
    T *get() { return data; }
};

template <typename T> struct Stride1D {
    T *data;
    size_t strideElem;
  
    template <typename S>
    Stride1D(DeviceBuffer<S> &data, size_t strideElem)
        : data(reinterpret_cast<T *>(data.get())),
          strideElem(strideElem) {}
  
    Stride1D(T *data, size_t strideElem)
        : data(data),
          strideElem(strideElem) {}
  };
  
  template <typename T> struct Stride2D {
    T *data;
    size_t strideElem;
    size_t strideRow;
  
    template <typename S>
    Stride2D(DeviceBuffer<S> &data, size_t strideElem, size_t strideRow)
        : data(reinterpret_cast<T *>(data.get())),
          strideElem(strideElem),
          strideRow(strideRow) {}
  
    Stride2D(T *data, size_t strideElem, size_t strideRow)
        : data(data),
          strideElem(strideElem),
          strideRow(strideRow) {}
  };

} // namespace ship

#endif