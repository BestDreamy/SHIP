#include <cstdint>
namespace ship {

template <typename T> class HostBuffer;
template <typename T> class DeviceBuffer;

// Host Buffer class
template <typename T> class HostBuffer final {
public:
  HostBuffer(uint32_t size) : size_(size) {
    cudaMallocHost(&data_, size * sizeof(T));
  }

  HostBuffer(const DeviceBuffer<T> &device_buffer) {
    size_ = device_buffer.size();
    cudaMallocHost(&data_, size_ * sizeof(T));
    cudaMemcpy(data_, device_buffer.get(), size_ * sizeof(T), cudaMemcpyDeviceToHost);
  }

  HostBuffer(const HostBuffer &) = delete;

  ~HostBuffer() {
    cudaFreeHost(data_);
    data_ = nullptr;
  }

  HostBuffer &operator=(const HostBuffer &) = delete;

  uint32_t size() const { return size_; }
  T &operator[](uint32_t i) { return data_[i]; }
  const T &operator[](uint32_t i) const { return data_[i]; }

  const T *get() const { return data_; }

  void copyFromDevice(const DeviceBuffer<T> &device_buffer) {
    cudaMemcpy(data_, device_buffer.get(), size_ * sizeof(T), cudaMemcpyDeviceToHost);
  }

private:
  uint32_t size_;
  T *data_ = nullptr;
};

// Device Buffer class
template <typename T> class DeviceBuffer final {
public:
  DeviceBuffer(uint32_t size): size_(size) {
    cudaMalloc(&data_, size * sizeof(T));
  }

  DeviceBuffer(const HostBuffer<T> &host_buffer) {
    size_ = host_buffer.size();
    cudaMalloc(&data_, size_ * sizeof(T));
    cudaMemcpy(data_, host_buffer.get(), size_ * sizeof(T), cudaMemcpyHostToDevice);
  }

  DeviceBuffer(const DeviceBuffer &) = delete;

  ~DeviceBuffer() { cudaFree(data_); }

  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  uint32_t size() const { return size_; }
  T *operator&() { return data_; }

  T *get() { return data_; }

  void copyFromHost(const HostBuffer<T> &host_buffer) {
    cudaMemcpy(data_, host_buffer.get(), size_ * sizeof(T), cudaMemcpyHostToDevice);
  }

  void copyFromHost(const T *host_data, uint32_t num_elements) {
    cudaMemcpy(data_, host_data, num_elements * sizeof(T), cudaMemcpyHostToDevice);
  }

private:
  uint32_t size_;
  T *data_;
};

} // namespace ship