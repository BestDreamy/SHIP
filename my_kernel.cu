#include <torch/extension.h>

__global__ void custom_kernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 1.0f; // 每个元素加 1
    }
}

void launch_kernel(torch::Tensor data) {
    float* ptr = data.data_ptr<float>();
    int size = data.numel();
    
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    custom_kernel<<<blocks, threads>>>(ptr, size);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_kernel", &launch_kernel, "Custom CUDA Kernel");
}
