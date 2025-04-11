#include <torch/extension.h>
#include <api.h>

__global__ void reduce_kernel(float* c,
			const float* a, const float* b, int n) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
		c[i] = a[i] + b[i];
	}
}

/*
void reduce(float* c,
		const float* a, const float* b, int n) {
	dim3 grid((n + 1023) / 1024);
	dim3 block(1024);
	reduce<<<grid, block>>>(c, a, b, n);
}
*/

void torch_intranode_dispatch(torch::Tensor &c,
		const torch::Tensor &a, const torch::Tensor &b,int n) {
	dim3 blocksPerGrid((n + 1023) / 1024);
	dim3 threadsPerBlock(1024);
	reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>((float *)c.data_ptr(), 
	       (const float *)a.data_ptr(), (const float *)b.data_ptr(), n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("torchReduce", &torchReduce);
}
