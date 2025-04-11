#include <torch/extension.h>
#include "include/api.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("torchReduce", &torchReduce);
}