# all:
# nvcc -x cu -rdc=true -ccbin g++ -gencode=${NVCC_GENCODE} \
# -I ${NVSHMEM_HOME}/include \
# test_all_to_all.cc -o test_all_to_all \
# -L ${NVSHMEM_HOME}/lib \
# -lnvshmem -lnvidia-ml -lcuda -lcudart

# run:
# 	nvshmrun -np 4 ./test_all_to_all


CC := g++
NVCC := nvcc
NVCC_GENCODE ?= arch=compute_90,code=sm_90a

SRC_DIR := ..
INC_DIR := $(addprefix -I, $(abspath ../all2all) \
                           $(abspath ../include))
BUILD_DIR := ../build

CU_SRCS := $(shell find $(abspath $(SRC_DIR)) -name "*.cu")
TEST_CPP_SRCS := $(shell find $(abspath $(SRC_DIR)) -name "test_all_to_all.cpp")
BENCH_CPP_SRCS := $(shell find $(abspath $(SRC_DIR)) -name "bench_all_to_all.cpp")

# CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.cu.o,$(CU_SRCS))
# CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.cpp.o,$(CPP_SRCS))

CFLAGS := -I$(INC_DIR) -I${NVSHMEM_HOME}/include -I${CUDA_HOME}/include -std=c++17
NVCC_FLAGS := -x cu -rdc=true -ccbin $(CC) -gencode=$(NVCC_GENCODE)
LDFLAGS := -L${NVSHMEM_HOME}/lib -L${CUDA_HOME}/lib -lnvshmem -lnvidia-ml -lcuda -lcudart

# support for building with mpi
CFLAGS += -I${MPI_HOME}/include
LDFLAGS += -L${MPI_HOME}/lib -lmpi

test: test_all_to_all run_test
bench: bench_all_to_all run_bench

prepare:
	mkdir -p $(BUILD_DIR)/all2all

# $(BUILD_DIR)/%.cu.o: $(SRC_DIR)/%.cu
# 	$(NVCC) $(NVCC_FLAGS) $(CFLAGS) -c $< -o $@

# $(BUILD_DIR)/%.cpp.o: $(SRC_DIR)/%.cpp
# 	$(CC) $(CFLAGS) -c $< -o $@

test_all_to_all: $(TEST_CPP_SRCS) $(CU_SRCS)
	$(NVCC) $(NVCC_FLAGS) $(CFLAGS) -o $@ $^ $(LDFLAGS)

bench_all_to_all: $(BENCH_CPP_SRCS) $(CU_SRCS)
	$(NVCC) $(NVCC_FLAGS) $(CFLAGS) -o $@ $^ $(LDFLAGS)

run_test:
	nvshmrun -np 4 ./test_all_to_all
# NVSHMEM_REMOTE_TRANSPORT=none mpirun -np 4 --allow-run-as-root ./test_all_to_all

run_bench:
	nvshmrun -np 4 ./bench_all_to_all

nsys:
	nsys profile -t cuda,nvtx,oshmem -o intranode_dispatch --gpu-metrics-devices=all nvshmrun -np 4 ./test_all_to_all

clean:
	rm -rf $(BUILD_DIR) test_all_to_all *.log bench_all_to_all

.PHONY: prepare clean test run_test nsys bench run_bench