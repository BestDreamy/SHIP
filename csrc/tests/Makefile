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
CPP_SRCS := $(shell find $(abspath $(SRC_DIR)) -name "*.cpp")

# CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.cu.o,$(CU_SRCS))
# CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.cpp.o,$(CPP_SRCS))

CFLAGS := -I$(INC_DIR) -I${NVSHMEM_HOME}/include -I${CUDA_HOME}/include
NVCC_FLAGS := -x cu -rdc=true -ccbin $(CC) -gencode=$(NVCC_GENCODE)
LDFLAGS := -L${NVSHMEM_HOME}/lib -L${CUDA_HOME}/lib -lnvshmem -lnvidia-ml -lcuda -lcudart


all: test_all_to_all

prepare:
	mkdir -p $(BUILD_DIR)/all2all

# $(BUILD_DIR)/%.cu.o: $(SRC_DIR)/%.cu
# 	$(NVCC) $(NVCC_FLAGS) $(CFLAGS) -c $< -o $@

# $(BUILD_DIR)/%.cpp.o: $(SRC_DIR)/%.cpp
# 	$(CC) $(CFLAGS) -c $< -o $@

test_all_to_all: $(CPP_SRCS) $(CU_SRCS)
	$(NVCC) $(NVCC_FLAGS) $(CFLAGS) -o $@ $^ $(LDFLAGS)

run:
	nvshmrun -np 4 ./test_all_to_all

nsys:
	nsys profile -t cuda,nvtx,oshmem -o intranode_dispatch --gpu-metrics-devices=all nvshmrun -np 4 ./test_all_to_all

clean:
	rm -rf $(BUILD_DIR) test_all_to_all

.PHONY: all prepare run clean