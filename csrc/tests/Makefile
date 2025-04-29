all:
	nvcc -x cu -rdc=true -ccbin g++ -gencode=${NVCC_GENCODE} \
	-I ${NVSHMEM_HOME}/include \
	test_all_to_all.cc -o test_all_to_all \
	-L ${NVSHMEM_HOME}/lib \
	-lnvshmem -lnvidia-ml -lcuda -lcudart

run:
	nvshmrun -np 4 ./test_all_to_all