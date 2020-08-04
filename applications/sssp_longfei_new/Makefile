NVCC=/usr/local/cuda/bin/nvcc
CFLAGS=-O3

all: sssp

sssp: sssp.o main.o
	$(NVCC) $^ -o $@

sssp.o: sssp.cu sssp_config.cuh heap_with_aux.cuh heap.cuh heaputil.cuh util.cuh
	$(NVCC) $< $(CFLAGS) -c -o $@

main.o: main.cu sssp_config.cuh heap_with_aux.cuh heap.cuh heaputil.cuh util.cuh
	$(NVCC) $< $(CFLAGS) -c -o $@
