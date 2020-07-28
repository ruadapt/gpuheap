NVCC=/usr/local/cuda/bin/nvcc
CFLAGS=-O3

sssp_with_aux.o: sssp_with_aux.cu sssp_with_aux.cuh heap_with_aux.cuh heap.cuh heaputil.cuh
	$(NVCC) sssp_with_aux.cu $(CFLAGS) -c -o sssp_with_aux.o

main_with_aux.o: main_with_aux.cu sssp_with_aux.cuh heap_with_aux.cuh heap.cuh heaputil.cuh
	$(NVCC) main_with_aux.cu $(CFLAGS) -c -o main_with_aux.o

sssp: sssp_with_aux.o main_with_aux.o
	$(NVCC) sssp_with_aux.o main_with_aux.o -o sssp
