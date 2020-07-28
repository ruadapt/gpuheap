NVCC=/usr/local/cuda/bin/nvcc

bug_demo: bug_demo.cu heap.cuh
	$(NVCC) bug_demo.cu -std=c++11 -arch=sm_61 -g -G -o bug_demo
