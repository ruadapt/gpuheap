NVCC=/usr/local/cuda/bin/nvcc

merge_path_test: merge_path_test.cu heaputil.cuh
	$(NVCC) merge_path_test.cu -std=c++11 -arch=sm_61 -g -G -o merge_path_test

clean:
	rm merge_path_test
