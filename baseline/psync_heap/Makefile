nvcc = nvcc
nvccflag = -g -G -std=c++11 -O3 -arch=sm_61
debug = -DDEBUG

all: insdel insdel_api

insdel: insdel.cu heap.cuh heap_api.cuh sort.cuh util.hpp
	$(nvcc) $(nvccflag) $< -o $@

insdel_api: insdel_api.cu heap.cuh heap_api.cuh sort.cuh util.hpp
	$(nvcc) $(nvccflag) $< -o $@

clean:
	rm -rf insdel insdel_api
