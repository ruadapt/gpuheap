nvcc = nvcc
nvccflag = -g -G -std=c++11 -O3 -arch=sm_61
debug = -DDEBUG

all: insdel

insdel: insdel.cu heap.cuh heap_api.cuh sort.cuh util.hpp
	$(nvcc) $(nvccflag) $< -o $@

clean:
	rm -rf insdel
