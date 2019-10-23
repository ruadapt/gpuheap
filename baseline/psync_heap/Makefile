nvcc = nvcc
nvccflag = -g -G -std=c++11 -O0 -arch=sm_61
debug = -DDEBUG

all: app

app: app.cu heap.cuh sort.cuh util.hpp
	$(nvcc) $(nvccflag) $< -o $@

debug: app.cu heap.cuh sort.cuh util.hpp
	$(nvcc) $(nvccflag) $(debug) $< -o app

test: test.cu heap.cuh sort.cuh util.hpp
	$(nvcc) $(nvccflag) $< -o $@

clean:
	rm -rf app test
