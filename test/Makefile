NVCC=nvcc
BHEAPNPPATH=../heap/buitdd_np/
THEAPNPPATH=../heap/tditdd_np/
BHEAPPATH=../heap/buitdd/
THEAPPATH=../heap/tditdd/
NVCCFLAG= -O3 -arch=sm_61 -std=c++11

all: Bheap_test Theap_test Bheap_np_test Theap_np_test

Bheap_test: partial_heap_test.cu
	${NVCC} ${NVCCFLAG} -I${BHEAPPATH} -DHEAP_SORT $< -o $@ 

Theap_test: partial_heap_test.cu
	${NVCC} ${NVCCFLAG} -I${THEAPPATH} -DHEAP_SORT $< -o $@

Bheap_np_test: heap_test.cu
	${NVCC} ${NVCCFLAG} -I${BHEAPNPPATH} -DHEAP_SORT $< -o $@ 

Theap_np_test: heap_test.cu
	${NVCC} ${NVCCFLAG} -I${THEAPNPPATH} -DHEAP_SORT $< -o $@

clean:
	rm -rf *_test
