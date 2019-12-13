NVCC=nvcc
NVCCFLAG= -g -G -arch=sm_61 -O3 -std=c++11
KNAPSACK_FLAG=-DPBS_MODEL
DEBUG_FLAG=-DPERF_DEBUG #-DPRINT_DEBUG

HEAP_DIR=../
BUFFER_DIR=../../../buffer/

all: knapsack

knapsack: main.cu models.cuh
	${NVCC} ${NVCCFLAG} -I${HEAP_DIR} -I${BUFFER_DIR} $< -o $@

knapsack_debug: main.cu models.cuh
	${NVCC} ${NVCCFLAG} ${DEBUG_FLAG} -I${HEAP_DIR} -I${BUFFER_DIR} $< -o $@

clean:
	rm -rf knapsack knapsack_debug
