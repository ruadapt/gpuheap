NVCC=nvcc
NVCCFLAG= -g -G -arch=sm_61 -O0 -std=c++11
KNAPSACK_FLAG=-DPBS_MODEL
DEBUG_FLAG=-DPERF_DEBUG #-DPRINT_DEBUG

TD_HEAP_DIR=../heap/tditdd/
BU_HEAP_DIR=../heap/buitdd/

all: knapsack

knapsack: main.cu models.cuh gc.cuh
	${NVCC} ${NVCCFLAG} -I${TD_HEAP_DIR} ${KNAPSACK_FLAG} $< -o $@

knapsack_debug: main.cu models.cuh gc.cuh
	${NVCC} ${NVCCFLAG} -I${TD_HEAP_DIR} ${KNAPSACK_FLAG} ${DEBUG_FLAG} $< -o $@

clean:
	rm -rf knapsack knapsack_debug
