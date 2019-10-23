NVCC=nvcc
NVCCFLAG= -g -G -arch=sm_61 -O0 -std=c++11 #-DTEST_INVALID #-DPERF_DEBUG

HEAP_DIR=../../heap/bu/
MODEL_DIR=../../models/knapsack-model/

all: knapsack

tdmodel: main.cu
	${NVCC} ${NVCCFLAG} -I${HEAP_DIR} -I${MODEL_DIR} main.cu -o $@

knapsack: main.cu
	${NVCC} ${NVCCFLAG} -I${HEAP_DIR} -I${MODEL_DIR} main.cu -o $@

clean:
	rm -rf tdmodel knapsack
