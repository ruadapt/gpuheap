NVCC=nvcc
NVCCFLAG= -g -G -arch=sm_61 -O3 -std=c++11
 
all: knapsack_seq
knapsack_seq: knapsack_seq.cu
	${NVCC} ${NVCCFLAG} knapsack_seq.cu -o $@

clean:
	rm -rf knapsack_seq
