#ifndef EXPAND_CUH
#define EXPAND_CUH

#include "heap.cuh"
#include "datastructure.hpp"

#include "knapsackKernel.cuh"

__global__ void twoHeapApplication(Heap *delHeap, Heap *insHeap, int batchSize, 
                            int *weight, int *benefit, float *benefitPerWeight,
                            int capacity, int inputSize, int alpha,
                            int *max_benefit)
{
    extern __shared__ int smem[];
    uint128 *delItem = (uint128 *)&smem[0];
    uint128 *insItem = (uint128 *)&delItem[batchSize];
    uint32 *delSize = (uint32 *)&insItem[2 * batchSize];
    uint32 *insSize = (uint32 *)&delSize[1];


    int smemOffset = (sizeof(uint128) * batchSize * 3 +
                      sizeof(uint32) * 2 +
                      sizeof(uint128)) / sizeof(int);

	uint32 totalDel = 0;

    while(1) {

        if (delHeap->deleteRoot(delItem, *delSize) == true) {
            __syncthreads();
            delHeap->deleteUpdate(smemOffset);
        }
        __syncthreads();


		totalDel++;
		if (*delSize == 0) break;

		if (threadIdx.x == 0) {
            *insSize = 0;
        }
        __syncthreads();

        if (*delSize > 0) {
            appKernel(weight, benefit, benefitPerWeight,
                      max_benefit, inputSize, capacity,
                      delItem, delSize,
                      insItem, insSize);
        }
        __syncthreads();

        if (*insSize > 0) {
            for (int batchOffset = 0; batchOffset < *insSize; batchOffset += batchSize) {
                int partialSize = min(batchSize, *insSize - batchOffset);
                insHeap->insertion(insItem + batchOffset, partialSize, smemOffset);
                __syncthreads();
            }
        }
        __syncthreads();

		if (totalDel == alpha) break;
    }
}


void expand2Heap(Heap *heap1, Heap *heap2, int batchSize,
        int blockNum, int blockSize, int smemOffset,
        int *weight, int *benefit, float *benefitPerWeight,
        int capacity, int inputSize, int delAllowed,
        int *max_benefit, int &expandFlag) {


    if (expandFlag == 0) {
        twoHeapApplication<<<blockNum, blockSize, smemOffset>>>
                        (heap1, heap2, batchSize, 
                         weight, benefit, benefitPerWeight,
                         capacity, inputSize, delAllowed,
                         max_benefit);
        cudaDeviceSynchronize();
        expandFlag = 1 - expandFlag;
    }
    else if (expandFlag == 1) {
        twoHeapApplication<<<blockNum, blockSize, smemOffset>>>
                        (heap2, heap1, batchSize, 
                         weight, benefit, benefitPerWeight,
                         capacity, inputSize, delAllowed,
                         max_benefit);
        cudaDeviceSynchronize();
        expandFlag = 1 - expandFlag;
    }
}

#endif
