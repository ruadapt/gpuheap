#ifndef END_CUH
#define END_CUH

#include <iostream>

#include "heap.cuh"
#include "datastructure.hpp"

#include "knapsackKernel.cuh"

using namespace std;

__global__ void endingApplication(uint128 *delItems, int *delSize,
								  uint128 *insItems, int *insSize,
								  int smemReservedSpace,
								  int *weight, int *benefit, float *benefitPerWeight,
								  int capacity, int inputSize, int *max_benefit) 
{

	/* prepare smem space that is preset to 1024 * sizeof(uint128) */
	__shared__ extern int smem[];
	uint128 *smemItems = (uint128 *)&smem[0];
	uint32 *smemItemsSize = (uint32 *)&smemItems[smemReservedSpace];
	int *insertOffset = (int *)&smemItemsSize[1];
	uint32 *newCounter = (uint32 *)&insertOffset[1];

	/*calculate the range of items to process*/
	int delWorkPerBlock = (*delSize + gridDim.x - 1) / gridDim.x;
	int delStartIdx = blockIdx.x * delWorkPerBlock;
	int delEndIdx = min(delStartIdx + delWorkPerBlock, *delSize);

	/*move items in delItems to smemItems*/
	for (int i = delStartIdx + threadIdx.x; i < delEndIdx; i += blockDim.x) {
		smemItems[i - delStartIdx] = delItems[i];
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		*smemItemsSize = delEndIdx - delStartIdx;
	}
	__syncthreads();

	while (*smemItemsSize != 0 && *smemItemsSize < smemReservedSpace / 3) {
		if (threadIdx.x == 0) {
			*newCounter = 0;
		}
		__syncthreads();
        appKernel(weight, benefit, benefitPerWeight,
                  max_benefit, inputSize, capacity,
                  smemItems, smemItemsSize,
                  smemItems + *smemItemsSize, newCounter);
		__syncthreads();

		/*compact smemItems*/
        uint32 shiftOffset = *smemItemsSize >= *newCounter ? 
            *smemItemsSize : *newCounter;
        uint32 shiftMax = *smemItemsSize >= *newCounter ? *newCounter : *smemItemsSize;
		for (int i = threadIdx.x; i < shiftMax; i += blockDim.x) {
			smemItems[i] = smemItems[shiftOffset + i];
		}
		__syncthreads();
		if (threadIdx.x == 0) {
//            printf("%d %d %d %d\n", blockIdx.x, currentOffset, *newCounter, *max_benefit);
			*smemItemsSize = *newCounter;
		}
		__syncthreads();

	}

	/*reserve space in insItems*/
	if (threadIdx.x == 0) {
		*insertOffset = atomicAdd(insSize, *smemItemsSize);
	}
	__syncthreads();
	/*move items from smemItems to insItems*/
	for (int i = threadIdx.x; i < *smemItemsSize; i += blockDim.x) {
		insItems[*insertOffset + i] = smemItems[i];
	}
	__syncthreads();
}

void heapDataToArray(Heap heap, uint128 *array, int batchSize)
{
    int pSize = 0, bCount = 0;
    cudaMemcpy(&pSize, heap.partialBatchSize, sizeof(uint32), cudaMemcpyDeviceToHost);
    cudaMemcpy(&bCount, heap.batchCount, sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(array, heap.heapItems, sizeof(uint128) * pSize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(array + pSize, heap.heapItems + batchSize, sizeof(uint128) * bCount * batchSize, cudaMemcpyDeviceToDevice);
}

bool prepareAfterExpandData(uint128 **delItems, int **delSize, uint128 **insItems, int **insSize,
                            Heap heap1, Heap heap2, Heap *d_heap1, Heap *d_heap2,
                            int batchSize, int smemReservedSpace, int blockNum)
{
    int itemCount1 = heap1.itemCount();
    int itemCount2 = heap2.itemCount();

    if (itemCount1 + itemCount2 > smemReservedSpace * blockNum / 3) {
        cout << "Limited Smem Space\n";
        return false;
    }

    cudaMalloc((void **)delItems, sizeof(uint128) * smemReservedSpace * blockNum);
    cudaMalloc((void **)delSize, sizeof(int));
    heapDataToArray(heap1, *delItems, batchSize);
    heapDataToArray(heap2, *delItems + itemCount1, batchSize);
   
    cudaMalloc((void **)insItems, sizeof(uint128) * smemReservedSpace * blockNum);
    cudaMemset(*insItems, 0, sizeof(uint128) * smemReservedSpace * blockNum);
    cudaMalloc((void **)insSize, sizeof(int));
    cudaMemset(*insSize, 0, sizeof(int));

    return true;
}

void afterExpandStage(Heap heap1, Heap heap2, int batchSize,
                      Heap *d_heap1, Heap *d_heap2, 
                      int *weight, int *benefit, float *benefitPerWeight,
                      int capacity, int inputSize, int endingBlockNum,
                      int *max_benefit, int &endFlag) 
{
    int batchCount1 = heap1.nodeCount();
    int batchCount2 = heap2.nodeCount();
    int itemCount1 = heap1.itemCount();
    int itemCount2 = heap2.itemCount();
    if (itemCount1 == 0 && itemCount2 == 0) {
        endFlag = 1;
        return;
    }
    int h_benefit, prev_benefit;
    int h_insSize = -1;
    cudaMemcpy(&prev_benefit, max_benefit, sizeof(int), cudaMemcpyDeviceToHost);
    h_benefit = prev_benefit;

    int smemReservedSpace = 3900;
    int blockSize = batchSize;
    uint128 *delItems, *insItems;
    int *delSize, *insSize;

    if (prepareAfterExpandData(&delItems, &delSize, &insItems, &insSize,
                            heap1, heap2, d_heap1, d_heap2,
                            batchSize, smemReservedSpace, endingBlockNum) == false) {
        // not enough space, data is still in the heap, so still use the heap
        endFlag = 1;
        return;
    }

    int smemOffset = smemReservedSpace * sizeof(uint128) + 4 * sizeof(int);

    while (prev_benefit == h_benefit && h_insSize != 0) {

        endingApplication<<<endingBlockNum, blockSize, smemOffset>>>
            (delItems, delSize,
             insItems, insSize,
             smemReservedSpace,
             weight, benefit, benefitPerWeight,
             capacity, inputSize, max_benefit);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_benefit, max_benefit, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_insSize, insSize, sizeof(int), cudaMemcpyDeviceToHost);
    }
    if (h_insSize == 0) endFlag = 1;

    cudaFree(insItems); insItems = NULL;
    cudaFree(delItems); delItems = NULL;
    cudaFree(insSize); insSize = NULL;
    cudaFree(delSize); delSize = NULL;
}

#endif
