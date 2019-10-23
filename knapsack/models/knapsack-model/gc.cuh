#ifndef GC_CUH
#define GC_CUH

#include "heap.cuh"
#include "datastructure.hpp"

#include "knapsackKernel.cuh"

__global__ void garbageCollection(Heap *heap, int batchSize, int batchCount,
                        int *weight, int *benefit, float *benefitPerWeight,
                        int capacity, int inputSize,
                        int *max_benefit, int k,
                        uint128 *insItems, int *insSize)
{
    uint128 dummy_item(INIT_LIMITS, 0, 0, 0);
    // TODO now only support batchsize == blockSize
    /*handle parital batch first*/
    if (blockIdx.x == 0) {
        for (int i = threadIdx.x; i < *heap->partialBatchSize; i += blockDim.x) {
            int oldBenefit = -(heap->heapItems[i]).first;
            int oldWeight = (heap->heapItems[i]).second;
            short oldIndex = (heap->heapItems[i]).third;

            int _bound = ComputeBound(oldBenefit, oldWeight, oldIndex, inputSize, capacity, weight, benefit);
            if (oldBenefit + _bound > *max_benefit) {
                int index = atomicAdd(insSize, 1);
                insItems[index] = heap->heapItems[i];
            }
            heap->heapItems[i] = dummy_item;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            *heap->partialBatchSize = 0;
        }
        __syncthreads();
    }
    int startBatchIdx = batchCount - k >= 0 ? batchCount - k + 1 : 1;
    for (int batchIndex = startBatchIdx + blockIdx.x; batchIndex <= batchCount; batchIndex += gridDim.x) {
        if(threadIdx.x == 0) {
            heap->status[batchIndex] = UNUSED;
        }

        int i = batchIndex * batchSize + threadIdx.x;
        int oldBenefit = -(heap->heapItems[i]).first;
        int oldWeight = (heap->heapItems[i]).second;
        short oldIndex = (heap->heapItems[i]).third;

        int _bound = ComputeBound(oldBenefit, oldWeight, oldIndex, inputSize, capacity, weight, benefit);
		if (oldBenefit + _bound > *max_benefit) {
            int index = atomicAdd(insSize, 1);
            insItems[index] = heap->heapItems[i];
		}
        heap->heapItems[i] = dummy_item;
    }

    if(!threadIdx.x && !blockIdx.x){
        *heap->batchCount = startBatchIdx - 1;
    }
    __syncthreads();

}

__global__ void HeapInsert(Heap *heap, uint128 *insItems, int *insSize, int batchSize){	
	if(!(*insSize))
		return;
	uint32 batchNeed = (*insSize + batchSize - 1) / batchSize;
    for (uint32 i = blockIdx.x; i < batchNeed; i += gridDim.x) {
        uint32 size = min(batchSize, *insSize - i * batchSize);
        heap->insertion(insItems + i * batchSize,
                        size, 0);
        __syncthreads();
    }
	
}

void invalidFilter(Heap heap, Heap *d_heap, int batchSize, int batchCount,
                    int *weight, int *benefit, float *benefitPerWeight,
                    int capacity, int inputSize,
                    int *max_benefit, int k)
{

    uint128 *insItems;
    int *insSize;
    cudaMalloc((void **)&insItems, k * batchSize * sizeof(uint128));
    cudaMalloc((void **)&insSize, sizeof(int));
    cudaMemset(insSize, 0, sizeof(int));

    int blockNum = 32;
    int blockSize = batchSize;

    size_t smemOffset = sizeof(uint128) * batchSize * 3 +
                        sizeof(uint32) * 2 +
                        sizeof(uint128) +
                        5 * batchSize * sizeof(uint128);

    garbageCollection<<<blockNum, blockSize>>>(d_heap, batchSize, batchCount,
                                                weight, benefit, benefitPerWeight,
                                                capacity, inputSize,
                                                max_benefit, k,
                                                insItems, insSize);
    cudaDeviceSynchronize();

    HeapInsert<<<blockNum, blockSize, smemOffset>>>(d_heap, insItems, insSize, batchSize);

    cudaDeviceSynchronize();

    cudaFree(insItems); insItems = NULL;
    cudaFree(insSize); insSize = NULL;

}

void invalidFilter2Heap(Heap heap1, Heap heap2, 
                        Heap *d_heap1, Heap *d_heap2, int batchSize,
                        int *weight, int *benefit, float *benefitPerWeight,
                        int capacity, int inputSize, int *max_benefit,
                        int expandFlag, int gcThreshold, int k)
{
    if (expandFlag == 0) {
        int batchCount = heap1.nodeCount();
        if (batchCount > gcThreshold) {
            cout << "gc..." << batchCount;
            invalidFilter(heap1, d_heap1, batchSize, batchCount,
                          weight, benefit, benefitPerWeight,
                          capacity, inputSize, max_benefit, 1024000);
            cout << heap2.nodeCount();
        }
    }
    else if (expandFlag == 1) {
        int batchCount = heap2.nodeCount();
        if (batchCount > gcThreshold) {
            cout << "gc..." << batchCount;
            invalidFilter(heap2, d_heap2, batchSize, batchCount,
                          weight, benefit, benefitPerWeight,
                          capacity, inputSize, max_benefit, 1024000);
            cout << heap2.nodeCount();
        }
    }
}



#endif
