#ifndef MODELS_CUH
#define MODELS_CUH

#include <functional>
#include <iostream>
#include "util.cuh"
#include "heap.cuh"
#include "gc.cuh"
#include "datastructure.hpp"

using namespace std;

__global__ void init(Heap *heap) {
    uint128 a(0, 0, -1, 0);
    heap->insertion(&a, 1, 0);
}

__device__ int ComputeBound(int newBenefit, int newWeight, int index, int inputSize, int capacity, int *weight, int *benefit)
{	
    // if weight overcomes the knapsack capacity, return
    // 0 as expected bound
    if (newWeight >= capacity)
        return 0;

    // initialize bound on profit by current profit
    int profit_bound = 0;

    // start including items from index 1 more to current
    // item index
    int j = index + 1;
    double totweight = newWeight;
	
    // checking index condition and knapsack capacity
    // condition
    while ((j < inputSize) && (totweight + ((double) weight[j]) <= capacity))
    {
        totweight    += (double) weight[j];
        profit_bound += benefit[j];
        j++;
    }

    // If k is not n, include last item partially for
    // upper bound on profit
    if (j < inputSize)
        profit_bound += (capacity - totweight) * benefit[j] / ((double)weight[j]);

    return profit_bound;
}

__device__ void appKernel(int *weight, int *benefit, float *benefitPerWeight,
                          int *max_benefit, int inputSize, int capacity,
                          uint128 *delItem, uint32 *delSize,
                          uint128 *insItem, uint32 *insSize) 
{
    for(int i = threadIdx.x; i < *delSize; i += blockDim.x){
        uint128 item = delItem[i];
        int oldBenefit = -item.first;
        int oldWeight = item.second;
        short oldIndex = item.third;
//        short oldSeq = item.fourth;

        int _bound = ComputeBound(oldBenefit, oldWeight, oldIndex, inputSize, capacity, weight, benefit);
        if (oldBenefit + _bound < *max_benefit) continue;

        short index = oldIndex + 1;

        if (index == inputSize) continue;

        // check for 1: accept item at current level
        int newBenefit = oldBenefit + benefit[index];
        int newWeight = oldWeight + weight[index];
        int newBound = ComputeBound(newBenefit, newWeight, index, inputSize, capacity, weight, benefit);
        // int newBound = bound[index + 1];

        if(newWeight <= capacity){
            int oldMax = atomicMax(max_benefit, newBenefit);
        }
        
        // printf("bid: %d, processing: %d %u %d, %llu\n", blockIdx.x, -oldBenefit, oldWeight, index, oldSeq);
        if(newWeight <= capacity && newBenefit + newBound > *max_benefit){
            int insIndex = atomicAdd(insSize, 1);
            // printf("choose 1: %d %u %llu\n", -oldBenefit, oldWeight, oldSeq, -newBenefit, newWeight, ((oldSeq << 1) + 1));
            insItem[insIndex].first = -newBenefit;
            insItem[insIndex].second = newWeight;
            insItem[insIndex].third = index;
//            insItem[insIndex].fourth = ((oldSeq << 1) + 1);
        }
        int newBound1 = ComputeBound(oldBenefit, oldWeight, index, inputSize, capacity, weight, benefit);
        // newBound = bound[index + 1];
//        printf("%d-%d i: %d 0: %d %d 1: %d %d\n", blockIdx.x, threadIdx.x, index,
//				oldWeight <= capacity, oldBenefit + newBound1 > *max_benefit, 
//				newWeight <= capacity, newBenefit + newBound > *max_benefit);
        // check for 0: reject current item
		if(oldWeight <= capacity && oldBenefit + newBound1 > *max_benefit){
            int insIndex = atomicAdd(insSize, 1);
            // printf("old: %d %u %llu, choose 0: %d %u %llu\n", -oldBenefit, oldWeight, oldSeq, -oldBenefit, oldWeight, oldSeq << 1);
            insItem[insIndex].first = -oldBenefit;
            insItem[insIndex].second = oldWeight;
            insItem[insIndex].third = index;
//            insItem[insIndex].fourth = oldSeq << 1;
        }
    }
}

#ifdef TEST_INVALID

__device__ void checkInvalidKernel(int *weight, int *benefit, float *benefitPerWeight,
                          int *max_benefit, int *invalidCounter, int inputSize, int capacity,
                          uint128 *delItem, uint32 *delSize,
                          uint128 *insItem, uint32 *insSize) 
{
    for(int i = threadIdx.x; i < *delSize; i += blockDim.x){
        uint128 item = delItem[i];
        int oldBenefit = -item.first;
        int oldWeight = item.second;
        short oldIndex = item.third;
        // uint64 oldSeq = item.fourth;

        int _bound = ComputeBound(oldBenefit, oldWeight, oldIndex, inputSize, capacity, weight, benefit);
		if (oldBenefit + _bound <= *max_benefit) {
			atomicAdd(invalidCounter, 1);
		}
    }
}


__global__ void checkInvalid(Heap *heap, int batchSize, 
                            int *weight, int *benefit, float *benefitPerWeight,
                            int capacity, int inputSize,
                            int *max_benefit, int *invalidCounter)
{
    extern __shared__ int smem[];
    uint128 *delItem = (uint128 *)&smem[0];
    uint128 *insItem = (uint128 *)&delItem[batchSize];
    uint32 *delSize = (uint32 *)&insItem[2 * batchSize];
    uint32 *insSize = (uint32 *)&delSize[1];

    int smemOffset = (sizeof(uint128) * batchSize * 3 +
                      sizeof(uint32) * 2 +
                      sizeof(uint128)) / sizeof(int);

    while(*heap->batchCount != 0 || *heap->partialBatchSize != 0) {

        if (heap->deleteRoot(delItem, *delSize) == true) {
            __syncthreads();
            heap->deleteUpdate(smemOffset);
        }
        __syncthreads();

        if (*delSize > 0) {
            checkInvalidKernel(weight, benefit, benefitPerWeight,
                      max_benefit, invalidCounter, inputSize, capacity,
                      delItem, delSize,
                      insItem, insSize);
        }
        __syncthreads();
    }
}

__global__ void oneHeapCheckValidApplication(Heap *heap, int batchSize, 
                            int *weight, int *benefit, float *benefitPerWeight,
                            int capacity, int inputSize,
                            int *max_benefit, int *old_benefit)
{
    extern __shared__ int smem[];
    uint128 *delItem = (uint128 *)&smem[0];
    uint128 *insItem = (uint128 *)&delItem[batchSize];
    uint32 *delSize = (uint32 *)&insItem[2 * batchSize];
    uint32 *insSize = (uint32 *)&delSize[1];

    int smemOffset = (sizeof(uint128) * batchSize * 3 +
                      sizeof(uint32) * 2 +
                      sizeof(uint128)) / sizeof(int);

	bool init = true;

	if (blockIdx.x == 0 && threadIdx.x == 0) {
		*delSize = 1;
		delItem[0].first = 0;
		delItem[0].second = 0;
		delItem[0].third = -1;
		delItem[0].fourth = 0;
	}
	__syncthreads();

    while(*max_benefit != *old_benefit) {

        if (!init && heap->deleteRoot(delItem, *delSize) == true) {
            __syncthreads();
            heap->deleteUpdate(smemOffset);
        }
        __syncthreads();

		init = false;

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
                heap->insertion(insItem + batchOffset, partialSize, smemOffset);
                __syncthreads();
            }
        }
        __syncthreads();

    }
}


#endif

__global__ void oneHeapApplication(Heap *heap, int batchSize, 
                            int *weight, int *benefit, float *benefitPerWeight,
                            int capacity, int inputSize,
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

	bool init = true;

	if (blockIdx.x == 0 && threadIdx.x == 0) {
		*delSize = 1;
		delItem[0].first = 0;
		delItem[0].second = 0;
		delItem[0].third = -1;
		delItem[0].fourth = 0;
	}
	__syncthreads();

    while(1) {

        if (!init && heap->deleteRoot(delItem, *delSize) == true) {
            __syncthreads();
            heap->deleteUpdate(smemOffset);
        }
        __syncthreads();

		init = false;

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
                heap->insertion(insItem + batchOffset, partialSize, smemOffset);
                __syncthreads();
            }
			if (threadIdx.x == 0) {
//				printf("%u %u %u\n", *insSize, *max_benefit, *heap->batchCount);
			}
			__syncthreads();
        }
        __syncthreads();
		if (threadIdx.x == 0) {
			*delSize = heap->ifTerminate();
		}
		__syncthreads();
		if (*delSize == 1) break;

    }
}

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


//		if (*delSize == 0) break;
		totalDel++;
		if (*delSize == 0) break;
        // XXX
//        if (totalDel == alpha) break;

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


__global__ void endingApplication(uint128 *delItems, int *delSize,
								  uint128 *insItems, int *insSize,
								  int smemReservedSpace, int *totalCount,
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
			atomicAdd(totalCount, *smemItemsSize);
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

void oneheap(int *weight, int *benefit, float *benefitPerWeight,
             int *max_benefit, int capacity, int inputSize,
             int batchNum, int batchSize, int blockNum, int blockSize)
{

    /* prepare heap data */
    Heap heap(batchNum, batchSize);

    Heap *d_heap;
    cudaMalloc((void **)&d_heap, sizeof(Heap));
    cudaMemcpy(d_heap, &heap, sizeof(Heap), cudaMemcpyHostToDevice);

    size_t smemOffset = sizeof(uint128) * batchSize * 3 +
                        sizeof(uint32) * 2 +
                        sizeof(uint128) +
                        5 * batchSize * sizeof(uint128);

    /* initialize step for applications */
//    init<<<1, blockSize, smemOffset>>>(d_heap);
//    cudaDeviceSynchronize();

	struct timeval startTime, endTime;
	setTime(&startTime);

    oneHeapApplication<<<blockNum, blockSize, smemOffset>>>(d_heap, batchSize, 
                                                     weight, benefit, benefitPerWeight,
                                                     capacity, inputSize,
                                                     heap.max_benefit);
//                                                     max_benefit);
    cudaDeviceSynchronize();

	setTime(&endTime);
//	cudaMemcpy(&h_benefit, max_benefit, sizeof(int), cudaMemcpyDeviceToHost);
//	cout << getTime(&startTime, &endTime) << endl;

#ifdef TEST_INVALID
	int *max_benefit1;
	cudaMalloc((void **)&max_benefit1, sizeof(int));
	cudaMemset(max_benefit1, 0, sizeof(int));

    oneHeapCheckValidApplication<<<blockNum, blockSize, smemOffset>>>(d_heap, batchSize, 
                                                     weight, benefit, benefitPerWeight,
                                                     capacity, inputSize,
                                                     max_benefit1, max_benefit);
    cudaDeviceSynchronize();

	cudaMemcpy(&heap, d_heap, sizeof(Heap), cudaMemcpyDeviceToHost);
	uint32 partialNum = 0, fullNum = 0;
	cudaMemcpy(&partialNum, heap.partialBatchSize, sizeof(uint32), cudaMemcpyDeviceToHost);
	cudaMemcpy(&fullNum, heap.batchCount, sizeof(uint32), cudaMemcpyDeviceToHost);
	int totalNum = partialNum + fullNum * batchSize;

	int *d_invalidCounter;
	int h_invalidCounter;
	cudaMalloc((void **)&d_invalidCounter, sizeof(int));
	cudaMemset(d_invalidCounter, 0, sizeof(int));
	checkInvalid<<<blockNum, blockSize, smemOffset>>>(d_heap, batchSize, 
													 weight, benefit, benefitPerWeight,
													 capacity, inputSize,
													 max_benefit, d_invalidCounter);
	cudaDeviceSynchronize();
	cudaMemcpy(&h_invalidCounter, d_invalidCounter, sizeof(int), cudaMemcpyDeviceToHost);
	cout << h_invalidCounter <<  " " << totalNum << " " << 
		(double)h_invalidCounter / (double)totalNum << endl;

#endif

#ifdef PERF_DEBUG
	cudaMemcpy(&heap, d_heap, sizeof(Heap), cudaMemcpyDeviceToHost);
   /* int maxHeapItems = 0;*/
	/*cudaMemcpy(&maxHeapItems, heap.maxHeapItems, sizeof(uint32), cudaMemcpyDeviceToHost);*/
	/*cout << maxHeapItems << endl;*/
	int totalInsNum = 0;
	cudaMemcpy(&totalInsNum, heap.totalInsNum, sizeof(uint32), cudaMemcpyDeviceToHost);
	cout << totalInsNum << endl;
	uint32 partialNum = 0, fullNum = 0;
	cudaMemcpy(&partialNum, heap.partialNum, sizeof(uint32), cudaMemcpyDeviceToHost);
	cudaMemcpy(&fullNum, heap.fullNum, sizeof(uint32), cudaMemcpyDeviceToHost);
	cout << partialNum << " " << fullNum << endl;
#endif

    cudaFree(d_heap); d_heap = NULL;
}

void twoheap(int *weight, int *benefit, float *benefitPerWeight,
             int *max_benefit, int capacity, int inputSize, int alpha, int k,
             int batchNum, int batchSize, int blockNum, int blockSize,
			 bool enableEnding, int endingBlockNum)
{
    Heap heap1(batchNum, batchSize);
    Heap heap2(batchNum, batchSize);

    Heap *d_heap1;
    Heap *d_heap2;
    cudaMalloc((void **)&d_heap1, sizeof(Heap));
    cudaMalloc((void **)&d_heap2, sizeof(Heap));
    cudaMemcpy(d_heap1, &heap1, sizeof(Heap), cudaMemcpyHostToDevice);
    cudaMemcpy(d_heap2, &heap2, sizeof(Heap), cudaMemcpyHostToDevice);

    size_t smemOffset = sizeof(uint128) * batchSize * 3 +
                        sizeof(uint32) * 2 +
                        sizeof(uint128) +
                        5 * batchSize * sizeof(uint128);

    init<<<1, blockSize, smemOffset>>>(d_heap1);
    cudaDeviceSynchronize();


	int iter = 0;

	struct timeval startTime, endTime;
	setTime(&startTime);


//	vector<int> explored;
//	int count1 = 0, count2 = 0;
//	vector<int> maxBenefit;
	int h_benefit = 0;
//	vector<int> heapS;

    while (1) {
//		count2= heap2.itemCount();
        cudaMemcpy(heap2.max_benefit, heap1.max_benefit, sizeof(int), cudaMemcpyDeviceToDevice);
        twoHeapApplication<<<blockNum, blockSize, smemOffset>>>(d_heap1, d_heap2, batchSize, 
                                                         weight, benefit, benefitPerWeight,
                                                         capacity, inputSize, alpha,
                                                         heap2.max_benefit);
//                                                         max_benefit);
        cudaDeviceSynchronize();
		iter++;
		cudaMemcpy(&heap1, d_heap1, sizeof(Heap), cudaMemcpyDeviceToHost);
        cudaMemcpy(&heap2, d_heap2, sizeof(Heap), cudaMemcpyDeviceToHost);
//		explored.push_back(heap2.itemCount() - count2);
//		cudaMemcpy(&h_benefit, max_benefit, sizeof(int), cudaMemcpyDeviceToHost);
//		maxBenefit.push_back(h_benefit);
//		heapS.push_back(heap2.itemCount()/batchSize);
        if (heap1.isEmpty() && heap2.isEmpty() == true) break;
        cudaMemcpy(heap1.max_benefit, heap2.max_benefit, sizeof(int), cudaMemcpyDeviceToDevice);

//		count1 = heap1.itemCount();
        twoHeapApplication<<<blockNum, blockSize, smemOffset>>>(d_heap2, d_heap1, batchSize, 
                                                         weight, benefit, benefitPerWeight,
                                                         capacity, inputSize, alpha,
                                                         heap1.max_benefit);
//                                                         max_benefit);
        cudaDeviceSynchronize();
		iter++;
		cudaMemcpy(&heap1, d_heap1, sizeof(Heap), cudaMemcpyDeviceToHost);
        cudaMemcpy(&heap2, d_heap2, sizeof(Heap), cudaMemcpyDeviceToHost);
//		explored.push_back(heap1.itemCount() - count1);
//		cudaMemcpy(&h_benefit, max_benefit, sizeof(int), cudaMemcpyDeviceToHost);
//		maxBenefit.push_back(h_benefit);
//		heapS.push_back(heap1.itemCount()/batchSize);
        if (heap1.isEmpty() && heap2.isEmpty() == true) break;

 	}

	setTime(&endTime);
//	cout << getTime(&startTime, &endTime) << endl;
//	cudaMemcpy(&h_benefit, max_benefit, sizeof(int), cudaMemcpyDeviceToHost);
//	cout << h_benefit << endl;

/*
	for (int i = 0; i < explored.size(); i++) {
		if (i != 0) {
			explored[i] += explored[i - 1];
		}
		cout << (double)i * (knapsackTime / (double)explored.size()) << 
			" " << explored[i] << " " << maxBenefit[i] << " " << heapS[i] << endl;
	}
*/


	// do it second time for testing performance with gc
	bool gcflag = true;
	int h_benefit1 = 0;
	cudaMemset(max_benefit, 0, sizeof(int));
    init<<<1, blockSize, smemOffset>>>(d_heap1);
    cudaDeviceSynchronize();

	struct timeval expandStartTime, expandEndTime;
	struct timeval gcStartTime, gcEndTime;
	struct timeval endingStartTime, endingEndTime;

	int batchCount1, batchCount2;
	int count1 = 0, count2 = 0;
	int totalCount = 0;

	setTime(&expandStartTime);

    while (1) {
        twoHeapApplication<<<blockNum, blockSize, smemOffset>>>(d_heap1, d_heap2, batchSize, 
                                                         weight, benefit, benefitPerWeight,
                                                         capacity, inputSize, alpha,
                                                         max_benefit);
        cudaDeviceSynchronize();
        cudaMemcpy(&heap1, d_heap1, sizeof(Heap), cudaMemcpyDeviceToHost);
        cudaMemcpy(&heap2, d_heap2, sizeof(Heap), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_benefit1, max_benefit, sizeof(int), cudaMemcpyDeviceToHost);
		totalCount += heap2.itemCount() - count2;
		count1 = heap1.itemCount();
		if (gcflag && h_benefit1 == h_benefit) {
			setTime(&expandEndTime);
			batchCount1 = heap1.nodeCount();
			batchCount2 = heap2.nodeCount();
//            cout << heap1.itemCount() << " " << heap2.itemCount() << endl;
//			cout << batchCount1 + batchCount2 << " ";
            setTime(&gcStartTime);
            invalidFilter(d_heap1, d_heap1, batchSize, batchCount1,
                          blockNum, blockSize, smemOffset,
                          weight, benefit, benefitPerWeight,
                          capacity, inputSize, max_benefit, k);
            invalidFilter(d_heap2, d_heap1, batchSize, batchCount2,
                          blockNum, blockSize, smemOffset,
                          weight, benefit, benefitPerWeight,
                          capacity, inputSize, max_benefit, k);
			gcflag = false;
            alpha = 1024000;
			setTime(&gcEndTime);
            batchCount1 = heap1.nodeCount();
			batchCount2 = heap2.nodeCount();
//            cout << heap1.itemCount() << " " << heap2.itemCount() << endl;
//			cout << batchCount1 + batchCount2 << " ";
			setTime(&endingStartTime);
			totalCount = 0;
			count2 = 0;
			if (enableEnding) break;
		}
//        if (heap2.isEmpty()) break;
        if (heap1.isEmpty() == true && heap2.isEmpty() == true) break;

		twoHeapApplication<<<blockNum, blockSize, smemOffset>>>(d_heap2, d_heap1, batchSize, 
                                                         weight, benefit, benefitPerWeight,
                                                         capacity, inputSize, alpha,
                                                         max_benefit);
        cudaDeviceSynchronize();
        cudaMemcpy(&heap1, d_heap1, sizeof(Heap), cudaMemcpyDeviceToHost);
        cudaMemcpy(&heap2, d_heap2, sizeof(Heap), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_benefit1, max_benefit, sizeof(int), cudaMemcpyDeviceToHost);
		totalCount += heap1.itemCount() - count1;
		count2 = heap2.itemCount();
		if (gcflag && h_benefit1 == h_benefit) {
			setTime(&expandEndTime);
            batchCount1 = heap1.nodeCount();
			batchCount2 = heap2.nodeCount();
//			cout << batchCount1 + batchCount2 << " ";
//            cout << heap1.itemCount() << " " << heap2.itemCount() << endl;
            setTime(&gcStartTime);
            invalidFilter(d_heap1, d_heap1, batchSize, batchCount1,
                          blockNum, blockSize, smemOffset,
                          weight, benefit, benefitPerWeight,
                          capacity, inputSize, max_benefit, k);
            invalidFilter(d_heap2, d_heap1, batchSize, batchCount2,
                          blockNum, blockSize, smemOffset,
                          weight, benefit, benefitPerWeight,
                          capacity, inputSize, max_benefit, k);
            gcflag = false;
            alpha = 1024000;
			setTime(&gcEndTime);
            batchCount1 = heap1.nodeCount();
			batchCount2 = heap2.nodeCount();
//            cout << heap1.itemCount() << " " << heap2.itemCount() << endl;
//			cout << batchCount1 + batchCount2 << " ";
			setTime(&endingStartTime);
			totalCount = 0;
			count2 = 0;
			if (enableEnding) break;
		}
//        if (heap1.isEmpty() == true) break;
        if (heap1.isEmpty() == true && heap2.isEmpty() == true) break;

 	}

	if (enableEnding) {


		// TODO need to make sure heap item is less than smemReservedSpace * blockNum now
		int smemReservedSpace = 2048;
		uint128 *delItems, *insItems;
		int *delSize, *insSize;

		cudaMalloc((void **)&delItems, sizeof(uint128) * smemReservedSpace * blockNum);
		cudaMemcpy(delItems, heap1.heapItems + batchSize, sizeof(uint128) * batchCount1 * batchSize, cudaMemcpyDeviceToDevice);
		int pSize = 0;
		cudaMemcpy(&pSize, heap1.partialBatchSize, sizeof(uint32), cudaMemcpyDeviceToHost);
		cudaMemcpy(delItems + batchCount1 * batchSize, heap1.heapItems, sizeof(uint128) * pSize, cudaMemcpyDeviceToDevice);

		int heapItemCount = pSize + batchCount1 * batchSize;
		cudaMalloc((void **)&delSize, sizeof(int));
		cudaMemcpy(delSize, &heapItemCount, sizeof(int), cudaMemcpyHostToDevice);

		cudaMalloc((void **)&insItems, sizeof(uint128) * smemReservedSpace * blockNum);
		cudaMemset(insItems, 0, sizeof(uint128) * smemReservedSpace * blockNum);

		cudaMalloc((void **)&insSize, sizeof(int));
		cudaMemset(insSize, 0, sizeof(int));

		int *d_totalCount;
		cudaMalloc((void **)&d_totalCount, sizeof(int));
		cudaMemset(d_totalCount, 0, sizeof(int));

		int h_insSize = 0;
		int endingIter = 0;

		smemOffset = smemReservedSpace * sizeof(uint128) + 4 * sizeof(int);
        blockNum = endingBlockNum;
		// blockSize = 32;

		setTime(&endingStartTime);
		
		while (1) {
			endingApplication<<<blockNum, blockSize, smemOffset>>>
				(delItems, delSize,
				 insItems, insSize,
				 smemReservedSpace, d_totalCount,
				 weight, benefit, benefitPerWeight,
				 capacity, inputSize, max_benefit);
			cudaDeviceSynchronize();

			endingIter++;

			cudaMemcpy(&h_insSize, insSize, sizeof(int), cudaMemcpyDeviceToHost);
        	cudaMemcpy(&h_benefit1, max_benefit, sizeof(int), cudaMemcpyDeviceToHost);
//            cout << endingIter << " " << h_insSize << " " << h_benefit1 << endl;
			if (h_insSize == 0) break;
			cudaMemcpy(delSize, insSize, sizeof(int), cudaMemcpyDeviceToDevice);
			cudaMemcpy(delItems, insItems, sizeof(uint128) * h_insSize, cudaMemcpyDeviceToDevice);
			cudaMemset(insSize, 0, sizeof(int));
		}

//		cout << endingIter << endl;
//   		cudaMemcpy(&totalCount, d_totalCount, sizeof(int), cudaMemcpyDeviceToHost);

//        cout << totalCount << endl;
        cudaFree(delItems); delItems = NULL;
        cudaFree(delSize); delSize = NULL;
        cudaFree(insItems); insItems = NULL;
        cudaFree(insSize); insSize = NULL;
        cudaFree(d_totalCount); d_totalCount = NULL;

	}
	
	setTime(&endingEndTime);
//	cout << totalCount << endl;
//	cout << getTime(&expandStartTime, &expandEndTime) << " ";
//	cout << getTime(&gcStartTime, &gcEndTime) << " ";
//	cout << getTime(&endingStartTime, &endingEndTime) << " ";
//	cout << getTime(&expandStartTime, &expandEndTime) +
//			getTime(&gcStartTime, &gcEndTime) +
//			getTime(&endingStartTime, &endingEndTime) << endl;
	cudaMemcpy(&h_benefit1, max_benefit, sizeof(int), cudaMemcpyDeviceToHost);
// 	cout << h_benefit1 << endl;


#ifdef TEST_INVALID
	int h_benefit1 = 0;
	cudaMemset(max_benefit, 0, sizeof(int));
    init<<<1, blockSize, smemOffset>>>(d_heap1);
    cudaDeviceSynchronize();

	uint128 h_first;

    while (1) {
        twoHeapApplication<<<blockNum, blockSize, smemOffset>>>(d_heap1, d_heap2, batchSize, 
                                                         weight, benefit, benefitPerWeight,
                                                         capacity, inputSize, alpha,
                                                         max_benefit);
        cudaDeviceSynchronize();
        cudaMemcpy(&heap2, d_heap2, sizeof(Heap), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_benefit1, max_benefit, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_first, &heap2.heapItems[batchSize], sizeof(uint128), cudaMemcpyDeviceToHost);
		if (h_benefit1 == h_benefit) break;
        if (heap2.isEmpty() == true && heap1.isEmpty() == true) break;


        twoHeapApplication<<<blockNum, blockSize, smemOffset>>>(d_heap2, d_heap1, batchSize, 
                                                         weight, benefit, benefitPerWeight,
                                                         capacity, inputSize, alpha,
                                                         max_benefit);
        cudaDeviceSynchronize();
        cudaMemcpy(&heap1, d_heap1, sizeof(Heap), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_benefit1, max_benefit, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_first, &heap1.heapItems[batchSize], sizeof(uint128), cudaMemcpyDeviceToHost);
		if (h_benefit1 == h_benefit) break;
        if (heap1.isEmpty() == true && heap2.isEmpty() == true) break;

 	}

	cudaMemcpy(&heap1, d_heap1, sizeof(Heap), cudaMemcpyDeviceToHost);
	cudaMemcpy(&heap2, d_heap2, sizeof(Heap), cudaMemcpyDeviceToHost);
	uint32 partialNum1 = 0, fullNum1 = 0;
	uint32 partialNum2 = 0, fullNum2 = 0;
	cudaMemcpy(&partialNum1, heap1.partialBatchSize, sizeof(uint32), cudaMemcpyDeviceToHost);
	cudaMemcpy(&fullNum1, heap1.batchCount, sizeof(uint32), cudaMemcpyDeviceToHost);
	cudaMemcpy(&partialNum2, heap2.partialBatchSize, sizeof(uint32), cudaMemcpyDeviceToHost);
	cudaMemcpy(&fullNum2, heap2.batchCount, sizeof(uint32), cudaMemcpyDeviceToHost);
	int totalNum = partialNum1 + fullNum1 * batchSize + partialNum2 + fullNum2 * batchSize;

	int *d_invalidCounter;
	int h_invalidCounter;
	cudaMalloc((void **)&d_invalidCounter, sizeof(int));
	cudaMemset(d_invalidCounter, 0, sizeof(int));
	checkInvalid<<<blockNum, blockSize, smemOffset>>>(d_heap1, batchSize, 
													 weight, benefit, benefitPerWeight,
													 capacity, inputSize,
													 max_benefit, d_invalidCounter);
	cudaDeviceSynchronize();
	checkInvalid<<<blockNum, blockSize, smemOffset>>>(d_heap2, batchSize, 
													 weight, benefit, benefitPerWeight,
													 capacity, inputSize,
													 max_benefit, d_invalidCounter);
	cudaDeviceSynchronize();
	cudaMemcpy(&h_invalidCounter, d_invalidCounter, sizeof(int), cudaMemcpyDeviceToHost);
	cout << h_invalidCounter <<  " " << totalNum << " " << 
		(double)h_invalidCounter / (double)totalNum << endl;
//	cout << h_first << endl;

#endif

#ifdef PERF_DEBUG
	cout << iter << endl;
	cudaMemcpy(&heap1, d_heap1, sizeof(Heap), cudaMemcpyDeviceToHost);
	cudaMemcpy(&heap2, d_heap2, sizeof(Heap), cudaMemcpyDeviceToHost);
   /* int maxHeapItems = 0;*/
	/*cudaMemcpy(&maxHeapItems, heap.maxHeapItems, sizeof(uint32), cudaMemcpyDeviceToHost);*/
	/*cout << maxHeapItems << endl;*/
	int totalInsNum1 = 0, totalInsNum2 = 0;
	cudaMemcpy(&totalInsNum1, heap1.totalInsNum, sizeof(uint32), cudaMemcpyDeviceToHost);
	cudaMemcpy(&totalInsNum2, heap2.totalInsNum, sizeof(uint32), cudaMemcpyDeviceToHost);
	cout << totalInsNum1 + totalInsNum2 << endl;
	uint32 partialNum1 = 0, fullNum1 = 0;
	uint32 partialNum2 = 0, fullNum2 = 0;
	cudaMemcpy(&partialNum1, heap1.partialNum, sizeof(uint32), cudaMemcpyDeviceToHost);
	cudaMemcpy(&fullNum1, heap1.fullNum, sizeof(uint32), cudaMemcpyDeviceToHost);
	cudaMemcpy(&partialNum2, heap2.partialNum, sizeof(uint32), cudaMemcpyDeviceToHost);
	cudaMemcpy(&fullNum2, heap2.fullNum, sizeof(uint32), cudaMemcpyDeviceToHost);
	cout << partialNum1 << " " << fullNum1 << endl;
	cout << partialNum2 << " " << fullNum2 << endl;
#endif

    cudaFree(d_heap1); d_heap1 = NULL;
    cudaFree(d_heap2); d_heap2 = NULL;
}

#endif
