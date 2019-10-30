#ifndef MODELS_CUH
#define MODELS_CUH

#include <functional>
#include <iostream>
#include "util.cuh"
#include "heap.cuh"
#include "gc.cuh"
#include "datastructure.hpp"
#include "knapsackKernel.cuh"

using namespace std;

/*__global__ void init(Heap<KnapsackItem> *heap) {*/
    /*KnapsackItem a(0, 0, -1, 0);*/
    /*heap->insertion(&a, 1, 0);*/
/*}*/

/*__device__ int ComputeBound(int newBenefit, int newWeight, int index, int inputSize, int capacity, int *weight, int *benefit)*/
/*{	*/
    /*// if weight overcomes the knapsack capacity, return*/
    /*// 0 as expected bound*/
    /*if (newWeight >= capacity)*/
        /*return 0;*/

    /*// initialize bound on profit by current profit*/
    /*int profit_bound = 0;*/

    /*// start including items from index 1 more to current*/
    /*// item index*/
    /*int j = index + 1;*/
    /*double totweight = newWeight;*/
    
    /*// checking index condition and knapsack capacity*/
    /*// condition*/
    /*while ((j < inputSize) && (totweight + ((double) weight[j]) <= capacity))*/
    /*{*/
        /*totweight    += (double) weight[j];*/
        /*profit_bound += benefit[j];*/
        /*j++;*/
    /*}*/

    /*// If k is not n, include last item partially for*/
    /*// upper bound on profit*/
    /*if (j < inputSize)*/
        /*profit_bound += (capacity - totweight) * benefit[j] / ((double)weight[j]);*/

    /*return profit_bound;*/
/*}*/

/*__device__ void appKernel(int *weight, int *benefit, float *benefitPerWeight,*/
                          /*int *globalBenefit, int inputSize, int capacity,*/
                          /*KnapsackItem *delItem, int *delSize,*/
                          /*KnapsackItem *insItem, int *insSize) */
/*{*/
    /*for(int i = threadIdx.x; i < *delSize; i += blockDim.x){*/
        /*KnapsackItem item = delItem[i];*/
        /*int oldBenefit = -item.first;*/
        /*int oldWeight = item.second;*/
        /*short oldIndex = item.third;*/
/*//        short oldSeq = item.fourth;*/

        /*int _bound = ComputeBound(oldBenefit, oldWeight, oldIndex, inputSize, capacity, weight, benefit);*/
        /*if (oldBenefit + _bound < *globalBenefit) continue;*/

        /*short index = oldIndex + 1;*/

        /*if (index == inputSize) continue;*/

        /*// check for 1: accept item at current level*/
        /*int newBenefit = oldBenefit + benefit[index];*/
        /*int newWeight = oldWeight + weight[index];*/
        /*int newBound = ComputeBound(newBenefit, newWeight, index, inputSize, capacity, weight, benefit);*/
        /*// int newBound = bound[index + 1];*/

        /*if(newWeight <= capacity){*/
            /*int oldMax = atomicMax(globalBenefit, newBenefit);*/
        /*}*/
        
        /*// printf("bid: %d, processing: %d %u %d, %llu\n", blockIdx.x, -oldBenefit, oldWeight, index, oldSeq);*/
        /*if(newWeight <= capacity && newBenefit + newBound > *globalBenefit){*/
            /*int insIndex = atomicAdd(insSize, 1);*/
            /*// printf("choose 1: %d %u %llu\n", -oldBenefit, oldWeight, oldSeq, -newBenefit, newWeight, ((oldSeq << 1) + 1));*/
            /*insItem[insIndex].first = -newBenefit;*/
            /*insItem[insIndex].second = newWeight;*/
            /*insItem[insIndex].third = index;*/
/*//            insItem[insIndex].fourth = ((oldSeq << 1) + 1);*/
        /*}*/
        /*int newBound1 = ComputeBound(oldBenefit, oldWeight, index, inputSize, capacity, weight, benefit);*/
        /*// newBound = bound[index + 1];*/
/*//        printf("%d-%d i: %d 0: %d %d 1: %d %d\n", blockIdx.x, threadIdx.x, index,*/
/*//				oldWeight <= capacity, oldBenefit + newBound1 > *globalBenefit, */
/*//				newWeight <= capacity, newBenefit + newBound > *globalBenefit);*/
        /*// check for 0: reject current item*/
        /*if(oldWeight <= capacity && oldBenefit + newBound1 > *globalBenefit){*/
            /*int insIndex = atomicAdd(insSize, 1);*/
            /*// printf("old: %d %u %llu, choose 0: %d %u %llu\n", -oldBenefit, oldWeight, oldSeq, -oldBenefit, oldWeight, oldSeq << 1);*/
            /*insItem[insIndex].first = -oldBenefit;*/
            /*insItem[insIndex].second = oldWeight;*/
            /*insItem[insIndex].third = index;*/
/*//            insItem[insIndex].fourth = oldSeq << 1;*/
        /*}*/
    /*}*/
/*}*/

__global__ void oneHeapApplication(Heap<KnapsackItem> *heap, int batchSize, 
                            int *weight, int *benefit, float *benefitPerWeight,
                            int capacity, int inputSize,
                            int *globalBenefit,
                            int *gc_flag, int gc_threshold,
                            bool init_flag = true)
{
    extern __shared__ int smem[];
    KnapsackItem *delItem = (KnapsackItem *)&smem[0];
    KnapsackItem *insItem = (KnapsackItem *)&delItem[batchSize];
    int *delSize = (int *)&insItem[2 * batchSize];
    int *insSize = (int *)&delSize[1];

    int smemOffset = (sizeof(KnapsackItem) * batchSize * 3 +
                      sizeof(int) * 2) / sizeof(int);

    bool init = init_flag;
    if (threadIdx.x == 0) {
        heap->tbstate[blockIdx.x] = 1;
    }
    __syncthreads();

    if (init && blockIdx.x == 0 && threadIdx.x == 0) {
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
#ifdef PRINT_DEBUG
            printf("thread %d delete items %d\n", blockIdx.x, *delSize);
            for (int i = 0; i < *delSize; i++) {
                printf("%d %d | ", delItem[i].first, delItem[i].second);
            }
            printf("\n");
#endif
            *insSize = 0;
        }
        __syncthreads();

        if (*delSize > 0) {
            appKernel(weight, benefit, benefitPerWeight,
                      globalBenefit, inputSize, capacity,
                      delItem, delSize,
                      insItem, insSize);
        }
        __syncthreads();
#ifdef PRINT_DEBUG
        if (threadIdx.x == 0) {
            printf("thread %d insert items %d\n", blockIdx.x, *insSize);
            for (int i = 0; i < *insSize; i++) {
                printf("%d %d | ", insItem[i].first, insItem[i].second);
            }
            printf("\n");
        }
        __syncthreads();
#endif

        if (*insSize > 0) {
            for (int batchOffset = 0; batchOffset < *insSize; batchOffset += batchSize) {
                int partialSize = min(batchSize, *insSize - batchOffset);
                heap->insertion(insItem + batchOffset, partialSize, smemOffset);
                __syncthreads();
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            *delSize = heap->ifTerminate();
            if (*heap->batchCount > gc_threshold) {
                *gc_flag  = 1;
            }
        }
        __syncthreads();
        if (*delSize == 1 || *gc_flag == 1) break;

    }
}

void oneheap(int *weight, int *benefit, float *benefitPerWeight,
             int *max_benefit, int capacity, int inputSize,
             int batchNum, int batchSize, int blockNum, int blockSize,
             int gc_threshold)
{

    /* prepare heap data */
    Heap<KnapsackItem> heap(batchNum, batchSize);

    Heap<KnapsackItem> *d_heap;
    cudaMalloc((void **)&d_heap, sizeof(Heap<KnapsackItem>));
    cudaMemcpy(d_heap, &heap, sizeof(Heap<KnapsackItem>), cudaMemcpyHostToDevice);

    size_t smemOffset = sizeof(KnapsackItem) * batchSize * 3 +
                        sizeof(int) * 2 +
                        sizeof(KnapsackItem) +
                        5 * batchSize * sizeof(KnapsackItem);

    bool init_flag = true;
    int *gc_flag;
    cudaMalloc((void **)&gc_flag, sizeof(int));
    cudaMemset(gc_flag, 0, sizeof(int));

	struct timeval startTime, endTime;
	setTime(&startTime);
#ifdef PERF_DEBUG
    struct timeval appStartTime, appEndTime;
    double appTime = 0;
    struct timeval gcStartTime, gcEndTime;
    double gcTime = 0;
#endif

    while (1) {
#ifdef PERF_DEBUG
        setTime(&appStartTime);
#endif
        oneHeapApplication<<<blockNum, blockSize, smemOffset>>>(d_heap, batchSize, 
                                                         weight, benefit, benefitPerWeight,
                                                         capacity, inputSize,
                                                         heap.globalBenefit,
                                                         gc_flag, gc_threshold,
                                                         init_flag);
        cudaDeviceSynchronize();
#ifdef PERF_DEBUG
        setTime(&appEndTime);
        appTime += getTime(&appStartTime, &appEndTime);
        cudaMemcpy(&heap, d_heap, sizeof(Heap<KnapsackItem>), cudaMemcpyDeviceToHost);
        int batchCount = 0;
        cudaMemcpy(&batchCount, heap.batchCount, sizeof(int), cudaMemcpyDeviceToHost);
        int cur_benefit = 0;
        cudaMemcpy(&cur_benefit, heap.globalBenefit, sizeof(int), cudaMemcpyDeviceToHost);
        cout << appTime << " " << batchCount << " " << cur_benefit << " "; 
#endif
        int app_terminate = 0;
        cudaMemcpy(&app_terminate, heap.terminate, sizeof(int), cudaMemcpyDeviceToHost);
        if (app_terminate) break;
#ifdef PERF_DEBUG
        setTime(&gcStartTime);
#endif
        // garbage collection
        invalidFilter(heap, d_heap, batchSize,
                      weight, benefit, benefitPerWeight,
                      capacity, inputSize, heap.globalBenefit);
#ifdef PERF_DEBUG
        setTime(&gcEndTime);
        gcTime += getTime(&gcStartTime, &gcEndTime);
        cudaMemcpy(&batchCount, heap.batchCount, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&cur_benefit, heap.globalBenefit, sizeof(int), cudaMemcpyDeviceToHost);
        cout << gcTime << " " << batchCount << " " << cur_benefit << endl; 
#endif
        // reset gc flag
        cudaMemset(gc_flag, 0, sizeof(int));
        init_flag = false;
    }

#ifdef PERF_DEBUG
    cout << endl;
#endif

	setTime(&endTime);
	cout << getTime(&startTime, &endTime) << endl;
    cudaMemcpy((int *)max_benefit, heap.globalBenefit, sizeof(int), cudaMemcpyDeviceToDevice);
    cudaFree(d_heap); d_heap = NULL;
    cudaFree(gc_flag); gc_flag = NULL;
}

#endif
