#ifndef MODELS_CUH
#define MODELS_CUH

#include <functional>
#include <iostream>
#include "util.cuh"
#include "heap.cuh"
#include "buffer.cuh"
#include "gc.cuh"
#include "datastructure.hpp"
#include "knapsackKernel.cuh"
#include "models_fifo.cuh"
#include "knapsack_util.cuh"

using namespace std;

__global__ void oneHeapApplicationEarlyStop(Heap<KnapsackItem> *heap, int batchSize, 
                            int *weight, int *benefit, float *benefitPerWeight,
                            int capacity, int inputSize,
                            int *globalBenefit, int *activeCount,
                            int *gc_flag, int gc_threshold, int max_benefit,
#ifdef PERF_DEBUG 
                            int *explored_nodes,
#endif
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
        *delSize = 0;
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
#ifdef PERF_DEBUG 
            if (threadIdx.x == 0) {
                atomicAdd(explored_nodes, *insSize);
            }
            __syncthreads();
#endif
            for (int batchOffset = 0; batchOffset < *insSize; batchOffset += batchSize) {
                int partialSize = min(batchSize, *insSize - batchOffset);
                heap->insertion(insItem + batchOffset, partialSize, smemOffset);
                __syncthreads();
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(activeCount, (*insSize - *delSize));
            if (atomicAdd(activeCount, 0) == 0 || 
                    *globalBenefit == max_benefit) {
                *delSize = -1; // TODO: use one global flag for termination
            }
            /**delSize = heap->ifTerminate();*/
            if (*heap->batchCount > gc_threshold) {
                *gc_flag  = 1;
            }
        }
        __syncthreads();
        if (*delSize == -1 || *gc_flag == 1) break;
    }
}

void oneheapearlystop(int *weight, int *benefit, float *benefitPerWeight,
             int *max_benefit, int capacity, int inputSize,
             int batchNum, int batchSize, int blockNum, int blockSize,
             int gc_threshold, int global_max_benefit)
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
    int *activeCount;
    cudaMalloc((void **)&activeCount, sizeof(int));
    int initActiveCount = 1;
    cudaMemcpy(activeCount, &initActiveCount, sizeof(int), cudaMemcpyHostToDevice);

	struct timeval startTime, endTime;
    double heapTime = 0, fifoTime = 0;
	setTime(&startTime);
#ifdef PERF_DEBUG
    int *explored_nodes;
    cudaMalloc((void **)&explored_nodes, sizeof(int));
    cudaMemset(explored_nodes, 0, sizeof(int));

    struct timeval appStartTime, appEndTime;
    double appTime = 0;
    struct timeval gcStartTime, gcEndTime;
    double gcTime = 0;
    cout << global_max_benefit << endl;
#endif

    // TODO: change the name
    int tmpItemCount = 0, tmpBenefit = 0;

    while (1) {
#ifdef PERF_DEBUG
        setTime(&appStartTime);
#endif
        oneHeapApplicationEarlyStop<<<blockNum, blockSize, smemOffset>>>(d_heap, batchSize, 
                                                         weight, benefit, benefitPerWeight,
                                                         capacity, inputSize,
                                                         heap.globalBenefit, activeCount,
                                                         gc_flag, gc_threshold, global_max_benefit,
#ifdef PERF_DEBUG
                                                         explored_nodes,
#endif
                                                         init_flag);
        cudaDeviceSynchronize();
        init_flag = false;
#ifdef PERF_DEBUG
        setTime(&appEndTime);
        appTime += getTime(&appStartTime, &appEndTime);
        cudaMemcpy(&heap, d_heap, sizeof(Heap<KnapsackItem>), cudaMemcpyDeviceToHost);
        int batchCount = 0;
        cudaMemcpy(&batchCount, heap.batchCount, sizeof(int), cudaMemcpyDeviceToHost);
        int cur_benefit = 0;
        cudaMemcpy(&cur_benefit, heap.globalBenefit, sizeof(int), cudaMemcpyDeviceToHost);
        int h_explored_nodes = 0;
        cudaMemcpy(&h_explored_nodes, explored_nodes, sizeof(int), cudaMemcpyDeviceToHost);
        cout << appTime << " " << batchCount << " " << cur_benefit << " " << h_explored_nodes << " "; 
#endif
        /*int app_terminate = 0;*/
        /*cudaMemcpy(&app_terminate, heap.terminate, sizeof(int), cudaMemcpyDeviceToHost);*/
        /*if (app_terminate) break;*/
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

        // if no items rest break
        tmpItemCount = heap.itemCount();
        if (tmpItemCount == 0) break;

        // if benefit == global_max_benefit break
        cudaMemcpy(&tmpBenefit, heap.globalBenefit, sizeof(int), cudaMemcpyDeviceToHost);
        if (tmpBenefit == global_max_benefit) break;
        cudaMemcpy(activeCount, &tmpItemCount, sizeof(int), cudaMemcpyHostToDevice);
    }

#ifdef PERF_DEBUG
    cout << endl;
#endif
    setTime(&endTime);
    heapTime = getTime(&startTime, &endTime);
    cout << "heap time: " << heapTime << endl;

    if (tmpItemCount != 0) {
        // we switch to the fifo queue mode

        setTime(&startTime);
        // prepare fifo queue (buffer)
        Buffer<KnapsackItem> buffer(batchSize * batchNum);
        Buffer<KnapsackItem> *d_buffer;
        cudaMalloc((void **)&d_buffer, sizeof(Buffer<KnapsackItem>));
        cudaMemcpy(d_buffer, &buffer, sizeof(Buffer<KnapsackItem>), cudaMemcpyHostToDevice);

        // copy data items in heap to fifo queue
        // TODO this is not a good implementation that manually change the fifo queue ptr
        unsigned long long int remain_item_number;
        heapDataToArray<KnapsackItem>(heap, buffer.bufferItems, remain_item_number);
        cudaMemcpy(buffer.writePos, &remain_item_number, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
        cudaMemcpy(buffer.endPos, &remain_item_number, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
        cudaMemcpy(buffer.globalBenefit, heap.globalBenefit, sizeof(int), cudaMemcpyDeviceToDevice);

        cudaMemcpy(activeCount, &remain_item_number, sizeof(int), cudaMemcpyHostToDevice);
        oneBufferApplication<<<blockNum, blockSize, smemOffset>>>(d_buffer, batchSize, 
                                                         weight, benefit, benefitPerWeight,
                                                         capacity, inputSize,
                                                         buffer.globalBenefit, activeCount,
                                                         gc_flag, gc_threshold,
#ifdef PERF_DEBUG
                                                         explored_nodes,
#endif
                                                         init_flag);
        cudaDeviceSynchronize();

        setTime(&endTime);
        fifoTime = getTime(&startTime, &endTime);
        cout << "gc time: " << fifoTime << endl;
        cudaMemcpy((int *)max_benefit, buffer.globalBenefit, sizeof(int), cudaMemcpyDeviceToDevice);
        cudaFree(d_buffer); d_buffer = NULL;
    }


    int h_explored_nodes;
    cudaMemcpy(&h_explored_nodes, explored_nodes, sizeof(int), cudaMemcpyDeviceToHost);
    cout << "total time: " << heapTime + fifoTime << " explored nodes: " << h_explored_nodes << endl;

    cudaFree(d_heap); d_heap = NULL;
    cudaFree(gc_flag); gc_flag = NULL;
    cudaFree(activeCount); activeCount = NULL;
}

#endif
