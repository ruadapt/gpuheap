#ifndef MODELS_CUH
#define MODELS_CUH

#include <functional>
#include <iostream>
#include "util.cuh"
#include "heap.cuh"
#include "buffer.cuh"
#include "heap_api.cuh"
#include "datastructure.hpp"
#include "knapsackKernel.cuh"
#include "knapsack_util.cuh"

using namespace std;

__global__ void oneBufferApplication(Buffer<KnapsackItem> *buffer, int batchSize, 
                            int *weight, int *benefit, float *benefitPerWeight,
                            int capacity, int inputSize,
                            int *globalBenefit, unsigned long long int *activeCount,
                               int *explored_nodes)
   {
    extern __shared__ int smem[];
    KnapsackItem *delItem = (KnapsackItem *)&smem[0];
    KnapsackItem *insItem = (KnapsackItem *)&delItem[batchSize];
    int *delSize = (int *)&insItem[2 * batchSize];
    int *insSize = (int *)&delSize[1];

    int smemOffset = (sizeof(KnapsackItem) * batchSize * 3 +
                      sizeof(int) * 2) / sizeof(int);

    if (threadIdx.x == 0) {
        *delSize = 0;
    }
    __syncthreads();


    while(1) {
        buffer->deleteFromBuffer(delItem, *delSize, smemOffset);
        __syncthreads();

        if (threadIdx.x == 0) {
            *insSize = 0;
        }
        __syncthreads();

        if (*delSize > 0) {
            appKernelWrapper(weight, benefit, benefitPerWeight,
                      globalBenefit, inputSize, capacity,
                      explored_nodes,
                      delItem, delSize,
                      insItem, insSize);
        }
        __syncthreads();
        if (*insSize > 0) {
            buffer->insertToBuffer(insItem, *insSize, smemOffset);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            atomicAdd(activeCount, (*insSize - *delSize));
//            printf("bid %d ac %llu en %d b %d\n", blockIdx.x, *activeCount, *explored_nodes, *globalBenefit);
            if (atomicAdd(activeCount, 0) == 0) {
                *delSize = -1;
            }
        }
        __syncthreads();
        if (*delSize == -1) break;
        __syncthreads();
    }
}

__global__ void psyncAppKernel(int *weight, int *benefit, float *benefitPerWeight,
                  int *globalBenefit, int inputSize, int capacity,
                  KnapsackItem *delItems, int delSize,
                  KnapsackItem *insItems, int *insSize,
                  int batchSize) {

    int batchCount = delSize / batchSize;
    if (blockIdx.x >= batchCount) return;
    appKernel(weight, benefit, benefitPerWeight,
              globalBenefit, inputSize, capacity,
              delItems + blockIdx.x * batchSize, &batchSize,
              insItems, insSize);
    __syncthreads();
}

void oneheapfifoswitch(int *weight, int *benefit, float *benefitPerWeight,
             int *max_benefit, int capacity, int inputSize,
             int batchNum, int batchSize, int blockNum, int blockSize,
             int gc_threshold, int switch_counter, int global_max_benefit)
{
    int table_size = 1024000;
    /* prepare heap data */
    Heap<KnapsackItem> heap(batchNum, batchSize, table_size);
    Heap<KnapsackItem> *d_heap;
    cudaMalloc((void **)&d_heap, sizeof(Heap<KnapsackItem>));
    cudaMemcpy(d_heap, &heap, sizeof(Heap<KnapsackItem>), cudaMemcpyHostToDevice);

    /* prepare ins/del table buffer*/
    TB<KnapsackItem> h_insTB(table_size, batchSize, 0);
    TB<KnapsackItem> *d_insTB;
    cudaMalloc((void **)&d_insTB, sizeof(TB<KnapsackItem>));
    cudaMemcpy(d_insTB, &h_insTB, sizeof(TB<KnapsackItem>), cudaMemcpyHostToDevice);

    TB<KnapsackItem> h_delTB(table_size, batchSize, 1);
    TB<KnapsackItem> *d_delTB;
    cudaMalloc((void **)&d_delTB, sizeof(TB<KnapsackItem>));
    cudaMemcpy(d_delTB, &h_delTB, sizeof(TB<KnapsackItem>), cudaMemcpyHostToDevice);

    //  prepare fifo queue (buffer) 
    Buffer<KnapsackItem> buffer(batchSize * batchNum);
    Buffer<KnapsackItem> *d_buffer;
    cudaMalloc((void **)&d_buffer, sizeof(Buffer<KnapsackItem>));
    cudaMemcpy(d_buffer, &buffer, sizeof(Buffer<KnapsackItem>), cudaMemcpyHostToDevice);	

    size_t smemOffset = sizeof(KnapsackItem) * batchSize * 3 +
                        sizeof(int) * 2 +
                        sizeof(KnapsackItem) +
                        5 * batchSize * sizeof(KnapsackItem);

    KnapsackItem *h_ins_items = new KnapsackItem[2 * batchSize * batchNum]();
    h_ins_items[0] = KnapsackItem(0, 0, -1, 0);
    KnapsackItem *d_ins_items;
    cudaMalloc((void **)&d_ins_items, sizeof(KnapsackItem) * 2 * batchSize * batchNum);
    cudaMemcpy(d_ins_items, h_ins_items, sizeof(KnapsackItem) * 2 * batchSize * batchNum, cudaMemcpyHostToDevice);
    h_ins_items[0] = KnapsackItem();


    KnapsackItem *h_del_items = new KnapsackItem[batchSize * batchNum]();
    KnapsackItem *d_del_items;
    cudaMalloc((void **)&d_del_items, sizeof(KnapsackItem) * batchSize * batchNum);
    cudaMemcpy(d_del_items, h_del_items, sizeof(KnapsackItem) * batchSize * batchNum, cudaMemcpyHostToDevice);

    int h_ins_size = batchSize;
    int *d_ins_size;
    cudaMalloc((void **)&d_ins_size, sizeof(int));
    cudaMemcpy(d_ins_size, &h_ins_size, sizeof(int), cudaMemcpyHostToDevice);

    int h_del_size = batchSize;
    /*int *d_del_size;*/
    /*cudaMalloc((void **)&d_del_size, sizeof(int));*/
    /*cudaMemcpy(d_del_size, &h_del_size, sizeof(int), cudaMemcpyHostToDevice);*/

    psyncHeapInsert<KnapsackItem>
                    (d_heap, d_insTB, d_delTB,
                    d_ins_items, batchSize,
                    blockNum, blockSize, batchSize);
    cudaMemcpy(&heap, d_heap, sizeof(Heap<KnapsackItem>), cudaMemcpyDeviceToHost);
//    heap.printHeap();
    /*cout << "init done\n";*/

    int h_benefit;
    struct timeval startTime, endTime;
	setTime(&startTime);
	
    int h_explored_nodes = 0;
    int counter = 0;
    int previous_benefit = 0;
    int *explored_nodes;
    cudaMalloc((void **)&explored_nodes, sizeof(int));
    cudaMemset(explored_nodes, 0, sizeof(int));

    /*cudaMemcpy(&heap, d_heap, sizeof(Heap<KnapsackItem>), cudaMemcpyDeviceToHost);*/
    /*cout << "init batchCount: " << heap.GetBatchCount() << endl;*/

    int iter = 1;

	while (1) {
        psyncHeapDelete<KnapsackItem>(d_heap, d_insTB, d_delTB,
                        d_del_items, h_del_size,
                        blockNum, blockSize, batchSize);

        /*cudaMemcpy(h_del_items, d_del_items, sizeof(KnapsackItem) * h_del_size, cudaMemcpyDeviceToHost);*/
        /*for (int i = 0; i < h_del_size; i++) {*/
            /*cout << h_del_items[i] << " | ";*/
            /*if (i % 8 == 7) cout << endl;*/
        /*}*/

        cudaMemset(d_ins_size, 0, sizeof(int));

        cudaMemcpy(&heap, d_heap, sizeof(Heap<KnapsackItem>), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_benefit, heap.globalBenefit, sizeof(int), cudaMemcpyDeviceToHost);
        /*cout << "delete check\n";*/
        /*cout << "iter " << iter << " delete: delSize: " << h_del_size << " benefit: " << h_benefit << " " << endl;*/
        heap.checkHeap();
//        heap.printHeap();

        psyncAppKernel<<<blockNum, blockSize>>>
                 (weight, benefit, benefitPerWeight,
                  heap.globalBenefit, inputSize, capacity,
                  d_del_items, h_del_size,
                  d_ins_items, d_ins_size,
                  batchSize);

        cudaMemcpy(&h_ins_size, d_ins_size, sizeof(int), cudaMemcpyDeviceToHost);
        /*cout << "real insSize " << h_ins_size << endl;*/
        h_explored_nodes += h_ins_size;
        h_ins_size = (h_ins_size + batchSize - 1) / batchSize * batchSize;

        /*cudaMemcpy(h_ins_items, d_ins_items, sizeof(KnapsackItem) * h_ins_size, cudaMemcpyDeviceToHost);*/
        /*for (int i = 0; i < h_ins_size; i++) {*/
            /*cout << h_ins_items[i] << " | ";*/
            /*if (i % 8 == 7) cout << endl;*/
        /*}*/
       
        psyncHeapInsert(d_heap, d_insTB, d_delTB,
                        d_ins_items, h_ins_size,
                        blockNum, blockSize, batchSize);


        cudaMemcpy(&h_benefit, heap.globalBenefit, sizeof(int), cudaMemcpyDeviceToHost);

        cudaMemcpy(&heap, d_heap, sizeof(Heap<KnapsackItem>), cudaMemcpyDeviceToHost);
        /*cout << "insert check\n";*/
        heap.checkHeap();
//        heap.printHeap();
        /*cout << "iter " << iter << " insert: insSize: " << h_ins_size << " benefit: " << h_benefit << " " << endl;*/

        cudaMemcpy(d_ins_items, h_ins_items, sizeof(KnapsackItem) * 2 * batchSize * batchNum, cudaMemcpyHostToDevice);
        cudaMemcpy(&heap, d_heap, sizeof(Heap<KnapsackItem>), cudaMemcpyDeviceToHost);
        if (heap.batchCount == 0) break;
        h_del_size = heap.batchCount <  blockNum ? heap.batchCount * batchSize : blockNum * batchSize;

        cudaMemcpy(&h_benefit, heap.globalBenefit, sizeof(int), cudaMemcpyDeviceToHost);
        if (previous_benefit == h_benefit) counter++;
        else counter = 0;
//        if (counter == 5) break;
        if (h_benefit == global_max_benefit) break;
        previous_benefit = h_benefit;
        iter++;
    }

    setTime(&endTime);

    cout << "heap time: " << getTime(&startTime, &endTime) << " "
         << "explored nodes: " << h_explored_nodes << " "
         << "benefit: " << h_benefit << endl;

    setTime(&startTime);
    if (heap.batchCount != 0) {
        cudaMemcpy(&heap, d_heap, sizeof(Heap<KnapsackItem>), cudaMemcpyDeviceToHost);
        unsigned long long int remain_item_number = heap.batchCount * batchSize;
        cout << "remain items: " << remain_item_number << endl;

        heapDataToArray<KnapsackItem>(heap, buffer.bufferItems, remain_item_number);

        cudaMemcpy(buffer.writePos, &remain_item_number, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
        cudaMemcpy(buffer.endPos, &remain_item_number, sizeof(unsigned long long int), cudaMemcpyHostToDevice);

        unsigned long long int *activeCount;
        cudaMalloc((void **)&activeCount, sizeof(unsigned long long int));
        cudaMemcpy(activeCount, &remain_item_number, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
        cudaMemcpy(explored_nodes, &h_explored_nodes, sizeof(int), cudaMemcpyHostToDevice);
        blockNum = 32;
        oneBufferApplication<<<blockNum, blockSize, smemOffset>>>(d_buffer, batchSize, 
                                                         weight, benefit, benefitPerWeight,
                                                         capacity, inputSize,
                                                         heap.globalBenefit, activeCount,
                                                         explored_nodes);
        cudaDeviceSynchronize();
        cudaFree(activeCount); activeCount = NULL;
    }
    setTime(&endTime);

    cudaMemcpy(&h_explored_nodes, explored_nodes, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_benefit, heap.globalBenefit, sizeof(int), cudaMemcpyDeviceToHost);

    cout << "buffer time: " << getTime(&startTime, &endTime) << " " 
         << "explored nodes: " << h_explored_nodes << " "
         << h_benefit << endl;

    
    
    cudaFree(d_heap); d_heap = NULL;
    cudaFree(d_buffer); d_buffer = NULL;
    cudaFree(d_insTB); d_insTB = NULL;
    cudaFree(d_delTB); d_delTB = NULL;
    cudaFree(d_ins_items); d_ins_items = NULL;
    cudaFree(d_del_items); d_del_items = NULL;
    cudaFree(d_ins_size); d_ins_size = NULL;
//    cudaFree(d_del_size); d_del_size = NULL;
}

#endif
