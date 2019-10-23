#ifndef MODELS2H_CUH
#define MODELS2H_CUH

#include <functional>
#include <iostream>
#include "util.cuh"
#include "heap.cuh"
#include "datastructure.hpp"

#include "gc.cuh"
#include "expand.cuh"
#include "end.cuh"

using namespace std;

__global__ void init(Heap *heap) {
    uint128 a(0, 0, -1, 0);
    heap->insertion(&a, 1, 0);
}

void twoheap(int *weight, int *benefit, float *benefitPerWeight,
             int *max_benefit, int capacity, int inputSize, 
             int delAllowed, int gcThreshold, int expandThreshold, int endingBlockNum,
             int batchNum, int batchSize, int blockNum, int blockSize)
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


    int h_benefit = 0;
    int prev_benefit = -1;
    int benefitCtr = 0;
    int expandFlag = 0;
    int endFlag = 0;

    struct timeval expandStartTime, expandEndTime;
    struct timeval gcStartTime, gcEndTime;
    struct timeval endingStartTime, endingEndTime;
    struct timeval startTime, endTime;
    double expandTime = 0, gcTime = 0, endingTime = 0, totalTime = 0;

    int expandIter = 0, gcIter = 0, endingIter = 0;

    setTime(&startTime);

    while (!endFlag) {

        // expand stage
        // TODO can we not use expandFlag? in other word can we switch two heap's data? both device and host
        setTime(&expandStartTime);
        expandIter++;
        cout << "expand...";
        heap1.printHeap();
        heap2.printHeap();
        expand2Heap(d_heap1, d_heap2, batchSize,
                    blockNum, blockSize, smemOffset,
                    weight, benefit, benefitPerWeight,
                    capacity, inputSize, delAllowed,
                    max_benefit, expandFlag);
        setTime(&expandEndTime);
        expandTime += getTime(&expandStartTime, &expandEndTime);

        heap1.printHeap();
        heap2.printHeap();

        // do gc if heap size is too large
        setTime(&gcStartTime);
        gcIter++;
        invalidFilter2Heap(heap1, heap2, d_heap1, d_heap2, batchSize,
                           weight, benefit, benefitPerWeight,
                           capacity, inputSize, max_benefit, 
                           expandFlag, gcThreshold, 1024000);
        setTime(&gcEndTime);
        gcTime += getTime(&gcStartTime, &gcEndTime);
        heap1.printHeap();
        cudaMemcpy(&h_benefit, max_benefit, sizeof(int), cudaMemcpyDeviceToHost);
        // check if we should go to the ending stage
        if (h_benefit != prev_benefit) {
            benefitCtr = 0;
            prev_benefit = h_benefit;
        }
        else if (h_benefit == prev_benefit && ++benefitCtr == expandThreshold) {
            setTime(&endingStartTime);
            endingIter++;
            cout << "ending...";
            afterExpandStage(heap1, heap2, batchSize,
                             d_heap1, d_heap2,
                             weight, benefit, benefitPerWeight,
                             capacity, inputSize, endingBlockNum,
                             max_benefit, endFlag);
            setTime(&endingEndTime);
            endingTime += getTime(&endingStartTime, &endingEndTime);

            if (endFlag == 0) {
                // max benefit is changed, go to expand stage
                benefitCtr = 0;
                prev_benefit = h_benefit;
                expandFlag = 0;
            }
        }
        cout << endl;
    }
    setTime(&endTime);
    totalTime = getTime(&startTime, &endTime);

    cudaMemcpy(&h_benefit, max_benefit, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d %.4f\n%d %.4f\n%d %.4f\n%.4f %d\n", 
            expandTime, expandIter, gcTime, gcIter, endingTime, endingIter,
            totalTime, h_benefit);

    cudaFree(d_heap1); d_heap1 = NULL;
    cudaFree(d_heap2); d_heap2 = NULL;
}

#endif
