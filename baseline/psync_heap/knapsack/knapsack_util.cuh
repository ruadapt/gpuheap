#ifndef KNAPSACK_UTIL_CUH
#define KNAPSACK_UTIL_CUH

#include "heap.cuh"

template <typename K = int>
void heapDataToArray(Heap<K> &heap, K *array, unsigned long long int &number) {
    int bCount = heap.batchCount;
    int bSize = heap.batchSize;

    /*K *tmp = new K[bSize * (bCount + 1)];*/
    /*for (int i = 0; i < bSize * (bCount + 1); i++) {*/
        /*tmp[i] = INIT_LIMITS;*/
    /*}*/
    /*cudaMemcpy(heap.heapItems, tmp, sizeof(K) * bSize * (bCount + 1), cudaMemcpyHostToDevice);*/
    /*delete []tmp;*/
    /*cudaMemset(heap.status, AVAIL, sizeof(int) * (bCount + 1));*/
    /*cudaMemset(heap.batchCount, 0, sizeof(int));*/
    /*cudaMemset(heap.partialBufferSize, 0, sizeof(int));*/

    cudaMemcpy(array, heap.heapKeys, sizeof(K) * bCount * bSize, cudaMemcpyDeviceToDevice);
    number = bCount * bSize;
}

#endif
