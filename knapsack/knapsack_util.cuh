#ifndef KNAPSACK_UTIL_CUH
#define KNAPSACK_UTIL_CUH

#include "heap.cuh"

template <typename K = int>
void heapDataToArray(Heap<K> heap, K *array, unsigned long long int &number) {
    int pSize = 0, bCount = 0;
    int bSize = heap.batchSize;
    cudaMemcpy(&pSize, heap.partialBufferSize, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&bCount, heap.batchCount, sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(array, heap.heapItems, sizeof(K) * pSize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(array + pSize, heap.heapItems + bSize, sizeof(K) * bCount * bSize, cudaMemcpyDeviceToDevice);
    number = pSize + bCount * bSize;
}

#endif
