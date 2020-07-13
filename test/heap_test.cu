#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <time.h>
#include <algorithm>
#include <cmath>

#include "heap.cuh"
#include "util.hpp"

using namespace std;

template <class T>
__global__ void concurrentKernel(Heap<T> *heap, T *items, 
    uint32_t arraySize, uint32_t batchSize) {
    uint32_t batchNeed = arraySize / batchSize;
    if (blockIdx.x < gridDim.x / 2) {
        // insertion
        for (uint32_t i = blockIdx.x; i < batchNeed; i += gridDim.x / 2) {
            heap->insertion(items + arraySize + i * batchSize,
                            batchSize, 0);
            __syncthreads();
        }
    }
    else {
        int size = 0;
        // deletion
        for (uint32_t i = blockIdx.x - gridDim.x / 2; i < batchNeed; i += gridDim.x / 2) {
            // delete items from heap
            if (heap->deleteRoot(items, size) == true) {
                __syncthreads();

                heap->deleteUpdate(0);
            }
            __syncthreads();
        }
    }
}


template <class T>
__global__ void insertKernel(Heap<T> *heap, 
                             T *items, 
                             uint32_t arraySize, 
                             uint32_t batchSize) {

    uint32_t batchNeed = arraySize / batchSize;
    // insertion
    for (uint32_t i = blockIdx.x; i < batchNeed; i += gridDim.x) {
        // insert items to buffer
        heap->insertion(items + i * batchSize,
                        batchSize, 0);
        __syncthreads();
    }
}

template <class T>
__global__ void deleteKernel(Heap<T> *heap, T *items, 
    uint32_t arraySize, uint32_t batchSize) {

    uint32_t batchNeed = arraySize / batchSize;
    int size = 0;
    // deletion
    for (uint32_t i = blockIdx.x; i < batchNeed; i += gridDim.x) {

        // delete items from heap
        if (heap->deleteRoot(items, size) == true) {
            __syncthreads();

            heap->deleteUpdate(0);
        }
        __syncthreads();
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__);  }
inline void gpuAssert(cudaError_t code, const char *file, uint32_t line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


int main(int argc, char *argv[]) {

    if (argc != 3) {
        cout << argv[0] << " [batch size] [array num]\n";
        return -1;
    }

    srand(time(NULL));

    uint32_t batchSize = atoi(argv[1]);
    uint32_t arrayNum = pow(2, atoi(argv[2]));
    arrayNum = (arrayNum + batchSize - 1) / batchSize * batchSize;
    uint32_t batchNum = 1;
    while (batchNum * batchSize < arrayNum * 2) {
        batchNum <<= 1;
    }
    uint32_t blockNum = 32;
    uint32_t blockSize = batchSize / 2;

    printf("%s test size: 2^%u/%u heap[%u/%u] kernel[%u/%u]\n", argv[0], atoi(argv[2]), arrayNum, batchSize, batchNum, blockNum, blockSize);

    struct timeval startTime;
    struct timeval endTime;

    uint32_t *oriItems = new uint32_t[arrayNum]();
    uint32_t *oriItems_ = new uint32_t[arrayNum]();
    for (uint32_t i = 0; i < arrayNum / 2; ++i) {
        oriItems_[i] = oriItems[i] = rand() % (INT_MAX / 2);
    }
    for (uint32_t i = arrayNum / 2; i < arrayNum; ++i) {
        oriItems_[i] = oriItems[i] = INT_MAX / 2 + rand() % (INT_MAX / 2);
    }
    // sort original solutions
    std::sort(oriItems_, oriItems_ + arrayNum);

    // bitonic heap sort
    Heap<uint32_t> h_heap(batchNum, batchSize, UINT32_MAX);

    uint32_t *heapItems;
    Heap<uint32_t> *d_heap;

    cudaMalloc((void **)&heapItems, sizeof(uint32_t) * arrayNum);
    cudaMemcpy(heapItems, oriItems, sizeof(uint32_t) * arrayNum, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_heap, sizeof(Heap<uint32_t>));
    cudaMemcpy(d_heap, &h_heap, sizeof(Heap<uint32_t>), cudaMemcpyHostToDevice);

    uint32_t smemSize = batchSize * 3 * sizeof(uint32_t);
    smemSize += (blockSize + 1) * sizeof(uint32_t) + 2 * batchSize * sizeof(uint32_t);

    // concurrent insertion
    setTime(&startTime);

    insertKernel<uint32_t><<<blockNum, blockSize, smemSize>>>(d_heap, heapItems, arrayNum / 2, batchSize);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    setTime(&endTime);
    double insertTime = getTime(&startTime, &endTime);

    // concurrent insertion and deletion
    setTime(&startTime);
 
    concurrentKernel<uint32_t><<<blockNum, blockSize, smemSize>>>(d_heap, heapItems, arrayNum / 2, batchSize);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    setTime(&endTime);
    double concurrentTime = getTime(&startTime, &endTime);

    // concurrent deletion
    setTime(&startTime);

    deleteKernel<uint32_t><<<blockNum, blockSize, smemSize>>>(d_heap, heapItems, arrayNum / 2, batchSize);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    setTime(&endTime);
    double deleteTime = getTime(&startTime, &endTime);
    cudaMemcpy(oriItems, heapItems, sizeof(uint32_t) * arrayNum, cudaMemcpyDeviceToHost);

    for (uint32_t i = 0; i < arrayNum; ++i) {
        if (oriItems_[i] != oriItems[i]) {
            printf("Wrong Answer! id: %d correct %u heap %u\n", i, oriItems_[i], oriItems[i]);
            return -1;
        }
    }
    printf("Success %.f %.f %.f %.f\n", insertTime, concurrentTime, deleteTime, insertTime + concurrentTime + deleteTime);
    return 0;
}
