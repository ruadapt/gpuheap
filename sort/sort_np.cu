#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <time.h>

#ifdef THRUST_SORT
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#endif

#include <heap.cuh>
#include "util.hpp"

using namespace std;

__global__ void concurrentKernel(Heap<int> *heap, int *items, int arraySize, int batchSize) {

    if (blockIdx.x < gridDim.x / 2) {
        int batchNeed = (arraySize + batchSize - 1) / batchSize;
        // insertion
        for (int i = blockIdx.x; i < batchNeed; i += gridDim.x / 2) {
            int size = min(batchSize, arraySize - i * batchSize);

            // insert items to buffer
            heap->insertion(items + arraySize + i * batchSize,
                            size, 0);
            __syncthreads();
        }
    }
    else {
        int batchNeed = (arraySize + batchSize - 1) / batchSize;
        int size = 0;
        // deletion
        for (int i = blockIdx.x - gridDim.x / 2; i < batchNeed; i += gridDim.x / 2) {
            // delete items from heap
            if (heap->deleteRoot(items, size) == true) {
                __syncthreads();

                heap->deleteUpdate(0);
            }
            __syncthreads();
        }
    }
}



__global__ void insertKernel(Heap<int> *heap, 
                             int *items, 
                             int arraySize, 
                             int batchSize) {

//    batchSize /= 3;
    int batchNeed = (arraySize + batchSize - 1) / batchSize;
    // insertion
    for (int i = blockIdx.x; i < batchNeed; i += gridDim.x) {
        int size = min(batchSize, arraySize - i * batchSize);

        // insert items to buffer
        heap->insertion(items + i * batchSize,
                        size, 0);
        __syncthreads();
    }
}

__global__ void deleteKernel(Heap<int> *heap, int *items, int arraySize, int batchSize) {

    int batchNeed = (arraySize + batchSize - 1) / batchSize;
    int size = 0;
    // deletion
    for (int i = blockIdx.x; i < batchNeed; i += gridDim.x) {

        // delete items from heap
        if (heap->deleteRoot(items, size) == true) {
            __syncthreads();

            heap->deleteUpdate(0);
        }
        __syncthreads();
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__);  }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


int main(int argc, char *argv[]) {

    if (argc != 7) {
        cout << "./sortCP [arrayNum] [numLength] [batchSize] [batchNum] [blockNum] [blockSize]\n";
        return -1;
    }

    srand(time(NULL));

    int arrayNum = atoi(argv[1]);
    int numLength = atoi(argv[2]) - 1;

    int batchSize = atoi(argv[3]);
    int batchNum = atoi(argv[4]);

    int blockNum = atoi(argv[5]);
    int blockSize = atoi(argv[6]);

    struct timeval startTime;
    struct timeval endTime;

    int *oriItems = new int[2 * arrayNum];
    int beginNum = pow(10, numLength);
    for (int i = 0; i < arrayNum; ++i) {
        oriItems[i] = rand() % (5 * beginNum);
    }
    for (int i = 0; i < arrayNum; ++i) {
        oriItems[arrayNum + i] = rand() % (5 * beginNum) + 5 * beginNum;
    }
#ifdef THRUST_SORT
    // thrust sort
    int *thrustItems;

    cudaMalloc((void **)&thrustItems, sizeof(int) * 2 * arrayNum);
    cudaMemcpy(thrustItems, oriItems, sizeof(int) * 2 * arrayNum, cudaMemcpyHostToDevice);

    thrust::device_ptr<int> items_ptr(thrustItems);
    
    setTime(&startTime);
    
    thrust::sort(thrust::device, items_ptr, items_ptr + 2 * arrayNum);
    cudaDeviceSynchronize();

    setTime(&endTime);
    cout << "thrust time: " << getTime(&startTime, &endTime) << "ms\n";

    int *h_tItems = new int[2 * arrayNum];

    cudaMemcpy(h_tItems, thrustItems, sizeof(int) * 2 * arrayNum, cudaMemcpyDeviceToHost);
#endif
/*
    cout << "key: ";
    for (int i = 0; i < arrayNum; ++i) {
        cout << h_tItems[i] << " ";
    }
    cout << endl;
*/
    // bitonic heap sort
    Heap<int> h_heap(batchNum, batchSize);

    int *heapItems;
    Heap<int> *d_heap;

    cudaMalloc((void **)&heapItems, sizeof(int) * 2 * arrayNum);
    cudaMemcpy(heapItems, oriItems, sizeof(int) * 2 * arrayNum, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_heap, sizeof(Heap<int>));
    cudaMemcpy(d_heap, &h_heap, sizeof(Heap<int>), cudaMemcpyHostToDevice);

    int smemSize = batchSize * 3 * sizeof(int);
    smemSize += (blockSize + 1) * sizeof(int) + 2 * batchSize * sizeof(int);

    cout << "start:\n";
    double insertTime = 0, deleteTime = 0, concurrentTime = 0;

#ifdef INSERT_TEST
    // concurrent insertion
    setTime(&startTime);

    insertKernel<<<blockNum, blockSize, smemSize>>>(d_heap, heapItems, arrayNum, batchSize);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    setTime(&endTime);
    insertTime = getTime(&startTime, &endTime);
    cout << "heap insert time: " << insertTime << "ms" << endl;

    cudaMemcpy(&h_heap, d_heap, sizeof(Heap<int>), cudaMemcpyDeviceToHost);
    if (h_heap.checkInsertHeap()) cout << "Insert Result Correct\n";
    else cout << "Insert Result Wrong\n";
//    h_heap.printHeap();
#endif
#ifdef CONCURRENT_TEST
    // concurrent insertion and deletion
    setTime(&startTime);

    concurrentKernel<<<blockNum, blockSize, smemSize>>>(d_heap, heapItems, arrayNum, batchSize);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    setTime(&endTime);
    concurrentTime = getTime(&startTime, &endTime);
    cout << "heap concurrent insertion and deletion time: " << concurrentTime << "ms" << endl;
    cudaMemcpy(&h_heap, d_heap, sizeof(Heap<int>), cudaMemcpyDeviceToHost);
    if (h_heap.checkInsertHeap()) cout << "Concurrent Insert and Delete Result Correct\n";
    else cout << "Concurrent Insert and Delete Result Wrong\n";
//    h_heap.printHeap();
#endif
#ifdef DELETE_TEST
    // concurrent deletion
    setTime(&startTime);

    deleteKernel<<<blockNum, blockSize, smemSize>>>(d_heap, heapItems, arrayNum, batchSize);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    setTime(&endTime);
    deleteTime = getTime(&startTime, &endTime);
    cout << "heap deletion time: " << deleteTime << "ms" << endl;

    cout << "heap time: " << insertTime + concurrentTime + deleteTime << "ms" << endl;

    cudaMemcpy(oriItems, heapItems, sizeof(int) * 2 * arrayNum, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_heap, d_heap, sizeof(Heap<int>), cudaMemcpyDeviceToHost);
    if (h_heap.checkInsertHeap()) cout << "Concurrent Delete Result Correct\n";
    else cout << "Concurrent Delete Result Wrong\n";
//    h_heap.printHeap();
#endif
    cout << "delete result\n";

#ifdef THRUST_SORT
    for (int i = 0; i < 2 * arrayNum; ++i) {
        if (h_tItems[i] != oriItems[i]) {
            printf("Wrong Answer! id: %d thrust %d heap %d\n", i, h_tItems[i], oriItems[i]);
            return -1;
        }
    }
#else
    for (int i = 0; i < arrayNum; ++i) {
        if (oriItems[i] < oriItems[i - 1]) {
            printf("Wrong Answer! items[%d]:%d, items[%d]:%d\n", 
                    i, oriItems[i], i - 1, oriItems[i - 1]);
            return -1;
        }
    }

#endif

    printf("Correct!\n");
    return 0;

}
