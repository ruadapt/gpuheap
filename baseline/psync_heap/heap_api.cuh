#ifndef HEAP_API_CUH
#define HEAP_API_CUH

#include "heap.cuh"

template <typename K> 
void psyncHeapInsert(Heap<K> *heap, TB<K> *insTB, TB<K> *delTB,
                     K *keys, int insert_num,
                     int blockNum, int blockSize,
                     int batchSize) {

    bool h_status = false;
    bool *d_status;
    cudaMalloc((void **)&d_status, sizeof(bool));
    cudaMemcpy(d_status, &h_status, sizeof(bool), cudaMemcpyHostToDevice);

    int smemSize = 10 * batchSize * sizeof(K);
    for (int i = 0; i < insert_num / batchSize; i++) {
        insertItems<K><<<1, blockSize, smemSize>>>(heap, 
                                                keys + i * batchSize,
                                                insTB);
        cudaDeviceSynchronize();

        insertUpdate<K><<<blockNum, blockSize, smemSize>>>(heap, insTB);
        cudaDeviceSynchronize();

        updateTableBuffer<K><<<1, 1>>>(heap, insTB, delTB, d_status);
        cudaDeviceSynchronize();

        insertUpdate<K><<<blockNum, blockSize, smemSize>>>(heap, insTB);
        cudaDeviceSynchronize();

        updateTableBuffer<K><<<1, 1>>>(heap, insTB, delTB, d_status);
        cudaDeviceSynchronize();
    }

    while (1) {
        cudaMemcpy(&h_status, d_status, sizeof(bool), cudaMemcpyDeviceToHost);
        if (h_status) break;

        insertUpdate<K><<<blockNum, blockSize, smemSize>>>(heap, insTB);
        cudaDeviceSynchronize();

        updateTableBuffer<K><<<1, 1>>>(heap, insTB, delTB, d_status);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_status, d_status, sizeof(bool), cudaMemcpyDeviceToHost);
        if (h_status) break;

        insertUpdate<K><<<blockNum, blockSize, smemSize>>>(heap, insTB);
        cudaDeviceSynchronize();

        updateTableBuffer<K><<<1, 1>>>(heap, insTB, delTB, d_status);
        cudaDeviceSynchronize();
    }
}


template <typename K> 
void psyncHeapDelete(Heap<K> *heap, TB<K> *insTB, TB<K> *delTB,
                     K *keys, int delete_num,
                     int blockNum, int blockSize,
                     int batchSize) {

    bool h_status = false;
    bool *d_status;
    cudaMalloc((void **)&d_status, sizeof(bool));
    cudaMemcpy(d_status, &h_status, sizeof(bool), cudaMemcpyHostToDevice);

    int smemSize = 10 * batchSize * sizeof(K);
    for (int i = 0; i < delete_num / batchSize; i++) {
        deleteItems<K><<<1, blockSize, smemSize>>>(heap, 
                                                keys + i * batchSize,
                                                insTB, delTB);
        cudaDeviceSynchronize();

        deleteUpdate<K><<<blockNum, blockSize, smemSize>>>(heap, delTB);
        cudaDeviceSynchronize();

        updateTableBuffer<K><<<1, 1>>>(heap, insTB, delTB, d_status);
        cudaDeviceSynchronize();

        deleteUpdate<K><<<blockNum, blockSize, smemSize>>>(heap, delTB);
        cudaDeviceSynchronize();

        updateTableBuffer<K><<<1, 1>>>(heap, insTB, delTB, d_status);
        cudaDeviceSynchronize();
    }

    while (1) {
        cudaMemcpy(&h_status, d_status, sizeof(bool), cudaMemcpyDeviceToHost);
        if (h_status) break;

        deleteUpdate<K><<<blockNum, blockSize, smemSize>>>(heap, delTB);
        cudaDeviceSynchronize();

        updateTableBuffer<K><<<1, 1>>>(heap, insTB, delTB, d_status);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_status, d_status, sizeof(bool), cudaMemcpyDeviceToHost);
        if (h_status) break;

        deleteUpdate<K><<<blockNum, blockSize, smemSize>>>(heap, delTB);
        cudaDeviceSynchronize();

        updateTableBuffer<K><<<1, 1>>>(heap, insTB, delTB, d_status);
        cudaDeviceSynchronize();
    }
}

#endif
