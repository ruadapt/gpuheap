#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <time.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "util.hpp"
#include "heap.cuh"

using namespace std;

int main(int argc, char *argv[]) {

    if (argc != 7) {
        cout << "./test <arrayNum> <batchSize> <batchNum> <tableSize> <blockNum> <blockSize>\n";
        return -1;
    }

    srand(time(NULL));

    int arrayNum = atoi(argv[1]);

    int batchSize = atoi(argv[2]);
    int batchNum = atoi(argv[3]);
    int tableSize = atoi(argv[4]);

    int blockNum = atoi(argv[5]);
    int blockSize = atoi(argv[6]);

    struct timeval startTime;
    struct timeval endTime;
    double insertTime = 0, deleteTime = 0;

    // generate <keys, vals> sequence
    int *oriKeys = new int[2 * arrayNum];
    int *oriVals = new int[2 * arrayNum];
    for (int i = 0; i < 2 * arrayNum; ++i) {
        oriKeys[i] = rand() % INT_MAX;
        oriVals[i] = i;
    }

    // barrier heap sort
	/* Prepare Heap */
    Heap<int, int> h_heap(batchNum, batchSize, tableSize);
    Heap<int, int> *d_heap;
	cudaMalloc((void **)&d_heap, sizeof(Heap<int, int>));
    cudaMemcpy(d_heap, &h_heap, sizeof(Heap<int, int>), cudaMemcpyHostToDevice);
	
	/* Prepare data array */
	int *d_Keys, *d_Vals;
	cudaMalloc((void **)&d_Keys, sizeof(int) * 2 * arrayNum);
	cudaMalloc((void **)&d_Vals, sizeof(int) * 2 * arrayNum);
    cudaMemcpy(d_Keys, oriKeys, sizeof(int) * 2 * arrayNum, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vals, oriVals, sizeof(int) * 2 * arrayNum, cudaMemcpyHostToDevice);
	
	/* Prepare ins/del table buffer */
	TB h_insTB(tableSize, batchSize, 0);
	TB *d_insTB;
	cudaMalloc((void **)&d_insTB, sizeof(TB));
	cudaMemcpy(d_insTB, &h_insTB, sizeof(TB), cudaMemcpyHostToDevice);
	
    TB h_delTB(tableSize, batchSize, 1);
	TB *d_delTB;
	cudaMalloc((void **)&d_delTB, sizeof(TB));
	cudaMemcpy(d_delTB, &h_delTB, sizeof(TB), cudaMemcpyHostToDevice);

    int smemSize = 10 * batchSize * sizeof(int);

    // Insert Items
    for (int i = 0; i < arrayNum / batchSize; i++) {
        insertItems<<<1, blockSize, smemSize>>>(d_heap, 
                                             d_Keys + i * batchSize, 
                                             d_Vals + i * batchSize,
											 d_insTB);
        cudaDeviceSynchronize();

        // Even level insert update
        insertUpdate<<<blockNum, blockSize, smemSize>>>(d_heap,
														d_insTB);
        cudaDeviceSynchronize();

        updateTableBuffer<<<1, 1>>>(d_heap, d_insTB, d_delTB);
        cudaDeviceSynchronize();

        // Odd level insert update
        insertUpdate<<<blockNum, blockSize, smemSize>>>(d_heap,
														d_insTB);
        cudaDeviceSynchronize();
		
		updateTableBuffer<<<1, 1>>>(d_heap, d_insTB, d_delTB);
        cudaDeviceSynchronize();
    }

    setTime(&startTime);

    // Insert Items
    for (int i = 0; i < arrayNum / batchSize; i++) {
        insertItems<<<1, blockSize, smemSize>>>(d_heap, 
                                             d_Keys + arrayNum + i * batchSize, 
                                             d_Vals + arrayNum + i * batchSize,
											 d_insTB);
        cudaDeviceSynchronize();

        // Even level insert update
        insertUpdate<<<blockNum, blockSize, smemSize>>>(d_heap,
														d_insTB);
        cudaDeviceSynchronize();

        updateTableBuffer<<<1, 1>>>(d_heap, d_insTB, d_delTB);
        cudaDeviceSynchronize();

        // Odd level insert update
        insertUpdate<<<blockNum, blockSize, smemSize>>>(d_heap,
														d_insTB);
        cudaDeviceSynchronize();
		
		updateTableBuffer<<<1, 1>>>(d_heap, d_insTB, d_delTB);
        cudaDeviceSynchronize();
    }

    setTime(&endTime);
    insertTime += getTime(&startTime, &endTime);
    cout << insertTime << " ";
    setTime(&startTime);

    // Delete Items
	for (int i = 0; i < arrayNum / batchSize; i++) {
        deleteItems<<<1, blockSize>>>(d_heap, 
									  d_Keys + i * batchSize, 
									  d_Vals + i * batchSize,
									  d_insTB,
									  d_delTB);
        cudaDeviceSynchronize();

        // Even level delete update
        deleteUpdate<<<blockNum, blockSize, smemSize>>>(d_heap,
														d_delTB);
        cudaDeviceSynchronize();

        updateTableBuffer<<<1, 1>>>(d_heap, d_insTB, d_delTB);
        cudaDeviceSynchronize();
        // Odd level insert update
        deleteUpdate<<<blockNum, blockSize, smemSize>>>(d_heap,
														d_delTB);
        cudaDeviceSynchronize();
		
		updateTableBuffer<<<1, 1>>>(d_heap, d_insTB, d_delTB);
        cudaDeviceSynchronize();
    }
	
    setTime(&endTime);
    deleteTime = getTime(&startTime, &endTime);
    cout << deleteTime << " " << insertTime + deleteTime << " " << endl;
	
    return 0;

}
