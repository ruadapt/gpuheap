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

    if (argc != 8) {
        cout << "./sort <arrayNum> <numLength> <batchSize> <batchNum> <tableSize> <blockNum> <blockSize>\n";
        return -1;
    }

    srand(time(NULL));

    int arrayNum = atoi(argv[1]);
    int numLength = atoi(argv[2]);

    int batchSize = atoi(argv[3]);
    int batchNum = atoi(argv[4]);
    int tableSize = atoi(argv[5]);

    int blockNum = atoi(argv[6]);
    int blockSize = atoi(argv[7]);

    struct timeval startTime;
    struct timeval endTime;

    // generate <keys, vals> sequence
    int *oriKeys = new int[arrayNum];
    int *oriVals = new int[arrayNum];
    int beginNum = pow(10, numLength);
    for (int i = 0; i < arrayNum; ++i) {
        oriKeys[i] = rand() % (9 * beginNum) + beginNum;
        oriVals[i] = i;
    }

    // thrust sort
    int *thrustKeys;
    int *thrustVals;

    cudaMalloc((void **)&thrustKeys, sizeof(int) * arrayNum);
    cudaMemcpy(thrustKeys, oriKeys, sizeof(int) * arrayNum, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&thrustVals, sizeof(int) * arrayNum);
    cudaMemcpy(thrustVals, oriVals, sizeof(int) * arrayNum, cudaMemcpyHostToDevice);

    thrust::device_ptr<int> key_ptr(thrustKeys);
    thrust::device_ptr<int> val_ptr(thrustVals);
    
    setTime(&startTime);
    
    thrust::sort_by_key(thrust::device, key_ptr, key_ptr + arrayNum, val_ptr);
    cudaDeviceSynchronize();

    setTime(&endTime);
    cout << "thrust time: " << getTime(&startTime, &endTime) << "ms\n";

    int *h_tKeys = new int[arrayNum];
    int *h_tVals = new int[arrayNum];

    cudaMemcpy(h_tKeys, thrustKeys, sizeof(int) * arrayNum, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tVals, thrustVals, sizeof(int) * arrayNum, cudaMemcpyDeviceToHost);

    // bitonic heap sort
	/* Prepare Heap */
    Heap<int, int> h_heap(batchNum, batchSize, tableSize);
    Heap<int, int> *d_heap;
	cudaMalloc((void **)&d_heap, sizeof(Heap<int, int>));
    cudaMemcpy(d_heap, &h_heap, sizeof(Heap<int, int>), cudaMemcpyHostToDevice);
	
	/* Prepare data array */
	int *d_Keys, *d_Vals;
	cudaMalloc((void **)&d_Keys, sizeof(int) * arrayNum);
	cudaMalloc((void **)&d_Vals, sizeof(int) * arrayNum);
    cudaMemcpy(d_Keys, oriKeys, sizeof(int) * arrayNum, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vals, oriVals, sizeof(int) * arrayNum, cudaMemcpyHostToDevice);
	
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

    setTime(&startTime);

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

        updateTableBuffer<int, int><<<1, 1>>>(d_heap, d_insTB, d_delTB);
        cudaDeviceSynchronize();

#ifdef DEBUG		
 		// Result check
		cudaMemcpy(&h_heap, d_heap, sizeof(Heap<int, int>), cudaMemcpyDeviceToHost);
		h_heap.printHeap();

		cudaMemcpy(&h_insTB, d_insTB, sizeof(TB), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_delTB, d_delTB, sizeof(TB), cudaMemcpyDeviceToHost);
		h_insTB.printTB();
		h_delTB.printTB();
		cout << "--------------------------------------------" << endl;
#endif

        // Odd level insert update
        insertUpdate<<<blockNum, blockSize, smemSize>>>(d_heap,
														d_insTB);
        cudaDeviceSynchronize();
		
		updateTableBuffer<int, int><<<1, 1>>>(d_heap, d_insTB, d_delTB);
        cudaDeviceSynchronize();

#ifdef DEBUG		
 		// Result check
		cudaMemcpy(&h_heap, d_heap, sizeof(Heap<int, int>), cudaMemcpyDeviceToHost);
		h_heap.printHeap();

		cudaMemcpy(&h_insTB, d_insTB, sizeof(TB), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_delTB, d_delTB, sizeof(TB), cudaMemcpyDeviceToHost);
		h_insTB.printTB();
		h_delTB.printTB();
		cout << "--------------------------------------------" << endl;
		cout << "--------------------------------------------" << endl;
#endif
    }

    setTime(&endTime);
    cout << "barrier heap insert time: " << getTime(&startTime, &endTime) << "ms\n";
    setTime(&startTime);

#ifdef DEBUG
    cout << "\n###################################################\n";
#endif
	// Delete Items
	for (int i = 0; i < arrayNum / batchSize; i++) {
        deleteItems<<<1, blockSize>>>(d_heap, 
									  d_Keys + i * batchSize, 
									  d_Vals + i * batchSize,
									  d_insTB,
									  d_delTB);
        cudaDeviceSynchronize();

#ifdef DEBUG
        cudaMemcpy(oriKeys + i * batchSize, d_Keys + i * batchSize, sizeof(int) * batchSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(oriVals + i * batchSize, d_Vals + i * batchSize, sizeof(int) * batchSize, cudaMemcpyDeviceToHost);
        for (int j = 0; j < batchSize; j++) {
            cout << oriKeys[i * batchSize + j] << " " << oriVals[i * batchSize + j] << " | ";
        }
        cout << endl;
#endif

        // Even level delete update
        deleteUpdate<<<blockNum, blockSize, smemSize>>>(d_heap,
														d_delTB);
        cudaDeviceSynchronize();

        updateTableBuffer<int, int><<<1, 1>>>(d_heap, d_insTB, d_delTB);
        cudaDeviceSynchronize();
#ifdef DEBUG		
		// Result check
		cudaMemcpy(&h_heap, d_heap, sizeof(Heap<int, int>), cudaMemcpyDeviceToHost);
		h_heap.printHeap();

		cudaMemcpy(&h_insTB, d_insTB, sizeof(TB), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_delTB, d_delTB, sizeof(TB), cudaMemcpyDeviceToHost);
		h_insTB.printTB();
		h_delTB.printTB();
		cout << "--------------------------------------------" << endl;
#endif
        // Odd level insert update
        deleteUpdate<<<blockNum, blockSize, smemSize>>>(d_heap,
														d_delTB);
        cudaDeviceSynchronize();
		
		updateTableBuffer<int, int><<<1, 1>>>(d_heap, d_insTB, d_delTB);
        cudaDeviceSynchronize();
#ifdef DEBUG		
		// Result check
		cudaMemcpy(&h_heap, d_heap, sizeof(Heap<int, int>), cudaMemcpyDeviceToHost);
		h_heap.printHeap();

		cudaMemcpy(&h_insTB, d_insTB, sizeof(TB), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_delTB, d_delTB, sizeof(TB), cudaMemcpyDeviceToHost);
		h_insTB.printTB();
		h_delTB.printTB();
		cout << "--------------------------------------------" << endl;
		cout << "--------------------------------------------" << endl;
#endif
    }
	
    setTime(&endTime);
    cout << "barrier heap delete time: " << getTime(&startTime, &endTime) << "ms\n";

    // Result check
	cudaMemcpy(oriKeys, d_Keys, sizeof(int) * arrayNum, cudaMemcpyDeviceToHost);
	cudaMemcpy(oriVals, d_Vals, sizeof(int) * arrayNum, cudaMemcpyDeviceToHost);
#ifdef DEBUG
    for (int i = 0; i < arrayNum; i++) {
		cout << oriKeys[i] << " " << oriVals[i] << " | ";
	}
	cout << endl;
#endif
	for (int i = 0; i < arrayNum; i++) {
		if (oriKeys[i] != h_tKeys[i]) {
            printf("Wrong Answer! id: %d thrust %d heap %d\n", 
                    i, h_tKeys[i], oriKeys[i]);
			return -1;
		}
	}
	cout << "Correct!\n";
	
    return 0;

}
