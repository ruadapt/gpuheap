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
        cout << "./insert <initNum> <insertNum> <batchSize> <batchNum> <tableSize> <blockNum> <blockSize>\n";
        return -1;
    }

    srand(time(NULL));

    int initNum = atoi(argv[1]) == 0 ?
                    0 : pow(2, atoi(argv[1]));
    int insertNum = atoi(argv[2]) == 0 ?
                    0 : pow(2, atoi(argv[2]));

    int batchSize = atoi(argv[3]);
    int batchNum = atoi(argv[4]);
    int tableSize = atoi(argv[5]);

    int blockNum = atoi(argv[6]);
    int blockSize = atoi(argv[7]);

    initNum = initNum * batchSize;

    struct timeval startTime;
    struct timeval endTime;
    double insertTime = 0;

    printf("init %d insert %d\n", initNum, insertNum);
    // generate <keys, vals> sequence
    int *oriKeys = new int[initNum + insertNum];
    for (int i = 0; i < initNum + insertNum; ++i) {
        oriKeys[i] = rand() % INT_MAX;
//        oriKeys[i] = initNum + insertNum - 1 - i;
    }

    // barrier heap sort
	/* Prepare Heap */
    Heap<int> h_heap(batchNum, batchSize, tableSize);
    Heap<int> *d_heap;
	cudaMalloc((void **)&d_heap, sizeof(Heap<int>));
    cudaMemcpy(d_heap, &h_heap, sizeof(Heap<int>), cudaMemcpyHostToDevice);
	
	/* Prepare data array */
	int *d_Keys;
	cudaMalloc((void **)&d_Keys, sizeof(int) * (initNum + insertNum));
    cudaMemcpy(d_Keys, oriKeys, sizeof(int) * (initNum + insertNum), cudaMemcpyHostToDevice);
	
	/* Prepare ins/del table buffer */
	TB<int> h_insTB(tableSize, batchSize, 0);
	TB<int> *d_insTB;
	cudaMalloc((void **)&d_insTB, sizeof(TB<int>));
	cudaMemcpy(d_insTB, &h_insTB, sizeof(TB<int>), cudaMemcpyHostToDevice);
	
    TB<int> h_delTB(tableSize, batchSize, 1);
	TB<int> *d_delTB;
	cudaMalloc((void **)&d_delTB, sizeof(TB<int>));
	cudaMemcpy(d_delTB, &h_delTB, sizeof(TB<int>), cudaMemcpyHostToDevice);

    int smemSize = 10 * batchSize * sizeof(int);

    bool h_status;
    bool *d_status;
    cudaMalloc((void **)&d_status, sizeof(bool));

    // Insert Items
    for (int i = 0; i < initNum / batchSize; i++) {
        insertItems<<<1, blockSize, smemSize>>>(d_heap, 
                                             d_Keys + i * batchSize, 
											 d_insTB);
        cudaDeviceSynchronize();

        // Even level insert update
        insertUpdate<<<blockNum, blockSize, smemSize>>>(d_heap,
														d_insTB);
        cudaDeviceSynchronize();

        updateTableBuffer<<<1, 1>>>(d_heap, d_insTB, d_delTB, d_status);
        cudaDeviceSynchronize();

        // Odd level insert update
        insertUpdate<<<blockNum, blockSize, smemSize>>>(d_heap,
														d_insTB);
        cudaDeviceSynchronize();
		
		updateTableBuffer<<<1, 1>>>(d_heap, d_insTB, d_delTB, d_status);
        cudaDeviceSynchronize();

    }

    while (1) {

        cudaMemcpy(&h_status, d_status, sizeof(bool), cudaMemcpyDeviceToHost);
        if (h_status) break;

        // Even level insert update
        insertUpdate<<<blockNum, blockSize, smemSize>>>(d_heap,
														d_insTB);
        cudaDeviceSynchronize();

        updateTableBuffer<<<1, 1>>>(d_heap, d_insTB, d_delTB, d_status);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_status, d_status, sizeof(bool), cudaMemcpyDeviceToHost);
        if (h_status) break;

        // Odd level insert update
        insertUpdate<<<blockNum, blockSize, smemSize>>>(d_heap,
														d_insTB);
        cudaDeviceSynchronize();
		
		updateTableBuffer<<<1, 1>>>(d_heap, d_insTB, d_delTB, d_status);
        cudaDeviceSynchronize();
    }

    setTime(&startTime);

    // Insert
    for (int i = 0; i < insertNum / batchSize; i++) {
        insertItems<<<1, blockSize, smemSize>>>(d_heap, 
                                                d_Keys + initNum + i * batchSize, 
											    d_insTB);
        cudaDeviceSynchronize();

        // Even level insert update
        insertUpdate<<<blockNum, blockSize, smemSize>>>(d_heap,
														d_insTB);
        cudaDeviceSynchronize();

        updateTableBuffer<<<1, 1>>>(d_heap, d_insTB, d_delTB, d_status);
        cudaDeviceSynchronize();

        // Odd level insert update
        insertUpdate<<<blockNum, blockSize, smemSize>>>(d_heap,
														d_insTB);
        cudaDeviceSynchronize();
		
		updateTableBuffer<<<1, 1>>>(d_heap, d_insTB, d_delTB, d_status);
        cudaDeviceSynchronize();
    }

    while (1) {
        
        cudaMemcpy(&h_status, d_status, sizeof(bool), cudaMemcpyDeviceToHost);
        if (h_status) break;

        // Even level insert update
        insertUpdate<<<blockNum, blockSize, smemSize>>>(d_heap,
														d_insTB);
        cudaDeviceSynchronize();

        updateTableBuffer<<<1, 1>>>(d_heap, d_insTB, d_delTB, d_status);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_status, d_status, sizeof(bool), cudaMemcpyDeviceToHost);
        if (h_status) break;

        // Odd level insert update
        insertUpdate<<<blockNum, blockSize, smemSize>>>(d_heap,
														d_insTB);
        cudaDeviceSynchronize();
		
		updateTableBuffer<<<1, 1>>>(d_heap, d_insTB, d_delTB, d_status);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < (initNum + insertNum) / batchSize; i++) {
        deleteItems<<<1, blockSize>>>(d_heap, 
									  d_Keys + i * batchSize, 
									  d_insTB,
									  d_delTB);
        cudaDeviceSynchronize();

        // Even level delete update
        deleteUpdate<<<blockNum, blockSize, smemSize>>>(d_heap,
														d_delTB);
        cudaDeviceSynchronize();

        updateTableBuffer<<<1, 1>>>(d_heap, d_insTB, d_delTB, d_status);
        cudaDeviceSynchronize();
        // Odd level insert update
        deleteUpdate<<<blockNum, blockSize, smemSize>>>(d_heap,
														d_delTB);
        cudaDeviceSynchronize();
		
		updateTableBuffer<<<1, 1>>>(d_heap, d_insTB, d_delTB, d_status);
        cudaDeviceSynchronize();
    }

    while (1) {

        cudaMemcpy(&h_status, d_status, sizeof(bool), cudaMemcpyDeviceToHost);
        if (h_status) break;

        // Even level delete update
        deleteUpdate<<<blockNum, blockSize, smemSize>>>(d_heap,
														d_delTB);
        cudaDeviceSynchronize();

        updateTableBuffer<<<1, 1>>>(d_heap, d_insTB, d_delTB, d_status);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_status, d_status, sizeof(bool), cudaMemcpyDeviceToHost);
        if (h_status) break;

        // Odd level insert update
        deleteUpdate<<<blockNum, blockSize, smemSize>>>(d_heap,
														d_delTB);
        cudaDeviceSynchronize();
		
		updateTableBuffer<<<1, 1>>>(d_heap, d_insTB, d_delTB, d_status);
        cudaDeviceSynchronize();
    }


    setTime(&endTime);
    insertTime += getTime(&startTime, &endTime);
//    printf("%d %d %d %d %d %.4f\n",
//            batchSize, blockNum, blockSize, initNum, insertNum, insertTime);
    printf("%.4f\n", insertTime);

    cudaMemcpy(oriKeys, d_Keys, sizeof(int) * (initNum + insertNum), cudaMemcpyDeviceToHost);
    for (int i = 1; i < initNum + insertNum; i++) {
        if (oriKeys[i] < oriKeys[i - 1]) {
            printf("%d %d %d\n", i, oriKeys[i], oriKeys[i - 1]);
            return -1;
        }
        if (oriKeys[i] != i) {
            printf("%d %d\n", i, oriKeys[i]);
            return -1;
        }
    }
    return 0;

}
