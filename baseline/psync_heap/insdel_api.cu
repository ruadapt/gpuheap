#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <time.h>
#include <algorithm>

#include "util.hpp"
#include "heap.cuh"
#include "heap_api.cuh"

using namespace std;

int main(int argc, char *argv[]) {

    if (argc != 4) {
        cout << "./insdel_api <insertNum> <blockNum> <blockSize>\n";
        return -1;
    }

    srand(time(NULL));

    int insertBatchNum = atoi(argv[1]);
    int insertNum = pow(2, insertBatchNum);
    int initNum = insertNum / 2;

    int batchSize = 1024;
    int batchNum = 1024000;
    int tableSize = 1024000;

//    int batchNum = insertBatchNum * 2;
//    int tableSize = batchNum;

    int blockNum = atoi(argv[2]);
    int blockSize = atoi(argv[3]);

    printf("init %d insert %d\n", initNum, insertNum);
    // generate <keys, vals> sequence
    int *oriKeys = new int[initNum + insertNum];
    for (int i = 0; i < insertNum; ++i) {
        oriKeys[i] = rand() % (INT_MAX / 2) ;
//        oriKeys[i] = initNum + insertNum - 1 - i;
    }
    for (int i = 0; i < initNum; ++i) {
        oriKeys[insertNum + i] = rand() % (INT_MAX / 2) + INT_MAX / 2;
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

    psyncHeapInsert<int>(d_heap, d_insTB, d_delTB, 
            d_Keys, insertNum, 
            blockNum, blockSize, batchSize);

    cudaMemcpy(&h_heap, d_heap, sizeof(Heap<int>), cudaMemcpyDeviceToHost);
    if (!h_heap.checkHeap()) {
        h_heap.printHeap();
        cout << "first insert fail\n";
        return -1;
    } else {
        cout << "first insert success\n";
    }

    psyncHeapDelete<int>(d_heap, d_insTB, d_delTB, 
            d_Keys, initNum, 
            blockNum, blockSize, batchSize);

    cudaMemcpy(&h_heap, d_heap, sizeof(Heap<int>), cudaMemcpyDeviceToHost);
    if (!h_heap.checkHeap()) {
        h_heap.printHeap();
        cout << "fisrt delete fail\n";
        return -1;
    } else {
        cout << "first delete success\n";
    }

    psyncHeapInsert<int>(d_heap, d_insTB, d_delTB, 
            d_Keys + insertNum, initNum, 
            blockNum, blockSize, batchSize);

    cudaMemcpy(&h_heap, d_heap, sizeof(Heap<int>), cudaMemcpyDeviceToHost);
    if (!h_heap.checkHeap()) {
        h_heap.printHeap();
        cout << "second insert fail\n";
        return -1;
    } else {
        cout << "second insert success\n";
    }

    psyncHeapDelete<int>(d_heap, d_insTB, d_delTB, 
            d_Keys + initNum, insertNum, 
            blockNum, blockSize, batchSize);

    cudaMemcpy(&h_heap, d_heap, sizeof(Heap<int>), cudaMemcpyDeviceToHost);
    if (!h_heap.checkHeap()) {
        h_heap.printHeap();
        cout << "second delete fail\n";
        return -1;
    } else {
        cout << "second delete success\n";
    }

    int *h_keys = new int[initNum + insertNum];
    cudaMemcpy(h_keys, d_Keys, sizeof(int) * (initNum + insertNum), cudaMemcpyDeviceToHost);
    sort(oriKeys, oriKeys + initNum + insertNum);
    for (int i = 0; i < initNum + insertNum; i++) {
        if (h_keys[i] != oriKeys[i]) {
            cout << "Error: "<< i 
                << " oriKeys " << oriKeys[i] 
                << " heapKeys " << h_keys[i] << "\n";
//            return -1;
        }
    }
    return 0;

}
