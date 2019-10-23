#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

//#include "models.cuh"
#include "models2h.cuh"
#include "util.cuh"
#include "datastructure.hpp"

using namespace std;

bool cmp(uint128 a, uint128 b)
{
    double r1 = (double)a.first / (double)a.second;
    double r2 = (double)b.first / (double)b.second;
    return r1 > r2;
}

int main(int argc, char *argv[])
{
    if (argc != 11) {
        cout << "./model [dataset] [batchnum] [batchsize] [blocknum] [blocksize] [model] [delAllowed] [gcThreshold] [expandThreshold] [endingBlockNum]\n";
        return 1;
    }
    ifstream inputFile;

    int batchNum = atoi(argv[2]);
    int batchSize = atoi(argv[3]);
    int blockNum = atoi(argv[4]);
    int blockSize = atoi(argv[5]);
    int model = atoi(argv[6]);
    int delAllowed = atoi(argv[7]);
    int gcThreshold = atoi(argv[8]);
    int expandThreshold = atoi(argv[9]);
    int endingBlockNum = atoi(argv[10]);

    inputFile.open(argv[1]);

    int capacity, inputSize;
//    inputFile >> capacity >> inputSize;
    inputFile >> inputSize >> capacity;

    int *weight = new int[inputSize];
    int *benefit = new int[inputSize];
    float *benefitPerWeight = new float[inputSize];

    for (int i = 0; i < inputSize; i++) {
//        inputFile >> weight[i] >> benefit[i];
        inputFile >> benefit[i] >> weight[i];
        benefitPerWeight[i] = (float)benefit[i] / (float)weight[i];
    }

    inputFile.close();

	// Sort items by ppw
	//SortKnapsackItem(weight, benefit, benefitPerWeight, inputSize, st);
	uint128 *items = new uint128[inputSize];
	for (int i = 0; i < inputSize; i++){
		items[i] = uint128(benefit[i], weight[i], 0, 0);
	}
	sort(items, items + inputSize, cmp);
	for (int i = 0; i < inputSize; i++){
		benefit[i] = items[i].first;
		weight[i] = items[i].second;
		benefitPerWeight[i] = (float)(benefit[i]) / (float)(weight[i]);

	}
	delete[]items;

	int *d_weight, *d_benefit;
    float *d_benefitPerWeight;
    cudaMalloc((void **)&d_weight, sizeof(int) * inputSize);
    cudaMalloc((void **)&d_benefit, sizeof(int) * inputSize); 
    cudaMalloc((void **)&d_benefitPerWeight, sizeof(float) * inputSize);
    cudaMemcpy(d_weight, weight, sizeof(int) * inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_benefit, benefit, sizeof(int) * inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_benefitPerWeight, benefitPerWeight, sizeof(float) * inputSize, cudaMemcpyHostToDevice);

    int max_benefit = 0;
    int *d_max_benefit;
    cudaMalloc((void **)&d_max_benefit, sizeof(int));
    cudaMemcpy(d_max_benefit, &max_benefit, sizeof(int), cudaMemcpyHostToDevice);


    if (model == 1)
    twoheap(d_weight, d_benefit, d_benefitPerWeight,
            d_max_benefit, capacity, inputSize, 
            delAllowed, gcThreshold, expandThreshold, endingBlockNum,
            batchNum, batchSize, blockNum, blockSize);
    /*else if (model == 2)*/
    /*oneheap(d_weight, d_benefit, d_benefitPerWeight,*/
            /*d_max_benefit, capacity, inputSize,*/
            /*batchNum, batchSize, blockNum, blockSize);*/

//    cudaMemcpy(&max_benefit, d_max_benefit, sizeof(int), cudaMemcpyDeviceToHost);

//    cout << max_benefit << endl;

    delete[] weight; weight = NULL;
    delete[] benefit; benefit = NULL;
    delete[] benefitPerWeight; benefitPerWeight = NULL;
    cudaFree(d_weight);
    cudaFree(d_benefit);
    cudaFree(d_benefitPerWeight);
    cudaFree(d_max_benefit);

    return 0;

}
