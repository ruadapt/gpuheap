#ifndef UTIL_HPP
#define UTIL_HPP

#include <sys/time.h>
#include "datastructure.hpp"
#include <numeric>
#include <algorithm>

using namespace std;

enum SortType { BPW, B, W };

void setTime(struct timeval* StartingTime) {
	gettimeofday(StartingTime, NULL);
}

double getTime(struct timeval* StartingTime, struct timeval* EndingingTime) {
	struct timeval ElapsedTime;
	timersub(EndingingTime, StartingTime, &ElapsedTime);
	return (ElapsedTime.tv_sec*1000.0 + ElapsedTime.tv_usec / 1000.0);	// Returning in milliseconds.
}

template <class VarType>
void swap(VarType* a, VarType* b)
{
	VarType t = *a;
	*a = *b;
	*b = t;
}

template <class VarType>
int Partition(VarType *arr, int *index, int low, int high)
{
	VarType pivot = arr[high];
	int i = (low - 1);

	for (int j = low; j <= high - 1; j++)
	{
		if (arr[j] <= pivot)
		{
			i++;
			swap(&arr[i], &arr[j]);
			swap(&index[i], &index[j]);
		}
	}
	swap(&arr[i + 1], &arr[high]);
	swap(&index[i + 1], &index[high]);
	return (i + 1);
}

template <class VarType>
void QuickSortWithIndex(VarType *arr, int *index, int low, int high)
{
	if (low < high)
	{
		int p = Partition(arr, index, low, high);

		QuickSortWithIndex(arr, index, low, p - 1);
		QuickSortWithIndex(arr, index, p + 1, high);
	}
}


void SortKnapsackItem(int *weight, int *benefit, float *benefitPerWeight, int size, SortType type) {

	int *index = new int[size];
    for (int i = 0; i < size; i++) {
        index[i] = i;
    }

	int *newWeight = new int[size];
	int *newBenefit = new int[size];
	float *newBenefitPerWeight = new float[size];

	switch (type)
	{
	case BPW: // sort by benefit per weight
		QuickSortWithIndex<float>(benefitPerWeight, index, 0, size - 1);
		// descending order
		reverse(benefitPerWeight, benefitPerWeight + size);
		reverse(index, index + size);

		for (int i = 0; i < size; i++) {
			newWeight[i] = weight[index[i]];
			newBenefit[i] = benefit[index[i]];
		}
		copy(newWeight, newWeight + size, weight);
		copy(newBenefit, newBenefit + size, benefit);

		break;

	case B: // sort by benefit
		QuickSortWithIndex<int>(benefit, index, 0, size - 1);

		// descending order
		reverse(benefit, benefit + size);
		reverse(index, index + size);

		for (int i = 0; i < size; i++) {
			newWeight[i] = weight[index[i]];
			newBenefitPerWeight[i] = benefitPerWeight[index[i]];
		}
		copy(newWeight, newWeight + size, weight);
		copy(newBenefitPerWeight, newBenefitPerWeight + size, benefitPerWeight);

		break;

	case W:	// sort by weight
		QuickSortWithIndex<int>(weight, index, 0, size - 1);

		// descending order
		reverse(weight, weight + size);
		reverse(index, index + size);

		for (int i = 0; i < size; i++) {
			newBenefit[i] = benefit[index[i]];
			newBenefitPerWeight[i] = benefitPerWeight[index[i]];
		}
		copy(newBenefit, newBenefit + size, benefit);
		copy(newBenefitPerWeight, newBenefitPerWeight + size, benefitPerWeight);

		break;

	default:
		break;
	}


	delete[]newWeight;
	delete[]newBenefit;
	delete[]newBenefitPerWeight;
	delete[]index;
}



#endif
