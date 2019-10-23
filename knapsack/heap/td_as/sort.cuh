#ifndef SORT_CUH
#define SORT_CUH

#include "datastructure.hpp"

template<typename T>
__device__ __forceinline__ void _swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

__device__ __forceinline__ void ibitonicSort(uint128 *items, uint32 size) {

    for (uint32 k = 2; k <= size; k *= 2) {
        for (uint32 j = k / 2; j > 0; j /= 2) {
            for (uint32 i =  threadIdx.x; i < size; i += blockDim.x) {
                uint32 ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0) {
                        if (items[i] > items[ixj]) {
                            _swap<uint128>(items[i], items[ixj]);
                        }
                    }
                    else {
                        if (items[i] < items[ixj]) {
                            _swap<uint128>(items[i], items[ixj]);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

}

__device__ __forceinline__ void dbitonicSort(uint128 *items, uint32 size) {

    for (uint32 k = 2; k <= size; k *= 2) {
        for (uint32 j = k / 2; j > 0; j /= 2) {
            for (uint32 i =  threadIdx.x; i < size; i += blockDim.x) {
                uint32 ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0) {
                        if (items[i] < items[ixj]) {
                            _swap<uint128>(items[i], items[ixj]);
                        }
                    }
                    else {
                        if (items[i] > items[ixj]) {
                            _swap<uint128>(items[i], items[ixj]);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

}

__device__ __forceinline__ void dbitonicMerge(uint128 *items, uint32 size) {
    for (int j = size / 2; j > 0; j /= 2) {
        for (int i =  threadIdx.x; i < size; i += blockDim.x) {
            int ixj = i ^ j;
            if ((ixj > i) && (items[i] < items[ixj]))
                _swap<uint128>(items[i], items[ixj]);
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void ibitonicMerge(uint128 *items, uint32 size) {
    for (int j = size / 2; j > 0; j /= 2) {
        for (int i =  threadIdx.x; i < size; i += blockDim.x) {
            int ixj = i ^ j;
            if ((ixj > i) && (items[i] > items[ixj]))
                _swap<uint128>(items[i], items[ixj]);
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void imergePath(uint128 *aItems, uint128 *bItems,
                                           uint128 *smallItems, uint128 *largeItems,
                                           uint32 size, uint32 smemOffset) {

    extern __shared__ int s[];
    uint128 *tmpItems = (uint128 *)&s[smemOffset];

    int lengthPerThread = size * 2 / blockDim.x;

    int index= threadIdx.x * lengthPerThread;
    int aTop = (index > size) ? size : index;
    int bTop = (index > size) ? index - size : 0;
    int aBottom = bTop;
    
    int offset, aI, bI;
    
    // binary search for diagonal intersections
    while (1) {
        offset = (aTop - aBottom) / 2;
        aI = aTop - offset;
        bI = bTop + offset;

        if (aTop == aBottom || (bI < size && (aI == size || aItems[aI] > bItems[bI]))) {
            if (aTop == aBottom || aItems[aI - 1] <= bItems[bI]) {
                break;
            }
            else {
                aTop = aI - 1;
                bTop = bI + 1;
            }
        }
        else {
            aBottom = aI;
        }
     }

     // start from [aI, bI], found a path with lengthPerThread
    for (int i = lengthPerThread * threadIdx.x; i < lengthPerThread * threadIdx.x + lengthPerThread; ++i) {
        if (bI == size || (aI < size && aItems[aI] <= bItems[bI])) {
            tmpItems[i] = aItems[aI];
            aI++;
        }
        else if (aI == size || (bI < size && aItems[aI] > bItems[bI])) {
            tmpItems[i] = bItems[bI];
            bI++;
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        smallItems[i] = tmpItems[i];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        largeItems[i] = tmpItems[size + i];
    }
    __syncthreads();
}

#endif
