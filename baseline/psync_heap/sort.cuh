#ifndef SORT_CUH
#define SORT_CUH

#include <cstdio>
//#include <cub/cub.cuh>

template<typename T>
__device__ inline void _swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template<typename T>
__device__ void ibitonicSort(T *keys, int size) {

    for (int k = 2; k <= size; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i =  threadIdx.x; i < size; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0) {
                        if (keys[i] > keys[ixj]) {
                            _swap<T>(keys[i], keys[ixj]);
                        }
                    }
                    else {
                        if (keys[i] < keys[ixj]) {
                            _swap<T>(keys[i], keys[ixj]);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

}

template<typename T>
__device__ void dbitonicSort(T *keys, int size) {

    for (int k = 2; k <= size; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i =  threadIdx.x; i < size; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0) {
                        if (keys[i] < keys[ixj]) {
                            _swap<T>(keys[i], keys[ixj]);
                        }
                    }
                    else {
                        if (keys[i] > keys[ixj]) {
                            _swap<T>(keys[i], keys[ixj]);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

}


template<typename T>
__device__ void imergePath(T *aKeys,
                           T *bKeys,
                           T *smallKeys,
                           T *largeKeys,
                           int size, int smemOffset) {

    extern __shared__ int s[];
    T *tmpKeys = (T *)&s[smemOffset];

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

        if (aTop == aBottom || (bI < size && (aI == size || aKeys[aI] > bKeys[bI]))) {
            if (aTop == aBottom || aKeys[aI - 1] <= bKeys[bI]) {
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
        if (bI == size || (aI < size && aKeys[aI] <= bKeys[bI])) {
            tmpKeys[i] = aKeys[aI];
            aI++;
        }
        else if (aI == size || (bI < size && aKeys[aI] > bKeys[bI])) {
            tmpKeys[i] = bKeys[bI];
            bI++;
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        smallKeys[i] = tmpKeys[i];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        largeKeys[i] = tmpKeys[size + i];
    }
    __syncthreads();
}


#endif
