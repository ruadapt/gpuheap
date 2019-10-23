#ifndef SORT_CUH
#define SORT_CUH

#include <cstdio>
//#include <cub/cub.cuh>

template<typename T>
__device__ inline void swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template<typename K, typename V>
__device__ void ibitonicSort(K *keys, V *vals, int size) {

    for (int k = 2; k <= size; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i =  threadIdx.x; i < size; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0) {
                        if (keys[i] > keys[ixj]) {
                            swap<K>(keys[i], keys[ixj]);
                            swap<V>(vals[i], vals[ixj]);
                        }
                    }
                    else {
                        if (keys[i] < keys[ixj]) {
                            swap<K>(keys[i], keys[ixj]);
                            swap<V>(vals[i], vals[ixj]);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

}

template<typename K, typename V>
__device__ void dbitonicSort(K *keys, V *vals, int size) {

    for (int k = 2; k <= size; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i =  threadIdx.x; i < size; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0) {
                        if (keys[i] < keys[ixj]) {
                            swap<K>(keys[i], keys[ixj]);
                            swap<V>(vals[i], vals[ixj]);
                        }
                    }
                    else {
                        if (keys[i] > keys[ixj]) {
                            swap<K>(keys[i], keys[ixj]);
                            swap<V>(vals[i], vals[ixj]);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

}


template<typename K, typename V>
__device__ void imergePath(K *aKeys, V *aVals,
                           K *bKeys, V *bVals,
                           K *smallKeys, V *smallVals,
                           K *largeKeys, V *largeVals,
                           int size, int smemOffset) {

    extern __shared__ int s[];
    K *tmpKeys = (K *)&s[smemOffset];
    V *tmpVals = (V *)&tmpKeys[2 * size];

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
            tmpVals[i] = aVals[aI];
            aI++;
        }
        else if (aI == size || (bI < size && aKeys[aI] > bKeys[bI])) {
            tmpKeys[i] = bKeys[bI];
            tmpVals[i] = bVals[bI];
            bI++;
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        smallKeys[i] = tmpKeys[i];
        smallVals[i] = tmpVals[i];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        largeKeys[i] = tmpKeys[size + i];
        largeVals[i] = tmpVals[size + i];
    }
    __syncthreads();
}


#endif
