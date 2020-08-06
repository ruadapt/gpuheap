#ifndef HEAPUTIL_CUH
#define HEAPUTIL_CUH

#include "util.cuh"

template < typename A, typename B >
__forceinline__ __device__ void batchCopy(A * dest, B * source, int size){
  for(int i = threadIdx.x; i < size; i += blockDim.x){
    dest[i] = source[i];
  }
}

template < typename A, typename B >
__device__ void batchFill(A * dest, B value, int size){
  for(int i = threadIdx.x;i < size;i += blockDim.x){
    dest[i] = value;
  }
}

template<typename T>
__forceinline__ __device__ void _swap(T &a, T &b){
  T tmp = a;
  a = b;
  b = tmp;
}

template<typename K>
__device__ void ibitonicSort(K *items, int size) {
  for (int k = 2; k <= size; k <<= 1) {
    for (int j = k / 2; j > 0; j >>= 1) {
      for (int i =  threadIdx.x; i < size; i += blockDim.x) {
        int ixj = i ^ j;
        if (ixj > i) {
          if ((i & k) == 0) {
            if (items[i] > items[ixj]) {
              _swap<K>(items[i], items[ixj]);
            }
          } else {
            if (items[i] < items[ixj]) {
              _swap<K>(items[i], items[ixj]);
            }
          }
        }
        __syncthreads();
      }
    }
  }
}

template<typename K>
__device__ void dbitonicSort(K *items, int size) {
  for (int k = 2; k <= size; k <<= 1) {
    for (int j = k / 2; j > 0; j >>= 1) {
      for (int i =  threadIdx.x; i < size; i += blockDim.x) {
        int ixj = i ^ j;
        if (ixj > i) {
          if ((i & k) == 0) {
            if (items[i] < items[ixj]) {
              _swap<K>(items[i], items[ixj]);
            }
          } else {
            if (items[i] > items[ixj]) {
              _swap<K>(items[i], items[ixj]);
            }
          }
        }
        __syncthreads();
      }
    }
  }
}

template<typename K>
__device__ void dbitonicMerge(K *items, int size) {
  for (int j = size / 2; j > 0; j /= 2) {
    for (int i =  threadIdx.x; i < size; i += blockDim.x) {
      int ixj = i ^ j;
      if ((ixj > i) && (items[i] < items[ixj]))
        _swap<K>(items[i], items[ixj]);
      __syncthreads();
    }
  }
}

template<typename K>
__device__ void ibitonicMerge(K *items, int size) {
  for (int j = size / 2; j > 0; j /= 2) {
    for (int i =  threadIdx.x; i < size; i += blockDim.x) {
      int ixj = i ^ j;
      if ((ixj > i) && (items[i] > items[ixj]))
        _swap<K>(items[i], items[ixj]);
      __syncthreads();
    }
  }
}

template < typename A, typename B, typename C, typename D >
__device__ void imergePath(A * aItems, B * bItems, C * smallItems, D * largeItems, int size, int smemOffset) {
  extern __shared__ char smem[];
  int curr_smem_offset = smemOffset;
  ALIGN_SMEM_8;  
  A *tmpItems = (A *)&smem[curr_smem_offset];

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
      } else {
        aTop = aI - 1;
        bTop = bI + 1;
      }
    } else {
      aBottom = aI;
    }
  }

  // start from [aI, bI], found a path with lengthPerThread
  for (int i = lengthPerThread * threadIdx.x; i < lengthPerThread * threadIdx.x + lengthPerThread; ++i) {
    if (bI == size || (aI < size && aItems[aI] <= bItems[bI])) {
      tmpItems[i] = aItems[aI];
      aI++;
    } else if (aI == size || (bI < size && aItems[aI] > bItems[bI])) {
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
