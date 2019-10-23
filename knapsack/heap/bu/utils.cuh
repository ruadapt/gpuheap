#ifndef UTILS_CUH
#define UTILS_CUH

#include <stdio.h>
#include <stdlib.h>

#define AVAIL 0
#define INSHOLD 1
#define DELMOD 2
#define INUSE 3

typedef unsigned int uint32;

#ifndef INIT_LIMITS
//TODO a better way to define the init max value??
#define INIT_LIMITS INT_MAX
#endif

__inline__ __device__ void batchCopy(uint128 *dest, uint128 *source, int size, bool reset = false)
{
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        dest[i] = source[i];
        if (reset) source[i] = INIT_LIMITS;
    }
    __syncthreads();
}

#endif
