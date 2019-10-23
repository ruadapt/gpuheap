#ifndef UTILS_CUH
#define UTILS_CUH

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <stdarg.h>

#define AVAIL 0
#define INUSE 1
#define TARGET 2
#define MARKED 3

#ifndef INIT_LIMITS
//TODO a better way to define the init max value??
#define INIT_LIMITS INT_MAX
#endif

template<typename K>
__inline__ __device__ void batchCopy(K *dest, K *source, int size, bool reset = false)
{
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        dest[i] = source[i];
        if (reset) source[i] = INIT_LIMITS;
    }
    __syncthreads();
}

#endif
