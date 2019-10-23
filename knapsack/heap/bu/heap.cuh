#ifndef HEAP_CUH
#define HEAP_CUH

#include "sort.cuh"
#include "utils.cuh"
#include "datastructure.hpp"

using namespace std;

class Heap {
public:

        uint32 batchNum;
        uint32 batchSize;

        uint32 *batchCount;
        uint32 *partialBufferSize;
#ifdef HEAP_SORT
        uint32 *deleteCount;
#endif
        uint128 *heapItems;
        uint32 *status;

        uint32 *tbstate;
        uint32 *terminate;

        Heap(uint32 _batchNum,
            uint32 _batchSize) : batchNum(_batchNum), batchSize(_batchSize) {
            // prepare device heap
            cudaMalloc((void **)&heapItems, sizeof(uint128) * batchSize * (batchNum + 1));
            // initialize heap items with max value
            uint128 *tmp = new uint128[batchSize * (batchNum + 1)];
            for (uint32 i = 0; i < (batchNum + 1) * batchSize; i++) {
                tmp[i] = INIT_LIMITS;
            }
            cudaMemcpy(heapItems, tmp, sizeof(uint128) * batchSize * (batchNum + 1), cudaMemcpyHostToDevice);
            delete []tmp; tmp = NULL;

            cudaMalloc((void **)&status, sizeof(uint32) * (batchNum + 1));
            cudaMemset(status, AVAIL, sizeof(uint32) * (batchNum + 1));

            cudaMalloc((void **)&batchCount, sizeof(uint32));
            cudaMemset(batchCount, 0, sizeof(uint32));
            cudaMalloc((void **)&partialBufferSize, sizeof(uint32));
            cudaMemset(partialBufferSize, 0, sizeof(uint32));
#ifdef HEAP_SORT
            cudaMalloc((void **)&deleteCount, sizeof(uint32));
            cudaMemset(deleteCount, 0, sizeof(uint32));
#endif
            cudaMalloc((void **)&tbstate, 1024 * sizeof(uint32));
            cudaMemset(tbstate, 0, 1024 * sizeof(uint32));
            uint32 tmp1 = 1;
            cudaMemcpy(tbstate, &tmp1, sizeof(uint32), cudaMemcpyHostToDevice);
            cudaMalloc((void **)&terminate, sizeof(uint32));
            cudaMemset(terminate, 0, sizeof(uint32));
        }

        ~Heap() {
            cudaFree(heapItems);
            heapItems = NULL;
            cudaFree(status);
            status = NULL;
            cudaFree(batchCount);
            batchCount = NULL;
            cudaFree(partialBufferSize);
            partialBufferSize = NULL;
#ifdef HEAP_SORT
            cudaFree(deleteCount);
            deleteCount = NULL;
#endif
            cudaFree(tbstate);
            tbstate = NULL;
            batchNum = 0;
            batchSize = 0;
        }

        __device__ uint32 ifTerminate() {
            return *terminate;
        }

        bool checkInsertHeap() {
            uint32 h_batchCount;
            uint32 h_partialBufferSize;
            cudaMemcpy(&h_batchCount, batchCount, sizeof(uint32), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_partialBufferSize, partialBufferSize, sizeof(uint32), cudaMemcpyDeviceToHost);

            uint32 *h_status = new uint32[h_batchCount + 1];
            uint128 *h_items = new uint128[batchSize * (h_batchCount + 1)];
            cudaMemcpy(h_items, heapItems, sizeof(uint128) * batchSize * (h_batchCount + 1), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_status, status, sizeof(uint32) * (h_batchCount + 1), cudaMemcpyDeviceToHost);

            // check partial batch
            if (h_status[0] != AVAIL) {
                printf("Partial Batch State Error: state should be AVAIL = 0 while current is %d\n", h_status[0]);
                return false;
            }
            if (h_batchCount != 0 && h_partialBufferSize != 0) {
                if (h_items[batchSize * 2 - 1] > h_items[0]) {
                    printf("Partial Buffer Error: partial batch should be larger than root batch.\n");
                    return false;
                }
                for (uint32 i = 1; i < h_partialBufferSize; i++) {
                    if (h_items[i] < h_items[i - 1]) {
                        printf("Partial Buffer Error: partialBuffer[%d] is smaller than partialBuffer[%d-1]\n", i, i); 
                        return false;
                    }
                }
            }

            for (uint32 i = 1; i <= h_batchCount; ++i) {
                if (h_status[i] != AVAIL) {
                    printf("State Error @ batch %d, state should be AVAIL = 0 while current is %d\n", i, h_status[i]);
                    return false;
                }
                uint32 p = hostGetReversedIdx(hostGetReversedIdx(i) >> 1);
                if (i > 1) {
                    if (h_items[i * batchSize] < h_items[p * batchSize + batchSize - 1]){
                        printf("Batch Keys Error @ batch %d's first item is smaller than batch %d's last item\n", i, p);
                        return false;
                    }
                }
                for (uint32 j = 1; j < batchSize; ++j) {
                    if (h_items[i * batchSize + j] < h_items[i * batchSize + j - 1]) {
                        printf("Batch Keys Error @ batch %d item[%d] is smaller than item[%d]\n", i, j, j - 1);
                        return false;
                    }
                }
            }

            delete []h_items;
            delete []h_status;

            return true;

        }


        /*void printHeap() {*/
            
            /*// TODO if you need this, print each item of the K*/

            /*int h_batchCount;*/
            /*int h_partialBufferSize;*/
            /*cudaMemcpy(&h_batchCount, batchCount, sizeof(int), cudaMemcpyDeviceToHost);*/
            /*cudaMemcpy(&h_partialBufferSize, partialBufferSize, sizeof(int), cudaMemcpyDeviceToHost);*/

            /*int *h_status = new int[h_batchCount + 1];*/
            /*uint128 *h_items = new uint128[batchSize * (h_batchCount + 1)];*/
            /*cudaMemcpy(h_items, heapItems, sizeof(uint128) * batchSize * (h_batchCount + 1), cudaMemcpyDeviceToHost);*/
            /*cudaMemcpy(h_status, status, sizeof(int) * (h_batchCount + 1), cudaMemcpyDeviceToHost);*/

            /*printf("batch partial %d_%d:", h_partialBufferSize, h_status[0]);*/

            /*for (int i = 0; i < h_partialBufferSize; ++i) {*/
                /*printf(" %d", h_items[i]);*/
            /*}*/
            /*printf("\n");*/

            /*for (int i = 1; i <= h_batchCount; ++i) {*/
                /*printf("batch %d_%d:", i, h_status[i]);*/
                /*for (int j = 0; j < batchSize; ++j) {*/
                    /*printf(" %d", h_items[i * batchSize + j]);*/
                /*}*/
                /*printf("\n");*/
            /*}*/

        /*}*/

        __device__ uint32 getItemCount() {
            changeStatus(&status[0], AVAIL, INUSE);
            uint32 itemCount = *partialBufferSize + *batchCount * batchSize;
            changeStatus(&status[0], INUSE, AVAIL);
            return itemCount;
        }

        __host__ bool isEmpty() {
            uint32 psize, bsize;
            cudaMemcpy(&psize, partialBufferSize, sizeof(uint32), cudaMemcpyDeviceToHost);
            cudaMemcpy(&bsize, batchCount, sizeof(uint32), cudaMemcpyDeviceToHost);
            return !psize && !bsize;
        }

        __inline__ __device__ uint32 getReversedIdx(uint32 oriIdx) {
            uint32 l = __clz(oriIdx) + 1;
            return (__brev(oriIdx) >> l) | (1 << (32-l));
        }

        uint32 hostGetReversedIdx(uint32 oriIdx) {
            if (oriIdx == 1) return 1;
            uint32 i = oriIdx;
            uint32 l = 0;
            while (i > 0) {
                l++;
                i>>= 1;
            }
            l = 32 - (l - 1);
            uint32 res = 0;
            for (uint32 i = 0; i < 32; i++) {
                uint32 n = oriIdx % 2;
                oriIdx >>= 1;
                res <<= 1;
                res += n;
            }
            return (res >> l) | (1 << (32 - l));
        }

    // changeStatus must make sure that original status = ori and new status = new
    __device__ bool changeStatus(uint32 *status, uint32 oriS, uint32 newS) {
        if ((oriS == AVAIL   && newS == INUSE  ) ||
            (oriS == INUSE   && newS == AVAIL  ) ||
            (oriS == INUSE   && newS == INSHOLD) ||
            (oriS == INSHOLD && newS == INUSE  ) ||
            (oriS == INSHOLD && newS == DELMOD ) ||
            (oriS == DELMOD  && newS == INUSE  ) ||
            (oriS == INUSE   && newS == DELMOD )) {
                while (atomicCAS(status, oriS, newS) != oriS){
            }
            return true;
        }
        else {
            printf("LOCK ERROR ori %d new %d\n", oriS, newS);
            return false;
        }
    }

    // determine the next batch when insert operation updating the heap
    // given the current batch index and the target batch index
    // return the next batch index to the target batch
    __device__ uint32 getNextIdxToTarget(uint32 currentIdx, uint32 targetIdx) {
        return targetIdx >> (__clz(currentIdx) - __clz(targetIdx) - 1);
    }

     __device__ bool deleteRoot(uint128 *items, uint32 &size) {

        if (threadIdx.x == 0) {
            changeStatus(&status[0], AVAIL, INUSE);
        }
        __syncthreads();

#ifdef HEAP_SORT
        uint32 deleteOffset = *deleteCount;
#else
        uint32 deleteOffset = 0;
#endif

        if (*batchCount == 0 && *partialBufferSize == 0) {
            if (threadIdx.x == 0) {
                tbstate[blockIdx.x] = 0;
                int i;
                for (i = 0; i < gridDim.x; i++) {
                    if (tbstate[i] == 1) break;
                }
                if (i == gridDim.x) atomicCAS(terminate, 0, 1);
                changeStatus(&status[0], INUSE, AVAIL);
            }
            size = 0;
            __syncthreads();
            return false;
        }

        if (*batchCount == 0 && *partialBufferSize != 0) {
            // only partial batch has items
            // output the partial batch
            size = *partialBufferSize;
            batchCopy(items + deleteOffset, heapItems, size, true);

            if (threadIdx.x == 0) {
                tbstate[blockIdx.x] = 1;
#ifdef HEAP_SORT
                *deleteCount += *partialBufferSize;
#endif
                *partialBufferSize = 0;
                changeStatus(&status[0], INUSE, AVAIL);
            }
            __syncthreads();
            return false;
        }

        if (threadIdx.x == 0) {
            tbstate[blockIdx.x] = 1;
            changeStatus(&status[1], AVAIL, INUSE);
#ifdef HEAP_SORT
            *deleteCount += batchSize;
#endif
        }
        __syncthreads();

        size = batchSize;
        batchCopy(items + deleteOffset, heapItems + batchSize, size);
        /*
           if (threadIdx.x == 0) {
           printf("delete index: %d\n", *deleteIdx);
           for (int i = 0; i < batchSize; ++i) {
           printf("%d ", keys[*deleteIdx * batchSize + i]);
           }
           printf("\n");
           }
           __syncthreads();
         */
        return true;
    }

    
    // deleteUpdate is used to update the heap
    // it will fill the empty root batch(may be full)
    __device__ void deleteUpdate(uint32 smemOffset) {
        
        extern __shared__ int s[];
        uint128 *sMergedItems = (uint128 *)&s[smemOffset];
        uint32 *tmpIdx = (uint32 *)&s[smemOffset];
        smemOffset += sizeof(uint128) * 3 * batchSize / sizeof(uint32);
//        uint32 *tmpType = (uint32 *)&s[smemOffset - 1];

        if (threadIdx.x == 0) {
            *tmpIdx = atomicSub(batchCount, 1);
            if (*tmpIdx == 1) {
                changeStatus(&status[1], INUSE, AVAIL);
                changeStatus(&status[0], INUSE, AVAIL);
            }
        }
        __syncthreads();

        // no full batch exist just stop delete worker
        if (*tmpIdx == 1) return;
        __syncthreads();

        uint32 lastIdx = *tmpIdx;
        __syncthreads();

        if (threadIdx.x == 0) {
            uint32 lstatus = INUSE;
            while (lstatus == INUSE) {
                lstatus = atomicMax(&status[lastIdx], INUSE);
            }
        }
        __syncthreads();

        batchCopy(sMergedItems, 
                  heapItems + lastIdx * batchSize, 
                  batchSize, true);

        if (threadIdx.x == 0) {
            changeStatus(&status[lastIdx], INUSE, AVAIL);
        }
        __syncthreads();

        /* start handling partial batch */
        batchCopy(sMergedItems + batchSize, heapItems, batchSize);

        imergePath(sMergedItems, sMergedItems + batchSize,
                   sMergedItems, heapItems,
                   batchSize, smemOffset);
        __syncthreads();

        if (threadIdx.x == 0) {
            changeStatus(&status[0], INUSE, AVAIL);
        }
        __syncthreads();
        /* end handling partial batch */

        uint32 currentIdx = 1;
        uint32 curPrevStatus = AVAIL;
        while (1) {
            uint32 leftIdx = getReversedIdx(getReversedIdx(currentIdx) << 1);
            uint32 rightIdx = getReversedIdx(getReversedIdx(leftIdx) + 1);
            uint32 leftPrevStatus = INUSE, rightPrevStatus = INUSE;
            if (threadIdx.x == 0) {
                while (leftPrevStatus == INUSE) {
                    leftPrevStatus = atomicMax(&status[leftIdx], INUSE);
                }
                while (rightPrevStatus == INUSE) {
                    rightPrevStatus = atomicMax(&status[rightIdx], INUSE);
                }
                if (leftPrevStatus == INSHOLD) leftPrevStatus = DELMOD;
                if (rightPrevStatus == INSHOLD) rightPrevStatus = DELMOD;
            }
            __syncthreads();

            // move leftIdx and rightIdx to shared memory
            batchCopy(sMergedItems + batchSize, 
                      heapItems + leftIdx * batchSize,
                      batchSize);
            batchCopy(sMergedItems + 2 * batchSize,
                      heapItems + rightIdx * batchSize,
                      batchSize);

            uint32 targetIdx = sMergedItems[2 * batchSize - 1] < sMergedItems[3 * batchSize - 1] ? rightIdx : leftIdx;
            uint32 targetPrevStatus = targetIdx == rightIdx ? rightPrevStatus : leftPrevStatus;
            uint32 newIdx = targetIdx == rightIdx ? leftIdx : rightIdx;
            uint32 newPrevStatus = targetIdx == rightIdx ? leftPrevStatus : rightPrevStatus;
            __syncthreads();

            imergePath(sMergedItems + batchSize, sMergedItems + 2 * batchSize,
                          sMergedItems + batchSize, heapItems + targetIdx * batchSize,
                          batchSize, smemOffset);
            __syncthreads();
            
            if (threadIdx.x == 0) {
                changeStatus(&status[targetIdx], INUSE, targetPrevStatus);
            }
            __syncthreads();

            if (sMergedItems[0] >= sMergedItems[2 * batchSize - 1]) {
                __syncthreads();
                batchCopy(heapItems + currentIdx * batchSize, 
                          sMergedItems + batchSize, 
                          batchSize);
            }
            else if (sMergedItems[batchSize - 1] < sMergedItems[batchSize]) {
                __syncthreads();
                batchCopy(heapItems + currentIdx * batchSize,
                          sMergedItems,
                          batchSize);
                batchCopy(heapItems + newIdx * batchSize,
                          sMergedItems + batchSize,
                          batchSize);
                if (threadIdx.x == 0) {
                    changeStatus(&status[currentIdx], INUSE, curPrevStatus);
                    changeStatus(&status[newIdx], INUSE, newPrevStatus);
                }
                __syncthreads();
                return;
            }
            else {
                __syncthreads();
                imergePath(sMergedItems, sMergedItems + batchSize,
                              heapItems + currentIdx * batchSize, sMergedItems,
                              batchSize, smemOffset);
            }

            if (threadIdx.x == 0) {
                changeStatus(&status[currentIdx], INUSE, curPrevStatus);
            }
            __syncthreads();

            currentIdx = newIdx;
            curPrevStatus = newPrevStatus;
        }
       
    }

    __device__ void insertion(uint128 *items, uint32 size, uint32 smemOffset) {

#ifdef INSERT_SMEM // insert items is already in smem
            extern __shared__ int s[];
            uint128 *sMergedItems1 = (uint128 *)&items[0];
            uint128 *sMergedItems2 = (uint128 *)&s[smemOffset];
            smemOffset += sizeof(uint128) * batchSize / sizeof(uint32);
            uint32 *tmpIdx = (uint32 *)&s[smemOffset - 1];
#else
            // allocate shared memory space
            extern __shared__ int s[];
            uint128 *sMergedItems = (uint128 *)&s[smemOffset];
            smemOffset += sizeof(uint128) * 2 * batchSize / sizeof(uint32);
            uint32 *tmpIdx = (uint32 *)&s[smemOffset - 1];


            // move insert batch to shared memory
            // may be a partial batch, fill rest part with INT_MAX
            // TODO in this way, we can use bitonic sorting
            // but the performance may not be good when size is small
            for (uint32 i = threadIdx.x; i < batchSize; i += blockDim.x) {
                if (i < size) {
                    sMergedItems[i] = items[i];
                }
                else {
                    sMergedItems[i] = INIT_LIMITS;
                }
            }
            __syncthreads();
#endif
            ibitonicSort(sMergedItems, batchSize);
            __syncthreads();

            if (threadIdx.x == 0) {
                changeStatus(&status[0], AVAIL, INUSE);
            }
            __syncthreads();

            /* start handling partial batch */
            // Case 1: the heap has no full batch
            // TODO current not support size > batchSize, app should handle this
            if (*batchCount == 0 && size < batchSize) {
                // Case 1.1: partial batch is empty
                if (*partialBufferSize == 0) {
                    batchCopy(heapItems, sMergedItems, batchSize);
                    if (threadIdx.x == 0) {
                        *partialBufferSize = size;
                        changeStatus(&status[0], INUSE, AVAIL);
                    }
                    __syncthreads();
                    return;
                }
                // Case 1.2: no full batch is generated
                else if (size + *partialBufferSize < batchSize) {
                    batchCopy(sMergedItems + batchSize, heapItems, batchSize);
                    imergePath(sMergedItems, sMergedItems + batchSize,
                               heapItems, sMergedItems,
                               batchSize, smemOffset);
                    __syncthreads();
                    if (threadIdx.x == 0) {
                        *partialBufferSize += size;
                        changeStatus(&status[0], INUSE, AVAIL);
                    }
                    __syncthreads();
                    return;
                }
                // Case 1.3: a full batch is generated
                else if (size + *partialBufferSize >= batchSize) {
                    batchCopy(sMergedItems + batchSize, heapItems, batchSize);
                    if (threadIdx.x == 0) {
                        // increase batchCount and change root batch to INUSE
                        atomicAdd(batchCount, 1);
                        changeStatus(&status[1], AVAIL, INUSE);
                    }
                    __syncthreads();
                    imergePath(sMergedItems, sMergedItems + batchSize,
                               heapItems + batchSize, heapItems,
                               batchSize, smemOffset);
                    __syncthreads();
                    if (threadIdx.x == 0) {
                        *partialBufferSize += (size - batchSize);
                        changeStatus(&status[0], INUSE, AVAIL);
                        changeStatus(&status[1], INUSE, AVAIL);
                    }
                    __syncthreads();
                    return;
                }
            }
            // Case 2: the heap is non empty
            else {
                // Case 2.1: no full batch is generated
                if (size + *partialBufferSize < batchSize) {
                    batchCopy(sMergedItems + batchSize, heapItems, batchSize);
                    // Merge insert batch with partial batch
                    imergePath(sMergedItems, sMergedItems + batchSize,
                               sMergedItems, sMergedItems + batchSize,
                               batchSize, smemOffset);
                    __syncthreads();
                    if (threadIdx.x == 0) {
                        changeStatus(&status[1], AVAIL, INUSE);
                    }
                    __syncthreads();
                    batchCopy(sMergedItems + batchSize, heapItems + batchSize, batchSize);
                    imergePath(sMergedItems, sMergedItems + batchSize,
                               heapItems + batchSize, heapItems,
                               batchSize, smemOffset);
                    __syncthreads();
                    if (threadIdx.x == 0) {
                        *partialBufferSize += size;
                        changeStatus(&status[0], INUSE, AVAIL);
                        changeStatus(&status[1], INUSE, AVAIL);
                    }
                    __syncthreads();
                    return;
                }
                // Case 2.2: a full batch is generated and needed to be propogated
                else if (size + *partialBufferSize >= batchSize) {
                    batchCopy(sMergedItems + batchSize, heapItems, batchSize);
                    // Merge insert batch with partial batch, leave larger half in the partial batch
                    imergePath(sMergedItems, sMergedItems + batchSize,
                               sMergedItems, heapItems,
                               batchSize, smemOffset);
                    __syncthreads();
                    if (threadIdx.x == 0) {
                        // update partial batch size 
                        *partialBufferSize += (size - batchSize);
                    }
                    __syncthreads();
                }
            }
            /* end handling partial batch */

         if (threadIdx.x == 0) {
            *tmpIdx = atomicAdd(batchCount, 1) + 1;
//            printf("block %d insert target %d\n", blockIdx.x, *tmpIdx);
            changeStatus(&status[*tmpIdx], AVAIL, INUSE);
            changeStatus(&status[0], INUSE, AVAIL);
        }
        __syncthreads();

        uint32 currentIdx = *tmpIdx;
        __syncthreads();

        batchCopy(heapItems + currentIdx * batchSize,
                  sMergedItems,
                  batchSize);

        if (threadIdx.x == 0) {
            changeStatus(&status[currentIdx], INUSE, INSHOLD);
        }
        __syncthreads();

        while (currentIdx != 1) {
            uint32 parentIdx = getReversedIdx(getReversedIdx(currentIdx) >> 1);
            uint32 cstatus = INUSE;
            if (threadIdx.x == 0) {
                *tmpIdx = 0;
                changeStatus(&status[parentIdx], AVAIL, INUSE);
                while (cstatus == INUSE) {
                    cstatus = atomicMax(&status[currentIdx], INUSE);
                }
                if (cstatus == INSHOLD) {
                    *tmpIdx = 1;
                }
                else if (cstatus == DELMOD) {
                    *tmpIdx = 2;
                }
            }
            __syncthreads();

            if (*tmpIdx == 0) {
                __syncthreads();
                if (threadIdx.x == 0) {
                    changeStatus(&status[parentIdx], INUSE, AVAIL);
                    changeStatus(&status[currentIdx], INUSE, cstatus);
                }
                __syncthreads();
                return;
            }
            else if (*tmpIdx == 2) {
                __syncthreads();
                if (threadIdx.x == 0) {
                    changeStatus(&status[currentIdx], INUSE, AVAIL);
                    changeStatus(&status[parentIdx], INUSE, INSHOLD);
                }
                __syncthreads();
                currentIdx = parentIdx;
                continue;
            }
            __syncthreads();

            if (heapItems[currentIdx * batchSize] >= heapItems[parentIdx * batchSize + batchSize - 1]) {
                __syncthreads();

                if (threadIdx.x == 0) {
                    changeStatus(&status[currentIdx], INUSE, AVAIL);
                    changeStatus(&status[parentIdx], INUSE, AVAIL);
                }
                __syncthreads();
                return;
            }
            else if (heapItems[parentIdx * batchSize] >= heapItems[currentIdx * batchSize + batchSize - 1]) {
                __syncthreads();
                batchCopy(sMergedItems, 
                          heapItems + parentIdx * batchSize, 
                          batchSize);
                batchCopy(sMergedItems + batchSize,
                          heapItems + currentIdx * batchSize,
                          batchSize);
                batchCopy(heapItems + parentIdx * batchSize,
                          sMergedItems + batchSize,
                          batchSize);
                batchCopy(heapItems + currentIdx * batchSize,
                          sMergedItems,
                          batchSize);
            }
            else {
                __syncthreads();
                batchCopy(sMergedItems, 
                          heapItems + currentIdx * batchSize,
                          batchSize);
                batchCopy(sMergedItems + batchSize,
                          heapItems + parentIdx * batchSize,
                          batchSize);

                imergePath(sMergedItems, sMergedItems + batchSize,
                              heapItems + parentIdx * batchSize, heapItems + currentIdx * batchSize,
                              batchSize, smemOffset);
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                changeStatus(&status[parentIdx], INUSE, INSHOLD);
                changeStatus(&status[currentIdx], INUSE, AVAIL);
            }
            currentIdx = parentIdx;
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            changeStatus(&status[currentIdx], INSHOLD, INUSE);
            changeStatus(&status[currentIdx], INUSE, AVAIL);
        }
        __syncthreads();
    }
};

#endif
