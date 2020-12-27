#ifndef HEAP_CUH
#define HEAP_CUH

#include "../heaputil.cuh"

#define AVAIL 0
#define INUSE 1
#define TARGET 2
#define MARKED 3

template<typename K = int>
class Heap {
    public:
        K init_limits;

        int batchNum;
        int batchSize;

        int *batchCount;
        int *partialBufferSize;
#ifdef HEAP_SORT
        int *deleteCount;
#endif
        K *heapItems;
        int *status;

        Heap(int _batchNum,
            int _batchSize,
            K _init_limits = 0) : batchNum(_batchNum), batchSize(_batchSize), init_limits(_init_limits) {
            // prepare device heap
            cudaMalloc((void **)&heapItems, sizeof(K) * batchSize * (batchNum + 1));
            // initialize heap items with max value
            K *tmp = new K[batchSize * (batchNum + 1)];
            for (int i = 0; i < (batchNum + 1) * batchSize; i++) {
                tmp[i] = init_limits;
            }
            cudaMemcpy(heapItems, tmp, sizeof(K) * batchSize * (batchNum + 1), cudaMemcpyHostToDevice);
            delete []tmp; tmp = NULL;

            cudaMalloc((void **)&status, sizeof(int) * (batchNum + 1));
            cudaMemset(status, AVAIL, sizeof(int) * (batchNum + 1));

            cudaMalloc((void **)&batchCount, sizeof(int));
            cudaMemset(batchCount, 0, sizeof(int));
            cudaMalloc((void **)&partialBufferSize, sizeof(int));
            cudaMemset(partialBufferSize, 0, sizeof(int));
#ifdef HEAP_SORT
            cudaMalloc((void **)&deleteCount, sizeof(int));
            cudaMemset(deleteCount, 0, sizeof(int));
#endif
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
            batchNum = 0;
            batchSize = 0;
        }

        bool checkInsertHeap() {
            int h_batchCount;
            int h_partialBufferSize;
            cudaMemcpy(&h_batchCount, batchCount, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_partialBufferSize, partialBufferSize, sizeof(int), cudaMemcpyDeviceToHost);

            int *h_status = new int[h_batchCount + 1];
            K *h_items = new K[batchSize * (h_batchCount + 1)];
            cudaMemcpy(h_items, heapItems, sizeof(K) * batchSize * (h_batchCount + 1), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_status, status, sizeof(int) * (h_batchCount + 1), cudaMemcpyDeviceToHost);

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
                for (int i = 1; i < h_partialBufferSize; i++) {
                    if (h_items[i] < h_items[i - 1]) {
                        printf("Partial Buffer Error: partialBuffer[%d] is smaller than partialBuffer[%d-1]\n", i, i); 
                        return false;
                    }
                }
            }

            for (int i = 1; i <= h_batchCount; ++i) {
                if (h_status[i] != AVAIL) {
                    printf("State Error @ batch %d, state should be AVAIL = 0 while current is %d\n", i, h_status[i]);
                    return false;
                }
                if (i > 1) {
                    if (h_items[i * batchSize] < h_items[i/2 * batchSize + batchSize - 1]){
                        printf("Batch Keys Error @ batch %d's first item is smaller than batch %d's last item\n", i, i/2);
                        return false;
                    }
                }
                for (int j = 1; j < batchSize; ++j) {
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

        __host__ int itemCount() {
            int h_batchCount;
            int h_partialBufferSize;
            cudaMemcpy(&h_batchCount, batchCount, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_partialBufferSize, partialBufferSize, sizeof(int), cudaMemcpyDeviceToHost);
            return h_partialBufferSize + h_batchCount * batchSize;
        }

        __device__ int getItemCount() {
            changeStatus(&status[0], AVAIL, INUSE);
            int itemCount = *partialBufferSize + *batchCount * batchSize;
            changeStatus(&status[0], INUSE, AVAIL);
            return itemCount;
        }

        __host__ bool isEmpty() {
            int psize, bsize;
            cudaMemcpy(&psize, partialBufferSize, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&bsize, batchCount, sizeof(int), cudaMemcpyDeviceToHost);
            return !psize && !bsize;
        }

        __inline__ __device__ int getReversedIdx(int oriIdx) {
            int l = __clz(oriIdx) + 1;
            return (__brev(oriIdx) >> l) | (1 << (32-l));
        }

        __device__ bool changeStatus(int *_status, int oriS, int newS) {
            if ((oriS == AVAIL  && newS == TARGET) ||
                (oriS == TARGET && newS == MARKED) ||
                (oriS == MARKED && newS == TARGET) ||
                (oriS == TARGET && newS == AVAIL ) ||
                (oriS == TARGET && newS == INUSE ) ||
                (oriS == INUSE  && newS == AVAIL ) ||
                (oriS == AVAIL  && newS == INUSE )) {
                while (atomicCAS(_status, oriS, newS) != oriS){
                }
                return true;
            }
            else {
                printf("LOCK ERROR %d %d\n", oriS, newS);
                return false;
            }
        }

        // determine the next batch when insert operation updating the heap
        // given the current batch index and the target batch index
        // return the next batch index to the target batch
        __device__ int getNextIdxToTarget(int currentIdx, int targetIdx) {
            return targetIdx >> (__clz(currentIdx) - __clz(targetIdx) - 1);
        }

        __device__ bool deleteRoot(K *items, int &size) {
 
            if (threadIdx.x == 0) {
                changeStatus(&status[0], AVAIL, INUSE);
            }
            __syncthreads();

#ifdef HEAP_SORT
            int deleteOffset = *deleteCount;
#else
            int deleteOffset = 0;
#endif

            if (*batchCount == 0 && *partialBufferSize == 0) {
                if (threadIdx.x == 0) {
                    changeStatus(&status[0], INUSE, AVAIL);
                }
                size = 0;
                __syncthreads();
                return false;
            }
/*
            if (*batchCount == 0 && *partialBufferSize != 0) {
                // only partial batch has items
                // output the partial batch
                size = *partialBufferSize;
                batchCopy(items + deleteOffset, heapItems, size, true);

                if (threadIdx.x == 0) {
#ifdef HEAP_SORT
                    *deleteCount += *partialBufferSize;
#endif
                    *partialBufferSize = 0;
                    changeStatus(&status[0], INUSE, AVAIL);
                }
                __syncthreads();
                return false;
            }
*/
            if (threadIdx.x == 0) {
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
        // it will fill the empty root batch
        __device__ void deleteUpdate(int smemOffset) {

            extern __shared__ int s[];
            K *sMergedItems = (K *)&s[smemOffset];
            int *tmpIdx = (int *)&s[smemOffset];

            smemOffset += sizeof(K) * 3 * batchSize / sizeof(int);
            int *tmpType = (int *)&s[smemOffset - 1];


            if (threadIdx.x == 0) {
                //            *tmpIdx = getReversedIdx(atomicSub(batchCount, 1));
                *tmpIdx = atomicSub(batchCount, 1);
                // if no more batches in the heap
                if (*tmpIdx == 1) {
                    changeStatus(&status[1], INUSE, AVAIL);
                    changeStatus(&status[0], INUSE, AVAIL);
                }
            }
            __syncthreads();

            int lastIdx = *tmpIdx;
            __syncthreads();

            if (lastIdx == 1) return;

            if (threadIdx.x == 0) {
                while(1) {
                    if (atomicCAS(&status[lastIdx], AVAIL, INUSE) == AVAIL) {
                        *tmpType = 0;
                        break;
                    }
                    if (atomicCAS(&status[lastIdx], TARGET, MARKED) == TARGET) {
                        *tmpType = 1;
                        break;
                    }
                }
            }
            __syncthreads();

            if (*tmpType == 1) {
                // wait for insert worker
                if (threadIdx.x == 0) {
                    while (atomicCAS(&status[lastIdx], TARGET, AVAIL) != TARGET) {}
                }
                __syncthreads();

                batchCopy(sMergedItems, heapItems + batchSize, batchSize);
            }
            else if (*tmpType == 0){

                batchCopy(sMergedItems, heapItems + lastIdx * batchSize, batchSize, true, init_limits);

                if (threadIdx.x == 0) {
                    changeStatus(&status[lastIdx], INUSE, AVAIL);
                }
                __syncthreads();
            }

            /* start handling partial batch */
/*
            batchCopy(sMergedItems + batchSize, heapItems, batchSize);

            imergePath(sMergedItems, sMergedItems + batchSize,
                       sMergedItems, heapItems,
                       batchSize, smemOffset);
            __syncthreads();
*/
            /* end handling partial batch */
            if (threadIdx.x == 0) {
                changeStatus(&status[0], INUSE, AVAIL);
            }
            __syncthreads();

            int currentIdx = 1;
            while (1) {
                int leftIdx = currentIdx << 1;
                int rightIdx = leftIdx + 1;
                // Wait until status[] are not locked
                // After that if the status become unlocked, than child exists
                // If the status is not unlocked, than no valid child
                if (threadIdx.x == 0) {
                    int leftStatus, rightStatus;
                    leftStatus = atomicCAS(&status[leftIdx], AVAIL, INUSE);
                    while (leftStatus == INUSE) {
                        leftStatus = atomicCAS(&status[leftIdx], AVAIL, INUSE);
                    }
                    if (leftStatus != AVAIL) {
                        *tmpType = 0;
                    }
                    else {
                        rightStatus = atomicCAS(&status[rightIdx], AVAIL, INUSE);
                        while (rightStatus == INUSE) {
                            rightStatus = atomicCAS(&status[rightIdx], AVAIL, INUSE);
                        }
                        if (rightStatus != AVAIL) {
                            *tmpType = 1;
                        }
                        else {
                            *tmpType = 2;
                        }
                    }
                }
                __syncthreads();

                int deleteType = *tmpType;
                __syncthreads();

                if (deleteType == 0) { // no children
                    // move shared memory to currentIdx
                    batchCopy(heapItems + currentIdx * batchSize, sMergedItems, batchSize);
                    if (threadIdx.x == 0) {
                        changeStatus(&status[currentIdx], INUSE, AVAIL);
                    }
                    __syncthreads();
                    return;
                }
                else if (deleteType == 1) { // only has left child and left child is a leaf batch
                    // move leftIdx to shared memory
                    batchCopy(sMergedItems + batchSize, heapItems + leftIdx * batchSize, batchSize);

                    imergePath(sMergedItems, sMergedItems + batchSize,
                               heapItems + currentIdx * batchSize, heapItems + leftIdx * batchSize,
                               batchSize, smemOffset);
                    __syncthreads();

                    if (threadIdx.x == 0) {
                        // unlock batch[currentIdx] & batch[leftIdx]
                        changeStatus(&status[currentIdx], INUSE, AVAIL);
                        changeStatus(&status[leftIdx], INUSE, AVAIL);
                    }
                    __syncthreads();
                    return;
                }

                // move leftIdx and rightIdx to shared memory
                batchCopy(sMergedItems + batchSize,
                          heapItems + leftIdx * batchSize,
                          batchSize);
                batchCopy(sMergedItems + 2 * batchSize,
                          heapItems + rightIdx * batchSize,
                          batchSize);

                int largerIdx = (heapItems[leftIdx * batchSize + batchSize - 1] < heapItems[rightIdx * batchSize + batchSize - 1]) ? rightIdx : leftIdx;
                int smallerIdx = 4 * currentIdx - largerIdx + 1;
                __syncthreads();

                imergePath(sMergedItems + batchSize, sMergedItems + 2 * batchSize,
                           sMergedItems + batchSize, heapItems + largerIdx * batchSize,
                           batchSize, smemOffset);
                __syncthreads();

                if (threadIdx.x == 0) {
                    changeStatus(&status[largerIdx], INUSE, AVAIL);
                }
                __syncthreads();

                if (sMergedItems[0] >= sMergedItems[2 * batchSize - 1]) {
                    batchCopy(heapItems + currentIdx * batchSize,
                              sMergedItems + batchSize,
                              batchSize);
                }
                else if (sMergedItems[batchSize - 1] <= sMergedItems[batchSize]) {
                    batchCopy(heapItems + currentIdx * batchSize, 
                              sMergedItems,
                              batchSize);
                    batchCopy(heapItems + smallerIdx * batchSize, 
                              sMergedItems + batchSize,
                              batchSize);
                    if (threadIdx.x == 0) {
                        changeStatus(&status[currentIdx], INUSE, AVAIL);
                        changeStatus(&status[smallerIdx], INUSE, AVAIL);
                    }
                    __syncthreads();
                    return;
                }
                else {
                    imergePath(sMergedItems, sMergedItems + batchSize,
                               heapItems + currentIdx * batchSize, sMergedItems,
                               batchSize, smemOffset);
                }
                __syncthreads();

                if (threadIdx.x == 0) {
                    changeStatus(&status[currentIdx], INUSE, AVAIL);
                }
                __syncthreads();
                currentIdx = smallerIdx;
            }

        }

        __device__ void insertion(K *items, 
                                  int size, 
                                  int smemOffset) {
#ifdef INSERT_SMEM // insert items is already in smem
            extern __shared__ int s[];
            K *sMergedItems1 = (K *)&items[0];
            K *sMergedItems2 = (K *)&s[smemOffset];
            smemOffset += sizeof(K) * batchSize / sizeof(int);
            int *tmpIdx = (int *)&s[smemOffset - 1];
#else
            // allocate shared memory space
            extern __shared__ int s[];
            K *sMergedItems1 = (K *)&s[smemOffset];
            K *sMergedItems2 = (K *)&sMergedItems1[batchSize];
            smemOffset += sizeof(K) * 2 * batchSize / sizeof(int);
            int *tmpIdx = (int *)&s[smemOffset - 1];

            // move insert batch to shared memory
            // may be a partial batch, fill rest part with INT_MAX
            // TODO in this way, we can use bitonic sorting
            // but the performance may not be good when size is small
            for (int i = threadIdx.x; i < batchSize; i += blockDim.x) {
                sMergedItems1[i] = i < size ? items[i] : init_limits;
            }
            __syncthreads();
#endif
            ibitonicSort(sMergedItems1, batchSize);
            __syncthreads();

            if (threadIdx.x == 0) {
                changeStatus(&status[0], AVAIL, INUSE);
            }
            __syncthreads();

            /* start handling partial batch */
    /*
            // Case 1: the heap has no full batch
            // TODO current not support size > batchSize, app should handle this
            if (*batchCount == 0 && size < batchSize) {
                // Case 1.1: partial batch is empty
                if (*partialBufferSize == 0) {
                    batchCopy(heapItems, sMergedItems1, batchSize);
                    if (threadIdx.x == 0) {
                        *partialBufferSize = size;
                        changeStatus(&status[0], INUSE, AVAIL);
                    }
                    __syncthreads();
                    return;
                }
                // Case 1.2: no full batch is generated
                else if (size + *partialBufferSize < batchSize) {
                    batchCopy(sMergedItems2, heapItems, batchSize);
                    imergePath(sMergedItems1, sMergedItems2,
                               heapItems, sMergedItems1,
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
                    batchCopy(sMergedItems2, heapItems, batchSize);
                    if (threadIdx.x == 0) {
                        // increase batchCount and change root batch to INUSE
                        atomicAdd(batchCount, 1);
                        changeStatus(&status[1], AVAIL, TARGET);
                        changeStatus(&status[1], TARGET, INUSE);
                    }
                    __syncthreads();
                    imergePath(sMergedItems1, sMergedItems2,
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
                    batchCopy(sMergedItems2, heapItems, batchSize);
                    // Merge insert batch with partial batch
                    imergePath(sMergedItems1, sMergedItems2,
                               sMergedItems1, sMergedItems2,
                               batchSize, smemOffset);
                    __syncthreads();
                    if (threadIdx.x == 0) {
                        changeStatus(&status[1], AVAIL, INUSE);
                    }
                    __syncthreads();
                    batchCopy(sMergedItems2, heapItems + batchSize, batchSize);
                    imergePath(sMergedItems1, sMergedItems2,
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
                    batchCopy(sMergedItems2, heapItems, batchSize);
                    // Merge insert batch with partial batch, leave larger half in the partial batch
                    imergePath(sMergedItems1, sMergedItems2,
                               sMergedItems1, heapItems,
                               batchSize, smemOffset);
                    __syncthreads();
                    if (threadIdx.x == 0) {
                        // update partial batch size 
                        *partialBufferSize += (size - batchSize);
                    }
                    __syncthreads();
                }
            }
*/
            /* end handling partial batch */

            if (threadIdx.x == 0) {
                //            *tmpIdx = getReversedIdx(atomicAdd(batchCount, 1) + 1);
                *tmpIdx = atomicAdd(batchCount, 1) + 1;
                changeStatus(&status[*tmpIdx], AVAIL, TARGET);
                if (*tmpIdx != 1) {
                    changeStatus(&status[1], AVAIL, INUSE);
                }
            }
            __syncthreads();

            int currentIdx = 1;
            int targetIdx = *tmpIdx;
            __syncthreads();

            while(currentIdx != targetIdx) {
                if (threadIdx.x == 0) {
                    *tmpIdx = 0;
                    if (status[targetIdx] == MARKED) {
                        *tmpIdx = 1;
                    }
                }
                __syncthreads();

                if (*tmpIdx == 1) break;
                __syncthreads();

                if (threadIdx.x == 0) {
                    changeStatus(&status[currentIdx / 2], INUSE, AVAIL);
                }
                __syncthreads();

                // move batch to shard memory
                batchCopy(sMergedItems2, 
                          heapItems + currentIdx * batchSize, 
                          batchSize);

                if (sMergedItems1[batchSize - 1] <= sMergedItems2[0]) {
                    // if insert batch is smaller than current batch
                    __syncthreads();
                    batchCopy(heapItems + currentIdx * batchSize,
                              sMergedItems1,
                              batchSize);

                    batchCopy(sMergedItems1, sMergedItems2, batchSize);
                }
                else if (sMergedItems2[batchSize - 1] > sMergedItems1[0]) {
                    __syncthreads();
                    imergePath(sMergedItems1, sMergedItems2,
                            heapItems + currentIdx * batchSize, sMergedItems1,
                            batchSize, smemOffset);
                    __syncthreads();
                }
                currentIdx = getNextIdxToTarget(currentIdx, targetIdx);
                if (threadIdx.x == 0) {
                    if (currentIdx != targetIdx) {
                        changeStatus(&status[currentIdx], AVAIL, INUSE);
                    }
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                atomicCAS(&status[targetIdx], TARGET, INUSE);
            }
            __syncthreads();

            if (status[targetIdx] == MARKED) {
                __syncthreads();
                batchCopy(heapItems + batchSize, sMergedItems1, batchSize);
                if (threadIdx.x == 0) {
                    changeStatus(&status[currentIdx / 2], INUSE, AVAIL);
                    if (targetIdx != currentIdx) {
                        changeStatus(&status[currentIdx], INUSE, AVAIL);
                    }
                    changeStatus(&status[targetIdx], MARKED, TARGET);
                }
                __syncthreads();
                return;
            }

            batchCopy(heapItems + targetIdx * batchSize, sMergedItems1, batchSize);

            if (threadIdx.x == 0) {
                changeStatus(&status[currentIdx / 2], INUSE, AVAIL);
                changeStatus(&status[currentIdx], INUSE, AVAIL);
            }
            __syncthreads();
        }
    };

#endif
