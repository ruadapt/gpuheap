#ifndef HEAP_CUH
#define HEAP_CUH

#include "sort.cuh"
#include <iostream>
#include <cstdio>

using namespace std;

#define UNUSED 0
#define INUSE 1
#define AVAIL 2
#define TARGET 3
#define MARKED 4

class Heap {
	public:

		uint32 batchNum;
		uint32 batchSize;

		uint32 *batchCount;
		uint32 *partialBatchSize;
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
			// initialize partial batch
			uint128 *tmp = new uint128[batchSize];
            for (uint32 i = 0; i < batchSize; i++) {
                uint128 n(INIT_LIMITS, 0, 0, 0);
                tmp[i] = n;
            }
			cudaMemcpy(heapItems, tmp, sizeof(uint128) * batchSize, cudaMemcpyHostToDevice);
			delete []tmp; tmp = NULL;

			cudaMalloc((void **)&status, sizeof(uint32) * (batchNum + 1));
			cudaMemset(status, 0, sizeof(uint32) * (batchNum + 1));
			// initialize status for partial buffer
			uint32 partialStatusValue = AVAIL;
			cudaMemcpy(status, &partialStatusValue, sizeof(uint32), cudaMemcpyHostToDevice);

			cudaMalloc((void **)&batchCount, sizeof(uint32));
			cudaMemset(batchCount, 0, sizeof(uint32));
			cudaMalloc((void **)&partialBatchSize, sizeof(uint32));
			cudaMemset(partialBatchSize, 0, sizeof(uint32));
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
			cudaFree(partialBatchSize);
			partialBatchSize = NULL;
#ifdef HEAP_SORT
			cudaFree(deleteCount);
			deleteCount = NULL;
#endif
            cudaFree(tbstate);
            tbstate = NULL;
            cudaFree(terminate);
            terminate = NULL;

			batchNum = 0;
			batchSize = 0;
		}

        __device__ uint32 ifTerminate() {
            return *terminate;
        }

		bool checkInsertHeap() {
			uint32 h_batchCount;
			uint32 h_partialBatchSize;
			cudaMemcpy(&h_batchCount, batchCount, sizeof(uint32), cudaMemcpyDeviceToHost);
			cudaMemcpy(&h_partialBatchSize, partialBatchSize, sizeof(uint32), cudaMemcpyDeviceToHost);

			uint32 *h_status = new uint32[h_batchCount + 1];
            uint128 *h_items = new uint128[batchSize * (h_batchCount + 1)];
            cudaMemcpy(h_items, heapItems, sizeof(uint128) * batchSize * (h_batchCount + 1), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_status, status, sizeof(uint32) * (h_batchCount + 1), cudaMemcpyDeviceToHost);

			// check partial batch
			if (h_status[0] != AVAIL) {
				printf("Partial Batch Error: status should be ACTIVE while current is %u\n", h_status[0]);
				return false;
			}
			if (h_batchCount != 0 && h_partialBatchSize != 0) {
				if (h_items[batchSize * 2 - 1] > h_items[0]) {
					printf("Partial Batch Error: partial batch should be larger than root batch.\n");
					return false;
				}
				for (uint32 i = 1; i < h_partialBatchSize; i++) {
					if (h_items[i] < h_items[i - 1]) {
						printf("Partial Batch Error: %u with key %hu is smaller than %u with key %hu\n", 
                                i, h_items[i].first, i - 1, h_items[i - 1].first);
						return false;
					}
				}
			}

			for (uint32 i = 1; i <= h_batchCount; ++i) {
				if (h_status[i] == UNUSED) 
					continue;
				else if (h_status[i] != AVAIL) {
					printf("Lock Error @ batch %u, the value is %u while is expected 2\n", i, h_status[i]);
					return false;
				}
				if (i > 1) {
					if (h_items[i * batchSize] < h_items[i/2 * batchSize + batchSize - 1]){
						printf("Batch Keys Error @ batch %u with key %hu smaller than key %hu\n", 
                                i, h_items[i * batchSize].first, h_items[i/2 * batchSize + batchSize - 1].first);
						return false;
					}
				}
				for (uint32 j = 1; j < batchSize; ++j) {
					if (h_items[i * batchSize + j] < h_items[i * batchSize + j - 1]) {
						printf("Batch Keys Error @ batch %u item %u with key %hu smaller than key %hu\n", 
                                i, j, h_items[i * batchSize + j].first, h_items[i * batchSize + j - 1].first);
						return false;
					}
				}
			}

            delete []h_items;
            delete []h_status;

			return true;

		}


		void printHeap() {
            
            // TODO if you need this, print each item of the uint128

			uint32 h_batchCount;
			uint32 h_partialBatchSize;
			cudaMemcpy(&h_batchCount, batchCount, sizeof(uint32), cudaMemcpyDeviceToHost);
			cudaMemcpy(&h_partialBatchSize, partialBatchSize, sizeof(uint32), cudaMemcpyDeviceToHost);

			uint32 *h_status = new uint32[h_batchCount + 1];
			uint128 *h_items = new uint128[batchSize * (h_batchCount + 1)];
			cudaMemcpy(h_items, heapItems, sizeof(uint128) * batchSize * (h_batchCount + 1), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_status, status, sizeof(uint32) * (h_batchCount + 1), cudaMemcpyDeviceToHost);

			cout << "batch partial size" << h_partialBatchSize << "_" << h_status[0] << ": ";

			for (uint32 i = 0; i < h_partialBatchSize; ++i) {
                cout << h_items[i] << endl;
			}
			cout << endl;

			for (uint32 i = 1; i <= h_batchCount; ++i) {
				cout << "batch " << i << "_" << h_status[i] << ": ";
				for (uint32 j = 0; j < batchSize; ++j) {
                    cout << h_items[i] << endl;
				}
				cout << endl;
			}

		}

		__device__ uint32 getItemCount() {
			changeStatus(&status[0], AVAIL, INUSE);
			uint32 itemCount = *partialBatchSize + *batchCount * batchSize;
			changeStatus(&status[0], INUSE, AVAIL);
			return itemCount;
		}

        bool isEmpty() {
            uint32 psize, bsize;
            cudaMemcpy(&psize, partialBatchSize, sizeof(uint32), cudaMemcpyDeviceToHost);
            cudaMemcpy(&bsize, batchCount, sizeof(uint32), cudaMemcpyDeviceToHost);
            return !psize && !bsize;
        }

		__device__ uint32 getReversedIdx(uint32 oriIdx) {
			uint32 l = __clz(oriIdx) + 1;
			return (__brev(oriIdx) >> l) | (1 << (32-l));
		}

		uint32 maxIdx(uint32 oriIdx) {
			uint32 i = 1;
			while (i <= oriIdx) {
				i <<= 1;
			}
			return i;
		}


		// changeStatus must make sure that original status = ori and new status = new
		__device__ bool changeStatus(uint32 *_status, uint32 oriS, uint32 newS) {
			if ((oriS == UNUSED && newS == TARGET) ||
				(oriS == TARGET && newS == MARKED) ||
				(oriS == MARKED && newS == UNUSED) ||
				(oriS == TARGET && newS == INUSE ) ||
				(oriS == INUSE  && newS == AVAIL ) ||
				(oriS == INUSE  && newS == UNUSED) ||
				(oriS == AVAIL  && newS == INUSE )) {
				while (atomicCAS(_status, oriS, newS) != oriS){
				}
				return true;
			}
			else {
				printf("LOCK ERROR %u %u\n", oriS, newS);
				return false;
			}
		}

		// determine the next batch when insert operation updating the heap
		// given the current batch index and the target batch index
		// return the next batch index to the target batch
		__device__ uint32 getNextIdxToTarget(uint32 currentIdx, uint32 targetIdx) {
			return targetIdx >> (__clz(currentIdx) - __clz(targetIdx) - 1);
		}

        // TODO consider items in already in shared memory
		__device__ bool deleteRoot(uint128 *items, uint32 &size) {

			if (!threadIdx.x) {
				changeStatus(&status[0], AVAIL, INUSE);
			}
			__syncthreads();

#ifdef HEAP_SORT
			uint32 deleteOffset = *deleteCount;
#else
			uint32 deleteOffset = 0;
#endif
			
			if (*batchCount == 0 && *partialBatchSize == 0) {
				if (!threadIdx.x) {
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

			if (*batchCount == 0 && *partialBatchSize != 0) {
				// only partial batch has items
				// output the partial batch
				for (uint32 i = threadIdx.x; i < *partialBatchSize; i += blockDim.x) {
					items[deleteOffset + i] = heapItems[i];
                    uint128 n(INIT_LIMITS, 0, 0, 0);
                    heapItems[i] = n;
				}
                size = *partialBatchSize;
                __syncthreads();

				if (!threadIdx.x) {
                    tbstate[blockIdx.x] = 1;
#ifdef HEAP_SORT
					*deleteCount += *partialBatchSize;
#endif
                    *partialBatchSize = 0;
					changeStatus(&status[0], INUSE, AVAIL);
				}
				__syncthreads();
				return false;
			}

			if (!threadIdx.x) {
                tbstate[blockIdx.x] = 1;
				changeStatus(&status[1], AVAIL, INUSE);
#ifdef HEAP_SORT
				*deleteCount += batchSize;
#endif
			}
			__syncthreads();

			for (uint32 i = threadIdx.x; i < batchSize; i += blockDim.x) {
				items[deleteOffset + i] = heapItems[batchSize + i];
			}
            size = batchSize;
			__syncthreads();
			/*
			   if (!threadIdx.x) {
			   printf("delete index: %d\n", *deleteIdx);
			   for (uint32 i = 0; i < batchSize; ++i) {
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
		__device__ void deleteUpdate(uint32 smemOffset) {

			extern __shared__ int s[];
			uint128 *sMergedItems = (uint128 *)&s[smemOffset];
			uint32 *tmpIdx = (uint32 *)&s[smemOffset];
            smemOffset += sizeof(uint128) * 3 * batchSize / sizeof(int);
			uint32 *tmpType = (uint32 *)&s[smemOffset - 1];


			if (!threadIdx.x) {
				//            *tmpIdx = getReversedIdx(atomicSub(batchCount, 1));
				*tmpIdx = atomicSub(batchCount, 1);
				// if no more batches in the heap
				if (*tmpIdx == 1) {
					changeStatus(&status[1], INUSE, UNUSED);
					changeStatus(&status[0], INUSE, AVAIL);
				}
			}
			__syncthreads();

			uint32 lastIdx = *tmpIdx;
			__syncthreads();

			if (lastIdx == 1) return;

			if (!threadIdx.x) {
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
				if (!threadIdx.x) {
					while (status[lastIdx] != UNUSED) {}
				}
				__syncthreads();

				// batch[lastIdx] has been moved to batch[root] by insert worker
				for (uint32 i = threadIdx.x; i < batchSize; i += blockDim.x) {
					sMergedItems[i] = heapItems[batchSize + i];
				}
				__syncthreads();
			}
			else if (*tmpType == 0){

				for (uint32 i = threadIdx.x; i < batchSize; i += blockDim.x) {
					sMergedItems[i] = heapItems[lastIdx * batchSize + i];
				}
				__syncthreads();

				if (!threadIdx.x) {
					changeStatus(&status[lastIdx], INUSE, UNUSED);
				}
				__syncthreads();
			}

			/* start handling partial batch */
			for (uint32 i = threadIdx.x; i < batchSize; i += blockDim.x) {
				sMergedItems[batchSize + i] = heapItems[i];
			}
			__syncthreads();

			imergePath(sMergedItems, sMergedItems + batchSize,
                       sMergedItems, heapItems,
					   batchSize, smemOffset);
			__syncthreads();

			if (!threadIdx.x) {
				changeStatus(&status[0], INUSE, AVAIL);
			}
			__syncthreads();
			/* end handling partial batch */

			uint32 currentIdx = 1;
			uint32 leftIdx = 2;
			uint32 rightIdx = 3;
			while (1) {
				// Wait until status[] are not locked
				// After that if the status become unlocked, than child exists
				// If the status is not unlocked, than no valid child
				// TODO leftIdx may larger than batchNum
				if (!threadIdx.x) {
					uint32 leftStatus, rightStatus;
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

				uint32 deleteType = *tmpType;
				__syncthreads();

				if (deleteType == 0) { // no children
					// move shared memory to currentIdx
					for (uint32 i = threadIdx.x; i < batchSize; i += blockDim.x) {
						heapItems[currentIdx * batchSize + i] = sMergedItems[i];
					}
					__syncthreads();
					if (!threadIdx.x) {
						changeStatus(&status[currentIdx], INUSE, AVAIL);
					}
					__syncthreads();
					return;
				}
				else if (deleteType == 1) { // only has left child and left child is a leaf batch

					// move leftIdx to shared memory
					for (uint32 i = threadIdx.x; i < batchSize; i += blockDim.x) {
						sMergedItems[batchSize + i] = heapItems[leftIdx * batchSize + i];
					}
					__syncthreads();

                    imergePath(sMergedItems, sMergedItems + batchSize,
                               heapItems + currentIdx * batchSize, heapItems + leftIdx * batchSize,
                               batchSize, smemOffset);
					__syncthreads();

					if (!threadIdx.x) {
						// unlock batch[currentIdx] & batch[leftIdx]
						changeStatus(&status[currentIdx], INUSE, AVAIL);
						changeStatus(&status[leftIdx], INUSE, AVAIL);
					}
					__syncthreads();
					return;
				}
				else {
					// move leftIdx and rightIdx to shared memory
					// rightIdx is stored in reversed order
					for (uint32 i = threadIdx.x; i < batchSize; i += blockDim.x) {
						sMergedItems[batchSize + i] = heapItems[leftIdx * batchSize + i];
					}
					for (uint32 i = threadIdx.x; i < batchSize; i += blockDim.x) {
						sMergedItems[2 * batchSize + i] = heapItems[rightIdx * batchSize + i];
					}
					__syncthreads();

					uint32 targetIdx = (heapItems[leftIdx * batchSize + batchSize - 1] < heapItems[rightIdx * batchSize + batchSize - 1]) ? rightIdx : leftIdx;
					__syncthreads();

                    imergePath(sMergedItems + batchSize, sMergedItems + 2 * batchSize,
                               sMergedItems + batchSize, heapItems + targetIdx * batchSize,
                               batchSize, smemOffset);
					__syncthreads();

					if (!threadIdx.x) {
						changeStatus(&status[targetIdx], INUSE, AVAIL);
					}
					__syncthreads();

					if (sMergedItems[0] >= sMergedItems[2 * batchSize - 1]) {
						__syncthreads();
						for (uint32 i = threadIdx.x; i < batchSize; i += blockDim.x) {
							heapItems[currentIdx * batchSize + i] = sMergedItems[batchSize + i];
						}
					}
					else {
						__syncthreads();
                        imergePath(sMergedItems, sMergedItems + batchSize,
                                   heapItems + currentIdx * batchSize, sMergedItems,
                                   batchSize, smemOffset);
					}
					__syncthreads();

					if (!threadIdx.x) {
						changeStatus(&status[currentIdx], INUSE, AVAIL);
					}
					__syncthreads();
					// update currentIdx leftIdx rightIdx
					currentIdx = 4 * currentIdx - targetIdx + 1;
					leftIdx = 2 * currentIdx;
					rightIdx = 2 * currentIdx + 1;
				}
			}

		}

		__device__ void insertion(uint128 *items, 
								  uint32 size, 
								  uint32 smemOffset) {

			// allocate shared memory space
			extern __shared__ int s[];
			uint128 *sMergedItems = (uint128 *)&s[smemOffset];
			uint32 *tmpIdx = (uint32 *)&s[smemOffset];

            smemOffset += sizeof(uint128) * 2 * batchSize / sizeof(int);

			// move insert batch to shared memory
			// may be a partial batch, fill rest part with INT_MAX
			// TODO in this way, we can use bitonic sorting
			// but the performance may not be good when size is small
			for (uint32 i = threadIdx.x; i < batchSize; i += blockDim.x) {
                if (i < size) {
                    sMergedItems[batchSize + i] = items[i];
                }
                else {
                    uint128 tmp(INIT_LIMITS, 0, 0, 0);
                    sMergedItems[batchSize + i] = tmp;
                }
			}
			__syncthreads();

			ibitonicSort(sMergedItems + batchSize, 
						 batchSize);
			__syncthreads();

			if (!threadIdx.x) {
				changeStatus(&status[0], AVAIL, INUSE);
			}
			__syncthreads();

			/* start handling partial batch */
			// Case 1: the heap has no full batch
			// TODO current not support size > batchSize, app should handle this
			if (*batchCount == 0 && size < batchSize) {
				// Case 1.1: partial batch is empty
				if (*partialBatchSize == 0) {
					for (uint32 i = threadIdx.x; i < batchSize; i += blockDim.x) {
						heapItems[i] = sMergedItems[batchSize + i];
					}
					__syncthreads();
					if (!threadIdx.x) {
						*partialBatchSize = size;
						changeStatus(&status[0], INUSE, AVAIL);
					}
					__syncthreads();
					return;
				}
				// Case 1.2: no full batch is generated
				else if (size + *partialBatchSize < batchSize) {
					for (uint32 i = threadIdx.x; i < batchSize; i += blockDim.x) {
						sMergedItems[i] = heapItems[i];
					}
					__syncthreads();
                    imergePath(sMergedItems, sMergedItems + batchSize,
                               heapItems, sMergedItems,
                               batchSize, smemOffset);
					__syncthreads();
					if (!threadIdx.x) {
						*partialBatchSize += size;
						changeStatus(&status[0], INUSE, AVAIL);
					}
					__syncthreads();
					return;
				}
				// Case 1.3: a full batch is generated
				else if (size + *partialBatchSize >= batchSize) {
					for (uint32 i = threadIdx.x; i < batchSize; i += blockDim.x) {
						sMergedItems[i] = heapItems[i];
					}
					__syncthreads();
					if (!threadIdx.x) {
						// increase batchCount and change root batch to INUSE
						atomicAdd(batchCount, 1);
						changeStatus(&status[1], UNUSED, TARGET);
						changeStatus(&status[1], TARGET, INUSE);
					}
					__syncthreads();
                    imergePath(sMergedItems, sMergedItems + batchSize,
                               heapItems + batchSize, heapItems,
                               batchSize, smemOffset);
					__syncthreads();
					if (!threadIdx.x) {
						*partialBatchSize += (size - batchSize);
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
				if (size + *partialBatchSize < batchSize) {
					for (uint32 i = threadIdx.x; i < batchSize; i += blockDim.x) {
						sMergedItems[i] = heapItems[i];
					}
					__syncthreads();
					// Merge insert batch with partial batch
                    imergePath(sMergedItems, sMergedItems + batchSize,
                               sMergedItems, sMergedItems + batchSize,
                               batchSize, smemOffset);
					__syncthreads();
					if (!threadIdx.x) {
						changeStatus(&status[1], AVAIL, INUSE);
					}
					__syncthreads();
					for (uint32 i = threadIdx.x; i < batchSize; i += blockDim.x) {
						sMergedItems[batchSize + i] = heapItems[batchSize + i];
					}
					__syncthreads();
                    imergePath(sMergedItems, sMergedItems + batchSize,
                               heapItems + batchSize, heapItems,
                               batchSize, smemOffset);
					__syncthreads();
					if (!threadIdx.x) {
						*partialBatchSize += size;
						changeStatus(&status[0], INUSE, AVAIL);
						changeStatus(&status[1], INUSE, AVAIL);
					}
					__syncthreads();
					return;
				}
				// Case 2.2: a full batch is generated and needed to be propogated
				else if (size + *partialBatchSize >= batchSize) {
					for (uint32 i = threadIdx.x; i < batchSize; i += blockDim.x) {
						sMergedItems[i] = heapItems[i];
					}
					__syncthreads();
					// Merge insert batch with partial batch, leave larger half in the partial batch
                    imergePath(sMergedItems, sMergedItems + batchSize,
                               sMergedItems + batchSize, heapItems,
                               batchSize, smemOffset);
					__syncthreads();
					if (!threadIdx.x) {
						// update partial batch size 
						*partialBatchSize += (size - batchSize);
					}
					__syncthreads();
				}
			}
			/* end handling partial batch */

			if (!threadIdx.x) {
				//            *tmpIdx = getReversedIdx(atomicAdd(batchCount, 1) + 1);
				*tmpIdx = atomicAdd(batchCount, 1) + 1;
				changeStatus(&status[*tmpIdx], UNUSED, TARGET);
			}
			__syncthreads();

			uint32 prevIdx = 0;
			uint32 currentIdx = 1;
			uint32 targetIdx = *tmpIdx;
			__syncthreads();

			while(currentIdx != targetIdx) {
				if (!threadIdx.x) {
					*tmpIdx = currentIdx;
					if (status[targetIdx] == MARKED) {
						*tmpIdx = 0;
					}
					else {
						while(atomicCAS(&status[currentIdx], AVAIL, INUSE) != AVAIL) {
							if (status[targetIdx] == MARKED) {
								*tmpIdx = 0;
								break;
							}
						}
						if (*tmpIdx) changeStatus(&status[prevIdx], INUSE, AVAIL);
					}
				}
				__syncthreads();

				currentIdx = *tmpIdx;
				__syncthreads();

				// insert target has been required by another delete worker
				if (!currentIdx) break;

				// move batch to shard memory
				for (uint32 i = threadIdx.x; i < batchSize; i += blockDim.x) {
					sMergedItems[i] = heapItems[currentIdx * batchSize + i];
				}
				__syncthreads();

				if (sMergedItems[2 * batchSize - 1] <= sMergedItems[0]) {
					for (uint32 i = threadIdx.x; i < batchSize; i += blockDim.x) {
						heapItems[currentIdx * batchSize + i] = sMergedItems[batchSize + i];
					}
					__syncthreads();

					for (uint32 i = threadIdx.x; i < batchSize; i += blockDim.x) {
						sMergedItems[batchSize + i] = sMergedItems[i];
					}
					__syncthreads();
				}
				else if (sMergedItems[batchSize - 1] > sMergedItems[batchSize]) {
                    imergePath(sMergedItems, sMergedItems + batchSize,
                            heapItems + currentIdx * batchSize, sMergedItems + batchSize,
                            batchSize, smemOffset);
					__syncthreads();
				}
				prevIdx = currentIdx;
				currentIdx = getNextIdxToTarget(currentIdx, targetIdx);
			}

			if (currentIdx) {
				if (!threadIdx.x) {
					*tmpIdx = currentIdx;
					uint32 prevStatus = atomicCAS(&status[targetIdx], TARGET, INUSE);
					while (prevStatus != TARGET) {
						if (prevStatus == MARKED) {
							*tmpIdx = 0;
							break;
						}
						else {
							prevStatus = atomicCAS(&status[targetIdx], TARGET, INUSE);
						}
					}
					if (prevStatus == TARGET) {
						changeStatus(&status[prevIdx], INUSE, AVAIL);
					}
				}
				__syncthreads();

				currentIdx = *tmpIdx;
				__syncthreads();

				if (currentIdx) {
					// move larger half from shared memory to batch
					for (uint32 i = threadIdx.x; i < batchSize; i += blockDim.x) {
						heapItems[targetIdx * batchSize + i] = sMergedItems[batchSize + i];
					}
					__syncthreads();

					if (!threadIdx.x) {
						changeStatus(&status[targetIdx], INUSE, AVAIL);
					}
					__syncthreads();
				}
			}
			__syncthreads();

			if (currentIdx == 0) {

				// move larger half from shared memory to batch[root]
				for (uint32 i = threadIdx.x; i < batchSize; i += blockDim.x) {
					heapItems[batchSize + i] = sMergedItems[batchSize + i];
				}
				__syncthreads();

				if (!threadIdx.x) {
					changeStatus(&status[targetIdx], MARKED, UNUSED);
					if (prevIdx != 1) {
						changeStatus(&status[prevIdx], INUSE, AVAIL);
					}
				}
				__syncthreads();
			}
			__syncthreads();
		}
};

#endif
