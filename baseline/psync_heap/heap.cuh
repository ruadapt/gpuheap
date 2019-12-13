#ifndef HEAP_CUH
#define HEAP_CUH

#include "sort.cuh"
#include <iostream>
#include <cstdio>

#define UNUSED 0
#define USED 1
#define LASTONE -1

using namespace std;

template <class T>
struct TB {
	/* Number of available entries */
	int tableSize;
	
	/* Index of where next update should starts at */
	int startIdx;
	int startOffset;
	/* Number of entries that next update should works on */
	int validEntryNum;
	/* 
		The index where last available entries locates
		This index will be increased when we update/insert
		new entries into the table buffer
	*/
	int endIdx;
	/* The type of the table buffer */
	int type;
	int batchSize;
	
	/* Current batch */
	int *node;
	/* Target batch node */
	int *target;
	
	T *bufferKeys;
	
	TB (int _tableSize, int _batchSize, int _type)
	{
		initial(_tableSize, _batchSize, _type);
	}

    bool isEmpty() {
        return (startIdx == endIdx);
    }
	
    void printTB() {
		int *h_node = new int[tableSize];
		int *h_target = new int[tableSize];
		T *h_bufferKeys = new T[tableSize * batchSize];
		
		cudaMemcpy(h_node, node, sizeof(int) * tableSize, cudaMemcpyDeviceToHost);
		if (type == 0) {
			cudaMemcpy(h_target, target, sizeof(int) * tableSize, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_bufferKeys, bufferKeys, sizeof(T) * tableSize * batchSize, cudaMemcpyDeviceToHost);

			cout << "insert table buffer:\n";
			for (int i = startIdx; i != endIdx; ++i) {
				cout << "entry " << i << " node: " << h_node[i] << " target: " << h_target[i] << endl;
				for (int j = 0; j < batchSize; ++j) {
					cout << h_bufferKeys[i * batchSize + j];
					cout << " | ";
				}
				cout << endl;
			}
		}
		else {
			cout << "delete table buffer:\n";
			for (int i = startIdx; i != endIdx; ++i) {
				cout << "entry " << i << " node: " << h_node[i] << endl;
			}
		}
    }


	// type = 0: insert table buffer
	// type = 1: delete table buffer
	void initial(int _tableSize, int _batchSize, int _type)
	{
		tableSize = _tableSize;
		batchSize = _batchSize;
		type = _type;
		startIdx = 0;
		startOffset = 0;
		validEntryNum = 0;
		endIdx = 0;
		cudaMalloc((void **)&node, sizeof(int) * tableSize);
		/* 
			if type = 1 this is a delete table buffer
				It only need to store the node it locates at, in the paper, the author
				also uses buffer to store largest batch at each updates which seems to
				be redundant, since you still need to move everything into shared memory
				when do merging or sorting
			if type = 0 this is an insert table buffer
				For
		*/
		if (type == 0) {
			cudaMalloc((void **)&target, sizeof(int) * tableSize);
			cudaMalloc((void **)&bufferKeys, sizeof(T) * tableSize * batchSize);
			// target = (int *)malloc(sizeof(int) * tableSize);
			// bufferKeys = (int *)malloc(sizeof(int) * tableSize * batchSize);
			// bufferVals = (int *)malloc(sizeof(int) * tableSize * batchSize);
		}
	}

};

template <typename T>
class Heap {
public:

	/* Max batch number */
    int batchNum;
	/* The number of data items that a batch can contain */
    int batchSize;
	/* The number of used batches*/
	int batchCount;
	
	/* The number of entries in table buffer */
    int tableSize;
	
    T *heapKeys;
	/* 
		Instead of using the queue in controller (on CPU),
		store every batch's status(USED/UNUSED/LASTONE).
	*/
	int *status;

    int *globalBenefit;

    Heap (int _batchNum, int _batchSize, int _tableSize) :
		batchNum(_batchNum), batchSize(_batchSize), 
		batchCount(0), tableSize(_tableSize)
    {
		// prepare device heap
		cudaMalloc((void **)&heapKeys, sizeof(T) * batchSize * batchNum);
        T *h_keys = new T[batchSize * batchNum]();
        cudaMemcpy(heapKeys, h_keys, batchSize * batchNum * sizeof(T), cudaMemcpyHostToDevice);
        delete []h_keys;

		cudaMalloc((void **)&status, sizeof(int) * batchNum);
        int *h_status = new int[batchNum];
        for (int i = 0; i < batchNum; ++i) h_status[i] = UNUSED;
        cudaMemcpy(status, h_status, batchNum * sizeof(int), cudaMemcpyHostToDevice);
        delete []h_status;

        cudaMalloc((void **)&globalBenefit, sizeof(int));
        cudaMemset(globalBenefit, 0, sizeof(int));
    }

    void reset() {
		cudaMalloc((void **)&heapKeys, sizeof(T) * batchSize * batchNum);
        T *h_keys = new T[batchSize * batchNum]();
        cudaMemcpy(heapKeys, h_keys, batchSize * batchNum * sizeof(T), cudaMemcpyHostToDevice);
        delete []h_keys;
		cudaMalloc((void **)&status, sizeof(int) * batchNum);
        int *h_status = new int[batchNum];
        for (int i = 0; i < batchNum; ++i) h_status[i] = UNUSED;
        cudaMemcpy(status, h_status, batchNum * sizeof(int), cudaMemcpyHostToDevice);
        delete []h_status;
        batchCount = 0;
    }

    ~Heap() 
	{
        cudaFree(heapKeys);
        heapKeys = NULL;
		cudaFree(status);
		status = NULL;
        cudaFree(globalBenefit);
        globalBenefit = NULL;
    }

	int maxIdx(int oriIdx) {
        return oriIdx;
        /*int i = 1;*/
        /*while (i < oriIdx) {*/
            /*i <<= 1;*/
        /*}*/
        /*return i;*/
    }
	
    int GetBatchCount() {
        return batchCount;
    }

    void checkHeap() {
				
        T *h_key = new T[batchSize * (maxIdx(batchCount) + 1)];
        int *h_status = new int[maxIdx(batchCount) + 1];
        int h_batchCount = batchCount;
		
        cudaMemcpy(h_key, heapKeys, sizeof(T) * batchSize * (maxIdx(h_batchCount) + 1), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_status, status, sizeof(int) * (maxIdx(batchCount) + 1), cudaMemcpyDeviceToHost);

        for (int i = 0; i < maxIdx(h_batchCount) + 1; ++i) {
            if (i != 0  && h_key[i * batchSize] < h_key[i / 2 * batchSize]) {
                cout << "Error1" << endl;
                return;
            }
            for (int j = 1; j < batchSize; ++j) {
                if (h_key[i * batchSize + j] < h_key[i * batchSize + j - 1]) {
                    cout << "Error2" << endl;
                    return;
                }
            }
        }

    }

    void printHeap() {
		
//		delTableBuffer->printTB();
//		insTableBuffer->printTB();
		
        T *h_key = new T[batchSize * (maxIdx(batchCount) + 1)];
        int *h_status = new int[maxIdx(batchCount) + 1];
        int h_batchCount = batchCount;
		
        cudaMemcpy(h_key, heapKeys, sizeof(T) * batchSize * (maxIdx(h_batchCount) + 1), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_status, status, sizeof(int) * (maxIdx(batchCount) + 1), cudaMemcpyDeviceToHost);

        for (int i = 0; i < maxIdx(h_batchCount) + 1; ++i) {
            cout << "batch " << i << "_" << h_status[i] << ": ";
            for (int j = 0; j < batchSize; ++j) {
                cout << h_key[i * batchSize + j];
                cout << " | ";
            }
            cout << endl;
        }

    }

};


__device__ int getReversedIdx(int oriIdx) {
    return oriIdx;
	/*int l = __clz(oriIdx + 1) + 1;*/
	/*return ((__brev(oriIdx + 1) >> l) | (1 << (32-l))) - 1;*/
}

__device__ int getNextIdxToTarget(int currentIdx, int targetIdx)
{
	return ((targetIdx + 1) >> (__clz(currentIdx + 1) - __clz(targetIdx + 1) - 1)) - 1;
}

template <typename T>
__global__ void insertItems(Heap<T> *heap, 
							T *insertKeys,
							TB<T> *insTableBuffer)
{		
	int batchSize = heap->batchSize;
	extern __shared__ int s[];
	T *sKeys = (T *)&s[0];
	
	/* Load data from insert batch to shared memory */
	for (int j = threadIdx.x; j < batchSize; j += blockDim.x) {
		sKeys[batchSize + j] = insertKeys[j];
	}
	__syncthreads();
		
	if (heap->batchCount != 0) {
		/* Load data from root batch to shared memory */
		for (int j = threadIdx.x; j < batchSize; j += blockDim.x) {
			sKeys[j] = heap->heapKeys[j];
		}
		ibitonicSort(sKeys, 
                     2 * batchSize);
		__syncthreads();
		
		/* restore data from shared memory to root batch */
		for (int j = threadIdx.x; j < batchSize; j += blockDim.x) {
			heap->heapKeys[j] = sKeys[j];
		}

        /* Initialized an insert entry */
        int index = insTableBuffer->endIdx;
        for (int j = threadIdx.x; j < batchSize; j += blockDim.x) {
            insTableBuffer->bufferKeys[index * batchSize + j] = sKeys[batchSize + j];
        }
        __syncthreads();
        
        if (!threadIdx.x) {
            insTableBuffer->target[index] = getReversedIdx(heap->batchCount);
            insTableBuffer->node[index] = getNextIdxToTarget(0, insTableBuffer->target[index]);
            insTableBuffer->endIdx++;
            heap->batchCount++;
        }
        __syncthreads();
	}
	else {
        ibitonicSort(sKeys + batchSize,
                     batchSize);
		__syncthreads();

		/* restore data from shared memory to root batch */
		for (int j = threadIdx.x; j < batchSize; j += blockDim.x) {
			heap->heapKeys[j] = sKeys[batchSize + j];
		}
        __syncthreads();

        if (!threadIdx.x) {
            heap->batchCount++;
        }
        __syncthreads();
	}
	
}

template <typename T>
__global__ void deleteItems(Heap<T> *heap, 
							T *deleteKeys,
							TB<T> *insTableBuffer,
							TB<T> *delTableBuffer)
{		
	int batchSize = heap->batchSize;

	/* Load data from root batch to delete batch */
	for (int j = threadIdx.x; j < batchSize; j += blockDim.x) {
		deleteKeys[j] = heap->heapKeys[j];
	}
	__syncthreads();
	
	/* Load an entry from insert table buffer or the last batch */
	if (insTableBuffer->validEntryNum != 0) {
		int index = insTableBuffer->endIdx - 1;
		for (int j = threadIdx.x; j < batchSize; j += blockDim.x) {
			heap->heapKeys[j] = insTableBuffer->bufferKeys[index * batchSize + j];
		}
		__syncthreads();
		/* Update insert table buffer */
		if (!threadIdx.x) {
			insTableBuffer->endIdx--;
			insTableBuffer->endIdx += insTableBuffer->tableSize;
			insTableBuffer->endIdx %= insTableBuffer->tableSize;
			insTableBuffer->validEntryNum--;
		}
		__syncthreads();
	}
	else {
		int index = getReversedIdx(heap->batchCount - 1);
		for (int j = threadIdx.x; j < batchSize; j += blockDim.x) {
			heap->heapKeys[j] = heap->heapKeys[index * batchSize + j];
		}
		__syncthreads();
		/* Update heap status */
		if (!threadIdx.x) {
			heap->status[index] = UNUSED;
		}
		__syncthreads();
	}
	
	/* Initialized an delete entry */
	if (!threadIdx.x) {
		delTableBuffer->node[delTableBuffer->endIdx] = 0;
		delTableBuffer->endIdx++;
		delTableBuffer->endIdx %= delTableBuffer->tableSize;
		heap->batchCount--;
	}
	__syncthreads();		
}

// return if the table buffer is empty or not
// FIXME should be separate
template <typename T>
__global__ void updateTableBuffer(Heap<T> *heap,
								  TB<T> *insTableBuffer,
								  TB<T> *delTableBuffer,
                                  bool *status)
{
	if (!threadIdx.x && !blockIdx.x) {
		delTableBuffer->startIdx += delTableBuffer->validEntryNum;
		delTableBuffer->validEntryNum = delTableBuffer->endIdx - delTableBuffer->startIdx;
		delTableBuffer->startIdx %= delTableBuffer->tableSize;
		delTableBuffer->endIdx %= delTableBuffer->tableSize;
							
		insTableBuffer->startIdx += insTableBuffer->startOffset;
		insTableBuffer->validEntryNum = insTableBuffer->endIdx - insTableBuffer->startIdx;
		insTableBuffer->startIdx %= insTableBuffer->tableSize;
		insTableBuffer->endIdx %= insTableBuffer->tableSize;
		insTableBuffer->startOffset = 0; 

        status[0] = false;

        if ((delTableBuffer->startIdx == delTableBuffer->endIdx) && 
            (insTableBuffer->startIdx == insTableBuffer->endIdx)) {
            status[0] = true;
        }
    }
}

template <typename T>
__global__ void insertUpdate(Heap<T> *heap,
							 TB<T> *insTableBuffer)
{
	int blockNum = gridDim.x;
	int blockSize = blockDim.x;
	
	int entryNum = insTableBuffer->validEntryNum;
	int entryStartIdx = insTableBuffer->startIdx;
	
	int batchSize = heap->batchSize;
	
	extern __shared__ int s[];
	T *sKeys = (T *)&s[0];

	for (int i = blockIdx.x; i < entryNum; i += blockNum) {
		int tableIdx = entryStartIdx + i;
		int currentIdx = insTableBuffer->node[tableIdx];
		int targetIdx = insTableBuffer->target[tableIdx];
		
		if (currentIdx == targetIdx) {
			/* Restore data back to target batch */
			for (int j = threadIdx.x; j < batchSize; j += blockSize) {
				heap->heapKeys[targetIdx * batchSize + j] = insTableBuffer->bufferKeys[tableIdx * batchSize + j];
			}
			__syncthreads();
			
			if (!threadIdx.x) {
				insTableBuffer->startOffset++;
				heap->status[targetIdx] = USED;
			}
			__syncthreads();
			
			continue;
		}
		
		/* Load data from current batch to shared memory */
		for (int j = threadIdx.x; j < batchSize; j += blockSize) {
			sKeys[j] = heap->heapKeys[currentIdx * batchSize + j];
		}
		
		/* Load data from buffer to shared memory */
		for (int j = threadIdx.x; j < batchSize; j += blockSize) {
			sKeys[batchSize + j] = insTableBuffer->bufferKeys[tableIdx * batchSize + j];
		}
		
		__syncthreads();
		
        imergePath(sKeys,
                   sKeys + batchSize,
                   &heap->heapKeys[currentIdx * batchSize], 
                   sKeys + batchSize, 
                   batchSize, 4 * batchSize);
		__syncthreads();
				
		/* Update currentIdx to one of its child batch, decided by targetIdx */
		currentIdx = getNextIdxToTarget(currentIdx, targetIdx);
		
		if (currentIdx == targetIdx) {
			/* If we reach target batch */
			
			/* Restore data to target node */
			for (int j = threadIdx.x; j < batchSize; j += blockSize) {
				heap->heapKeys[targetIdx * batchSize + j] = sKeys[batchSize + j];
			}
			
			if (!threadIdx.x) {
				insTableBuffer->startOffset++;
				heap->status[targetIdx] = USED;
			}
			__syncthreads();
		}
		else {
			/* If we do not reach the target batch */
			// update the entry in the insTableBuffer			
			for (int j = threadIdx.x; j < batchSize; j += blockDim.x) {
				insTableBuffer->bufferKeys[tableIdx * batchSize + j] = sKeys[batchSize + j];
			}
			__syncthreads();
			
			if (!threadIdx.x) {
				insTableBuffer->node[tableIdx] = currentIdx;
				insTableBuffer->target[tableIdx] = targetIdx;
			}
			__syncthreads();
		}				
	}
}

template <typename T>
__global__ void deleteUpdate(Heap<T> *heap,
							 TB<T> *delTableBuffer) 
{
	int blockNum = gridDim.x;
	int blockSize = blockDim.x;
	
	int entryNum = delTableBuffer->validEntryNum;
	int entryStartIdx = delTableBuffer->startIdx;
	
	int batchSize = heap->batchSize;
	
	extern __shared__ int s[];
	T *sKeys = (T *)&s[0];

	/* each thread block is assigned an entry in delTableBuffer */
	for (int i = blockIdx.x; i < entryNum; i += blockNum) {
		
		int tableIdx = entryStartIdx + i;
		int parentIdx = delTableBuffer->node[tableIdx];
		int lchildIdx = parentIdx * 2 + 1;
		int rchildIdx = lchildIdx + 1;
				
		if (heap->status[lchildIdx] == UNUSED) {
			/* No used child batches */
			continue;
		}
		else {
			/* Load data from parent batch to shared memory */
			for (int j = threadIdx.x; j < batchSize; j += blockSize) {
				sKeys[j] = heap->heapKeys[parentIdx * batchSize + j];
			}
			/* Load data from left child batch to shared memory */
			for (int j = threadIdx.x; j < batchSize; j += blockSize) {
				sKeys[batchSize + j] = heap->heapKeys[lchildIdx * batchSize + j];
			}
			__syncthreads();
			/* We need to check if there is a used right child batch */
			if (heap->status[rchildIdx] == UNUSED) {
				/* 
				Right child batch does not exist, so doing sorting, then
				store data back to heap and exit this update
				*/
                imergePath(sKeys,
                           sKeys + batchSize,
                           &heap->heapKeys[parentIdx * batchSize],
                           &heap->heapKeys[lchildIdx * batchSize],
                           batchSize, 6 * batchSize);
                __syncthreads();
			}
			else {
				/* Right child batch exists  */
				
				/* Load data from right child batch to shared memory */
				for (int j = threadIdx.x; j < batchSize; j += blockDim.x) {
					sKeys[2 * batchSize + j] = heap->heapKeys[rchildIdx * batchSize + j];
				}
				__syncthreads();
				
				/* 
					Need to know where child is the larger one
					Original larger child batch will always be the next parent batch
				*/
				int smallIdx = (sKeys[2 * batchSize - 1] < sKeys[3 * batchSize - 1]) ? 
								lchildIdx : rchildIdx;
				int largeIdx = (sKeys[2 * batchSize - 1] < sKeys[3 * batchSize - 1]) ? 
								rchildIdx : lchildIdx;
				__syncthreads();

                imergePath(sKeys + batchSize,
                           sKeys + 2 * batchSize,
                           sKeys + batchSize,
                           &heap->heapKeys[largeIdx * batchSize],
                           batchSize, 6 * batchSize);
                __syncthreads();

                imergePath(sKeys,
                           sKeys + batchSize,
                           &heap->heapKeys[parentIdx * batchSize],
                           &heap->heapKeys[smallIdx * batchSize],
                           batchSize, 6 * batchSize);
                __syncthreads();
			
				/* Check if smallIdx batch has any child batch and if early stop apply */
				if (!threadIdx.x && 
					(heap->status[smallIdx * 2 + 1] != UNUSED || 
					 heap->heapKeys[smallIdx * batchSize + batchSize - 1] < 
					 heap->heapKeys[(smallIdx * 2 + 1) * batchSize])) {
					/* Add a new entry to delete table buffer */
					int index = atomicAdd(&delTableBuffer->endIdx, 1) % delTableBuffer->tableSize;
					delTableBuffer->node[index] = smallIdx;
				}
				__syncthreads();
			}
		}
	}
}


#endif
