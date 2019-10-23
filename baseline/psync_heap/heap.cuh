#ifndef HEAP_CUH
#define HEAP_CUH

#include "sort.cuh"
#include <iostream>
#include <cstdio>

#define UNUSED 0
#define USED 1
#define LASTONE -1

using namespace std;

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
	
	int *bufferKeys;
	int *bufferVals;
	
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
		int *h_bufferKeys = new int[tableSize * batchSize];
		int *h_bufferVals = new int[tableSize * batchSize];
		
		cudaMemcpy(h_node, node, sizeof(int) * tableSize, cudaMemcpyDeviceToHost);
		if (type == 0) {
			cudaMemcpy(h_target, target, sizeof(int) * tableSize, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_bufferKeys, bufferKeys, sizeof(int) * tableSize * batchSize, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_bufferVals, bufferVals, sizeof(int) * tableSize * batchSize, cudaMemcpyDeviceToHost);

			cout << "insert table buffer:\n";
			for (int i = startIdx; i != endIdx; ++i) {
				cout << "entry " << i << " node: " << h_node[i] << " target: " << h_target[i] << endl;
				for (int j = 0; j < batchSize; ++j) {
					cout << h_bufferKeys[i * batchSize + j] << " " << h_bufferVals[i * batchSize + j];
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
			cudaMalloc((void **)&bufferKeys, sizeof(int) * tableSize * batchSize);
			cudaMalloc((void **)&bufferVals, sizeof(int) * tableSize * batchSize);
			// target = (int *)malloc(sizeof(int) * tableSize);
			// bufferKeys = (int *)malloc(sizeof(int) * tableSize * batchSize);
			// bufferVals = (int *)malloc(sizeof(int) * tableSize * batchSize);
		}
	}

};

template <typename K, typename V>
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
	
    K *heapKeys;
    V *heapVals;
	/* 
		Instead of using the queue in controller (on CPU),
		store every batch's status(USED/UNUSED/LASTONE).
	*/
	int *status;

    Heap (int _batchNum, int _batchSize, int _tableSize) :
		batchNum(_batchNum), batchSize(_batchSize), 
		batchCount(0), tableSize(_tableSize)
    {
		// prepare device heap
		cudaMalloc((void **)&heapKeys, sizeof(K) * batchSize * batchNum);
		cudaMalloc((void **)&heapVals, sizeof(V) * batchSize * batchNum);
		cudaMalloc((void **)&status, sizeof(int) * batchNum);
    }

    ~Heap() 
	{
        cudaFree(heapKeys);
        heapKeys = NULL;
        cudaFree(heapVals);
        heapVals = NULL;
		cudaFree(status);
		status = NULL;
    }

	int maxIdx(int oriIdx) {
        int i = 1;
        while (i < oriIdx) {
            i <<= 1;
        }
        return i;
    }
	
    void printHeap() {
		
//		delTableBuffer->printTB();
//		insTableBuffer->printTB();
		
        K *h_key = new K[batchSize * (maxIdx(batchCount) + 1)];
        V *h_value = new V[batchSize * (maxIdx(batchCount) + 1)];
        int *h_status = new int[maxIdx(batchCount) + 1];
        int h_batchCount = batchCount;
		
        cudaMemcpy(h_key, heapKeys, sizeof(K) * batchSize * (maxIdx(h_batchCount) + 1), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_value, heapVals, sizeof(V) * batchSize * (maxIdx(h_batchCount) + 1), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_status, status, sizeof(int) * (maxIdx(batchCount) + 1), cudaMemcpyDeviceToHost);

        for (int i = 0; i < maxIdx(h_batchCount) + 1; ++i) {
            cout << "batch " << i << "_" << h_status[i] << ": ";
            for (int j = 0; j < batchSize; ++j) {
                cout << h_key[i * batchSize + j] << " " << h_value[i * batchSize + j];
                cout << " | ";
            }
            cout << endl;
        }

    }

};


__device__ int getReversedIdx(int oriIdx) {
	int l = __clz(oriIdx + 1) + 1;
	return ((__brev(oriIdx + 1) >> l) | (1 << (32-l))) - 1;
}

__device__ int getNextIdxToTarget(int currentIdx, int targetIdx)
{
	return ((targetIdx + 1) >> (__clz(currentIdx + 1) - __clz(targetIdx + 1) - 1)) - 1;
}

template <typename K, typename V>
__global__ void insertItems(Heap<K, V> *heap, 
							K *insertKeys, V *insertVals,
							TB *insTableBuffer)
{		
	int batchSize = heap->batchSize;
	extern __shared__ int s[];
	K *sKeys = (K *)&s[0];
	V *sVals = (V *)&sKeys[2 * batchSize];
	
	/* Load data from insert batch to shared memory */
	for (int j = threadIdx.x; j < batchSize; j += blockDim.x) {
		sKeys[batchSize + j] = insertKeys[j];
		sVals[batchSize + j] = insertVals[j];
	}
	__syncthreads();
		
	
	if (heap->batchCount != 0) {
		/* Load data from root batch to shared memory */
		for (int j = threadIdx.x; j < batchSize; j += blockDim.x) {
			sKeys[j] = heap->heapKeys[j];
			sVals[j] = heap->heapVals[j];
		}
		ibitonicSort(sKeys, 
                     sVals,
                     2 * batchSize);
		__syncthreads();
		
		/* restore data from shared memory to root batch */
		for (int j = threadIdx.x; j < batchSize; j += blockDim.x) {
			heap->heapKeys[j] = sKeys[j];
			heap->heapVals[j] = sVals[j];
		}

        /* Initialized an insert entry */
        int index = insTableBuffer->endIdx;
        for (int j = threadIdx.x; j < batchSize; j += blockDim.x) {
            insTableBuffer->bufferKeys[index * batchSize + j] = sKeys[batchSize + j];
            insTableBuffer->bufferVals[index * batchSize + j] = sVals[batchSize + j];
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
                     sVals + batchSize,
                     batchSize);
		__syncthreads();

		/* restore data from shared memory to root batch */
		for (int j = threadIdx.x; j < batchSize; j += blockDim.x) {
			heap->heapKeys[j] = sKeys[batchSize + j];
			heap->heapVals[j] = sVals[batchSize + j];
		}
        __syncthreads();

        if (!threadIdx.x) {
            heap->batchCount++;
        }
        __syncthreads();
	}
	
}

template <typename K, typename V>
__global__ void deleteItems(Heap<K, V> *heap, 
							K *deleteKeys, V *deleteVals, 
							TB *insTableBuffer,
							TB *delTableBuffer)
{		
	int batchSize = heap->batchSize;

	/* Load data from root batch to delete batch */
	for (int j = threadIdx.x; j < batchSize; j += blockDim.x) {
		deleteKeys[j] = heap->heapKeys[j];
		deleteVals[j] = heap->heapVals[j];
	}
	__syncthreads();
	
	/* Load an entry from insert table buffer or the last batch */
	if (insTableBuffer->validEntryNum != 0) {
		int index = insTableBuffer->endIdx - 1;
		for (int j = threadIdx.x; j < batchSize; j += blockDim.x) {
			heap->heapKeys[j] = insTableBuffer->bufferKeys[index * batchSize + j];
			heap->heapVals[j] = insTableBuffer->bufferVals[index * batchSize + j];
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
			heap->heapVals[j] = heap->heapVals[index * batchSize + j];
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
template <typename K, typename V>
__global__ void updateTableBuffer(Heap<K, V> *heap,
								  TB *insTableBuffer,
								  TB *delTableBuffer,
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

template <typename K, typename V>
__global__ void insertUpdate(Heap<K, V> *heap,
							 TB *insTableBuffer)
{
	int blockNum = gridDim.x;
	int blockSize = blockDim.x;
	
	int entryNum = insTableBuffer->validEntryNum;
	int entryStartIdx = insTableBuffer->startIdx;
	
	int batchSize = heap->batchSize;
	
	extern __shared__ int s[];
	K *sKeys = (K *)&s[0];
	V *sVals = (V *)&sKeys[2 * batchSize];

	for (int i = blockIdx.x; i < entryNum; i += blockNum) {
		int tableIdx = entryStartIdx + i;
		int currentIdx = insTableBuffer->node[tableIdx];
		int targetIdx = insTableBuffer->target[tableIdx];
		
		if (currentIdx == targetIdx) {
			/* Restore data back to target batch */
			for (int j = threadIdx.x; j < batchSize; j += blockSize) {
				heap->heapKeys[targetIdx * batchSize + j] = insTableBuffer->bufferKeys[tableIdx * batchSize + j];
				heap->heapVals[targetIdx * batchSize + j] = insTableBuffer->bufferVals[tableIdx * batchSize + j];
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
			sVals[j] = heap->heapVals[currentIdx * batchSize + j];
		}
		
		/* Load data from buffer to shared memory */
		for (int j = threadIdx.x; j < batchSize; j += blockSize) {
			sKeys[batchSize + j] = insTableBuffer->bufferKeys[tableIdx * batchSize + j];
			sVals[batchSize + j] = insTableBuffer->bufferVals[tableIdx * batchSize + j];
		}
		
		__syncthreads();
		
        imergePath(sKeys, sVals,
                   sKeys + batchSize, sVals + batchSize,
                   &heap->heapKeys[currentIdx * batchSize], 
                   &heap->heapVals[currentIdx * batchSize],
                   sKeys + batchSize, sVals + batchSize,
                   batchSize, 4 * batchSize);
		__syncthreads();
				
		/* Update currentIdx to one of its child batch, decided by targetIdx */
		currentIdx = getNextIdxToTarget(currentIdx, targetIdx);
		
		if (currentIdx == targetIdx) {
			/* If we reach target batch */
			
			/* Restore data to target node */
			for (int j = threadIdx.x; j < batchSize; j += blockSize) {
				heap->heapKeys[targetIdx * batchSize + j] = sKeys[batchSize + j];
				heap->heapVals[targetIdx * batchSize + j] = sVals[batchSize + j];
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
				insTableBuffer->bufferVals[tableIdx * batchSize + j] = sVals[batchSize + j];
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

template <typename K, typename V>
__global__ void deleteUpdate(Heap<K, V> *heap,
							 TB *delTableBuffer) 
{
	int blockNum = gridDim.x;
	int blockSize = blockDim.x;
	
	int entryNum = delTableBuffer->validEntryNum;
	int entryStartIdx = delTableBuffer->startIdx;
	
	int batchSize = heap->batchSize;
	
	extern __shared__ int s[];
	K *sKeys = (K *)&s[0];
	V *sVals = (V *)&sKeys[3 * batchSize];

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
				sVals[j] = heap->heapVals[parentIdx * batchSize + j];
			}
			/* Load data from left child batch to shared memory */
			for (int j = threadIdx.x; j < batchSize; j += blockSize) {
				sKeys[batchSize + j] = heap->heapKeys[lchildIdx * batchSize + j];
				sVals[batchSize + j] = heap->heapVals[lchildIdx * batchSize + j];
			}
			__syncthreads();
			/* We need to check if there is a used right child batch */
			if (heap->status[rchildIdx] == UNUSED) {
				/* 
				Right child batch does not exist, so doing sorting, then
				store data back to heap and exit this update
				*/
                imergePath(sKeys, sVals,
                           sKeys + batchSize, sVals + batchSize,
                           &heap->heapKeys[parentIdx * batchSize],
                           &heap->heapVals[parentIdx * batchSize],
                           &heap->heapKeys[lchildIdx * batchSize],
                           &heap->heapVals[lchildIdx * batchSize],
                           batchSize, 6 * batchSize);
                __syncthreads();
			}
			else {
				/* Right child batch exists  */
				
				/* Load data from right child batch to shared memory */
				for (int j = threadIdx.x; j < batchSize; j += blockDim.x) {
					sKeys[2 * batchSize + j] = heap->heapKeys[rchildIdx * batchSize + j];
					sVals[2 * batchSize + j] = heap->heapVals[rchildIdx * batchSize + j];
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

                imergePath(sKeys + batchSize, sVals + batchSize,
                           sKeys + 2 * batchSize, sVals + 2 * batchSize,
                           sKeys + batchSize, sVals + batchSize,
                           &heap->heapKeys[largeIdx * batchSize],
                           &heap->heapVals[largeIdx * batchSize],
                           batchSize, 6 * batchSize);
                __syncthreads();

                imergePath(sKeys, sVals,
                           sKeys + batchSize, sVals + batchSize,
                           &heap->heapKeys[parentIdx * batchSize],
                           &heap->heapVals[parentIdx * batchSize],
                           &heap->heapKeys[smallIdx * batchSize],
                           &heap->heapVals[smallIdx * batchSize],
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
