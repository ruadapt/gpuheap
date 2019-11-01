#ifndef BUFFER_CUH
#define BUFFER_CUH
// TODO we may need to reuse the space and let it be a buffer
using namespace std;

template <typename K=int>
class Buffer {
public:

    K *bufferItems;

/*
beginPos------------readPos------writePos------------endPos
       reading data                      writing data
*/
    int capacity;
    int *begPos;
    int *endPos;
    int *readPos;
    int *writePos;

    int *bufferLock;

    Buffer(int _capacity) : capacity(_capacity) {
        cudaMalloc((void **)&bufferItems, sizeof(K) * capacity);
        cudaMalloc((void **)&begPos, sizeof(int));
        cudaMemset(begPos, 0, sizeof(int));
        cudaMalloc((void **)&readPos, sizeof(int));
        cudaMemset(readPos, 0, sizeof(int));
        cudaMalloc((void **)&writePos, sizeof(int));
        cudaMemset(writePos, 0, sizeof(int));
        cudaMalloc((void **)&endPos, sizeof(int));
        cudaMemset(endPos, 0, sizeof(int));
        cudaMalloc((void **)&bufferLock, sizeof(int));
        cudaMemset(bufferLock, 0, sizeof(int));
    }

    ~Buffer() {
        cudaFree(bufferItems);
        bufferItems = NULL;
        cudaFree(begPos);
        begPos = NULL;
        cudaFree(readPos);
        readPos = NULL;
        cudaFree(writePos);
        writePos = NULL;
        cudaFree(endPos);
        endPos = NULL;
        cudaFree(bufferLock);
        bufferLock = NULL;
    }

    void printBufferPtr() {
        int h_read, h_write, h_begin, h_end;
        cudaMemcpy(&h_read, readPos, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_write, writePos, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_begin, begPos, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_end, endPos, sizeof(int), cudaMemcpyDeviceToHost);

        cout << h_begin << " " << h_read << " " << h_write << " " << h_end << endl;
    }
    void printBuffer() {
        K *h_items = new K[capacity];
        cudaMemcpy(h_items, bufferItems, capacity * sizeof(K), cudaMemcpyDeviceToHost);
        int h_read, h_write, h_begin, h_end;
        cudaMemcpy(&h_read, readPos, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_write, writePos, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_begin, begPos, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_end, endPos, sizeof(int), cudaMemcpyDeviceToHost);

        cout << "item: ";
        for (int i = 0; i < capacity; ++i)
            cout << h_items[i] << " ";
        cout << endl;
        cout << h_begin << " " << h_read << " " << h_write << " " << h_end << endl;
    }

    __device__ bool isEmpty() {
        return *endPos == *readPos;
    }

    __device__ int getSize() {
        return *endPos - *readPos;
    }

    // current insert/delete implementation can provide maximum sizeof(int) items
    __device__ void insertToBuffer(K *items,
                                   int size,
                                   int smemOffset) {
        extern __shared__ int s[];
        int *insertStartPos = (int *)&s[smemOffset];
        int *numItemsForRound = (int *)&insertStartPos[1];

        if (!threadIdx.x) {
            // Get the begin position in the buffer for this insertion
            *insertStartPos = atomicAdd(endPos, size);
        }
        __syncthreads();

        int offset = 0;

        // Loop until all items are added to the buffer
        while (offset < size) {
            if (!threadIdx.x) {
                // Wait until there is some available space in the buffer
                while (*insertStartPos - *begPos >= capacity) {}
                // Determine the number of items for this round
                int remain = capacity - (*insertStartPos - *begPos);
                *numItemsForRound = size < remain ? size : remain;
                /**numItemsForRound = thrust::min(size, capacity - (*insertStartPos - *begPos));*/
            }
            __syncthreads();

            for (int i = threadIdx.x; i < *numItemsForRound; i += blockDim.x) {
                bufferItems[*insertStartPos + offset + i] = items[offset + i];
            }
            __syncthreads();

            offset += *numItemsForRound;
            __syncthreads();
        }
        if (!threadIdx.x && size) {
            while (atomicCAS(writePos, *insertStartPos, *insertStartPos + size) != *insertStartPos) {}
        }
        __syncthreads();
    }

    __device__ bool deleteFromBuffer(K *items,
                                     int &size,
                                     int smemOffset) {
        extern __shared__ int s[];
        int *deleteStartPos = (int *)&s[smemOffset];
        int *deleteSize = (int *)&deleteStartPos[1];

        if (!threadIdx.x) {
            *deleteSize = 0;
            while (1) {
                int tmpSize = *writePos - *readPos;
                *deleteSize = tmpSize < blockDim.x ? tmpSize : blockDim.x;
                if (*deleteSize == 0) break;
                int tmpReadPos = *readPos;
                if (tmpReadPos == atomicCAS(readPos, tmpReadPos, tmpReadPos + *deleteSize)) {
                    *deleteStartPos = tmpReadPos;
                    break;
                }
            }
        }
        __syncthreads();

        size = *deleteSize;
        __syncthreads();

        if (size == 0) return false;

        for (int i = threadIdx.x; i < size; i += blockDim.x) {
            items[i] = bufferItems[*deleteStartPos + i];
        }
        __syncthreads();

        if (!threadIdx.x) {
            while (atomicCAS(begPos, *deleteStartPos, *deleteStartPos + size) != *deleteStartPos) {}
        }
        __syncthreads();

        return true;
    }
};

#endif
