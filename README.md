# Batched-based GPU Priority Queue

## Quick Start
In test/ there are two tests for testing sorting with the heap (both version with and without partial buffer) and a script file for verifying the results under different test size (# of keys to be sorted) and batch size. Thread block number and thread block size are currently fixed while the # of batched initially allocated is determined by the test size and the batch size automatically.

Use `make` to compile the test programs and `./test.sh` for executing a series of test cases. The `np` in `Bheap_np_test` stands for "no partial buffer".
```bash
$ make
nvcc -O3 -arch=sm_61 -std=c++11 -I../heap/buitdd/ -DHEAP_SORT partial_heap_test.cu -o Bheap_test
nvcc -O3 -arch=sm_61 -std=c++11 -I../heap/tditdd/ -DHEAP_SORT partial_heap_test.cu -o Theap_test
nvcc -O3 -arch=sm_61 -std=c++11 -I../heap/buitdd_np/ -DHEAP_SORT heap_test.cu -o Bheap_np_test
nvcc -O3 -arch=sm_61 -std=c++11 -I../heap/tditdd_np/ -DHEAP_SORT heap_test.cu -o Theap_np_test
$ ./test.sh
All 12 testcases passed
Success
```

When you test the code, please use -O3 and remove both -G and -g since it will cause a huge degrading of the performance. 

## heap
There are four versions of heap
- buitdd: Bottom-Up Insertion Top-Down Deletion with partial buffer
- buitdd_np: Bottom-Up Insertion Top-Down Deletion without partial buffer
- tditdd: Top-Down Insertion Top-Down Deletion with partial buffer
- tditdd_np: Top-Down Insertion Top-Down Deletion without partial buffer

### Useful APIs/functions related to heap 
- `Heap<K>(batch_num, batch_size, init_limits)` init_limits specifies the default max value (since the heap is a min-heap) of a key in the heap. For the case that you want to use a max heap, I usually use a signed key with negative value.
- `bool checkInsertHeap()` host function that checks whether the data (in the device heap) is correct (heap properties are satisfied).
- `bool deleteRoot(K *items, int &size)` delete the root batch from the heap, size = batchSize when there are at least K = batchSize items in the heap. If K is less than batchSize size = K and only the K items will be deleted. Returns true if the heap properties are destroyed and a delete update is needed. Otherwise return false.
- `void deleteUpdate(int smemoffset)` should be invoked when deleteRoot() return true. smemoffset should be specified when some shared memory space are reserved for other use like application itself.
- `void insertion(K *items, int size, int smemoffset)` inserts *size* new items into the heap, smemoffset is the same thing you specified in `deleteUpdate()`. Current code does not support size > batchSize, so to handle this, multiple calls of insertion() should be invoked.

There are some MACROs that are used for applications which I will add later.

### MACRO (TODO)

**-DHEAP_SORT**
The -DHEAP_SORT is only used when you want the delete keys are maintained with a global order. In most cases, this macro should not be used when you are implementing heap with real applications. For instance:

## Applications (TODO)
