/*
  Heap with auxiliary data
  This is a two-layered data structure.
  The core is the batched heap structure, as described in "BGPQ: A Batch Based Priority Queue Design on GPUs".
  Beyond this structure, we add another buffer, called the aux buffer.
  The aux buffer is just like a batch in the heap. However each key in the buffer is associated with additional data (aux_data).
  When a thread retrieves keys from the structure, both the key and its associated aux_data is returned. The aux buffer is then filled with new keys extracted from the heap. The aux_data for each new key is initialized from init_aux.
  We also implement a reader-writer lock. This is used to implement garbage collection, since no insertion or retrieval may be performed during GC.
*/

#include "heap.cuh"

#define CONFIG_GC_FLAG_NUM 256

#define DECLARE_SMEM(Type, Name, Count) Type * Name = (Type *)(smem + curr_smem_offset); curr_smem_offset += sizeof(Type) * (Count); curr_smem_offset = (curr_smem_offset + 3) & ~0x03
#define SINGLE_THREADED if(threadIdx.x == 0)

template < typename T, typename Aux >
struct Heap_With_Aux {
  Heap < T > heap;
  T * aux_key_buffer;
  Aux * aux_data_buffer;
  Aux init_aux;
  T * gc_buffer;
  int curr_aux_buf_size;
  volatile int rwlock_x, rwlock_y, rwlock_z;
  //X counter is thread ticket counter
  //Threads with ticket <= Y counter are allowed to run
  //Threads with ticket < Z have finished executing
  //A spinlock takes 4 bytes, yet only uses a single bit. This is wasteful. CUDA purports to support atomic operations on 16-bit integers, yet it seems that this is only supported on devices with capability >= 7.0
  int spinlock;

  Heap_With_Aux(int batchNum, int batchSize, T init_limit, Aux init_aux_) : heap(batchNum, batchSize, init_limit), init_aux(init_aux_), curr_aux_buf_size(0), rwlock_x(0), rwlock_y(0), rwlock_z(0), spinlock(0) {
    cudaMalloc(&aux_key_buffer, sizeof(T) * batchSize * 2);
    cudaMalloc(&aux_data_buffer, sizeof(Aux) * batchSize * 2);
    cudaMalloc(&gc_buffer, sizeof(T) * CONFIG_GC_FLAG_NUM);
  }

  __device__ void spinlock_lock(){
    while (atomicCAS(&spinlock, 0, 1) != 0) {}
  }

  __device__ void spinlock_unlock(){
    atomicExch(&spinlock, 0);
  }

  __device__ void retrieve(T * key_dst, Aux * aux_dst, int * count, bool (* chk_func)(T *, Aux *, void *), void * priv_data, int smemOffset){

    extern __shared__ char smem[];
    int curr_smem_offset = smemOffset;

    DECLARE_SMEM(int, p_ticket, 1);
    SINGLE_THREADED {*p_ticket = atomicAdd((int *)&rwlock_x, 1);}
    __syncthreads();
    while(rwlock_y < *p_ticket) {}
    SINGLE_THREADED {atomicAdd((int *)&rwlock_y, 1);}

    DECLARE_SMEM(int, p_retrieve_size, 1);
    char need_update = 0; //Always have the same value in all threads

    SINGLE_THREADED {spinlock_lock();}
    __syncthreads();

    if(curr_aux_buf_size < *count){
      //We hold the spinlock when we fill the aux buffer. However we temporarily release the lock when we do deleteUpdate(). This gives other threads a chance to retrieve concurrently.
      //There are two possible ways to do this. The first way is to release the lock immediately after deleteRoot(), execute deleteUpdate(), then lock it again. The second way is to postpone deleteUpdate() until we finish all processing.
      //The first way may seem simple, but it has a serious drawback. Suppose the first thread group calls deleteRoot() and goes into deleteUpdate(). Then other thread groups come and lock, seeing that there are enough keys in the aux buffer, simply copy them away. When the first thread group finishes deleteUpdate() and comes back, there isn't enough keys in the buffer again.
      //We adopt the second approach here.

      need_update = heap.deleteRoot(aux_key_buffer + curr_aux_buf_size, *p_retrieve_size);
      for(int i = threadIdx.x;i < *p_retrieve_size;i += blockDim.x){
        aux_data_buffer[i + curr_aux_buf_size] = init_aux;
      }
      SINGLE_THREADED {curr_aux_buf_size += *p_retrieve_size;}
    }
    __syncthreads();

    SINGLE_THREADED {if(*count > curr_aux_buf_size) {*count = curr_aux_buf_size;}}
    __syncthreads();
    DECLARE_SMEM(char, p_should_keep, *count);
    for(int i = threadIdx.x;i < *count;i += blockDim.x){
      key_dst[i] = aux_key_buffer[i];
      aux_dst[i] = aux_data_buffer[i];
      p_should_keep[i] = chk_func(aux_key_buffer + i, aux_data_buffer + i, priv_data);
    }
    __syncthreads();

    DECLARE_SMEM(int, p_keep_count, 1);
    SINGLE_THREADED {
      *p_keep_count = 0;
      for(int i = 0;i < *count;++i){
        if(p_should_keep[i]){
	  aux_key_buffer[*p_keep_count] = aux_key_buffer[i];
	  aux_data_buffer[*p_keep_count] = aux_data_buffer[i];
	  ++(*p_keep_count);
	}
      }
      int shift_size = curr_aux_buf_size - *count;
      for(int i = 0;i < shift_size;++i){
        aux_key_buffer[*p_keep_count + i] = aux_key_buffer[*count + i];
	aux_data_buffer[*p_keep_count + i] = aux_data_buffer[*count + i];
      }
      curr_aux_buf_size -= (*count - *p_keep_count);
    }
    __syncthreads();

    SINGLE_THREADED {spinlock_unlock();}
    __syncthreads();

    //We don't need any of the shared memory now.
    //We measure offsets in bytes, but legacy code measures offsets in ints. Therefore divide offset by 4.
    if(need_update) {heap.deleteUpdate(smemOffset / 4);}

    SINGLE_THREADED {atomicAdd((int *)&rwlock_z, 1);}
    __syncthreads();
  }

  __device__ void garbage_collection(int smemOffset, int * vert_distance){
    //This is inefficient, but just enough to make the program fully functional. Optimize later.
    //Putting the vert_distance parameter here is really inelegant, ugly hack. Should be abstracted away using a callback function or something.
    extern __shared__ char smem[];
    int curr_smem_offset = smemOffset;

    DECLARE_SMEM(int, p_key_count, 1);
    SINGLE_THREADED {*p_key_count = *(heap.batchCount) * heap.batchSize;}
    T * heapItems = heap.heapItems + heap.batchSize; //The first batch in heap.heapItems array is the partial buffer.
    //We move the partial buffer to the end of heap
    for(int i = threadIdx.x;i < *(heap.partialBufferSize);i += blockDim.x){
      heapItems[*p_key_count + i] = heap.heapItems[i];
    }
    SINGLE_THREADED {
      *p_key_count += *(heap.partialBufferSize);
      *(heap.batchCount) = 0;
      *(heap.partialBufferSize) = 0;
    }

    //In the worst case there will be a key for every single vertex in the graph. We can't really keep a flag for every key in the heap.

    DECLARE_SMEM(char, p_should_keep, CONFIG_GC_FLAG_NUM);
    int gc_batches = *p_key_count / CONFIG_GC_FLAG_NUM + 1;

    DECLARE_SMEM(int, p_kept_key_count, 1);
    DECLARE_SMEM(int, p_need_gc, 1);
    SINGLE_THREADED {*p_need_gc = 2;}
    for(int i = 0;i < gc_batches;++i){
      for(int j = threadIdx.x, idx = i * CONFIG_GC_FLAG_NUM;j < CONFIG_GC_FLAG_NUM;j += blockDim.x, idx += blockDim.x){
        if(idx >= *p_key_count || heapItems[idx].curr_dist > vert_distance[heapItems[idx].vert]) {p_should_keep[j] = 0;} else {p_should_keep[j] = 1;}
      }
      __syncthreads();

      SINGLE_THREADED {
        *p_kept_key_count = 0;
        for(int j = 0, idx = i * CONFIG_GC_FLAG_NUM;j < CONFIG_GC_FLAG_NUM;++j, ++idx){
          if(p_should_keep[j]){
	    gc_buffer[*p_kept_key_count] = heapItems[idx];
	    ++(*p_kept_key_count);
	  }
        }
      }
      __syncthreads();

      int new_batch_count = *p_kept_key_count / heap.batchSize;
      for(int j = 0;j < new_batch_count;++j){
	  heap.insertion(gc_buffer + j * heap.batchSize, heap.batchSize, curr_smem_offset / 4, p_need_gc, NULL);
	  __syncthreads();
      }
      int rem = *p_kept_key_count % heap.batchSize;
      heap.insertion(gc_buffer + new_batch_count * heap.batchSize, rem, curr_smem_offset / 4, p_need_gc, NULL);
      __syncthreads();
    }
  }

  __device__ void insert(T * new_items, int count, int smemOffset, int * vert_distance){

    extern __shared__ char smem[];
    int curr_smem_offset = smemOffset;

    DECLARE_SMEM(int, p_ticket, 1);
    SINGLE_THREADED {*p_ticket = atomicAdd((int *)&rwlock_x, 1);}
    __syncthreads();
    while(rwlock_y < *p_ticket) {}

    //We don't need to lock here, as insertions go directly into the heap and do not touch the Aux buffer.
    DECLARE_SMEM(int, p_need_gc, 1);
    do {
      SINGLE_THREADED {*p_need_gc = 0;}
      __syncthreads();
      heap.insertion(new_items, count, curr_smem_offset / 4, p_need_gc, (int *)&rwlock_y);
      //If insertion is successful, will increase rwlock_y, hence we do not increase it again in this function.
      if(*p_need_gc == 1) {
        while(rwlock_z < *p_ticket) {}
        //Perform GC, then try again
	garbage_collection(curr_smem_offset, vert_distance);
      }
    } while(*p_need_gc == 1);

    SINGLE_THREADED {atomicAdd((int *)&rwlock_z, 1);}
    __syncthreads();
  }
};
