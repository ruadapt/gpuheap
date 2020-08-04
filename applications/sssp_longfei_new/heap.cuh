#ifndef HEAP_CUH
#define HEAP_CUH

#include "util.cuh"
#include "heaputil.cuh"

template < typename T >
struct Heap {
public:
  const int batch_num;
  const int batch_size;
  const T init_limit;
  volatile int curr_batch_count;
  volatile int curr_pb_size;

  volatile T * items;
  volatile T * partial_buffer;
  int * spinlocks; //Spinlocks need not be volatile, because we are only accessing them through atomicX functions.
  volatile int * is_taken;

/*
  Notes on volatile int * is_taken:
  This variable is introduced so that retrieve() and insert() may work concurrently.
  When a thread group enters insert(), it locks batch 0 first. Then it increases curr_heap_size, and sets is_taken[target_batch] to 1. We implement the top-down version of insertion.
  Now suppose another thread group enters retrieve(). It locks batch 0 first. Then it decreases curr_heap_size, locks the last valid batch in the heap (which is the target of an ongoing insert()). It then checks is_taken[last_batch] and sees that it is 1. It sets it to 2. When the insert() thread group notices that is_taken is set to 2, it will realize that the elements have been taken, so it abandons insertion immediately, and fills the to-be-inserted elements into the root batch. Finally, it sets is_taken to 3, to notify the retrieve() thread group that the root batch has been filled. The retrieve() thread group acknowledges this by setting is_taken back to 0.
  The insert() thread group should never see that is_taken[target_batch] is 1, 2, or 3. If it is 1, then another thread group is also trying to insert into this same batch. If it is 2 or 3, then it is witnessing a key transfer between retrieve() and insert(). In that case the retrieve() thread should have batch 0 locked, so this new insert() thread should not be executing.

  In summary, the valid transitions for is_taken are:
  0 -> 1: Set by insert() thread, with batch 0 locked.
  1 -> 2: Set by retrieve() thread, with batch 0 and target batch locked.
  1 -> 0: Set by insert() thread, with target batch locked.
  2 -> 3: Set by insert() thread, with target batch locked. Meanwhile, batch 0 must be locked by another retrieve() thread.
  3 -> 0: Set by retrieve() thread, with batch 0 locked.
*/

private:
  //It has been suggested that __threadfence is not required during lock, but required during unlock. See http://people.tamu.edu/~abdullah.muzahid/pdfs/issre18.pdf.
  __forceinline__ __device__ void spinlock_lock(int idx){
    SINGLE_THREADED {
      while (atomicCAS(spinlocks + idx, 0, 1) != 0) {}
    }
    __syncthreads();
  }

  __forceinline__ __device__ void spinlock_unlock(int idx){
    __threadfence();
    __syncthreads();
    SINGLE_THREADED {atomicExch(spinlocks + idx, 0);}
  }

  __forceinline__ __device__ int get_next_idx(int curr, int target){
    int p = curr + 1, q = target + 1;
    int t = q >> (__clz(p) - __clz(q) - 1);
    return t - 1;
  }

public:
  Heap(int batch_num_, int batch_size_, T init_limit_) : batch_num(batch_num_), batch_size(batch_size_), init_limit(init_limit_), curr_batch_count(0), curr_pb_size(0){
    cudaMalloc((void **)&items, sizeof(T) * batch_size * batch_num);
    cudaMalloc((void **)&partial_buffer, sizeof(T) * batch_size);
    cudaMalloc((void **)&spinlocks, sizeof(int) * batch_num);
    cudaMemset((int *)spinlocks, 0, sizeof(int) * batch_num);
    cudaMalloc((void **)&is_taken, sizeof(int) * batch_num);
    cudaMemset((int *)is_taken, 0, sizeof(int) * batch_num);
  }

  ~Heap(){
    cudaFree((void *)items);
    cudaFree((void *)partial_buffer);
    cudaFree((void *)spinlocks);
    cudaFree((void *)is_taken);
  }

  int itemCount(){
    return curr_batch_count * batch_size + curr_pb_size;
  }

  __device__ void retrieve(T * ret_buffer, int * p_size, int smem_offset){
    extern __shared__ char smem[];
    int curr_smem_offset = smem_offset;

    spinlock_lock(0);
    DECLARE_SMEM(int, p_curr_batch_count, 1);
    DECLARE_SMEM(int, p_curr_pb_size, 1);
    ALIGN_SMEM_8;
    DECLARE_SMEM(T, p_key_buffer1, batch_size);
    DECLARE_SMEM(T, p_key_buffer2, batch_size);
    DECLARE_SMEM(T, p_key_buffer3, batch_size);

    SINGLE_THREADED {
      *p_curr_batch_count = curr_batch_count;
      *p_curr_pb_size = curr_pb_size;
    }
    __syncthreads();

    if(*p_curr_batch_count == 0){
      //Case 1: Directly copy from partial buffer

      SINGLE_THREADED {
        if(*p_size > *p_curr_pb_size) {*p_size = *p_curr_pb_size;}
      }
      __syncthreads();

      batchCopy(ret_buffer, partial_buffer, *p_size);
      int remaining_pb = *p_curr_pb_size - *p_size;
      batchCopy(p_key_buffer1, partial_buffer + *p_size, remaining_pb);
      __syncthreads();
      batchCopy(partial_buffer, p_key_buffer1, remaining_pb);

      SINGLE_THREADED {curr_pb_size = remaining_pb;}
      spinlock_unlock(0);
      return;
    }

    batchCopy(ret_buffer, items, *p_size);
    int remaining = batch_size - *p_size;

    if(*p_curr_batch_count == 1){ 
      //Case 2: Only root batch is non-empty. Either we move everything to partial buffer, or we fill the batch with keys from partial buffer.
      if(*p_curr_pb_size >= *p_size){
        int remaining_pb = *p_curr_pb_size - *p_size;
	batchCopy(p_key_buffer1, items + *p_size, remaining);
	batchCopy(p_key_buffer1 + remaining, partial_buffer, *p_size);
	batchCopy(p_key_buffer2, partial_buffer + *p_size, remaining_pb);
	__syncthreads();
	batchCopy(items, p_key_buffer1, batch_size);
	batchCopy(partial_buffer, p_key_buffer2, remaining_pb);

	SINGLE_THREADED {curr_pb_size = remaining_pb;}

	spinlock_unlock(0);
	return;
      } else {
        batchCopy(p_key_buffer1, items + *p_size, remaining);
	batchCopy(p_key_buffer1 + remaining, partial_buffer, *p_curr_pb_size);
	batchCopy(partial_buffer, p_key_buffer1, remaining + *p_curr_pb_size);

	SINGLE_THREADED {
	  curr_batch_count = 0;
	  curr_pb_size = remaining + *p_curr_pb_size;
	}

	spinlock_unlock(0);
	return;
      }
    }

    //Case 3: curr_batch_count > 1. In this case, first use partial buffer to fill root batch. If this is not enough, use the last valid batch to fill. The remaining keys in the last batch are put into partial buffer.
    batchCopy(p_key_buffer1, items + *p_size, remaining);

    if(*p_curr_pb_size >= *p_size){
      int remaining_pb = *p_curr_pb_size - *p_size;
      batchCopy(p_key_buffer1 + remaining, partial_buffer, *p_size);
      batchCopy(p_key_buffer2, partial_buffer + *p_size, remaining_pb);
      batchFill(p_key_buffer2 + remaining_pb, init_limit, batch_size - remaining_pb);
      __syncthreads();
      SINGLE_THREADED {curr_pb_size = remaining_pb;}
    } else {
      batchCopy(p_key_buffer1 + remaining, partial_buffer, *p_curr_pb_size);
      //SINGLE_THREADED {curr_pb_size = 0;}
      int remaining_gap = *p_size - *p_curr_pb_size;
      *p_curr_batch_count -= 1;
      spinlock_lock(*p_curr_batch_count);
      SINGLE_THREADED {curr_batch_count = *p_curr_batch_count;} //This assignment must be done after the spinlock is locked, because we guarantee that, whenever a batch is locked, and that batch contains valid data, then curr_batch_count must never drop below that batch. This allows a thread to easily determine whether a batch contains valid data or not, by first locking that batch, then checking curr_batch_count.
      DECLARE_SMEM(int, p_last_batch_status, 1);
      SINGLE_THREADED {*p_last_batch_status = is_taken[*p_curr_batch_count];}
      __syncthreads();
      if(*p_last_batch_status == 0){
        volatile T * last_batch = items + *p_curr_batch_count * batch_size;
        batchCopy(p_key_buffer1 + (batch_size - remaining_gap), last_batch, remaining_gap);
	batchCopy(p_key_buffer2, last_batch + remaining_gap, batch_size - remaining_gap);
	batchFill(p_key_buffer2 + (batch_size - remaining_gap), init_limit, remaining_gap);
	spinlock_unlock(*p_curr_batch_count);
	ibitonicSort(p_key_buffer1, batch_size);
	imergePath(p_key_buffer1, p_key_buffer2, p_key_buffer1, p_key_buffer2, batch_size, curr_smem_offset);
      } else {
        //is_taken[target_batch] must be 1. It cannot be 2 or 3 since that means another retrieve() thread is locking the root batch.
        SINGLE_THREADED {is_taken[*p_curr_batch_count] = 2;}
	spinlock_unlock(*p_curr_batch_count);
	SINGLE_THREADED {
	  while(is_taken[*p_curr_batch_count] != 3) {}
	  is_taken[*p_curr_batch_count] = 0;
	} //Single-threaded to reduce number of loads from that address. It is impossible for is_taken[target_batch] to return to 0 at this point, because spinlock_unlock induces a memory barrier. When the insert() thread locks target_batch again (it will do so at least once), it is guaranteed to see this flag.
	__syncthreads();
	batchCopy(p_key_buffer1 + (batch_size - remaining_gap), items, remaining_gap);
	batchCopy(p_key_buffer2, items + remaining_gap, batch_size - remaining_gap);
	batchFill(p_key_buffer2 + (batch_size - remaining_gap), init_limit, remaining_gap);
	__syncthreads();
	ibitonicSort(p_key_buffer1, batch_size);
	imergePath(p_key_buffer1, p_key_buffer2, p_key_buffer1, p_key_buffer2, batch_size, curr_smem_offset);
      }
      SINGLE_THREADED {curr_pb_size = batch_size - remaining_gap;}
    }
    batchCopy(partial_buffer, p_key_buffer2, batch_size);

    //We now begin to adjust heap structure
    DECLARE_SMEM(int, p_curr_idx, 1);
    DECLARE_SMEM(int, p_adjust_type, 1); //0 -> Leaf node, no adjustment needed; 1 -> Only left node valid; 2 -> Only right node valid; 3 -> Both children valid
    SINGLE_THREADED {*p_curr_idx = 0;}
    __syncthreads();
    do{
      //The contents of batch curr_idx is stored in p_key_buffer1
      int left_idx = *p_curr_idx * 2 + 1;
      int right_idx = left_idx + 1;
      SINGLE_THREADED {*p_adjust_type = 0;}
      if(left_idx < batch_num){
	spinlock_lock(left_idx);
	SINGLE_THREADED {
	  if(curr_batch_count > left_idx && is_taken[left_idx] == 0){
	    *p_adjust_type += 1;
	  }
	  //It might occur that curr_batch_count <= left_idx when we perform the check, but become greater just after the check. This means some insert() thread has targeted this batch simultaneously. It doesn't matter, because we are holding the lock on curr_idx, and the insert() thread must crawl through curr_idx before modifying left_idx or right_idx.
	  //But one thing does matter: the insert() thread must set is_taken[target_batch] before increasing curr_batch_count. Otherwise, retrieve() thread might observe that curr_batch_count is increased, but is_taken has not been set to 1 yet, and mistakenly think this batch contains valid data.
	}
      }
      if(right_idx < batch_num){
	spinlock_lock(right_idx);
	SINGLE_THREADED {
	  if(curr_batch_count > right_idx && is_taken[right_idx] == 0){
	    *p_adjust_type += 2;
	  }
	}
      }
      __syncthreads();

      if(*p_adjust_type == 0){
      	if(left_idx < batch_num) {spinlock_unlock(left_idx);}
	if(right_idx < batch_num) {spinlock_unlock(right_idx);}
        batchCopy(items + *p_curr_idx * batch_size, p_key_buffer1, batch_size);
	spinlock_unlock(*p_curr_idx);
        break;
      } else if(*p_adjust_type == 1){
	if(right_idx < batch_num) {spinlock_unlock(right_idx);}
        batchCopy(p_key_buffer2, items + left_idx * batch_size, batch_size);
	imergePath(p_key_buffer1, p_key_buffer2, items + *p_curr_idx * batch_size, p_key_buffer1, batch_size, curr_smem_offset);
	spinlock_unlock(*p_curr_idx);
	SINGLE_THREADED {*p_curr_idx = left_idx;}
	__syncthreads();
      } else if(*p_adjust_type == 2){
        if(left_idx < batch_num) {spinlock_unlock(left_idx);}
        batchCopy(p_key_buffer2, items + right_idx * batch_size, batch_size);
	imergePath(p_key_buffer1, p_key_buffer2, items + *p_curr_idx * batch_size, p_key_buffer1, batch_size, curr_smem_offset);
	spinlock_unlock(*p_curr_idx);
	SINGLE_THREADED {*p_curr_idx = right_idx;}
	__syncthreads();
      } else {
        batchCopy(p_key_buffer2, items + left_idx * batch_size, batch_size);
	batchCopy(p_key_buffer3, items + right_idx * batch_size, batch_size);
	__syncthreads();
        if(p_key_buffer2[batch_size - 1] <= p_key_buffer3[batch_size - 1]){
	  imergePath(p_key_buffer2, p_key_buffer3, p_key_buffer2, p_key_buffer3, batch_size, curr_smem_offset);
	  batchCopy(items + right_idx * batch_size, p_key_buffer3, batch_size);
	  spinlock_unlock(right_idx);
	  imergePath(p_key_buffer1, p_key_buffer2, items + *p_curr_idx * batch_size, p_key_buffer1, batch_size, curr_smem_offset);
	  spinlock_unlock(*p_curr_idx);
	  SINGLE_THREADED {*p_curr_idx = left_idx;}
	  __syncthreads();
	} else {
	  imergePath(p_key_buffer2, p_key_buffer3, p_key_buffer3, p_key_buffer2, batch_size, curr_smem_offset);
	  batchCopy(items + left_idx * batch_size, p_key_buffer2, batch_size);
	  spinlock_unlock(left_idx);
	  imergePath(p_key_buffer1, p_key_buffer3, items + *p_curr_idx * batch_size, p_key_buffer1, batch_size, curr_smem_offset);
	  spinlock_unlock(*p_curr_idx);
	  SINGLE_THREADED {*p_curr_idx = right_idx;}
	  __syncthreads();
	}
      }
    } while(1);
  }

  __device__ void insert(T * new_items, int count, int smem_offset, int * need_gc, int * rwlock_y){
    extern __shared__ char smem[];
    int curr_smem_offset = smem_offset;

    DECLARE_SMEM(int, p_curr_batch_count, 1);
    DECLARE_SMEM(int, p_curr_pb_size, 1);
    DECLARE_SMEM(int, p_abandon, 1);
    ALIGN_SMEM_8;
    DECLARE_SMEM(T, p_key_buffer1, batch_size);
    DECLARE_SMEM(T, p_key_buffer2, batch_size);

    batchCopy(p_key_buffer1, new_items, count);

    spinlock_lock(0);

    SINGLE_THREADED {
      *p_curr_batch_count = curr_batch_count;
      *p_curr_pb_size = curr_pb_size;
    }
    __syncthreads();

    //Perform GC if needed
    SINGLE_THREADED {
      if(*p_curr_batch_count >= batch_num - 1){
        *need_gc = 1;
        spinlock_unlock(0);
      } else {
        atomicAdd(rwlock_y, 1);
      }
    }
    __syncthreads();
    if(*need_gc) {return;}

    //Case 1: There is enough space in partial buffer
    if(*p_curr_pb_size + count < batch_size){
      batchCopy(p_key_buffer1 + count, partial_buffer, *p_curr_pb_size);
      int remaining = batch_size - count - *p_curr_pb_size;
      batchFill(p_key_buffer1 + count + *p_curr_pb_size, init_limit, remaining);
      __syncthreads();
      ibitonicSort(p_key_buffer1, batch_size);
      if(*p_curr_batch_count == 0){
	batchCopy(partial_buffer, p_key_buffer1, batch_size);
      } else {
        batchCopy(p_key_buffer2, items, batch_size);
	__syncthreads();
	imergePath(p_key_buffer1, p_key_buffer2, items, partial_buffer, batch_size, curr_smem_offset);
      }
      SINGLE_THREADED {curr_pb_size = *p_curr_pb_size + count;}
      spinlock_unlock(0);
      return;
    }

    //Case 2: A new batch is created
    int remaining = batch_size - count;
    batchCopy(p_key_buffer1 + count, partial_buffer, remaining);
    int remaining_pb = *p_curr_pb_size - remaining;
    batchCopy(p_key_buffer2, partial_buffer + remaining, remaining_pb);
    batchFill(p_key_buffer2 + remaining_pb, init_limit, batch_size - remaining_pb);
    __syncthreads();
    ibitonicSort(p_key_buffer1, batch_size);
    if(*p_curr_batch_count == 0){
      //Case 2.1: Fill the new batch directly to root
      imergePath(p_key_buffer1, p_key_buffer2, items, partial_buffer, batch_size, curr_smem_offset);
      SINGLE_THREADED {
        curr_batch_count = 1;
	curr_pb_size = remaining_pb;
      }
      spinlock_unlock(0);
      return;
    }

    //Case 2.2: Merging and propagation needed
    batchCopy(partial_buffer, p_key_buffer2, batch_size);
    SINGLE_THREADED {curr_pb_size = remaining_pb;}
    int curr_idx = 0;
    int target_idx = *p_curr_batch_count;
    SINGLE_THREADED {
      is_taken[target_idx] = 1;
      __threadfence();
      curr_batch_count = *p_curr_batch_count + 1;
    }
    __syncthreads();

    do{
      batchCopy(p_key_buffer2, items + curr_idx * batch_size, batch_size);
      __syncthreads();
      imergePath(p_key_buffer1, p_key_buffer2, items + curr_idx * batch_size, p_key_buffer1, batch_size, curr_smem_offset);
      int next_idx = get_next_idx(curr_idx, target_idx);
      spinlock_lock(next_idx);
      spinlock_unlock(curr_idx);
      SINGLE_THREADED {
        if(is_taken[target_idx] == 2) {*p_abandon = 1;} else {*p_abandon = 0;}
      }
      __syncthreads();
      if(*p_abandon){
        spinlock_unlock(next_idx);
	batchCopy(items, p_key_buffer1, batch_size);
	__threadfence();
	__syncthreads();
	SINGLE_THREADED {is_taken[target_idx] = 3;}
	return;
      } else {
	curr_idx = next_idx;
      }
    } while(curr_idx != target_idx);

    batchCopy(items + target_idx * batch_size, p_key_buffer1, batch_size);
    SINGLE_THREADED {is_taken[target_idx] = 0;}
    spinlock_unlock(target_idx);
  }
};

#endif
