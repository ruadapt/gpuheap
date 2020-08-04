#ifndef HEAP_WITH_AUX_CUH
#define HEAP_WITH_AUX_CUH

#include "heap.cuh"
#include "heaputil.cuh"
#include "util.cuh"

#define CONFIG_GC_GROUP_SIZE 256

template < typename T, typename Aux >
struct Heap_With_Aux {
public:
  Heap < T > heap;
  volatile T * aux_key_buffer;
  volatile Aux * aux_data_buffer;
  const Aux init_aux;
  const int aux_buf_len;
  volatile int curr_aux_buf_size;
  int spinlock;
  volatile int rwlock_x, rwlock_y, rwlock_z;

  Heap_With_Aux(int batch_num, int batch_size, T init_limit, Aux init_aux_, int thread_group_num) : heap(batch_num, batch_size, init_limit), init_aux(init_aux_), aux_buf_len(batch_size * thread_group_num), curr_aux_buf_size(0), spinlock(0), rwlock_x(0), rwlock_y(0), rwlock_z(0) {
    //It turns out that allocating batch_size * thread_group_num number of slots is sufficient to prevent overflow
    cudaMalloc((void **)&aux_key_buffer, sizeof(T) * batch_size * thread_group_num);
    cudaMalloc((void **)&aux_data_buffer, sizeof(Aux) * batch_size * thread_group_num);
  }

  ~Heap_With_Aux(){
    cudaFree((void *)aux_key_buffer);
    cudaFree((void *)aux_data_buffer);
  }

private:
  __device__ void spinlock_lock(){
    SINGLE_THREADED {
      while (atomicCAS(&spinlock, 0, 1) != 0) {}
    }
    __syncthreads();
  }

  __device__ void spinlock_unlock(){
    __threadfence();
    __syncthreads();
    SINGLE_THREADED {atomicExch(&spinlock, 0);}
  }

public:
  __device__ void retrieve(T * key_dst, Aux * aux_dst, int * count, bool (* chk_func)(T *, Aux *, const void *), const void * priv_data, int smem_offset){
    extern __shared__ char smem[];
    int curr_smem_offset = smem_offset;
    DECLARE_SMEM(int, p_curr_aux_buf_size, 1);
    DECLARE_SMEM(int, p_remaining, 1);
    DECLARE_SMEM(int, p_curr_idx, 1);
    ALIGN_SMEM_8;
    DECLARE_SMEM(Aux, p_aux_tmpbuf, blockDim.x);

    spinlock_lock();
    SINGLE_THREADED {
      *p_curr_aux_buf_size = curr_aux_buf_size;
    }
    __syncthreads();
    if(*p_curr_aux_buf_size > *count){
      batchCopy(key_dst, aux_key_buffer + *p_curr_aux_buf_size - *count, *count);
      batchCopy(aux_dst, aux_data_buffer + *p_curr_aux_buf_size - *count, *count);
      SINGLE_THREADED {
	*p_curr_idx = *p_curr_aux_buf_size - *count;
      }
    } else {
      batchCopy(key_dst, aux_key_buffer, *p_curr_aux_buf_size);
      batchCopy(aux_dst, aux_data_buffer, *p_curr_aux_buf_size);
      SINGLE_THREADED {
        curr_aux_buf_size = 0;
	*p_remaining = *count - *p_curr_aux_buf_size;
	*count = *p_curr_aux_buf_size;
      }
      __syncthreads();
      if(*p_remaining > 0){
        spinlock_unlock();
        heap.retrieve(key_dst + *p_curr_aux_buf_size, p_remaining, curr_smem_offset);
	for(int i = threadIdx.x;i < *p_remaining;i += blockDim.x){
	  aux_dst[*p_curr_aux_buf_size + i] = init_aux;
	}
	SINGLE_THREADED {*count += *p_remaining;}
	spinlock_lock();
	SINGLE_THREADED {*p_curr_idx = curr_aux_buf_size;}
      } else {
        SINGLE_THREADED {*p_curr_idx = 0;}
      }
    }
    __syncthreads();

    //SINGLE_THREADED {*p_curr_idx = curr_aux_buf_size;}
    //__syncthreads();
    for(int i = threadIdx.x;i < *count;i += blockDim.x){
      p_aux_tmpbuf[threadIdx.x] = aux_dst[i];
      if(chk_func(key_dst + i, p_aux_tmpbuf + threadIdx.x, priv_data)){
        int new_idx = atomicAdd(p_curr_idx, 1);
	aux_key_buffer[new_idx] = key_dst[i];
	aux_data_buffer[new_idx] = p_aux_tmpbuf[threadIdx.x];
      }
    }
    __syncthreads();
    SINGLE_THREADED {curr_aux_buf_size = *p_curr_idx;}
    spinlock_unlock();
  }

  __device__ void garbage_collection(int smem_offset, int * vert_distance){
    //GC currently unimplemented
    return;
  }

  __device__ void insert(T * new_items, int count, int smem_offset, int * vert_distance){
    extern __shared__ char smem[];
    int curr_smem_offset = smem_offset;
    DECLARE_SMEM(int, unused, 2);
    SINGLE_THREADED {unused[0] = 0;}

    heap.insert(new_items, count, curr_smem_offset, unused, unused + 1);
  }
};

#endif