#include <stdio.h>
#include "heap.cuh"

#define CONFIG_THREAD_GROUP_NUM 5
#define CONFIG_THREAD_NUM 32
#define CONFIG_CHUNK_SIZE 4
#define CONFIG_BATCH_SIZE 32
#define CONFIG_BATCH_NUM 312252

#define DECLARE_SMEM(Type, Name, Count) Type * Name = (Type *)(smem + curr_smem_offset); curr_smem_offset += sizeof(Type) * (Count); curr_smem_offset = (curr_smem_offset + 3) & ~0x03
#define SINGLE_THREADED if(threadIdx.x == 0)

struct sssp_heap_node {
  int vert;
  int curr_dist;

  __device__ bool operator==(const sssp_heap_node &n) {return vert == n.vert && curr_dist == n.curr_dist;}
  __device__ bool operator!=(const sssp_heap_node &n) {return vert != n.vert || curr_dist != n.curr_dist;}
  __device__ bool operator<(const sssp_heap_node &n) {return curr_dist < n.curr_dist || (curr_dist == n.curr_dist && vert < n.vert);}
  __device__ bool operator>(const sssp_heap_node &n) {return curr_dist > n.curr_dist || (curr_dist == n.curr_dist && vert > n.vert);}
  __device__ bool operator<=(const sssp_heap_node &n) {return curr_dist <= n.curr_dist || (curr_dist == n.curr_dist && vert <= n.vert);}
  __device__ bool operator>=(const sssp_heap_node &n) {return curr_dist >= n.curr_dist || (curr_dist == n.curr_dist && vert >= n.vert);}

  sssp_heap_node() : vert(1234567890), curr_dist(1234567890) {}
/*
  heap.cuh requires this type to have a int constructor...
*/
  __device__ sssp_heap_node(int unused) : vert(876543210), curr_dist(876543210) {}

  __device__ int weakCmp(sssp_heap_node &n) {return vert == n.vert;}
};

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__);  }
inline void gpuAssert(cudaError_t code, const char *file, unsigned int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void retrieve_insert_test(Heap < sssp_heap_node > * heap){
  extern __shared__ char smem[];
  int curr_smem_offset = 0;
  DECLARE_SMEM(int, p_size, 1);
  DECLARE_SMEM(sssp_heap_node, p_batch, CONFIG_BATCH_SIZE);
  DECLARE_SMEM(int, p_inserted_node_count, 1);
  DECLARE_SMEM(sssp_heap_node, p_inserted_nodes, 15);

  if(heap->deleteRoot(p_batch, *p_size)){
    heap->deleteUpdate(curr_smem_offset);
  }

  SINGLE_THREADED {
    for(int i = 0;i < *p_size;++i){
      if(p_batch[i].vert == 1357924680){
        printf("Error!\n");
      }
    }
  }
  __syncthreads();

  for(int i = 0;i < 5;++i){
    heap->insertion(p_inserted_nodes + 3 * i, 3, curr_smem_offset / 4);
  }
}

int main(){
  sssp_heap_node init_limit;
  init_limit.vert = 1357924680;
  init_limit.curr_dist = 1357924680;

  Heap < sssp_heap_node > * gpu_heap, cpu_heap(CONFIG_BATCH_NUM, CONFIG_BATCH_SIZE, init_limit);
  cudaMalloc((void **)&gpu_heap, sizeof(Heap < sssp_heap_node >));
  cudaMemcpy(gpu_heap, &cpu_heap, sizeof(Heap < sssp_heap_node >), cudaMemcpyHostToDevice);

  for(int i = 0;i < 10;++i){
    retrieve_insert_test<<<CONFIG_THREAD_GROUP_NUM, CONFIG_THREAD_NUM, 32768>>>(gpu_heap);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpy(&cpu_heap, gpu_heap, sizeof(Heap < sssp_heap_node >), cudaMemcpyDeviceToHost);

    printf("Used heap batches: %d with %d keys.\n", cpu_heap.nodeCount(), cpu_heap.itemCount());
    
  }
}
