#include "heap_with_aux.cuh"
#include "sssp_config.cuh"

#define VERT(x) ((x) & 0xffffffff)
#define DISTANCE(x) ((x) >> 32)
#define MAKE_KEY(vert, distance) ((((unsigned long long)(distance)) << 32) + (vert))

__device__ bool update_offset(unsigned long long * p_key, int * p_offset, const void * vp_edge_list_index){
  const int * edge_list_index = reinterpret_cast < const int * >(vp_edge_list_index);
  int edge_count = edge_list_index[VERT(*p_key) + 1] - edge_list_index[VERT(*p_key)];
  if(edge_count - *p_offset > CONFIG_CHUNK_SIZE){
    *p_offset += CONFIG_CHUNK_SIZE;
    return true;
  } else {
    return false;
  }
}

__global__ void ssspKernel(Heap_With_Aux < unsigned long long, int > * heap, const int * edge_list_index, const int * edge_dst, const int * edge_weight, int * distance, unsigned long long * inserted_nodes, volatile int * term_sig){
  extern __shared__ char smem[];
  int curr_smem_offset = 0;
  DECLARE_SMEM(int, p_size, 1);
  DECLARE_SMEM(int, p_total_task, 1);
  DECLARE_SMEM(int, p_valid_size, 1);
  DECLARE_SMEM(int, p_batch_offset, CONFIG_BATCH_SIZE);
  DECLARE_SMEM(int, p_rem_edge_count, CONFIG_BATCH_SIZE);
  //DECLARE_SMEM(int, p_task_psum, CONFIG_BATCH_SIZE);
  DECLARE_SMEM(int, p_inserted_node_count, 1);
  DECLARE_SMEM(int, p_should_exit, 1);
  ALIGN_SMEM_8;
  DECLARE_SMEM(unsigned long long, p_batch, CONFIG_BATCH_SIZE);
  DECLARE_SMEM(unsigned long long, p_valid_batch, CONFIG_BATCH_SIZE);
  //DECLARE_SMEM(unsigned long long, p_inserted_nodes, 0);
  unsigned long long *p_inserted_nodes = inserted_nodes + blockIdx.x * CONFIG_BATCH_SIZE * CONFIG_CHUNK_SIZE;

  do{

    SINGLE_THREADED {*p_size = CONFIG_BATCH_SIZE;}
    __syncthreads();
    heap->retrieve(p_batch, p_batch_offset, p_size, update_offset, edge_list_index, curr_smem_offset);

    SINGLE_THREADED {
      *p_total_task = 0;
      *p_inserted_node_count = 0;
      *p_valid_size = 0;
    }
    __syncthreads();
    for(int i = threadIdx.x;i < *p_size;i += blockDim.x){
      if(DISTANCE(p_batch[i]) <= distance[VERT(p_batch[i])]){
        int new_idx = atomicAdd(p_valid_size, 1);
        p_valid_batch[new_idx] = p_batch[i];
        int remaining_edge = edge_list_index[VERT(p_batch[i]) + 1] - edge_list_index[VERT(p_batch[i])] - p_batch_offset[i];
        if(remaining_edge > CONFIG_CHUNK_SIZE) {remaining_edge = CONFIG_CHUNK_SIZE;}
        p_rem_edge_count[new_idx] = remaining_edge;
        atomicAdd(p_total_task, remaining_edge);
      }
    }
    __syncthreads();

    if(*p_valid_size == 0) {
      SINGLE_THREADED {
        *p_should_exit = 1;
        term_sig[blockIdx.x] = 1;
	for(int i = 0;i < gridDim.x;++i){
	  if(term_sig[i] == 0) {*p_should_exit = 0;break;}
	}
      }
      __syncthreads();
      if(*p_should_exit) {return;} else {continue;}
    } else {
      SINGLE_THREADED {term_sig[blockIdx.x] = 0;}
    }

    //Parallel scan from https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
    batchFill(p_rem_edge_count + *p_valid_size, 0, CONFIG_BATCH_SIZE - *p_valid_size);
    __syncthreads();
    
    int psum_offset = 1;
    for(int d = CONFIG_BATCH_SIZE >> 1;d > 0;d >>= 1){
      if(threadIdx.x < d){
        int ai = psum_offset * (2 * threadIdx.x + 1) - 1;
	int bi = psum_offset * (2 * threadIdx.x + 2) - 1;
	p_rem_edge_count[bi] += p_rem_edge_count[ai];
      }
      psum_offset <<= 1;
      __syncthreads();
    }
    SINGLE_THREADED {p_rem_edge_count[CONFIG_BATCH_SIZE - 1] = 0;}
    for(int d = 1;d < CONFIG_BATCH_SIZE;d *= 2){
      __syncthreads();
      psum_offset >>= 1;
      if(threadIdx.x < d){
        int ai = psum_offset * (2 * threadIdx.x + 1) - 1;
	int bi = psum_offset * (2 * threadIdx.x + 2) - 1;
	int tmp = p_rem_edge_count[ai];
	p_rem_edge_count[ai] = p_rem_edge_count[bi];
	p_rem_edge_count[bi] += tmp;
      }
    }
    __syncthreads();
    //batchCopy(p_task_psum, p_rem_edge_count, CONFIG_BATCH_SIZE);

    for(int i = threadIdx.x;i < *p_total_task;i += blockDim.x){
      int vert_idx = 0;
      for(int j = CONFIG_BATCH_SIZE_LOG - 1;j >= 0;--j){
        int new_idx = vert_idx + (1 << j);
        if(p_rem_edge_count[new_idx] <= i) {vert_idx = new_idx;}
      }
      int offset = i - p_rem_edge_count[vert_idx];
      int src_vert = VERT(p_valid_batch[vert_idx]);
      int edge_id = edge_list_index[src_vert] + offset;
      int dst_vert = edge_dst[edge_id];
      int curr_dist = DISTANCE(p_valid_batch[vert_idx]);
      int new_dist = curr_dist + edge_weight[edge_id];
      int old_dist = atomicMin((int *)(distance + dst_vert), new_dist);
      if(new_dist < old_dist){
        int new_idx = atomicAdd(p_inserted_node_count, 1);
        p_inserted_nodes[new_idx] = MAKE_KEY(dst_vert, new_dist);
      }
    }
    __syncthreads();

    int batches = *p_inserted_node_count / CONFIG_BATCH_SIZE;
    
    for(int i = 0;i < batches;++i){
      heap->insert(p_inserted_nodes + i * CONFIG_BATCH_SIZE, CONFIG_BATCH_SIZE, curr_smem_offset, distance);
    }
    int rem = *p_inserted_node_count % CONFIG_BATCH_SIZE;
    if(rem > 0) {heap->insert(p_inserted_nodes + batches * CONFIG_BATCH_SIZE, rem, curr_smem_offset, distance);}

  } while(1);
}

__global__ void insertInitNode(Heap_With_Aux < unsigned long long, int > * heap, unsigned long long init_node){
  heap->insert(&init_node, 1, 0, NULL);
}
