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

__global__ void ssspKernel(Heap_With_Aux < unsigned long long, int > * heap, const int * edge_list_index, const int * edge_dst, const int * edge_weight, int * distance){
  extern __shared__ char smem[];
  int curr_smem_offset = 0;
  DECLARE_SMEM(int, p_size, 1);
  DECLARE_SMEM(int, p_total_task, 1);
  DECLARE_SMEM(int, p_valid_size, 1);
  DECLARE_SMEM(int, p_batch_offset, CONFIG_BATCH_SIZE);
  DECLARE_SMEM(int, p_rem_edge_count, CONFIG_BATCH_SIZE);
  DECLARE_SMEM(int, p_task_psum, CONFIG_BATCH_SIZE);
  DECLARE_SMEM(int, p_inserted_node_count, 1);
  ALIGN_SMEM_8;
  DECLARE_SMEM(unsigned long long, p_batch, CONFIG_BATCH_SIZE);
  DECLARE_SMEM(unsigned long long, p_valid_batch, CONFIG_BATCH_SIZE);
  DECLARE_SMEM(unsigned long long, p_inserted_nodes, 0);

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

  if(*p_valid_size == 0) {return;}

  //Implement parallel scan later
  SINGLE_THREADED {
    p_task_psum[0] = 0;
    for(int i = 1;i < *p_valid_size;++i){
      p_task_psum[i] = p_task_psum[i - 1] + p_rem_edge_count[i - 1];
    }
    for(int i = *p_valid_size;i < CONFIG_BATCH_SIZE;++i){
      p_task_psum[i] = *p_total_task;
    }
  }
  __syncthreads();

  for(int i = threadIdx.x;i < *p_total_task;i += blockDim.x){
    int vert_idx = 0;
    for(int j = CONFIG_BATCH_SIZE_LOG - 1;j >= 0;--j){
      int new_idx = vert_idx + (1 << j);
      if(p_task_psum[new_idx] <= i) {vert_idx = new_idx;}
    }
    int offset = i - p_task_psum[vert_idx];
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
    heap->insert(p_inserted_nodes + i * CONFIG_BATCH_SIZE, CONFIG_BATCH_SIZE, curr_smem_offset + sizeof(unsigned long long) * *p_inserted_node_count, distance);
  }
  int rem = *p_inserted_node_count % CONFIG_BATCH_SIZE;
  if(rem > 0) {heap->insert(p_inserted_nodes + batches * CONFIG_BATCH_SIZE, rem, curr_smem_offset + sizeof(unsigned long long) * *p_inserted_node_count, distance);}
}

__global__ void insertInitNode(Heap_With_Aux < unsigned long long, int > * heap, unsigned long long init_node){
  heap->insert(&init_node, 1, 0, NULL);
}
