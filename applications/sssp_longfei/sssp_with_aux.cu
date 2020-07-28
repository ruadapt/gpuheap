/*
  Single source shortest path with concurrent heap
  Reimplemented with a heap that supports auxiliary data, but not partial retrieval
*/

#include "sssp_with_aux.cuh"

__device__ bool update_offset(sssp_heap_node * p_key, int * p_offset, void * vp_edge_list_index){
  int * edge_list_index = reinterpret_cast < int * >(vp_edge_list_index);
  int edge_count = edge_list_index[p_key->vert + 1] - edge_list_index[p_key->vert];
  if(edge_count - *p_offset > CONFIG_CHUNK_SIZE){
    *p_offset += CONFIG_CHUNK_SIZE;
    return true;
  } else {
    return false;
  }
}

__global__ void ssspKernel(Heap_With_Aux < sssp_heap_node, int > * heap, int * edge_list_index, int * edge_dst, int * edge_weight, int * distance){
  extern __shared__ char smem[];
  int curr_smem_offset = 0;
  DECLARE_SMEM(int, p_size, 1);
  DECLARE_SMEM(int, p_total_task, 1);
  DECLARE_SMEM(sssp_heap_node, p_batch, CONFIG_BATCH_SIZE);
  DECLARE_SMEM(int, p_batch_offset, CONFIG_BATCH_SIZE);

  *p_size = CONFIG_BATCH_SIZE;
  heap->retrieve(p_batch, p_batch_offset, p_size, update_offset, edge_list_index, curr_smem_offset);
  __syncthreads();

  DECLARE_SMEM(int, p_inserted_node_count, 1);
  DECLARE_SMEM(sssp_heap_node, p_inserted_nodes, 0);
  SINGLE_THREADED {
    *p_total_task = 0;
    *p_inserted_node_count = 0;
    for(int i = 0;i < *p_size;++i){
      if(p_batch[i].curr_dist > distance[p_batch[i].vert]){
        sssp_heap_node tmp = p_batch[*p_size - 1];
	p_batch[*p_size - 1] = p_batch[i];
	p_batch[i] = tmp;
	--(*p_size);
	--i;
      } else {
        int remaining_edge = edge_list_index[p_batch[i].vert + 1] - edge_list_index[p_batch[i].vert] - p_batch_offset[i];
        if(remaining_edge > CONFIG_CHUNK_SIZE) {remaining_edge = CONFIG_CHUNK_SIZE;}
        *p_total_task += remaining_edge;
      }
    }
  }
  __syncthreads();
  if(*p_size == 0) {return;}

  int curr_vert = 0;
  int curr_offset = p_batch_offset[0];
  int curr_diff = threadIdx.x;
  for(int i = threadIdx.x;i < *p_total_task;i += blockDim.x){
    /*
      Determine the edge to process.
      Binary search might be faster. But will it lead to worse warp divergence?
    */
    while(curr_diff > 0){
      int remaining_edge = edge_list_index[p_batch[curr_vert].vert + 1] - edge_list_index[p_batch[curr_vert].vert] - curr_offset;
      if(remaining_edge > CONFIG_CHUNK_SIZE) {remaining_edge = CONFIG_CHUNK_SIZE;}
      if(curr_diff < remaining_edge){
        curr_offset += curr_diff;
	curr_diff = 0;
      } else {
        curr_diff -= remaining_edge;
        ++curr_vert;
	curr_offset = p_batch_offset[curr_vert];
      }
    }

    /*
      Update best known distance
    */
    int idx = edge_list_index[p_batch[curr_vert].vert] + curr_offset;
    int curr_best_dist = p_batch[curr_vert].curr_dist + edge_weight[idx];
    int old_best_dist = atomicMin(distance + edge_dst[idx], curr_best_dist);
    if(curr_best_dist < old_best_dist){
      /*
        Write new node into shared memory
      */
      int new_idx = atomicAdd(p_inserted_node_count, 1);
      p_inserted_nodes[new_idx].vert = edge_dst[idx];
      p_inserted_nodes[new_idx].curr_dist = curr_best_dist;
    }

    curr_diff = blockDim.x;
  }
  __syncthreads();

  int batches = *p_inserted_node_count / CONFIG_BATCH_SIZE;
  for(int i = 0;i < batches;++i){
    heap->insert(p_inserted_nodes + i * CONFIG_BATCH_SIZE, CONFIG_BATCH_SIZE, curr_smem_offset + *p_inserted_node_count * sizeof(sssp_heap_node), distance);
    __syncthreads();
  }
  int rem = *p_inserted_node_count % CONFIG_BATCH_SIZE;
  if(rem != 0){
    heap->insert(p_inserted_nodes + batches * CONFIG_BATCH_SIZE, rem, curr_smem_offset + *p_inserted_node_count * sizeof(sssp_heap_node), distance);
  }
}

__global__ void insertInitNode(Heap_With_Aux < sssp_heap_node, int > * heap, sssp_heap_node * init_node){
  heap->insert(init_node, 1, 0, NULL);
}
