#include "heap_with_aux.cuh"

#define CONFIG_THREAD_GROUP_NUM 3
#define CONFIG_THREAD_NUM 4
#define CONFIG_BATCH_NUM 20
#define CONFIG_CHUNK_SIZE 40
#define CONFIG_BATCH_SIZE 32

struct sssp_heap_node {
  int vert;
  int curr_dist;

  __device__ bool operator==(const sssp_heap_node &n) {return vert == n.vert && curr_dist == n.curr_dist;}
  __device__ bool operator!=(const sssp_heap_node &n) {return vert != n.vert || curr_dist != n.curr_dist;}
  __device__ bool operator<(const sssp_heap_node &n) {return curr_dist < n.curr_dist || (curr_dist == n.curr_dist && vert < n.vert);}
  __device__ bool operator>(const sssp_heap_node &n) {return curr_dist > n.curr_dist || (curr_dist == n.curr_dist && vert > n.vert);}
  __device__ bool operator<=(const sssp_heap_node &n) {return curr_dist <= n.curr_dist || (curr_dist == n.curr_dist && vert <= n.vert);}
  __device__ bool operator>=(const sssp_heap_node &n) {return curr_dist >= n.curr_dist || (curr_dist == n.curr_dist && vert >= n.vert);}

  sssp_heap_node() : vert(2147483647), curr_dist(2147483647) {}
/*
  heap.cuh requires this type to have a int constructor...
*/
  __device__ sssp_heap_node(int unused) : vert(2147483647), curr_dist(2147483647) {}

  __device__ int weakCmp(sssp_heap_node &n) {return vert == n.vert;}
};

__global__ void ssspKernel(Heap_With_Aux < sssp_heap_node, int > *, int *, int *, int *, int *);

__global__ void insertInitNode(Heap_With_Aux < sssp_heap_node, int > *, sssp_heap_node *);
