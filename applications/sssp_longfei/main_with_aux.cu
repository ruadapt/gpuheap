#include "sssp_with_aux.cuh"
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <algorithm>
#define MAX_INT 2147483647

struct input_line {
  int src, dst;
};

//From StackOverflow
int64_t timespecDiff(struct timespec *timeA_p, struct timespec *timeB_p)
{
  return ((timeA_p->tv_sec * 1000000000) + timeA_p->tv_nsec) -
           ((timeB_p->tv_sec * 1000000000) + timeB_p->tv_nsec);
}

bool sort_input(const input_line &a, const input_line &b) {return a.src < b.src || (a.src == b.src && a.dst < b.dst);}

int main(int argc,char ** argv){
  if(argc != 3){
    printf("Usage: sssp [graph filename] [number of lines]\n");
    return 0;
  }

  int * edge_list_index;
  int * edge_dst;
  int * edge_weight;
  int * distance;

  int vert_count = 0;
  FILE * fin = fopen(argv[1],"r");
  FILE * fout = fopen("output.txt", "w");
  int input_line_count;
  sscanf(argv[2], " %d", &input_line_count);
  input_line * lines = new input_line[input_line_count * 2];
  for(int i = 0;i < input_line_count;++i){
    fscanf(fin, " %d %d", &(lines[i * 2].src), &(lines[i * 2].dst));
    if(lines[i * 2].src >= vert_count) {vert_count = lines[i * 2].src + 1;}
    if(lines[i * 2].dst >= vert_count) {vert_count = lines[i * 2].dst + 1;}
    lines[i * 2 + 1].src = lines[i * 2].dst;
    lines[i * 2 + 1].dst = lines[i * 2].src;
  }
  std::sort(lines, lines + input_line_count * 2, sort_input);
  int edge_count = input_line_count * 2;

  edge_list_index = new int[vert_count + 1];
  edge_dst = new int[edge_count];
  edge_weight = new int[edge_count];
  distance = new int[vert_count];
  int curr_vert = 0;
  edge_list_index[0] = 0;
  for(int i = 0;i < edge_count;++i){
    while(curr_vert < lines[i].src){++curr_vert; edge_list_index[curr_vert] = i;}
    edge_dst[i] = lines[i].dst;
    edge_weight[i] = 1;
  }
  edge_list_index[vert_count] = edge_count;
  for(int i = 0;i < vert_count;++i){distance[i] = 2147483647;}
  distance[0] = 0;

  int * gpu_edge_list_index, * gpu_edge_dst, * gpu_edge_weight, * gpu_distance;
  cudaMalloc((void **)&gpu_edge_list_index, sizeof(int) * (vert_count + 1));
  cudaMemcpy(gpu_edge_list_index, edge_list_index, sizeof(int) * (vert_count + 1), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&gpu_edge_dst, sizeof(int) * edge_count);
  cudaMemcpy(gpu_edge_dst, edge_dst, sizeof(int) * edge_count, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&gpu_edge_weight, sizeof(int) * edge_count);
  cudaMemcpy(gpu_edge_weight, edge_weight, sizeof(int) * edge_count, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&gpu_distance, sizeof(int) * vert_count);
  cudaMemcpy(gpu_distance, distance, sizeof(int) * vert_count, cudaMemcpyHostToDevice);

  sssp_heap_node max_node;
  max_node.vert = MAX_INT;
  max_node.curr_dist = MAX_INT;
  Heap_With_Aux < sssp_heap_node, int > cpu_heap(vert_count * 10 / CONFIG_BATCH_SIZE, CONFIG_BATCH_SIZE, max_node, 0);
  Heap_With_Aux < sssp_heap_node, int > * gpu_heap;

  cudaMalloc((void **)&gpu_heap, sizeof(Heap_With_Aux < sssp_heap_node, int >));
  cudaMemcpy(gpu_heap, &cpu_heap, sizeof(Heap_With_Aux < sssp_heap_node, int >), cudaMemcpyHostToDevice);

  sssp_heap_node init_node;
  init_node.vert = 0;
  init_node.curr_dist = 0;
  sssp_heap_node * gpu_init_node;
  cudaMalloc((void **)&gpu_init_node, sizeof(sssp_heap_node));
  cudaMemcpy(gpu_init_node, &init_node, sizeof(sssp_heap_node), cudaMemcpyHostToDevice);
  insertInitNode<<<1, 1, 1024>>>(gpu_heap, gpu_init_node);

  printf("Preparation complete\n");

  struct timespec start_time, end_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);

  int iteration = 0;
  do{
    ssspKernel<<<CONFIG_THREAD_GROUP_NUM, CONFIG_THREAD_NUM, 32768>>>(gpu_heap, gpu_edge_list_index, gpu_edge_dst, gpu_edge_weight, gpu_distance);
    ++iteration;
    cudaMemcpy(&cpu_heap, gpu_heap, sizeof(Heap_With_Aux < sssp_heap_node, int >), cudaMemcpyDeviceToHost);
  } while(cpu_heap.heap.itemCount() > 0 || cpu_heap.curr_aux_buf_size > 0);

  clock_gettime(CLOCK_MONOTONIC, &end_time);
  printf("Finished in %d iterations\n", iteration);
  int64_t duration = timespecDiff(&end_time, &start_time);
  printf("Microseconds: %lld\n", duration / 1000);

  cudaMemcpy(distance, gpu_distance, sizeof(int) * vert_count, cudaMemcpyDeviceToHost);
  for(int i = 0;i < vert_count;++i){
    fprintf(fout, "%d %d\n", i, distance[i]);
  }
  return 0;
}
