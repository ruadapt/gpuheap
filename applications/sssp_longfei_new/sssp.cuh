#ifndef SSSP_CUH
#define SSSP_CUH

#include "heap_with_aux.cuh"

__global__ void ssspKernel(Heap_With_Aux < unsigned long long, int > * heap, const int * edge_list_index, const int * edge_dst, const int * edge_weight, int * distance, unsigned long long * inserted_nodes, volatile int * term_sig);

__global__ void insertInitNode(Heap_With_Aux < unsigned long long, int > * heap, unsigned long long init_node);

#endif
