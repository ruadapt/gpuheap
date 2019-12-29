#include "astar_item.cuh"
#include "pq_model.cuh"
#include "astar_map.cuh"
#include "heap.cuh"
#include "util.hpp"

#include <cstdio>
#include <cstdint>
#include <assert.h>

int main(int argc, char *argv[]) {

    if (argc != 5) {
        fprintf(stderr, "Usage: ./astar [map_file] [batchSize] [blockNum] [blockSize]\n");
        return 1;
    }

    uint32_t batch_size = atoi(argv[2]);
    uint32_t block_num = atoi(argv[3]);
    uint32_t block_size = atoi(argv[4]);

    struct timeval start_time, end_time;
    double astar_time;

    printf("read map\n");

    astar::AstarMap map(argv[1]);
    uint32_t map_h = map.h_;
    uint32_t map_w = map.w_;

    uint32_t batch_num = 1280000;
    uint32_t table_size = 1024000;

	Heap<astar::AstarItem> heap(batch_num, batch_size, table_size);
    TB<astar::AstarItem> insTB(table_size, batch_size, 0);
    TB<astar::AstarItem> delTB(table_size, batch_size, 1);
  
    astar::AppItem app_item(&map);

	astar::PQModel<astar::AstarItem> h_model(block_num, block_size, 
                                            batch_num, batch_size,
                                            &heap, &insTB, &delTB, &app_item);
    astar::PQModel<astar::AstarItem> *d_model;
	cudaMalloc((void **)&d_model, sizeof(astar::PQModel<astar::AstarItem>));
    cudaMemcpy(d_model, &h_model, sizeof(astar::PQModel<astar::AstarItem>), cudaMemcpyHostToDevice);

    printf("init item\n");

    astar::Init<astar::AstarItem>(&h_model, d_model);
    cudaDeviceSynchronize();

    printf("run astar\n");

    setTime(&start_time);

    astar::Run<astar::AstarItem>(&h_model, d_model, &heap, &app_item, map_w, map_h);
    cudaDeviceSynchronize();

    setTime(&end_time);
    astar_time = getTime(&start_time, &end_time);
    printf("astar time %.4f\n", astar_time);

    cudaMemcpy(&h_model, d_model, sizeof(astar::PQModel<astar::AstarItem>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&app_item, h_model.d_app_item_, sizeof(astar::AppItem), cudaMemcpyDeviceToHost);
    uint32_t *h_dist = new uint32_t[map_h * map_w];
    cudaMemcpy(h_dist, app_item.d_dist, map_h * map_w * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    uint32_t visited_nodes_count = 0;
    for (int i = 0; i < map_h * map_w; ++i) {
        if (h_dist[i] != UINT_MAX) visited_nodes_count++;
    }
    printf("visited_nodes_count: %u\n", visited_nodes_count);

    uint32_t gpu_path_weight = 0;
    cudaMemcpy(&gpu_path_weight, &app_item.d_dist[map_h * map_w - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (gpu_path_weight == UINT_MAX) {
        printf("No Available Path\n");
    } else {
        printf("Gpu Result: %u\n", gpu_path_weight);
    }
    cudaFree(d_model); d_model = NULL;
	return 0;
}
