#include "pq_model.cuh"
#include "astar_map.cuh"
#include "heap.cuh"
#include "util.hpp"
#include "seq_astar.hpp"

#include <cstdio>
#include <cstdint>
#include <assert.h>

int main(int argc, char *argv[]) {
/*
	if (argc != 8) {
		fprintf(stderr, "Usage: ./astar [mapH] [mapW] [block_rate] [batchNum] [batchSize] [blockNum] [blockSize]\n");
		return 1;
	}

	uint32_t map_h = atoi(argv[1]);
	uint32_t map_w = atoi(argv[2]);
	uint32_t block_rate = atoi(argv[3]);
	assert(block_rate < 100);
	uint32_t batch_num = atoi(argv[4]);
	uint32_t batch_size = atoi(argv[5]);
	uint32_t block_num = atoi(argv[6]);
	uint32_t block_size = atoi(argv[7]);
	astar::AstarMap map(map_h, map_w, block_rate);
*/
	if (argc != 7) {
        fprintf(stderr, "Usage: ./astar [map_file] [batchNum] [batchSize] [blockNum] [blockSize] [seq_flag] [mode: 0/pq]\n");
        return 1;
    }

    uint32_t batch_num = atoi(argv[2]);
    uint32_t batch_size = atoi(argv[3]);
    uint32_t block_num = atoi(argv[4]);
    uint32_t block_size = atoi(argv[5]);
    bool seq_flag = atoi(argv[6]) == 1 ? true : false;
    astar::AstarMap map(argv[1]);
    uint32_t mode = atoi(argv[7]);
    uint32_t map_h = map.h_;
    uint32_t map_w = map.w_;

    struct timeval start_time, end_time;

	Heap<astar::AstarItem> heap(batch_num, batch_size);
#ifdef DEBUG_PRINT
    map.PrintMap();
#else
    if (map_h <= 20) map.PrintMap();
#endif
    astar::AppItem app_item(&map);

	astar::PQModel<astar::AstarItem> h_model(block_num, block_size, 
											batch_num, batch_size,
											&heap, &app_item);
	astar::PQModel<astar::AstarItem> *d_model;
	cudaMalloc((void **)&d_model, sizeof(astar::PQModel<astar::AstarItem>));

    vector<uint32_t> g_list(map_h * map_w, UINT_MAX);
    if (seq_flag == true) {
        // Start handling sequential astar.
        uint32_t *seq_map = new uint32_t[map.size_]();
        cudaMemcpy(seq_map, map.d_map, map.size_ * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        setTime(&start_time);

        seq_astar::SeqAstarSearch1(seq_map, map_h, map_w, g_list);
        uint32_t seq_path_weight = g_list[map_h * map_w - 1];
        if (seq_path_weight == UINT_MAX) {
            printf("No Available Path\n");
        } else {
            h_model.seq_path_weight_ = seq_path_weight;
            printf("Sequential Result: %u\n", seq_path_weight);
        }
        setTime(&end_time);
        printf("Sequential Time: %.4f ms\n", getTime(&start_time, &end_time));
        delete []seq_map;
        // End handling sequential astar.
    }

    cudaMemcpy(d_model, &h_model, sizeof(astar::PQModel<astar::AstarItem>), cudaMemcpyHostToDevice);
	size_t smemSize = batch_size * sizeof(astar::AstarItem) // del_items
					+ 2 * sizeof(uint32_t) // ins_size, del_size
					+ 5 * batch_size * sizeof(astar::AstarItem); // ins/del operations

    setTime(&start_time);

    astar::Init<astar::AstarItem><<<1, 1, smemSize>>>(d_model);
    cudaDeviceSynchronize();

    setTime(&end_time);
    printf("Init Time: %.4f ms\n", getTime(&start_time, &end_time));
    setTime(&start_time);

    if (mode == /* pure pq mode */ 0) {
        astar::Run<astar::AstarItem><<<block_num, block_size, smemSize>>>(d_model);
        cudaDeviceSynchronize();
    }

    setTime(&end_time);
    printf("Run Time: %.4f ms\n", getTime(&start_time, &end_time));

    cudaMemcpy(&h_model, d_model, sizeof(astar::PQModel<astar::AstarItem>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&app_item, h_model.d_app_item_, sizeof(astar::AppItem), cudaMemcpyDeviceToHost);
#ifdef DEBUGi
    uint32_t *h_dist = new uint32_t[map_h * map_w];
    cudaMemcpy(h_dist, app_item.d_dist, map_h * map_w * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    uint32_t visited_nodes_count = 0;
    for (int i = 0; i < map_h * map_w; ++i) {
        if (h_dist[i] != UINT_MAX) visited_nodes_count++;
    }
    printf("visited_nodes_count: %u\n", visited_nodes_count);
    uint32_t tmp_counter = 0;
    for (int i = 0; i < map_h * map_w; ++i) {
        if (h_dist[i] == UINT_MAX && g_list[i] != UINT_MAX) {
            if (g_list[i] + (map_h + map_w - 2 - i % map_h - i / map_w) >= h_model.seq_path_weight_) {
                tmp_counter++;
            }
        }
    }
    printf("tmp counter: %u\n", tmp_counter);
#endif
    uint32_t gpu_path_weight = 0;
    cudaMemcpy(&gpu_path_weight, &app_item.d_dist[map_h * map_w - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (gpu_path_weight == UINT_MAX) {
        printf("No Available Path\n");
    } else {
        printf("Gpu Result: %u\n", gpu_path_weight);
    }
    if (h_model.seq_path_weight_ != gpu_path_weight) {
        printf("Error: Sequential Result (%u) is not equal to GPU Result (%u).\n", 
                h_model.seq_path_weight_, gpu_path_weight);
    }

    cudaFree(d_model); d_model = NULL;
/*
    astar::RunRemain<astar::AstarItem><<<block_num, block_size, smemSize>>>(d_model);
    cudaDeviceSynchronize();
    cudaMemcpy(&gpu_path_weight, &app_item.d_dist[map_h * map_w - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (gpu_path_weight == UINT_MAX) {
        printf("No Available Path\n");
    } else {
        printf("Remain Gpu Result: %u\n", gpu_path_weight);
    }
*/
	return 0;
}
