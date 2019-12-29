#ifndef PQ_MODEL_CUH
#define PQ_MODEL_CUH

#include "heap_api.cuh"
#include "heap.cuh"
#include "astar_map.cuh"

#include <cstdio>
#include <iostream>

#define EXPAND_NUMBER 8

namespace astar {

__device__ __forceinline__ uint32_t ManhattanDistance(uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2) {
	return x2 - x1 + y2 - y1;
}


// This class contains all application specific data and functions.
class AppItem {
public:
	AppItem(AstarMap *map) {
	    cudaMalloc((void **)&d_map, sizeof(AstarMap));
	    cudaMemcpy(d_map, map, sizeof(AstarMap), cudaMemcpyHostToDevice);
        uint32_t node_num = map->h_ * map->w_;
	    uint32_t *h_dist = new uint32_t[node_num]();
        for (int i = 1; i < node_num; i++) {
            h_dist[i] = UINT_MAX;
        }
        cudaMalloc((void **)&d_dist, node_num * sizeof(uint32_t));
	    cudaMemcpy(d_dist, h_dist, node_num * sizeof(uint32_t), cudaMemcpyHostToDevice);
        delete []h_dist;
    }
    
    ~AppItem() {
        cudaFree(d_map); d_map = nullptr;
        cudaFree(d_dist); d_dist = nullptr;
    }

    __device__ __forceinline__ uint32_t H(uint32_t i) {
        return ManhattanDistance(d_map->GetX(i), d_map->GetY(i), d_map->w_ - 1, d_map->h_ - 1);
    }

	AstarMap *d_map;
	uint32_t *d_dist;
};	

template <class HeapItem>
__forceinline__ __device__ void AppKernel(AppItem *app_item, uint32_t w, uint32_t h,
				HeapItem *del_items, int *del_size, 
				HeapItem *ins_items, int *ins_size) {
	uint32_t *dist = app_item->d_dist;

	for (int i = threadIdx.x; i < *del_size; i += blockDim.x) {
		HeapItem item = del_items[i];
		uint32_t item_node = item.node_;
        uint32_t item_f = item.f_;
		uint32_t item_g = item_f - app_item->H(item_node);
		if (item_g > dist[item_node]) continue;
        int wmap[8] = {3, 2, 2, 2, 1, 3, 1, 2};
		/*uint8_t *wmap = &app_item->d_map->d_wmap[8 * item_node];*/
        
		// 1 2 3
		// 4 * 5
		// 6 7 8
/*
        uint32_t node2 = item_node < w ? UINT_MAX : item_node - w;
        uint32_t node1 = (node2 == UINT_MAX || node2 % w == 0) ? UINT_MAX : node2 - 1;
        uint32_t node3 = (node2 == UINT_MAX || node2 % w == w - 1) ? UINT_MAX : node2 + 1;

        uint32_t node4 = item_node % w == 0 ? UINT_MAX : item_node - 1;
        uint32_t node5 = item_node % w == w - 1 ? UINT_MAX : item_node + 1;

        uint32_t node7 = item_node / w == h - 1 ? UINT_MAX : item_node + w;
        uint32_t node6 = (node7 == UINT_MAX || node7 % w == 0) ? UINT_MAX : node7 - 1;
        uint32_t node8 = (node7 == UINT_MAX || node7 % w == w - 1) ? UINT_MAX : node7 + 1;
*/
		uint32_t node1 = app_item->d_map->Get1(item_node);
		if (app_item->d_map->GetMap(node1) && atomicMin(&dist[node1], item_g + wmap[0]) > item_g + wmap[0]) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + wmap[0] + app_item->H(node1), node1);
		}
		uint32_t node2 = app_item->d_map->Get2(item_node);
		if (app_item->d_map->GetMap(node2) && atomicMin(&dist[node2], item_g + wmap[1]) > item_g + wmap[1]) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + wmap[1] + app_item->H(node2), node2);
		}
		uint32_t node3 = app_item->d_map->Get3(item_node);
		if (app_item->d_map->GetMap(node3) && atomicMin(&dist[node3], item_g + wmap[2]) > item_g + wmap[2]) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + wmap[2] + app_item->H(node3), node3);
		}
		uint32_t node4 = app_item->d_map->Get4(item_node);
		if (app_item->d_map->GetMap(node4) && atomicMin(&dist[node4], item_g + wmap[3]) > item_g + wmap[3]) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + wmap[3] + app_item->H(node4), node4);
		}
		uint32_t node5 = app_item->d_map->Get5(item_node);
		if (app_item->d_map->GetMap(node5) && atomicMin(&dist[node5], item_g + wmap[4]) > item_g + wmap[4]) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + wmap[4] + app_item->H(node5), node5);
//            if (node5 == 999999) printf("node 5 %u %u %u %u %u\n", item_node, item_g, item_f, item_g + 1 + app_item->H(node5), dist[999999]);
		}
		uint32_t node6 = app_item->d_map->Get6(item_node);
		if (app_item->d_map->GetMap(node6) && atomicMin(&dist[node6], item_g + wmap[5]) > item_g + wmap[5]) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + wmap[5] + app_item->H(node6), node6);
		}
		uint32_t node7 = app_item->d_map->Get7(item_node);
		if (app_item->d_map->GetMap(node7) && atomicMin(&dist[node7], item_g + wmap[6]) > item_g + wmap[6]) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + wmap[6] + app_item->H(node7), node7);
//            if (node7 == 999999) printf("node 7 %u %u %u %u %u\n", item_node, item_g, item_f, item_g + 1 + app_item->H(node7), dist[999999]);
		}
		uint32_t node8 = app_item->d_map->Get8(item_node);
		if (app_item->d_map->GetMap(node8) && atomicMin(&dist[node8], item_g + wmap[7]) > item_g + wmap[7]) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + wmap[7] + app_item->H(node8), node8);
//	        if (node8 == 999999) printf("node 8 %u %u %u %u %u\n", item_node, item_g, item_f, item_g + 2 + app_item->H(node8), dist[999999]);
	    }
	}
}

template <class HeapItem>
class PQModel {
public:
	PQModel(uint32_t block_num, uint32_t block_size, 
		uint32_t batch_num, uint32_t batch_size,
		Heap<HeapItem> *heap, TB<HeapItem> *insTB, TB<HeapItem> *delTB, 
        AppItem *app_item) :
	block_num_(block_num), block_size_(block_size),
	batch_num_(batch_num), batch_size_(batch_size) {
		cudaMalloc((void **)&d_heap_, sizeof(Heap<HeapItem>));
		cudaMemcpy(d_heap_, heap, sizeof(Heap<HeapItem>), cudaMemcpyHostToDevice);

        cudaMalloc((void **)&d_insTB_, sizeof(TB<HeapItem>));
        cudaMemcpy(d_insTB_, insTB, sizeof(TB<HeapItem>), cudaMemcpyHostToDevice);

        cudaMalloc((void **)&d_delTB_, sizeof(TB<HeapItem>));
        cudaMemcpy(d_delTB_, delTB, sizeof(TB<HeapItem>), cudaMemcpyHostToDevice);

		cudaMalloc((void **)&d_app_item_, sizeof(AppItem));
		cudaMemcpy(d_app_item_, app_item, sizeof(AppItem), cudaMemcpyHostToDevice);
	}
	~PQModel() {
		cudaFree(d_heap_); d_heap_ = nullptr;
		cudaFree(d_app_item_); d_app_item_ = nullptr;
        cudaFree(d_insTB_); d_insTB_ = nullptr;
        cudaFree(d_delTB_); d_delTB_ = nullptr;

	}

	Heap<HeapItem> *d_heap_;
    TB<HeapItem> *d_insTB_;
    TB<HeapItem> *d_delTB_;
	AppItem *d_app_item_;

	uint32_t block_num_;
	uint32_t block_size_;
	uint32_t batch_num_;
	uint32_t batch_size_;

};

template <class HeapItem>
__global__ void InitInsertItems(PQModel<HeapItem> *model, HeapItem *ins_items) {
    ins_items[0] = HeapItem(model->d_app_item_->H(0), 0);
}

template <class HeapItem>
void Init(PQModel<HeapItem> *h_model, PQModel<HeapItem> *d_model) {
    int batch_size = h_model->batch_size_;
    int block_num = h_model->block_num_;
    int block_size = h_model->block_size_;

    HeapItem *h_ins_items = new HeapItem[batch_size];
    HeapItem *d_ins_items;
    cudaMalloc((void **)&d_ins_items, sizeof(HeapItem) * batch_size);
    cudaMemcpy(d_ins_items, h_ins_items, sizeof(HeapItem) * batch_size, cudaMemcpyHostToDevice);

    InitInsertItems<HeapItem><<<1, 1>>>(d_model, d_ins_items);
    cudaDeviceSynchronize();

    psyncHeapInsert(h_model->d_heap_,
                    h_model->d_insTB_, h_model->d_delTB_,
                    d_ins_items, batch_size,
                    block_num, block_size, batch_size);

    cudaFree(d_ins_items); d_ins_items = nullptr;
}

template <class HeapItem>
__global__ void AppKernelWrapper(PQModel<HeapItem> *model,
                        HeapItem *del_items, int del_size,
                        HeapItem *ins_items, int *ins_size) {
    int batch_size = model->batch_size_;
    int batch_count = del_size / batch_size;
    if (blockIdx.x >= batch_count) return;
    AppKernel(model->d_app_item_, 
            model->d_app_item_->d_map->w_, 
            model->d_app_item_->d_map->h_,
            del_items + blockIdx.x * batch_size, &del_size, 
            ins_items, ins_size);
    __syncthreads();
}

template <class HeapItem>
__global__ void UpdateDelSize(PQModel<HeapItem> *model, int *del_size) {
    int batch_count = model->d_heap_->batchCount;
    int block_num = model->block_num_;
    int block_size = model->block_size_;
    *del_size = batch_count < block_num ? 
        batch_count * block_size : 
        block_num * block_size;
}

template <class HeapItem>
__global__ void CheckTerminate(PQModel<HeapItem> *model, int *terminate_flag) {
    uint32_t w = model->d_app_item_->d_map->w_;
    uint32_t h = model->d_app_item_->d_map->h_;
    if (model->d_heap_->batchCount == 0 || 
            model->d_heap_->heapKeys[0].f_ >= model->d_app_item_->d_dist[w * h - 1]) {
        atomicCAS(terminate_flag, 0, 1);
    }
}


template <class HeapItem>
void Run(PQModel<HeapItem> *h_model, PQModel<HeapItem> *d_model, 
        Heap<HeapItem> *h_heap, AppItem *h_app_item, uint32_t map_w, uint32_t map_h) {
	int batch_size = h_model->batch_size_;
    int block_num = h_model->block_num_;
    int block_size = h_model->block_size_;

    Heap<HeapItem> *d_heap = h_model->d_heap_;
    TB<HeapItem> *d_insTB = h_model->d_insTB_;
    TB<HeapItem> *d_delTB = h_model->d_delTB_;

    HeapItem *h_ins_items = new HeapItem[EXPAND_NUMBER * batch_size * block_num];
    HeapItem *d_ins_items;
    cudaMalloc((void **)&d_ins_items, sizeof(HeapItem) * 2 * batch_size * block_num);
    cudaMemcpy(d_ins_items, h_ins_items, sizeof(HeapItem) * 2 * batch_size * block_num, cudaMemcpyHostToDevice);

    HeapItem *h_del_items = new HeapItem[batch_size * block_num];
    HeapItem *d_del_items;
    cudaMalloc((void **)&d_del_items, sizeof(HeapItem) * batch_size * block_num);
    cudaMemcpy(d_del_items, h_del_items, sizeof(HeapItem) * batch_size * block_num, cudaMemcpyHostToDevice);

    int h_ins_size = 0;
    int *d_ins_size;
    cudaMalloc((void **)&d_ins_size, sizeof(int));
    cudaMemcpy(d_ins_size, &h_ins_size, sizeof(int), cudaMemcpyHostToDevice);

    int h_del_size = batch_size;
    int *d_del_size;
    cudaMalloc((void **)&d_del_size, sizeof(int));
    cudaMemcpy(d_del_size, &h_del_size, sizeof(int), cudaMemcpyHostToDevice);

    int h_terminate_flag = 0;
    int *d_terminate_flag;
    cudaMalloc((void **)&d_terminate_flag, sizeof(int));
    cudaMemset(d_terminate_flag, 0, sizeof(int));

    int iter = 0;

    while (1) {
        cudaMemcpy(h_heap, h_model->d_heap_, sizeof(Heap<HeapItem>), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_app_item, h_model->d_app_item_, sizeof(AppItem), cudaMemcpyDeviceToHost);
        uint32_t opt_value = 0;
        cudaMemcpy(&opt_value, &h_app_item->d_dist[map_h * map_w - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost);
        uint32_t heap_val = 0;
        cudaMemcpy(&heap_val, &h_heap->heapKeys[0].f_, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        printf("iter %d heap count %d opt value %u heap_val: %u\n", ++iter, h_heap->batchCount, opt_value, heap_val);
        printf("del size: %d\n", h_del_size);

        psyncHeapDelete(d_heap, d_insTB, d_delTB,
                d_del_items, h_del_size,
                block_num, block_size, batch_size);

        /*cudaMemcpy(h_del_items, d_del_items, sizeof(HeapItem) * h_del_size, cudaMemcpyDeviceToHost);*/
        /*for (int i = 0; i < h_del_size; i++) {*/
            /*std::cout << h_del_items[i] << " | ";*/
        /*}*/
        /*std::cout << std::endl;*/


        cudaMemset(d_ins_size, 0, sizeof(int));

        AppKernelWrapper<HeapItem>
            <<<block_num, block_size>>>(d_model,
                        d_del_items, h_del_size,
                        d_ins_items, d_ins_size);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_ins_size, d_ins_size, sizeof(int), cudaMemcpyDeviceToHost);
        printf("real ins size: %d ", h_ins_size);
        h_ins_size = (h_ins_size + batch_size - 1) / batch_size * batch_size;
        printf("ins size: %d\n", h_ins_size);

        psyncHeapInsert(d_heap, d_insTB, d_delTB,
                d_ins_items, h_ins_size,
                block_num, block_size, batch_size);

        cudaMemcpy(h_heap, h_model->d_heap_, sizeof(Heap<HeapItem>), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_app_item, h_model->d_app_item_, sizeof(AppItem), cudaMemcpyDeviceToHost);
        cudaMemcpy(&opt_value, &h_app_item->d_dist[map_h * map_w - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost);
        printf("iter %d heap count %d opt value %u\n", ++iter, h_heap->batchCount, opt_value);

        cudaMemset(d_terminate_flag, 0, sizeof(int));
        CheckTerminate<HeapItem><<<1, 1>>>(d_model, d_terminate_flag);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_terminate_flag, d_terminate_flag, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_terminate_flag) break;

        cudaMemcpy(d_del_size, &h_del_size, sizeof(int), cudaMemcpyHostToDevice);
        UpdateDelSize<HeapItem><<<1, 1>>>(d_model, d_del_size);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_del_size, d_del_size, sizeof(int), cudaMemcpyDeviceToHost);

	}

    cudaFree(d_ins_items); d_ins_items = nullptr;
    cudaFree(d_del_items); d_del_items = nullptr;
    cudaFree(d_ins_size); d_ins_size = nullptr;
    cudaFree(d_del_size); d_del_size = nullptr;
    cudaFree(d_terminate_flag); d_terminate_flag = nullptr;
}

} // astar

#endif
