#ifndef PQ_MODEL_CUH
#define PQ_MODEL_CUH

#include "heap.cuh"
#include "astar_map.cuh"

#define EXPAND_NUMBER 8

#ifdef DEBUG
__device__ uint32_t explored_node_number;
#endif

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
__device__ void AppKernel(AppItem *app_item, 
				HeapItem *del_items, uint32_t *del_size, 
				HeapItem *ins_items, uint32_t *ins_size) {
	uint32_t *dist = app_item->d_dist;
	for (int i = threadIdx.x; i < *del_size; i += blockDim.x) {
		HeapItem item = del_items[i];
		uint32_t item_node = item.node_;
        uint32_t item_f = item.f_;
		uint32_t item_g = item_f - app_item->H(item_node);
		if (item_g > dist[item_node]) continue;
        
		// 1 2 3
		// 4 * 5
		// 6 7 8
		uint32_t node1 = app_item->d_map->Get1(item_node);
		if (app_item->d_map->GetMap(node1) && atomicMin(&dist[node1], item_g + 3) > item_g + 3) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + 3 + app_item->H(node1), node1);
		}
		uint32_t node2 = app_item->d_map->Get2(item_node);
		if (app_item->d_map->GetMap(node2) && atomicMin(&dist[node2], item_g + 2) > item_g + 2) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + 2 + app_item->H(node2), node2);
		}
		uint32_t node3 = app_item->d_map->Get3(item_node);
		if (app_item->d_map->GetMap(node3) && atomicMin(&dist[node3], item_g + 2) > item_g + 2) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + 2 + app_item->H(node3), node3);
		}
		uint32_t node4 = app_item->d_map->Get4(item_node);
		if (app_item->d_map->GetMap(node4) && atomicMin(&dist[node4], item_g + 2) > item_g + 2) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + 2 + app_item->H(node4), node4);
		}
		uint32_t node5 = app_item->d_map->Get5(item_node);
		if (app_item->d_map->GetMap(node5) && atomicMin(&dist[node5], item_g + 1) > item_g + 1) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + 1 + app_item->H(node5), node5);
//            if (node5 == 999999) printf("node 5 %u %u %u %u %u\n", item_node, item_g, item_f, item_g + 1 + app_item->H(node5), dist[999999]);
		}
		uint32_t node6 = app_item->d_map->Get6(item_node);
		if (app_item->d_map->GetMap(node6) && atomicMin(&dist[node6], item_g + 3) > item_g + 3) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + 3 + app_item->H(node6), node6);
		}
		uint32_t node7 = app_item->d_map->Get7(item_node);
		if (app_item->d_map->GetMap(node7) && atomicMin(&dist[node7], item_g + 1) > item_g + 1) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + 1 + app_item->H(node7), node7);
//            if (node7 == 999999) printf("node 7 %u %u %u %u %u\n", item_node, item_g, item_f, item_g + 1 + app_item->H(node7), dist[999999]);
		}
		uint32_t node8 = app_item->d_map->Get8(item_node);
		if (app_item->d_map->GetMap(node8) && atomicMin(&dist[node8], item_g + 2) > item_g + 2) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + 2 + app_item->H(node8), node8);
//	        if (node8 == 999999) printf("node 8 %u %u %u %u %u\n", item_node, item_g, item_f, item_g + 2 + app_item->H(node8), dist[999999]);
	    }
	}
}

template <class HeapItem>
class PQModel {
public:
	PQModel(uint32_t block_num, uint32_t block_size, 
		uint32_t batch_num, uint32_t batch_size,
		Heap<HeapItem> *heap, AppItem *app_item) :
	block_num_(block_num), block_size_(block_size),
	batch_num_(batch_num), batch_size_(batch_size) {
		cudaMalloc((void **)&d_heap_, sizeof(Heap<HeapItem>));
		cudaMemcpy(d_heap_, heap, sizeof(Heap<HeapItem>), cudaMemcpyHostToDevice);

		cudaMalloc((void **)&d_app_item_, sizeof(AppItem));
		cudaMemcpy(d_app_item_, app_item, sizeof(AppItem), cudaMemcpyHostToDevice);

		cudaMalloc((void **)&ins_items, EXPAND_NUMBER * batch_size * block_num * sizeof(HeapItem));		
	}
	~PQModel() {
		cudaFree(d_heap_); d_heap_ = nullptr;
		cudaFree(d_app_item_); d_app_item_ = nullptr;

		cudaFree(ins_items); ins_items = nullptr;
	}

	Heap<HeapItem> *d_heap_;
	AppItem *d_app_item_;

	uint32_t block_num_;
	uint32_t block_size_;
	uint32_t batch_num_;
	uint32_t batch_size_;

	HeapItem *ins_items;
};

template <class HeapItem>
__global__ void Init(PQModel<HeapItem> *model) {

    if (blockIdx.x == 0) {
		if (threadIdx.x == 0) {
			// This is a hacky, our heap is min heap.
			model->ins_items[0] = HeapItem(model->d_app_item_->H(0), 0);
		}
		__syncthreads();
		model->d_heap_->insertion(model->ins_items, 1, 0);
		__syncthreads();
	}
#ifdef DEBUG
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        explored_node_number = 0;
    }
#endif
}

template <class HeapItem>
__global__ void Run(PQModel<HeapItem> *model) {

	uint32_t batch_size = model->batch_size_;
	Heap<HeapItem> *d_heap = model->d_heap_;
	// HeapItem *ins_items is not on smem due to undetermined large size
	HeapItem *ins_items = model->ins_items + blockIdx.x * batch_size * EXPAND_NUMBER;

	extern __shared__ int smem[];
	HeapItem *del_items = (HeapItem *)&smem[0];
	uint32_t *del_size = (uint32_t *)&del_items[batch_size];
	// HeapItem *ins_items is not on smem due to undetermined large size
	uint32_t *ins_size = (uint32_t *)&del_size[1];

	uint32_t smemOffset = (sizeof(HeapItem) * batch_size
							+ sizeof(uint32_t) * 2) / sizeof(int);
	
	while(!d_heap->ifTerminate()) {
        if (threadIdx.x == 0) {
    		*del_size = model->d_app_item_->d_dist[model->d_app_item_->d_map->target_];
        }
        __syncthreads();
        uint32_t app_t_val = *del_size;
		if(d_heap->deleteRoot(del_items, *del_size, app_t_val) == true){
			__syncthreads();
			d_heap->deleteUpdate(smemOffset);
			__syncthreads();
		}
		__syncthreads();
		if(threadIdx.x == 0){
			*ins_size = 0;
		}
		__syncthreads();
		
		if(*del_size > 0){
			AppKernel(model->d_app_item_, del_items, del_size, ins_items, ins_size);
			__syncthreads();
		}
#ifdef DEBUG
        if (threadIdx.x == 0) {
            atomicAdd(&explored_node_number, *ins_size);
        }
        __syncthreads();
#endif
		if(*ins_size > 0){
			for(uint32_t offset = 0; offset < *ins_size; offset += batch_size){
				uint32_t size = min(batch_size, *ins_size - offset);
				__syncthreads();
				d_heap->insertion(ins_items + offset, size, smemOffset);
				__syncthreads();
			}
		}
	}	
#ifdef DEBUG
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("explored_nodes_number: %u\n", explored_node_number);
    }
#endif
}

template <class HeapItem>
__global__ void RunRemain(PQModel<HeapItem> *model) {

	uint32_t batch_size = model->batch_size_;
	Heap<HeapItem> *d_heap = model->d_heap_;
	// HeapItem *ins_items is not on smem due to undetermined large size
	HeapItem *ins_items = model->ins_items + blockIdx.x * batch_size * EXPAND_NUMBER;

	extern __shared__ int smem[];
	HeapItem *del_items = (HeapItem *)&smem[0];
	uint32_t *del_size = (uint32_t *)&del_items[batch_size];
	// HeapItem *ins_items is not on smem due to undetermined large size
	uint32_t *ins_size = (uint32_t *)&del_size[1];

	uint32_t smemOffset = (sizeof(HeapItem) * batch_size
							+ sizeof(uint32_t) * 2) / sizeof(int);
	
	while(1) {
		if(d_heap->deleteRoot(del_items, *del_size) == true){
			__syncthreads();
			d_heap->deleteUpdate(smemOffset);
			__syncthreads();
		}
		__syncthreads();
		if(threadIdx.x == 0){
			*ins_size = 0;
		}
		__syncthreads();
		
		if(*del_size > 0){
			AppKernel(model->d_app_item_, del_items, del_size, ins_items, ins_size);
			__syncthreads();
        } else {
            break;
        }
#ifdef DEBUG
        if (threadIdx.x == 0) {
            atomicAdd(&explored_node_number, *ins_size);
        }
        __syncthreads();
#endif
		if(*ins_size > 0){
			for(uint32_t offset = 0; offset < *ins_size; offset += batch_size){
				uint32_t size = min(batch_size, *ins_size - offset);
				__syncthreads();
				d_heap->insertion(ins_items + offset, size, smemOffset);
				__syncthreads();
			}
		}
	}	
#ifdef DEBUG
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("explored_nodes_number: %u\n", explored_node_number);
    }
#endif
}

} // astar

#endif
