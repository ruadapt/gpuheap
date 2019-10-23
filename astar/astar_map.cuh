#ifndef ASTAR_MAP_CUH
#define ASTAR_MAP_CUH

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <fstream>

namespace astar {

class AstarMap {
public:
	AstarMap(uint32_t h, uint32_t w, uint32_t block_rate)
		 : h_(h), w_(w), start_(0), target_(h * w - 1) {
		size_ = (h_ * w_ + 31) / 32;
		uint32_t *h_map = new uint32_t[size_]();
		GenerateMap(h_map, block_rate);
		cudaMalloc((void **)&d_map, size_ * sizeof(uint32_t));
		cudaMemcpy(d_map, h_map, size_ * sizeof(uint32_t), cudaMemcpyHostToDevice);
		delete []h_map;
	}
    
    AstarMap(const char *filename) {
        ReadMap(filename);
    }

	~AstarMap() {
		cudaFree(d_map);
	}

    void WriteMap(const char *filename) {
        std::ofstream fout(filename);
        fout << h_ << " " << w_ << std::endl;
        uint32_t *h_map = new uint32_t[size_]();
        cudaMemcpy(h_map, d_map, size_ * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        for (int i = 0; i < h_; ++i) {
            for (int j = 0; j < w_; ++j) {
                fout << GetMap(i * w_ + j, h_map) << " ";
            }
            fout << std::endl;
        }
        delete []h_map;
        fout.close();
    }

    void ReadMap(const char *filename) {
        std::ifstream fin(filename);
        fin >> h_ >> w_;
        start_ = 0;
        target_ = h_ * w_ - 1;
        size_ = (h_ * w_ + 31) / 32;
        uint32_t *h_map = new uint32_t[size_]();
        for (int i = 0; i < h_; ++i) {
            for (int j = 0; j < w_; ++j) {
                uint32_t v;
                fin >> v;
                SetMap(i * w_ + j, v, h_map);
            }
        }
		cudaMalloc((void **)&d_map, size_ * sizeof(uint32_t));
        cudaMemcpy(d_map, h_map, size_ * sizeof(uint32_t), cudaMemcpyHostToDevice);
        delete []h_map;
        fin.close();
    }

    void SetMap(uint32_t i, uint32_t v, uint32_t *map) {
        uint32_t index = i / 32;
        uint32_t offset = i % 32;
        if (v == 1) {
            map[index] |= (1U << offset);
        } else {
            map[index] &= ~(1U << offset);
        }
    }
    void GenerateMap(uint32_t *map, uint32_t block_rate) {
        srand(time(NULL));
        for (int i = 0; i < h_ * w_; ++i) {
            SetMap(i, rand() % 100 > block_rate ? 1 : 0, map);
        }
        SetMap(0, 1, map);
        SetMap(h_ * w_ - 1, 1, map);
    }

	__device__ __forceinline__ uint32_t GetX(uint32_t i) {
		return i % w_;
	}

	__device__ __forceinline__ uint32_t GetY(uint32_t i) {
		return i / w_;
	}

	// 1 2 3
	// 4 * 5
	// 6 7 8
	__device__ __forceinline__ uint32_t GetNodeFromXY(int x, int y) {
		if (x < 0 || y < 0 || x >= w_ || y >= h_) return UINT_MAX;
		else return w_ * y + x;
	}
	__device__ __forceinline__ uint32_t Get1(uint32_t i) {
		return GetNodeFromXY((int)GetX(i) - 1, (int)GetY(i) - 1);
	}
	__device__ __forceinline__ uint32_t Get2(uint32_t i) {
		return GetNodeFromXY((int)GetX(i), (int)GetY(i) - 1);
	}
	__device__ __forceinline__ uint32_t Get3(uint32_t i) {
		return GetNodeFromXY((int)GetX(i) + 1, (int)GetY(i) - 1);
	}
	__device__ __forceinline__ uint32_t Get4(uint32_t i) {
		return GetNodeFromXY((int)GetX(i) - 1, (int)GetY(i));
	}
	__device__ __forceinline__ uint32_t Get5(uint32_t i) {
		return GetNodeFromXY((int)GetX(i) + 1, (int)GetY(i));
	}
	__device__ __forceinline__ uint32_t Get6(uint32_t i) {
		return GetNodeFromXY((int)GetX(i) - 1, (int)GetY(i) + 1);
	}
	__device__ __forceinline__ uint32_t Get7(uint32_t i) {
		return GetNodeFromXY((int)GetX(i), (int)GetY(i) + 1);
	}
	__device__ __forceinline__ uint32_t Get8(uint32_t i) {
		return GetNodeFromXY((int)GetX(i) + 1, (int)GetY(i) + 1);
	}	

	__device__ __forceinline__ uint32_t GetMap(uint32_t i) {
		// XXX: This is a hacky that we use UINT_MAX to represent invalid node.
		if (i == UINT_MAX) return 0;
		uint32_t index = i / 32;
		uint32_t offset = i % 32;
		return (d_map[index] >> offset) & 1U;
	}

	uint32_t GetMap(uint32_t i, uint32_t *map) {
		uint32_t index = i / 32;
		uint32_t offset = i % 32;
		return (map[index] >> offset) & 1U;
	}

	void PrintMap() {
		uint32_t *h_map = new uint32_t[size_];
		cudaMemcpy(h_map, d_map, size_ * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		for (uint32_t i = 0; i < h_; ++i) {
			for (uint32_t j = 0; j < w_; ++j) {
				printf("%3u ", GetMap(i * w_ + j, h_map));
			}
			printf("\n");
		}
	}
	/// 1 represents path, 0 represents obstacle.
	uint32_t *d_map;
	uint32_t h_;
	uint32_t w_;
	uint32_t size_;
	uint32_t start_;
	uint32_t target_;
};

} // namespace astar

#endif
