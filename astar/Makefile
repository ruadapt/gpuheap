NVCCFLAG= -g -G -arch=sm_61 -O0 -std=c++11
NVCC=nvcc ${NVCCFLAG}

all: astar generate_astar_map

astar: astar_main.cu *.cuh *.hpp
	${NVCC} astar_main.cu -o astar

astar_debug: astar_main.cu *.cuh *.hpp
	${NVCC} -DDEBUG astar_main.cu -o astar

generate_astar_map: generate_astar_map.cu astar_map.cuh
	${NVCC} generate_astar_map.cu -o generate_astar_map

clean:
	rm -rf astar generate_astar_map
	
