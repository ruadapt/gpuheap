NVCCFLAG= -g -G -arch=sm_61 -O3 -std=c++11 
NVCC=nvcc ${NVCCFLAG}

all: astar

astar: astar_main.cu *.cuh *.hpp
	${NVCC} astar_main.cu -I../ -o psync-astar

clean:
	rm -rf psync-astar
	
