nvcc = nvcc
nvccflag = -g -G -std=c++11 -O3 -arch=sm_61 -DHEAP_SORT -DTHRUST_SORT #-DINSERT_SMEM
testflag = -DINSERT_TEST -DDELETE_TEST# -DCONCURRENT_TEST
heappath = ../heap/

all: sortTD sortBU sortTDNP sortBUNP

#sort: sort.cu $(heappath)tditdd_as/heap.cuh
#	$(nvcc) $(nvccflag) $(testflag) -I$(heappath)tditdd_as/ $< -o $@

sortTD: sort.cu $(heappath)tditdd/heap.cuh
	$(nvcc) $(nvccflag) $(testflag) -I$(heappath)tditdd/ $< -o $@

sortTDNP: sort_np.cu $(heappath)tditdd_np/heap.cuh
	$(nvcc) $(nvccflag) $(testflag) -I$(heappath)tditdd_np/ $< -o $@

sortBU: sort.cu $(heappath)buitdd/heap.cuh
	$(nvcc) $(nvccflag) $(testflag) -I$(heappath)buitdd/ $< -o $@

sortBUNP: sort_np.cu $(heappath)buitdd_np/heap.cuh
	$(nvcc) $(nvccflag) $(testflag) -I$(heappath)buitdd_np/ $< -o $@

clean:
	rm -rf sortTD sortTDNP sortBU sortBUNP
