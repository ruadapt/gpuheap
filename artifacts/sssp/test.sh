#!/bin/bash

repeat=1
#nthreads=40

for ((i=0;i<${repeat};i++)); do
    ./ssspT com-LiveJournal.out 
    ./ssspT mawi_201512020000.out 
    ./ssspT hollywood-2009.out

    ./ssspB com-LiveJournal.out 
    ./ssspB mawi_201512020000.out 
    ./ssspB hollywood-2009.out
done
