#!/bin/bash

for size in 100 200 300 400 500
do
    for rate in 10 20 30
    do
        ./generate_astar_map ${size} ${size} ${rate} ${FREESPACE}/astar_maps/${size}_${size}_${rate}.map
    done
done
