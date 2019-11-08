#!/bin/bash

# This script is used to test the fifo queue (buffer)

INDEX=11.motivate.mixed
OUTPUT=knapsack_test_${INDEX}.out

printf "auto test\n"
for ((i=1;i<=10;i++)); do
    for FILE in ../datasets/motivate/ks_3_*00; do
        printf "%s " ${FILE##*/} >> ${OUTPUT}
        #timeout 30m ./../seq/knapsack_seq $FILE 0 1 >> ${OUTPUT}
        #timeout 3m ./../seq/knapsack_seq $FILE 1 1 >> ${OUTPUT}
        #timeout 30m ./../knapsack_debug $FILE 1024000 256 16 256 150 0 0 0 0 >> ${OUTPUT}
        #timeout 30m ./../knapsack_debug $FILE 1024000 256 16 256 150 1 0 0 0 >> ${OUTPUT}
        timeout 3m ./../knapsack $FILE 1024000 256 16 256 152 3 1 0 0 >> ${OUTPUT}
        #timeout 30m ./../knapsack $FILE 1024000 256 16 256 150 1 0 0 0 >> ${OUTPUT}
    done
done


