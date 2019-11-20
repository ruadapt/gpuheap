#!/bin/bash

# This script is used to test the fifo queue (buffer)

INDEX=11.motivate.mixed
OUTPUT=knapsack_test_${INDEX}.out

FILE=$1

printf "auto test\n"
for ((i=1;i<=10;i++)); do
    printf "%s " ${FILE##*/}
    #timeout 30m ./../seq/knapsack_seq $FILE 0 1 >> ${OUTPUT}
    #timeout 3m ./../seq/knapsack_seq $FILE 1 1 >> ${OUTPUT}
    #timeout 30m ./../knapsack_debug $FILE 1024000 256 16 256 150 0 0 0 0 >> ${OUTPUT}
    #timeout 30m ./../knapsack_debug $FILE 1024000 256 16 256 150 1 0 0 0 >> ${OUTPUT}
    timeout 1m ./../knapsack $FILE 102400 256 16 256 152 2 1 0 0
    #timeout 30m ./../knapsack $FILE 1024000 256 16 256 150 1 0 0 0 >> ${OUTPUT}
    printf "\n"
done


