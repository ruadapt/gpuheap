#!/bin/bash

# This script is used to test the fifo queue (buffer)

INDEX=16.switch.pldi
OUTPUT=knapsack_test_${INDEX}.out

printf "test switch on pldi\n"
for FILE in ../datasets/pldi/ks_5_*00; do
    echo $FILE
#        ./../seq/knapsack_seq_test $FILE 1 1
#        printf "%s " ${FILE##*/} >> ${OUTPUT}
#        timeout 3m ./../seq/knapsack_seq $FILE 1 1 >> ${OUTPUT}
#        timeout 3m ./../seq/knapsack_seq $FILE 0 1 >> ${OUTPUT}
    #timeout 30m ./../knapsack_debug $FILE 1024000 256 16 256 150 0 0 0 0 >> ${OUTPUT}
    #timeout 30m ./../knapsack_debug $FILE 1024000 256 16 256 150 1 0 0 0 >> ${OUTPUT}
    ./../knapsack ${FILE} 102400 256 8 256 128 3 1 0 0 >> ${OUTPUT}
    #timeout 1m ./../knapsack $FILE 1024000 256 16 256 150 2 1 0 0 >> ${OUTPUT}
    printf "\n" >> ${OUTPUT}
done


