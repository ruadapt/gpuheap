#!/bin/bash

i=0
totalcases=26
for ((batchSize=32;batchSize<=2048;batchSize*=2));do
    max_blockSize=$(($batchSize<=1024 ? $batchSize:1024))
    for ((blockSize=32;blockSize<=$max_blockSize;blockSize*=2));do
        printf "\r[%d/%d]" $i $totalcases
        RES=`./merge_path_test $batchSize $blockSize | tail -1 | cut -d' ' -f1`
        if [ "$RES" != "Success" ]; then
            echo $batchSize $blockSize $RES
            ./merge_path_test $batchSize $blockSize
            exit 1
        fi
        i=$((i+1))
    done
done
printf "\rAll $totalcases testcases passed\nSuccess\n"
