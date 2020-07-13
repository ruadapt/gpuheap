#!/bin/bash

i=0
totalcases=12
for ((batchSize=128;batchSize<=1024;batchSize*=2));do
    for ((testSize=20;testSize<=22;testSize++));do
        printf "\r[%d/%d]" $i $totalcases
        RES1=`./Theap_test $batchSize $testSize | tail -1 | cut -d' ' -f1`
        RES2=`./Bheap_test $batchSize $testSize | tail -1 | cut -d' ' -f1`
        RES3=`./Theap_np_test $batchSize $testSize | tail -1 | cut -d' ' -f1`
        RES4=`./Bheap_np_test $batchSize $testSize | tail -1 | cut -d' ' -f1`
        if [ "$RES1" != "Success" ]; then
            echo $RES1
            ./Theap_test $batchSize $testSize
            exit 1
        fi
        if [ "$RES2" != "Success" ]; then
            echo $RES2
            ./Bheap_test $batchSize $testSize
            exit 1
        fi
        if [ "$RES3" != "Success" ]; then
            echo $RES3
            ./TPheap_np_test $batchSize $testSize
            exit 1
        fi
        if [ "$RES4" != "Success" ]; then
            echo $RES4
            ./BPheap_np_test $batchSize $testSize
            exit 1
        fi
        i=$((i+1))
    done
done
printf "\rAll $totalcases testcases passed\nSuccess\n"
