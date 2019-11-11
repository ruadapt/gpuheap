#!/bin/bash

INDEX=13.findlowlevel
OUTPUT=knapsack_test_${INDEX}.out

FILE=$1
printf "%s " ${FILE##*/}
upthres=100
lowthres=0
count=0

for ((n=400;n<=1000;n+=100));do
    for ((itemnum=10000;itemnum<=20000;itemnum+=1000));do
        for ((i=1;i<=10000;i*=2)); do
            timeout 70s ./generate_single_test.sh 3 $n $itemnum $i 1000 > tmp 
            level=$(head -n 1 tmp)
            content=$(head -n 2 tmp | tail -n 1)
            if ((level < upthres)); then
                if ((level > lowthres)); then
                    count=$((count+1))
                    printf "%d %d %d %s\n" $n $itemnum $i $content >> ${OUTPUT}
                fi
            fi
            rm -rf tmp
            printf "%d %d [%d/13] %d / level %d\n" $n $itemnum $i $count $level
        done
    done
done


