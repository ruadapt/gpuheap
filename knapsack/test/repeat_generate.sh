#!/bin/bash

INDEX=14.findlowlevel
OUTPUT=knapsack_test_${INDEX}.out

FILE=$1
printf "%s " ${FILE##*/}
upthres=100
lowthres=0
count=0

for ((n=400;n<=400;n+=100));do
    for ((itemnum=10000;itemnum<=20000;itemnum+=1000));do
        for ((i=5;i<=14;i++)); do
            timeout 70s ./generate_single_test.sh 5 $n $itemnum $i 1000 > tmp 
            level=$(head -n 1 tmp)
            content=$(tail -n 1 tmp)
            if ((level < upthres)); then
                if ((level > lowthres)); then
                    count=$((count+1))
                    printf "%d %d [%d/13] %d / level %d " $n $itemnum $i $count $level >> ${OUTPUT}
                    echo $content >> ${OUTPUT}
                fi
            fi
            rm -rf tmp
            printf "%d %d [%d/13] %d / level %d " $n $itemnum $i $count $level 
            echo $content
        done
    done
done


