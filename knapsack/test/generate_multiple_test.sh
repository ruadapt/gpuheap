#!/bin/bash

OUTPUT=11.18.pldi.1.out
count=1

printf "test heap size, opt location, time\n" >> ${OUTPUT}

for to in 1 10; do
    count=0
    for type in 5 3; do
        for nitem in 200 400 800 1000 2000 4000; do
            for r in 20000 40000 60000; do
                for index in 500 600 700 800 900; do
                    
                    count=$((count+1))
                    TESTFILE="ks_${type}_${nitem}_${r}_${index}_1000.res"
                    printf "to %d type: %d nitem %d r %d index %d [%d/180]..." $to $type $nitem $r $index $count

                    if [ -f ../datasets/tmp/${TESTFILE} ]; then
                        printf "\n"
                        continue
                    fi

                    printf "ks_%d_%d_%d_%d " $type $nitem $r $index >> ${OUTPUT}

                    ./generate_single_test.sh ${type} ${nitem} $r $index 1000 $to >> ${OUTPUT}

                    printf "\n" >> ${OUTPUT}
                    printf "\n"
                done
            done
        done
    done
done
