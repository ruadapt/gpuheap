#!/bin/bash

itemNum=10240000
for ((batchSize=512;batchSize<=512;batchSize*=2))
do
    for ((tblockNum=32;tblockNum<=128;tblockNum*=2))
    do
        for ((tblockSize=32;tblockSize<=$batchSize;tblockSize*=2))
        do
#            echo -n $itemNum $tblockNum $tblockSize $batchSize 
            ./test $itemNum $batchSize $((4*itemNum/tblockSize)) 100000 $tblockNum $tblockSize
        done
    done
done

