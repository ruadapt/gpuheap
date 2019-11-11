#!/bin/bash

# generate a new file
pushd ../datasets/final/ > /dev/null
./gen2 $2 $3 $1 $4 $5 > /dev/null
TESTFILE="ks_${1}_${2}_${3}"
mv test.in ${TESTFILE}
FILE=$(realpath ${TESTFILE})

#echo $FILE

popd > /dev/null

# run the knapsack program
#timeout 30m ./../seq/knapsack_seq $FILE 0 1 >> ${OUTPUT}
timeout 1m ./../seq/knapsack_seq $FILE 1 1
#timeout 30m ./../knapsack_debug $FILE 1024000 256 16 256 150 0 0 0 0 >> ${OUTPUT}
#timeout 30m ./../knapsack_debug $FILE 1024000 256 16 256 150 1 0 0 0 >> ${OUTPUT}
#timeout 30m ./../knapsack $FILE 1024000 256 16 256 150 2 1 0 0
#timeout 30m ./../knapsack $FILE 1024000 256 16 256 150 1 0 0 0 >> ${OUTPUT}

