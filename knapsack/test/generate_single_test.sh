#!/bin/bash

# generate a new file
pushd ../datasets/tmp/ > /dev/null

./../gen2 $2 $3 $1 $4 $5  > /dev/null

TESTFILE="ks_${1}_${2}_${3}_${4}_${5}"
mv test.in ${TESTFILE}
FILE=$(realpath ${TESTFILE})

#echo $FILE

popd > /dev/null

# run the knapsack program
totime="${6}m"
timeout ${totime} ./../seq/knapsack_seq $FILE 1 1
#./../knapsack_debug $FILE 102400 256 16 256 128 2 1 0 0

