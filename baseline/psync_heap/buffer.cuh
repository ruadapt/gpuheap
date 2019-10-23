#ifndef BUFFER_CUH
#define BUFFER_CUH

#include "datastructure.h"

struct TB {
    /* Number of available entries */
    uint tableSize;
:xa

    /* Index of where next update should starts at */
    uint startIdx;
    uint startOffset;

    /* Number of entries that next update should works on */
    uint validEntryNum;

    /* 
        The index where last available entries locates
        This index will be increased when we update/insert
        new entries into the table buffer
    */
    uint endIdx;
        
    /* Current batch */
    uint *node;
    /* Target batch node */
    int *target;

    int *bufferKeys;
    int *bufferVals;

    TB (int _tableSize, int _batchSize, int _type)
    {
        initial(_tableSize, _batchSize, _type);
    }

    bool isEmpty() {
        return (startIdx == endIdx);
    }

    void printTB() {
        int *h_node = new int[tableSize];
        int *h_target = new int[tableSize];
        int *h_bufferKeys = new int[tableSize * batchSize];
        int *h_bufferVals = new int[tableSize * batchSize];
        
        cudaMemcpy(h_node, node, sizeof(int) * tableSize, cudaMemcpyDeviceToHost);
        if (type == 0) {
            cudaMemcpy(h_target, target, sizeof(int) * tableSize, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_bufferKeys, bufferKeys, sizeof(int) * tableSize * batchSize, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_bufferVals, bufferVals, sizeof(int) * tableSize * batchSize, cudaMemcpyDeviceToHost);

            cout << "insert table buffer:\n";
            for (int i = startIdx; i != endIdx; ++i) {
                cout << "entry " << i << " node: " << h_node[i] << " target: " << h_target[i] << endl;
                for (int j = 0; j < batchSize; ++j) {
                    cout << h_bufferKeys[i * batchSize + j] << " " << h_bufferVals[i * batchSize + j];
                    cout << " | ";
                }
                cout << endl;
            }
        }
        else {
            cout << "delete table buffer:\n";
            for (int i = startIdx; i != endIdx; ++i) {
                cout << "entry " << i << " node: " << h_node[i] << endl;
            }
        }
    }


    // type = 0: insert table buffer
    // type = 1: delete table buffer
    void initial(int _tableSize, int _batchSize, int _type)
    {
        tableSize = _tableSize;
        batchSize = _batchSize;
        type = _type;
        startIdx = 0;
        startOffset = 0;
        validEntryNum = 0;
        endIdx = 0;
        cudaMalloc((void **)&node, sizeof(int) * tableSize);
        /* 
            if type = 1 this is a delete table buffer
                It only need to store the node it locates at, in the paper, the author
                also uses buffer to store largest batch at each updates which seems to
                be redundant, since you still need to move everything into shared memory
                when do merging or sorting
            if type = 0 this is an insert table buffer
                For
        */
        if (type == 0) {
            cudaMalloc((void **)&target, sizeof(int) * tableSize);
            cudaMalloc((void **)&bufferKeys, sizeof(int) * tableSize * batchSize);
            cudaMalloc((void **)&bufferVals, sizeof(int) * tableSize * batchSize);
            // target = (int *)malloc(sizeof(int) * tableSize);
            // bufferKeys = (int *)malloc(sizeof(int) * tableSize * batchSize);
            // bufferVals = (int *)malloc(sizeof(int) * tableSize * batchSize);
        }
    }

};

#endif
