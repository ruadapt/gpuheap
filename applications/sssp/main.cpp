#include <cstring>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "parse_graph.hpp"

int main( int argc, char** argv )
{
    if (argc != 2) {
        std::cout << "./" << argv[0] << " [graph_name]\n";
    }
    std::ifstream inputFile;
    inputFile.open(argv[1]);

    std::cout << "Collecting the input graph ...\n";

    // Read graph using parse_graph
    bool nonDirectedGraph = true;
    long long arbparam = 0;
    std::vector<initial_vertex> parsedGraph( 0 );
    uint nEdges = parse_graph::parse(
            inputFile, // Input file.
            parsedGraph, // The parsed graph.
            arbparam, // Start vertex (? please double check)
            nonDirectedGraph ); // nonDirected/Directed Graph.
    std::cout << "Input graph collected with " << parsedGraph.size() << " vertices and " << nEdges << " edges.\n";


    //Print out Graph
    //for (int i = 0; i < parsedGraph.size(); ++i) {
        //std::cout << parsedGraph[i].vertexValue.distance << " ";
        //for (int j = 0; j < parsedGraph[i].nbrs.size(); ++j)
            //std::cout << parsedGraph[i].nbrs[j].srcIndex << " ";
        //std::cout << std::endl;
    //}

    // Transfer the original data structure to edge list based structure
    int edgeNum, vertexNum;
    vertexNum = parsedGraph.size();
    edgeNum = nEdges;
    std::vector<edgeList> E;
    int *A = (int *) std::calloc(vertexNum, sizeof(int));
    int totalW = 0;
    for (int i = 0; i < vertexNum; ++i) {
        for (int j = 0; j < parsedGraph[i].nbrs.size(); ++j) {
            edgeList newEdge;
            newEdge.destIdx = i;
            newEdge.sourceIdx = parsedGraph[i].nbrs[j].srcIndex;
            newEdge.weight = parsedGraph[i].nbrs[j].edgeValue.weight;
            totalW += newEdge.weight;
            E.push_back(newEdge);
        }
    }
    std::cout << "Input graph collected with " << vertexNum << " vertices and " << edgeNum << " edges.\n";

    inputFile.close();

    // An alternative way to read the graph.
    inputFile.open(argv[1]);

    char garbage[256];
    while (inputFile.peek() == '#') {
        inputFile.getline(garbage, 256);
    }

    int edgeNum1, vertexNum1;
    edgeNum1 = 0;
    vertexNum1 = 0;
    std::vector<edgeList> E1;
    while (!inputFile.eof()) {
        unsigned int source, dest, weight;
        inputFile >> source >> dest;
        if (inputFile.peek() == ' ') {
            weight = 1;
        }
        else {
            inputFile >> weight;
        }
        // edge(src, dest)
        edgeList newEdge;
        newEdge.sourceIdx = source;
        newEdge.destIdx = dest;
        newEdge.weight = weight;
        E1.push_back(newEdge);
        edgeNum1++;
        if (nonDirectedGraph) {
            // edge(dest, src)
            edgeList newEdge1;
            newEdge1.sourceIdx = dest;
            newEdge1.destIdx = source;
            newEdge1.weight = weight;
            E1.push_back(newEdge1);
            edgeNum1++;
        }
        if (source + 1 > vertexNum1) vertexNum1 = source + 1;
        if (dest + 1 > vertexNum1) vertexNum1 = dest + 1;
        //printf("%u %u %u\n", source, dest, weight);
    }
    std::cout << "Input graph collected with " << vertexNum1 << " vertices and " << edgeNum1 << " edges.\n";

    inputFile.close();

    return 0;
}
