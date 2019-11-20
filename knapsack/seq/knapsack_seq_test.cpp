// C++ program to solve knapsack problem using
// branch and bound
#include <bits/stdc++.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include "util.hpp"


using namespace std;

// nvcc -g -G -arch=sm_61 -O0 -std=c++11 knapsack_seq.cu -o knapsack_seq

// Stucture for Item which store weight and corresponding
// value of Item
struct Item
{
    double weight;
    int value;
};

int maxPQSize = 0;

// Node structure to store information of decision
// tree
struct Node
{
    // level  --> Level of node in decision tree (or index
    //             in arr[]
    // profit --> Profit of nodes on path from root to this
    //            node (including this node)
    // bound ---> Upper bound of maximum profit in subtree
    //            of this node/
    int level, profit, bound;
    double weight;
	
	bool operator<(const Node &rhs) const {
        return profit < rhs.profit;
//        return (profit + bound < rhs.profit + rhs.bound);
    }
	
};

inline std::ostream& operator << (std::ostream& o, const Node& a)
{
    o << "Profit: " << a.profit;
    return o;
}

// Comparison function to sort Item according to
// val/weight ratio
bool cmp(Item a, Item b)
{
    double r1 = (double)a.value / (double)a.weight;
    double r2 = (double)b.value / (double)b.weight;
    return r1 > r2;
}

// Returns bound of profit in subtree rooted with u.
// This function mainly uses Greedy solution to find
// an upper bound on maximum profit.
int bound(Node u, int n, int W, Item arr[])
{
    // if weight overcomes the knapsack capacity, return
    // 0 as expected bound
    if (u.weight >= W)
        return 0;

    // initialize bound on profit by current profit
    int profit_bound = u.profit;

    // start including items from index 1 more to current
    // item index
    int j = u.level + 1;
    double totweight = u.weight;

    // checking index condition and knapsack capacity
    // condition
    while ((j < n) && (totweight + arr[j].weight <= W))
    {
        totweight    += arr[j].weight;
        profit_bound += arr[j].value;
        j++;
    }

    // If k is not n, include last item partially for
    // upper bound on profit
    if (j < n)
        profit_bound += (W - totweight) * arr[j].value /
                                         arr[j].weight;

    return profit_bound;
}


template<typename A> void print_queue(A& pq)
{
	while (!pq.empty())
	{
		cout << pq.top() << endl;
		pq.pop();
	}
	cout << endl;
}

unsigned long long int explored = 0;
unsigned long long int processed = 0;
unsigned long long int pruned = 0;
// Returns maximum profit we can get with capacity W
int knapsack_Q(int W, Item arr[], int n)
{
    // sorting Item on basis of value per unit
    // weight.
    sort(arr, arr + n, cmp);
	
	queue<Node> Q;
    Node u, v;

    // dummy node at starting
    u.level = -1;
    u.profit = u.weight = 0;
    Q.push(u);

    // One by one extract an item from decision tree
    // compute profit of all children of extracted item
    // and keep saving maxProfit
    int maxProfit = 0;
    while (!Q.empty())
    {	
        // Dequeue a node
        u = Q.front();
        Q.pop();

        // If it is starting node, assign level 0
        if (u.level == -1)
            v.level = 0;

        // If there is nothing on next level
        if (u.level == n-1)
            continue;

        // Else if not last node, then increment level,
        // and compute profit of children nodes.
        v.level = u.level + 1;

        // Taking current level's item add current
        // level's weight and value to node u's
        // weight and value
        v.weight = u.weight + arr[v.level].weight;
        v.profit = u.profit + arr[v.level].value;

        // If cumulated weight is less than W and
        // profit is greater than previous profit,
        // update maxprofit
        if (v.weight <= W && v.profit > maxProfit)
            maxProfit = v.profit;

        // Get the upper bound on profit to decide
        // whether to add v to Q or not.
        v.bound = bound(v, n, W, arr);

        // If bound value is greater than profit,
        // then only push into queue for further
        // consideration
        if (v.bound > maxProfit)
            Q.push(v);

        // Do the same thing,  but Without taking
        // the item in knapsack
        v.weight = u.weight;
        v.profit = u.profit;
        v.bound = bound(v, n, W, arr);
        if (v.bound > maxProfit)
            Q.push(v);
		explored++;
    }
	// cout << "explored: " << explored << endl;

    return maxProfit;
}

int knapsack_PQ(int W, Item arr[], int n)
{
    // sorting Item on basis of value per unit
    // weight.
    sort(arr, arr + n, cmp);
	
	priority_queue<Node> Q;
    Node u, v;
    Node best;
/*  
	for(int i = 0; i < n; i++){
		cout << arr[i].weight << " " << arr[i].value << endl;
	}
*/
    // dummy node at starting
    u.level = -1;
    u.profit = u.weight = 0;
    Q.push(u);

    // One by one extract an item from decision tree
    // compute profit of all children of extracted item
    // and keep saving maxProfit
    int maxProfit = 0;
	vector<int> cexplored;
    vector<int> cbenefit;
    int counter = 0;

    struct timeval startTime;
    struct timeval endTime;

    setTime(&startTime);

    while (!Q.empty())
    {	
        // Dequeue a node
        u = Q.top();
        Q.pop();

        // If it is starting node, assign level 0
        if (u.level == -1)
            v.level = 0;

        // If there is nothing on next level
        if (u.level == n-1)
            continue;

        // Else if not last node, then increment level,
        // and compute profit of children nodes.
        v.level = u.level + 1;

        // Taking current level's item add current
        // level's weight and value to node u's
        // weight and value
        v.weight = u.weight + arr[v.level].weight;
        v.profit = u.profit + arr[v.level].value;

        // If cumulated weight is less than W and
        // profit is greater than previous profit,
        // update maxprofit
        if (v.weight <= W && v.profit > maxProfit) {
            maxProfit = v.profit;
            best = v;
        }

        // Get the upper bound on profit to decide
        // whether to add v to Q or not.
        v.bound = bound(v, n, W, arr);

        // If bound value is greater than profit,
        // then only push into queue for further
        // consideration
        if (v.bound > maxProfit) {
            Q.push(v);
#ifdef XXX
            processed++;
#endif
        } else {
#ifdef XXX
            pruned++;
#endif
        }

        // Do the same thing,  but Without taking
        // the item in knapsack
        v.weight = u.weight;
        v.profit = u.profit;
        v.bound = bound(v, n, W, arr);
        if (v.bound > maxProfit) {
            Q.push(v);
#ifdef XXX
            processed++;
#endif
        } else {
#ifdef XXX
            pruned++;
#endif
        }

//	 printf("%d\n", Q.size());
    maxPQSize = maxPQSize > Q.size() ? maxPQSize : Q.size();
	
	explored++;
    counter++;
    if (counter == n * 10) {
        counter = 0;
        cexplored.push_back(explored);
        cbenefit.push_back(maxProfit);
#ifdef XXX
    cout << maxProfit << " " << processed << " " << pruned << " " << explored << endl;
#endif
    }
    }
    setTime(&endTime);

    double knapsackTime = getTime(&startTime, &endTime);
//    cout << getTime(&startTime, &endTime) << endl;
/*    for (int i = 0; i < cexplored.size(); i++) {*/
        /*cout << (double)i * (knapsackTime / (double)cexplored.size())*/
            /*<< " " << cexplored[i] << " " << cbenefit[i] << endl;*/
    /*}*/
//	 cout << "explored: " << explored << endl;
//    cout << best.level << endl;

    return maxProfit;
}

void knapsack_PQ_1(int W, Item arr[], int n, int res)
{
    // sorting Item on basis of value per unit
    // weight.
    sort(arr, arr + n, cmp);
	
	priority_queue<Node> Q;
    Node u, v;
    // dummy node at starting
    u.level = -1;
    u.profit = u.weight = 0;
    u.bound = 1;
    Q.push(u);

    // One by one extract an item from decision tree
    // compute profit of all children of extracted item
    // and keep saving maxProfit
    int maxProfit = 0;
	vector<int> cexplored;
    vector<int> cbenefit;
    int counter = 0;

    struct timeval startTime;
    struct timeval endTime;
    bool flag = true;

    setTime(&startTime);

    while (!Q.empty())
    {	
        // Dequeue a node
        u = Q.top();
        Q.pop();

        // If it is starting node, assign level 0
        if (u.level == -1)
            v.level = 0;
        // If there is nothing on next level
        if (u.level == n-1)
            continue;

        if (u.bound <= maxProfit) continue;

        // Else if not last node, then increment level,
        // and compute profit of children nodes.
        v.level = u.level + 1;

        // Taking current level's item add current
        // level's weight and value to node u's
        // weight and value
        v.weight = u.weight + arr[v.level].weight;
        v.profit = u.profit + arr[v.level].value;

        // If cumulated weight is less than W and
        // profit is greater than previous profit,
        // update maxprofit
        if (v.weight <= W && v.profit > maxProfit) {
            maxProfit = v.profit;
            if (flag && maxProfit == res) {
                flag = false;
                setTime(&endTime);
                cout << getTime(&startTime, &endTime) << " " << counter << " ";
                setTime(&startTime);
            }
        }

        // Get the upper bound on profit to decide
        // whether to add v to Q or not.
        v.bound = bound(v, n, W, arr);

        // If bound value is greater than profit,
        // then only push into queue for further
        // consideration
        if (v.bound > maxProfit)
            Q.push(v);

        // Do the same thing,  but Without taking
        // the item in knapsack
        v.weight = u.weight;
        v.profit = u.profit;
        v.bound = bound(v, n, W, arr);
        if (v.bound > maxProfit)
            Q.push(v);
//	 printf("%d\n", Q.size());
/*        maxPQSize = maxPQSize > Q.size() ? maxPQSize : Q.size();*/
        
        /*explored++;*/
        counter++;
        /*if (counter % 100 == 1) {*/
            /*if (flag) cout << "1 ";*/
            /*else cout << "0 ";*/
            /*cout << Q.size() << " " << maxProfit << " " << u.level << endl;*/
        /*}*/
        /*if (counter == n) {*/
            /*counter = 0;*/
            /*cexplored.push_back(explored);*/
            /*cbenefit.push_back(maxProfit);*/
        /*}*/
    }
    setTime(&endTime);
    cout << getTime(&startTime, &endTime) << " " << counter << " ";

//    double knapsackTime = getTime(&startTime, &endTime);
//    cout << getTime(&startTime, &endTime) << endl;
/*    for (int i = 0; i < cexplored.size(); i++) {*/
        /*cout << (double)i * (knapsackTime / (double)cexplored.size())*/
            /*<< " " << cexplored[i] << " " << cbenefit[i] << endl;*/
    /*}*/
//	 cout << "explored: " << explored << endl;

//    return maxProfit;
}

int main(int argc, char *argv[])
{

    struct timeval startTime;
    struct timeval endTime;

    if (argc != 4) {
        cout << "usage: ./knapsack_seq [file name] [0 for Q, 1 for PQ] [0 for default, 1 for large set]\n";
        return -1;
    }

    ifstream fin;
    fin.open(argv[1]);
	
	int queueType = atoi(argv[2]);
	int setType = atoi(argv[3]);
	
    int capacity, number;
	if(!setType){
		fin >> capacity >> number;
	}else{
		fin >> number >> capacity;
	}

	Item *w_p = new Item[number];
    for (int i = 0; i < number; ++i) {
		if(!setType){
			fin >> w_p[i].weight >> w_p[i].value;
		}else{
			fin >> w_p[i].value >> w_p[i].weight;
		}
    }

    fin.close();

    int res = 0;

    setTime(&startTime);
	if(!queueType){
		res = knapsack_Q(capacity, w_p, number);
	}else{
		res = knapsack_PQ(capacity, w_p, number);
	}
    setTime(&endTime);

//	knapsack_PQ_1(capacity, w_p, number, res);

    /*cout << res << " ";*/

//    cout << getTime(&startTime, &endTime) << " " << explored << " ";

    setTime(&startTime);
	if(!queueType){
		res = knapsack_Q(capacity, w_p, number);
	}else{
		knapsack_PQ_1(capacity, w_p, number, res);
	}
    setTime(&endTime);

    ofstream outputFile;

    outputFile.open(strcat(argv[1], ".res"));
    outputFile << res << endl;
    outputFile.close();

     /*cout << "sequential knapsack time: " << getTime(&startTime, &endTime) << " ms" << endl;*/
     /*cout << "maximum weight: " << res << endl;*/
    
    /*cout << argv[1] << "," */
         /*<< number << ","*/
         /*<< res << ","*/
         /*<< (int)getTime(&startTime, &endTime) << ","*/
         /*<< getTime(&startTime, &endTime) << ","*/
         /*<< explored << ","*/
         /*<< "SEQPQ"*/
         /*<< endl;*/

    /*cout << maxPQSize << endl;*/
    return 0;
}
