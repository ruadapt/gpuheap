all: knapsack_seq knapsack_seq_test
knapsack_seq: knapsack_seq.cpp
	g++ -std=c++11 -O3 knapsack_seq.cpp -o $@

knapsack_seq_test: knapsack_seq_test.cpp
	g++ -std=c++11 -O3 knapsack_seq_test.cpp -o $@


clean:
	rm -rf knapsack_seq knapsack_seq_test
