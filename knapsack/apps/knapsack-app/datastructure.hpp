#ifndef DATA_STRUCTURE_HPP
#define DATA_STRUCTURE_HPP

#include <bitset>

typedef unsigned short int uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

#define INIT_LIMITS INT_MAX


struct uint128 {
    int first;
    int second;
    short third;
    short fourth;

    __host__ __device__ uint128(int a = 0, int b = 0, short c = 0, short d = 0)
        : first(a), second(b), third(c), fourth(d) {
    }

    __host__ __device__ uint128& operator=(const uint128 &rhs) {
        first = rhs.first;
        second = rhs.second;
        third = rhs.third;
        fourth = rhs.fourth;
        return *this;
    }

    // rewrite this if you need more accurate comparison
    __host__ __device__ bool operator<(const uint128 &rhs) const {
        return (first < rhs.first);
    }
    __host__ __device__ bool operator<=(const uint128 &rhs) const {
        return (first <= rhs.first);
    }
    __host__ __device__ bool operator>(const uint128 &rhs) const {
        return (first > rhs.first);
    }
    __host__ __device__ bool operator>=(const uint128 &rhs) const {
        return (first >= rhs.first);
    }
    __host__ __device__ bool operator==(const uint128 &rhs) const {
        return (first == rhs.first);
    }
    __host__ __device__ bool operator!=(const uint128 &rhs) const {
        return (first != rhs.first || second != rhs.second ||
                third != rhs.third)|| fourth != rhs.fourth;
    }


};

inline std::ostream& operator << (std::ostream& o, const uint128& a)
{
    //o << "Benefit: " << -a.first << " Weight: " << a.second << " Index: " << a.third << " Sequence: " << std::bitset<64>(a.fourth);
    o << "Benefit: " << -a.first << " Weight: " << a.second << " Index: " << (int)a.third;
    return o;
}

inline __host__ __device__ void bin(uint64 n)
{
    if (n > 1)
    bin(n>>1);

    printf("%d", n & 1);
}
#endif
