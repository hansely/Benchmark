#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <glob.h>
#include <iostream>

inline int64_t clockCounter() {
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

inline int64_t clockFrequency() {
    return std::chrono::high_resolution_clock::period::den / std::chrono::high_resolution_clock::period::num;
}

std::vector<std::string> glob(std::string& pat);

template <typename T>
std::vector<std::vector<T>> splitVector(const std::vector<T> &vec, int size) {
    int vecSize = (vec.size() % size > 0) ?  vec.size()/size+1 : vec.size()/size;
    std::vector<std::vector<T>> split(vecSize);
    auto start = vec.begin();
    for (int i=0; i < vecSize; i++) {
        if (i == vecSize - 1) {
            split[i].assign(start, vec.end());
        } else {
            split[i].assign(start, start+size);
        }
        start+=size;
    }
    return split;
}

void printResult(int iteration, float time, int bufferSize, unsigned int totalByte);