#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <glob.h>
#include <iostream>
#include "FileQueue.h"

inline int64_t clockCounter() {
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

inline int64_t clockFrequency() {
    return std::chrono::high_resolution_clock::period::den / std::chrono::high_resolution_clock::period::num;
}

std::vector<std::string> glob(std::string& pat);

std::vector<std::vector<std::tuple<char*, int>>> splitQueue(FileQueue &imageQueue, int cores);

void printResult(float time, int bufferSize, unsigned int totalByte);