#pragma once

#include <vector>
#include <memory>
#include <tuple>
#include <queue>
#include <mutex>
#include <condition_variable>

class FileQueue {
 public:
    FileQueue();
    ~FileQueue();
    void enqueue(std::tuple<char*, int> &image);
    std::tuple<char*, int> dequeue();
    bool isEmpty();
    unsigned int getSize();
 private:
    std::queue<std::tuple<char*, int>> mQueue;
    std::mutex mMutex;
    std::condition_variable mSignal;
};

