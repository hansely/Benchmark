#include "FileQueue.h"
#include <iostream> 

using namespace std;

FileQueue::FileQueue() {}

FileQueue::~FileQueue() {}

void FileQueue::enqueue(const tuple<char*, int> &image) {
    unique_lock<mutex> lock(mMutex);
    mQueue.push(move(image));
    lock.unlock();
    mSignal.notify_one();
}

tuple<char*, int> FileQueue::dequeue() {
    unique_lock<mutex> lock(mMutex);
    tuple<char*, int> image = move(mQueue.front());
    mQueue.pop();
    return image;
}

bool FileQueue::isEmpty() {
    unique_lock<mutex> lock(mMutex);
    return mQueue.empty();
}

unsigned int FileQueue::getSize() {
    unique_lock<mutex> lock(mMutex);
    return mQueue.size();
}
