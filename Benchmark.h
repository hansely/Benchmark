#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "FileQueue.h"
#include "Utils.h"

class Benchmark {
 public:
    Benchmark(const std::string directory, bool use_fp16, int count);

    ~Benchmark();

    int getFileSize();

    bool isEmptyQueue();

    void run();

    unsigned int getQueueSize();

    void RGB_resize(unsigned char *Rgb_in, unsigned char *Rgb_out, unsigned int swidth, unsigned int sheight, unsigned int dwidth, unsigned int dheight);

    void readFile(int iteration);

    void decodeFile();

    void convertToTensor(const cv::Mat &matOrig);

 private:
    int reverseInputChannelOrder = 0;

    float preprocessMpy[3] = {1, 1, 1};
    float preprocessAdd[3] = {0, 0, 0};

    unsigned int mTotalbyte;
    std::vector<std::string> mFilenames;
    bool mUse_fp16;
    int mCount;
    std::unique_ptr<FileQueue> mFileQueue;
};
