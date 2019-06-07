#pragma once

#include "FileQueue.h"
#include "Utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class Benchmark {
 public:
    Benchmark(std::string directory, bool use_fp16);

    ~Benchmark();

    int getFileSize();

    bool isEmptyQueue();

    unsigned int getQueueSize();

    void RGB_resize(unsigned char *Rgb_in, unsigned char *Rgb_out, unsigned int swidth, unsigned int sheight, unsigned int dwidth, unsigned int dheight);

    unsigned int readFile(int i);

    void decodeFile();

    void convertToTensor(const cv::Mat &matOrig);

 private:
    int reverseInputChannelOrder = 0;

    float preprocessMpy[3] = {1, 1, 1};
    float preprocessAdd[3] = {0, 0, 0};

    unsigned int mTotalbyte;
    std::vector<std::string> mFilenames;
    bool mUse_fp16;
    std::unique_ptr<FileQueue> mFileQueue;

};
