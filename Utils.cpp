#include "Utils.h"

using namespace std;

std::vector<std::string> glob(std::string& pat) {
    glob_t glob_result;
    glob(pat.c_str(), GLOB_TILDE, NULL, &glob_result);
    std::vector<std::string> ret;
    int idx;
    std::string str;
    size_t i = 0;

    while (i < glob_result.gl_pathc) {
        str = std::string(glob_result.gl_pathv[i]);
        ret.push_back(str);
        i++;
    }
    globfree(&glob_result);
    return ret;
}

std::vector<std::vector<std::tuple<char*, int>>> splitQueue(FileQueue &imageQueue, int cores) {
    int queueSize = imageQueue.getSize() / cores;
    std::vector<std::vector<std::tuple<char*, int>>> split(cores);
    std::tuple<char*, int> image;
    for (int i = 0; i < cores; i++) {
        if (i == cores - 1) {
            while (!imageQueue.isEmpty()) {
                image = imageQueue.dequeue();
                split[i].push_back(image);
            }
        } else {
            for (int j=0; j<queueSize; j++) {
                image = imageQueue.dequeue();
                split[i].push_back(image);
            }
        }
    }
    return split;
}

void printResult(float time, int imageCount, unsigned int totalByte) {
    unsigned int FPS = (unsigned int)(1000/time*(float)imageCount);
    float MBS = totalByte/time/1000;
    printf(" | %u FPS | %.2f MB/s\n" , FPS, MBS);
}