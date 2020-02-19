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

void printResult(int iteration, float time, int imageCount, unsigned int totalByte) {
    unsigned int FPS = (unsigned int)(1000/time*(float)imageCount);
    float MBS = totalByte/time/1000;
    printf("Iteration %d | %u FPS | %.2f MB/s\n" , iteration, FPS, MBS);
    printf("-------------------------------------\n");
}