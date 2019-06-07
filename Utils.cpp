#include "Utils.h"

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

void printAverage(int mode, float avgSpeed, int bufferSize, unsigned int totalByte){
    if (mode == 0) {
            printf("Read\n");
        } else if (mode == 1) {
            printf("Decode\n");
        }

        printf("--------------------------------\n");
        printf("\nAverage speed per iteration(%d files): %.3f msec\n", bufferSize, avgSpeed);
        printf("Images per second: %u images\n", (unsigned int)(1000/avgSpeed*(float)bufferSize));
        printf("Bytes per second: %u bytes\n", (unsigned int)(1000*totalByte/avgSpeed));
        printf("MB per second: %.2f bytes\n\n", (float)(1000*totalByte/avgSpeed)/1000000);
}