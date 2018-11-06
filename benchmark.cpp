#include <iostream>
#include <glob.h>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <functional>
#include <thread>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

inline int64_t clockCounter()
{
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

inline int64_t clockFrequency()
{
    return std::chrono::high_resolution_clock::period::den / std::chrono::high_resolution_clock::period::num;
}

std::vector<string> glob(const string& pat){
    glob_t glob_result;
    glob(pat.c_str(),GLOB_TILDE,NULL,&glob_result);
    vector<string> ret;
    int idx;
    string str;
    size_t i=0;
    
    while (i< glob_result.gl_pathc) {
        str = string(glob_result.gl_pathv[i]);
        ret.push_back(str);
        i++;
    }
    globfree(&glob_result);
    return ret;
}

void decodeFile(long filelen, char* &buffer) {
    Mat img = imdecode(Mat(1, filelen, CV_8UC1, buffer), CV_LOAD_IMAGE_UNCHANGED);
}

void benchmarkFile(vector<string> &filenames, int count, bool show) {
    
    char *buffer;
    long filelen;
    float avgtime=0;
    long totalbyte=0;
    int64_t freq = clockFrequency(), t0, t1, d0, d1;
    FILE *file;
    int totalsize = filenames.size();

    for (int i=0; i<count; i++) {
        long byteread = 0;
        float timetook = 0;
        float decodetook = 0;

        for (int j=0; j<totalsize; j++) {
            file = fopen(filenames[j].c_str(), "r");
            if (file==NULL) {
                cout << "can't open" << filenames[j] << endl;
                exit(-1);
            }
            fseek(file, 0, SEEK_END);
            filelen = ftell(file);
            rewind(file);
            buffer = (char *)malloc((filelen+1)*sizeof(char));

            //read
            t0 = clockCounter();
            byteread+=fread(buffer, 1, filelen, file);
            t1 = clockCounter();
            float readtime = (float)(t1-t0)*1000.0f/(float)freq;
            timetook+=readtime;

            //decode
            d0 = clockCounter();
            decodeFile(filelen, buffer);
            d1 = clockCounter();
            float decodetime= (float)(d1-d0)*1000.0f/(float)freq;
            decodetook+=decodetime;

            free(buffer);
            fclose(file);
        }

        if (show) {
            printf("Iteration %d\n", i+1);
            // printf("Bytes read: %ld bytes\n", byteread);
            // printf("--------------------------------\n");
            // printf("Read\n");
            // printf("--------------------------------\n");
            // printf("Reading %d images took %.3f msec\n", totalsize, timetook);
            // printf("Images per second: %ld images\n", (long)(1000/timetook*(float)totalsize));
            // printf("Bytes per second: %ld bytes\n\n", (long)(1000*byteread/timetook));
            // printf("MB per second: %.2ld bytes\n\n", (long)(1000*byteread/timetook)/1000000);

            printf("Decode\n");
            printf("--------------------------------\n");
            printf("Decoding %d images took %.3f msec\n", totalsize, decodetook);
            printf("Images per second: %ld images\n", (long)(1000/decodetook*(float)totalsize));
            printf("Bytes per second: %ld bytes\n\n", (long)(1000*byteread/decodetook));
            printf("MB per second: %.2ld bytes\n\n", (long)(1000*byteread/decodetook)/1000000);
        }
        avgtime+=decodetook;
        totalbyte+=byteread;
    }
    totalbyte/=count;

    printf("\nAverage speed per iteration: %.3f msec\n", avgtime/count);
    printf("Average bytes decoded per iteration: %ld bytes\n", totalbyte);
    printf("Images per second: %ld images\n", (long)(1000/avgtime*count*(float)totalsize));
    printf("Bytes per second: %ld bytes\n\n", (long)(1000*totalbyte/avgtime*count));
    printf("MB per second: %.2ld bytes\n\n", (long)(1000*totalbyte/avgtime*count)/1000000);
}

template <typename T>
vector<vector<T>> splitVector (vector<T> &vec, int cores){
    vector<vector<T>> split(cores);
    size_t size = vec.size() / cores;
    auto start = vec.begin();
    for (int i=0; i<cores; i++) {
        split[i].assign(start, start+size);
        cout << *start << endl;
        start+=size;
    }
    return split;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        cout << "Usage: ./benchmark [Image Folder] [Iteration Count] [Core Count] [-s]" << endl;
        exit(-1);
    }

    vector<string> filenames;
    bool show = false;
    string dir = argv[1];
    dir.append("/");
    string pat = dir+"*";
    
    int count = atoi(argv[2]);
    int cores = atoi(argv[3]);
    if (argc >=5) {
        string option = argv[4];
        if (option == "-s") {
            show = true;
        }
    }
    filenames = glob(pat);
    vector<vector<string>> split = splitVector(filenames, cores);

    vector<thread> dec_threads(cores);
    for (unsigned int i=0; i<cores; i++) {
        dec_threads[i] = std::thread(bind(&benchmarkFile, split[i], count, show));
    }

    for (int i=0; i<cores; i++) {
        dec_threads[i].join();
    }
    //benchmarkFile(filenames, count, show);
    return 0;
}