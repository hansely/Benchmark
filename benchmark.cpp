#include <iostream>
#include <glob.h>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <functional>
#include <thread>
#include <climits>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

#define BUFFER_SIZE INT_MAX

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

void decodeFile(const vector<pair<char*, int>> &buffer) {
    for (int i=0; i<buffer.size(); i++) {
        Mat img = imdecode(Mat(1, buffer[i].second, CV_8UC1, buffer[i].first), CV_LOAD_IMAGE_UNCHANGED);
    }
}

vector<pair<char*, int>> readFile(vector<string> &filenames, int count, bool show) {
    vector<pair<char*, int>> buffers;
    char *buffer;
    long filelen;
    float avgtime=0;;
    long totalbyte=0;
    int64_t freq = clockFrequency(), t0, t1;
    FILE *file;
    int totalsize = filenames.size();
    for (int i=0; i<count; i++) {
        long byteread = 0;
        float timetook = 0;
        for (int j=0; j<totalsize; j++) {
            //read
            t0 = clockCounter();
            file = fopen(filenames[j].c_str(), "r");
            if (file==NULL) {
                cout << "can't open" << filenames[j] << endl;
                exit(-1);
            }
            fseek(file, 0, SEEK_END);
            filelen = ftell(file);
            rewind(file);
            buffer =  buffer = (char *)malloc(filelen);
            byteread+=fread(buffer, 1, filelen, file);
            t1 = clockCounter();

            float readtime = (float)(t1-t0)*1000.0f/(float)freq;
            timetook+=readtime;
            fclose(file);
            if (i==0) {
                buffers.push_back(make_pair(buffer, filelen));
            }

        }
        free(buffer);

        if (show) {
            printf("Iteration %d\n", i+1);
            printf("Bytes read: %ld bytes\n", byteread);
            printf("--------------------------------\n");
            printf("Read\n");
            printf("--------------------------------\n");
            printf("Reading %d images took %.3f msec\n", totalsize, timetook);
            printf("Images per second: %ld images\n", (long)(1000/timetook*(float)totalsize));
            printf("Bytes per second: %ld bytes\n", (long)(1000*byteread/timetook));
            printf("MB per second: %.2ld bytes\n\n", (long)(1000*byteread/timetook)/1000000);
        }
        avgtime+=timetook;
        totalbyte+=byteread;
    }
    totalbyte/=count;
    avgtime/=count;
    printf("Read\n");
    printf("--------------------------------\n");
    printf("\nAverage speed per iteration: %.3f msec\n", avgtime);
    printf("Average bytes decoded per iteration: %ld bytes\n", totalbyte);
    printf("Images per second: %ld images\n", (long)(1000/avgtime*(float)totalsize));
    printf("Bytes per second: %ld bytes\n\n", (long)(1000*totalbyte/avgtime));
    printf("MB per second: %.2f bytes\n\n", (float)(1000*totalbyte/avgtime)/1000000);

    return buffers;
}

template <typename T>
vector<vector<T>> splitVector (vector<T> &vec, int size){
    int vecSize = (vec.size() % size > 0) ?  vec.size()/size+1 : vec.size()/size;
    vector<vector<T>> split(vecSize);
    auto start = vec.begin();

    for (int i=0; i<vecSize; i++) {
        if (i == vecSize - 1) {
            split[i].assign(start, vec.end());
        }
        else {
            split[i].assign(start, start+size);
        }
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

    //read
    vector<pair<char*, int>> buffers = readFile(filenames, count, show);

    long totalbyte = 0;

    for (int i=0; i< buffers.size(); i++) {
        totalbyte+=buffers[i].second;
    }
    vector<vector<pair<char*, int>>> split = splitVector(buffers, 64);
    vector<thread> dec_threads(cores);
    int64_t freq = clockFrequency(), t0, t1;

    float avgdecode = 0;

    for (int c = 0; c<count; c++) {
        float decodetook = 0;
        float decodebyte = 0;
        for (int i=0; i<split.size(); i++) {
            vector<vector<pair<char*, int>>> temp = splitVector(split[i], split[i].size()/cores);
            t0 = clockCounter();
            for (int j=0; j<cores; j++) {
                //decode
                dec_threads[j] = thread(bind(&decodeFile, temp[j]));
            }
            for (int i=0; i<cores; i++) {
                dec_threads[i].join();
            }
            t1 = clockCounter();
            float decodetime = (float)(t1-t0)*1000.0f/(float)freq;
            decodetook+=decodetime;
        }
        avgdecode+=decodetook;
        //printf("Decoding %d files took %.3f msec\n", (int)buffers.size(), decodetook);
    }
    avgdecode /= count;

    printf("Decode\n");
    printf("--------------------------------\n");
    printf("\nAverage speed per iteration: %.3f msec\n", avgdecode);
    printf("Images per second: %ld images\n", (long)(1000/avgdecode*(float)buffers.size()));
    printf("Bytes per second: %ld bytes\n\n", (long)(1000*totalbyte/avgdecode));
    printf("MB per second: %.2f bytes\n\n", (float)(1000*totalbyte/avgdecode)/1000000);
    return 0;
}