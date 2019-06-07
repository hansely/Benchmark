#include <unistd.h>
#include "Benchmark.h"

using namespace std;

int main(int argc, char **argv) {
    if (argc < 3) {
        cout << "Usage: ./benchmark [Image Folder][Iteration Count][-f16][-s]" << endl;
        return 0;
    }

    bool show = false;
    bool use_fp16 = false;

    string dir = argv[1];
    dir.append("/");
    dir+="*";
    int count = atoi(argv[2]);
    if (argc >=5) {
        use_fp16 = true;
        show = true;
    } else if (argc >= 4) {
        string option = argv[3];
        if (option == "-f16") {
            use_fp16 = true;
        } else if (option == "-s") {
            show = true;
        }
    }

    int64_t freq = clockFrequency(), t0, t1;
    Benchmark benchmark(dir, use_fp16);
    // read
    unsigned int totalByte = 0;
    float totalReadTime = 0;
    for (int i = 0; i < count; i++) {
        unsigned int byteRead = 0;
        t0 = clockCounter();
        byteRead+=benchmark.readFile(i);
        t1 = clockCounter();
        float readTime = (float)(t1-t0)*1000.0f/(float)freq;
        if (show) {
            printf("Iteration %d\n", i+1);
            printf("--------------------------------\n");
            printAverage(0, readTime, benchmark.getFileSize(), byteRead);
        }
        totalReadTime+=readTime;
        totalByte+=byteRead;
    }
    printAverage(0, totalReadTime/count, benchmark.getFileSize(), totalByte/count);

    int num_of_cores = sysconf(_SC_NPROCESSORS_ONLN);
    cout << "The total number of threads: "<< num_of_cores << endl << endl;

    // decode & convert to tensor
    while (!benchmark.isEmptyQueue()) {
            benchmark.decodeFile();
    }
    // for (int cores = 1; cores<= num_of_cores; cores*=2) {
    //      //vector<thread> dec_threads(cores);
    // //     float avgdecode = 0;
    //      while(!mFileQueue.isEmpty()) {
    //         cout << mFileQueue.getSize() << endl;
    //         tuple<char*, int> image = mFileQueue.dequeue();
    //         char * bytestream = get<0>(image);
    //         int size = get<1>(image);
    //         //benchmark.decodeFile(bytestream, size);
    //      }
    // }
    //         float decodetook = 0;
    //         float decodebyte = 0;
    //         for (int i=0; i<split.size(); i++) {
    //             int size = split[i].size() / cores;
    //             size = size < 1 ? 1 : size;
    //             vector<vector<pair<char*, int>>> temp = splitVector(split[i], size);
    //             // if the batch size is smaller than the total number of threads, use only that are needed
    //             int minSize = min ((int)temp.size(), cores);
    //             t0 = clockCounter();
    //             for (int j=0; j<minSize; j++) {
    //                 //decode
    //                 dec_threads[j] = thread(bind(&decodeFile, temp[j], use_fp16));
    //             }
    //             for (int i=0; i<minSize; i++) {
    //                 dec_threads[i].join();
    //             }
    //             t1 = clockCounter();
    //             float decodetime = (float)(t1-t0)*1000.0f/(float)freq;
    //             decodetook+=decodetime;
    //         }
    //         avgdecode+=decodetook;
    //         //printf("Decoding %d files took %.3f msec\n", (int)buffers.size(), decodetook);
    //     }
    //     avgdecode /= count;
    //     printf("Decode with %d core(s)\n", cores);
    //     printAverage(1, avgdecode, (int)buffers.size(), totalbyte);
    // }
    cout << "done" << endl;
    return 0;
}