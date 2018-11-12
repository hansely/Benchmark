#include <iostream>
#include <glob.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <dlfcn.h>
#include <fstream>
#include <chrono>
#include <thread>
#include <climits>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <xmmintrin.h>
#include <nmmintrin.h>

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

void decodeFile(const vector<pair<char*, int>> &buffer, bool use_fp16) {
    int reverseInputChannelOrder = 0;

    float preprocessMpy[3] = {1,1,1};
    float preprocessAdd[3] = {0,0,0};
    for (int k=0; k<buffer.size(); k++) {
        Mat matOrig = imdecode(Mat(1, buffer[k].second, CV_8UC1, buffer[k].first), CV_LOAD_IMAGE_UNCHANGED);
        int length = matOrig.cols * matOrig.rows;
        unsigned char * img;
        img = matOrig.data;
        __m128i mask_B, mask_G, mask_R;
        if (reverseInputChannelOrder)
        {
            mask_B = _mm_setr_epi8((char)0x0, (char)0x80, (char)0x80, (char)0x80, (char)0x3, (char)0x80, (char)0x80, (char)0x80, (char)0x6, (char)0x80, (char)0x80, (char)0x80, (char)0x9, (char)0x80, (char)0x80, (char)0x80);
            mask_G = _mm_setr_epi8((char)0x1, (char)0x80, (char)0x80, (char)0x80, (char)0x4, (char)0x80, (char)0x80, (char)0x80, (char)0x7, (char)0x80, (char)0x80, (char)0x80, (char)0xA, (char)0x80, (char)0x80, (char)0x80);
            mask_R = _mm_setr_epi8((char)0x2, (char)0x80, (char)0x80, (char)0x80, (char)0x5, (char)0x80, (char)0x80, (char)0x80, (char)0x8, (char)0x80, (char)0x80, (char)0x80, (char)0xB, (char)0x80, (char)0x80, (char)0x80);
        }
        else
        {
            mask_R = _mm_setr_epi8((char)0x0, (char)0x80, (char)0x80, (char)0x80, (char)0x3, (char)0x80, (char)0x80, (char)0x80, (char)0x6, (char)0x80, (char)0x80, (char)0x80, (char)0x9, (char)0x80, (char)0x80, (char)0x80);
            mask_G = _mm_setr_epi8((char)0x1, (char)0x80, (char)0x80, (char)0x80, (char)0x4, (char)0x80, (char)0x80, (char)0x80, (char)0x7, (char)0x80, (char)0x80, (char)0x80, (char)0xA, (char)0x80, (char)0x80, (char)0x80);
            mask_B = _mm_setr_epi8((char)0x2, (char)0x80, (char)0x80, (char)0x80, (char)0x5, (char)0x80, (char)0x80, (char)0x80, (char)0x8, (char)0x80, (char)0x80, (char)0x80, (char)0xB, (char)0x80, (char)0x80, (char)0x80);
        }
        int alignedLength = (length-2)& ~3;
        if (!use_fp16) {
            float * buf = (float *)malloc(sizeof(float)*1*3*matOrig.cols*matOrig.rows);
            float * B_buf = buf;
            float * G_buf = B_buf + length;
            float * R_buf = G_buf + length;
            int i = 0;
            __m128 fR, fG, fB;
            __m128 fB_temp, fG_temp, fR_temp;
            for (; i < alignedLength; i += 4) {
                __m128i pix0 = _mm_loadu_si128((__m128i *) img);
                fB = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_B));
                fG = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_G));
                fR = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_R));
                /*
                fB = _mm_mul_ps(fB, _mm_set1_ps(preprocessMpy[0]));
                fG = _mm_mul_ps(fG, _mm_set1_ps(preprocessMpy[1]));
                fR = _mm_mul_ps(fR, _mm_set1_ps(preprocessMpy[2]));
                fB = _mm_add_ps(fB, _mm_set1_ps(preprocessAdd[0]));
                fG = _mm_add_ps(fG, _mm_set1_ps(preprocessAdd[1]));
                fR = _mm_add_ps(fR, _mm_set1_ps(preprocessAdd[2]));
                */
                fB = _mm_fmadd_ps(fB, _mm_set1_ps(preprocessMpy[0]), _mm_set1_ps(preprocessAdd[0]));
                fG = _mm_fmadd_ps(fG, _mm_set1_ps(preprocessMpy[1]), _mm_set1_ps(preprocessAdd[1]));
                fR = _mm_fmadd_ps(fR, _mm_set1_ps(preprocessMpy[2]), _mm_set1_ps(preprocessAdd[2]));
                _mm_storeu_ps(B_buf, fB);
                _mm_storeu_ps(G_buf, fG);
                _mm_storeu_ps(R_buf, fR);
                B_buf += 4; G_buf += 4; R_buf += 4;
                img += 12;
            }
            /*for (; i < length; i++, img += 3) {
                *B_buf++ = (img[0] * preprocessMpy[0]) + preprocessAdd[0];
                *G_buf++ = (img[1] * preprocessMpy[1]) + preprocessAdd[1];
                *R_buf++ = (img[2] * preprocessMpy[2]) + preprocessAdd[2];
            }*/
            for (; i < length; i+=4, img += 3) {
            	fB_temp = _mm_fmadd_ps(_mm_set1_ps(img[0]), _mm_set1_ps(preprocessMpy[0]), _mm_set1_ps(preprocessAdd[0]));
            	fG_temp = _mm_fmadd_ps(_mm_set1_ps(img[1]), _mm_set1_ps(preprocessMpy[1]), _mm_set1_ps(preprocessAdd[1]));
            	fR_temp = _mm_fmadd_ps(_mm_set1_ps(img[2]), _mm_set1_ps(preprocessMpy[2]), _mm_set1_ps(preprocessAdd[2]));
            	_mm_storeu_ps(B_buf, fB_temp);
                _mm_storeu_ps(G_buf, fG_temp);
                _mm_storeu_ps(R_buf, fR_temp);
                B_buf += 4; G_buf += 4; R_buf += 4;
                img += 12;
            }
        }
        else {
            unsigned short * buf = (unsigned short *)malloc(sizeof(unsigned short)*1*3*matOrig.cols*matOrig.rows);
            unsigned short * B_buf = (unsigned short *)buf;
            unsigned short * G_buf = B_buf + length;
            unsigned short * R_buf = G_buf + length;
            int i = 0;

            __m128 fR, fG, fB;
            __m128i hR, hG, hB;
            for (; i < alignedLength; i += 4)
            {
                __m128i pix0 = _mm_loadu_si128((__m128i *) img);
                fB = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_B));
                fG = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_G));
                fR = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_R));
                /*
                fB = _mm_mul_ps(fB, _mm_set1_ps(preprocessMpy[0]));
                fG = _mm_mul_ps(fG, _mm_set1_ps(preprocessMpy[1]));
                fR = _mm_mul_ps(fR, _mm_set1_ps(preprocessMpy[2]));
                fB = _mm_add_ps(fB, _mm_set1_ps(preprocessAdd[0]));
                fG = _mm_add_ps(fG, _mm_set1_ps(preprocessAdd[1]));
                fR = _mm_add_ps(fR, _mm_set1_ps(preprocessAdd[2]));
                */
                fB = _mm_fmadd_ps(fB, _mm_set1_ps(preprocessMpy[0]), _mm_set1_ps(preprocessAdd[0]));
                fG = _mm_fmadd_ps(fG, _mm_set1_ps(preprocessMpy[1]), _mm_set1_ps(preprocessAdd[1]));
                fR = _mm_fmadd_ps(fR, _mm_set1_ps(preprocessMpy[2]), _mm_set1_ps(preprocessAdd[2]));
                // convert to half
                hB = _mm_cvtps_ph(fB, 0xF);
                hG = _mm_cvtps_ph(fG, 0xF);
                hR = _mm_cvtps_ph(fR, 0xF);
                _mm_storel_epi64((__m128i*)B_buf, hB);
                _mm_storel_epi64((__m128i*)G_buf, hG);
                _mm_storel_epi64((__m128i*)R_buf, hR);
                B_buf += 4; G_buf += 4; R_buf += 4;
                img += 12;
            }
            for (; i < length; i++, img += 3) {
                *B_buf++ = _cvtss_sh((float)((img[0] * preprocessMpy[0]) + preprocessAdd[0]), 1);
                *G_buf++ = _cvtss_sh((float)((img[1] * preprocessMpy[1]) + preprocessAdd[1]), 1);
                *R_buf++ = _cvtss_sh((float)((img[2] * preprocessMpy[2]) + preprocessAdd[2]), 1);
            }
        }
        matOrig.release();
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
            buffer = (char *)malloc(filelen);
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
    printf("Bytes per second: %ld bytes\n", (long)(1000*totalbyte/avgtime));
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
    if (argc < 3) {
        cout << "Usage: ./benchmark [Image Folder] [Iteration Count] [-f16] [-s]" << endl;
        return 0;
    }

    vector<string> filenames;
    bool show = false;
    bool use_fp16 = false;
    string dir = argv[1];
    dir.append("/");
    string pat = dir+"*";
    
    int count = atoi(argv[2]);
    if (argc >=5) {
        use_fp16 = true;
        show = true;
    }
    else if (argc >=4) {
        string option = argv[3];
        if (option == "-f16") {
            use_fp16 = true;
        }
        else if (option == "-s") {
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

    int num_of_cores = sysconf(_SC_NPROCESSORS_ONLN);
    cout << "The number of core: "<< num_of_cores << endl << endl;

    vector<vector<pair<char*, int>>> split = splitVector(buffers, 64);
    int64_t freq = clockFrequency(), t0, t1;
    
    for (int cores = 1; cores<= num_of_cores; cores*=2) {
        vector<thread> dec_threads(cores);
        float avgdecode = 0;
        for (int c = 0; c<count; c++) {
            float decodetook = 0;
            float decodebyte = 0;
            for (int i=0; i<split.size(); i++) {
                int size = split[i].size() / cores;
                size = size < 1 ? 1 : size;
                vector<vector<pair<char*, int>>> temp = splitVector(split[i], size);
                // if the batch size is smaller than the total number of threads, use only that are needed
                int minSize = min ((int)temp.size(), cores);
                t0 = clockCounter();
                for (int j=0; j<minSize; j++) {
                    //decode
                    //dec_threads[j] = thread(bind(&decodeFile, temp[j], use_fp16));
                }
                for (int i=0; i<minSize; i++) {
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
        // printf("Decode with %d core(s)\n", cores);
        // printf("--------------------------------\n");
        // printf("\nAverage speed per iteration: %.3f msec\n", avgdecode);
        // printf("Images per second: %ld images\n", (long)(1000/avgdecode*(float)buffers.size()));
        // printf("Bytes per second: %ld bytes\n", (long)(1000*totalbyte/avgdecode));
        // printf("MB per second: %.2f bytes\n\n", (float)(1000*totalbyte/avgdecode)/1000000);
    }
    
    return 0;
}