#include "Benchmark.h"
#include <nmmintrin.h>
#include <x86intrin.h>
using namespace std;
using namespace cv;

#define FP_BITS     16
#define FP_MUL (1 << FP_BITS)

// #define BUFFER_SIZE INT_MAX
Benchmark::Benchmark(string directory, bool use_fp16) : mUse_fp16(use_fp16) {
    mFilenames = glob(directory);
    mFileQueue = make_unique<FileQueue>();
}

Benchmark::~Benchmark() {}

int Benchmark::getFileSize() { return mFilenames.size();}

bool Benchmark::isEmptyQueue() { return mFileQueue->isEmpty(); }

unsigned int Benchmark::getQueueSize() { return mFileQueue->getSize(); }

void Benchmark::RGB_resize(unsigned char *Rgb_in, unsigned char *Rgb_out, unsigned int swidth, unsigned int sheight, unsigned int dwidth, unsigned int dheight) {
    float xscale = (float)((double)swidth / (double)dwidth);
    float yscale = (float)((double)sheight / (double)dheight);
    int alignW = (dwidth + 15)&~15;
    unsigned int *Xmap = new unsigned int[alignW*2];
    unsigned short *Xf = (unsigned short *)(Xmap + alignW);
    unsigned short *Xf1 = Xf + alignW;

    int xpos = (int)(FP_MUL * (xscale*0.5 - 0.5));
    int xinc = (int)(FP_MUL * xscale);
    int yinc = (int)(FP_MUL * yscale);    // to convert to fixed point
    unsigned int aligned_width = dwidth;
    // generate xmap
    for (unsigned int x = 0; x < dwidth; x++, xpos += xinc)
    {
        int xf;
        int xmap = (xpos >> FP_BITS);
        if (xmap >= (int)(swidth - 4)){
            aligned_width = x;
        }
        if (xmap >= (int)(swidth - 1)){
            Xmap[x] = (swidth - 1)*3;
        }
        else
            Xmap[x] = (xmap<0)? 0: xmap*3;
        xf = ((xpos & 0xffff) + 0x80) >> 8;
        Xf[x] = xf;
        Xf1[x] = (0x100 - xf);
    }
    aligned_width &= ~3;
    int stride = swidth * 3;
    int dstride = dwidth * 3;
    unsigned char *pSrcBorder = Rgb_in + (sheight*stride) - 3;    // points to the last pixel

    int ypos = (int)(FP_MUL * (yscale*0.5 - 0.5));
    for (int y = 0; y < (int)dheight; y++, ypos += yinc)
    {
        int ym, fy, fy1;
        unsigned char *pSrc1, *pSrc2;
        unsigned char *pdst = Rgb_out + y*dstride;

        ym = (ypos >> FP_BITS);
        fy = ((ypos & 0xffff) + 0x80) >> 8;
        fy1 = (0x100 - fy);
        if (ym >= (int)(sheight - 1)){
            pSrc1 = pSrc2 = Rgb_in + (sheight - 1)*stride;
        }
        else
        {
            pSrc1 = (ym<0)? Rgb_in : (Rgb_in + ym*stride);
            pSrc2 = pSrc1 + stride;
        }
        __m128i w_y = _mm_setr_epi32(fy1, fy, fy1, fy);
        const __m128i mm_zeros = _mm_setzero_si128();
        const __m128i mm_round = _mm_set1_epi32((int)0x80);
        __m128i p01, p23, ps01, ps23, pRG1, pRG2, pRG3;
        unsigned int x = 0;
        for (; x < aligned_width; x += 4)
        {
            // load 2 pixels each
            p01 = _mm_loadl_epi64((const __m128i*) &pSrc1[Xmap[x]]);
            p23 = _mm_loadl_epi64((const __m128i*) &pSrc1[Xmap[x+1]]);
            ps01 = _mm_loadl_epi64((const __m128i*) &pSrc2[Xmap[x]]);
            ps23 = _mm_loadl_epi64((const __m128i*) &pSrc2[Xmap[x + 1]]);
            // unpcklo for p01 and ps01
            p01 = _mm_unpacklo_epi8(p01, ps01);
            p23 = _mm_unpacklo_epi8(p23, ps23);
            p01 = _mm_unpacklo_epi16(p01, _mm_srli_si128(p01, 6));     //R0R1R2R3 G0G1G2G3 B0B1B2B3 XXXX for first pixel
            p23 = _mm_unpacklo_epi16(p23, _mm_srli_si128(p23, 6));      //R0R1R2R3 G0G1G2G3 B0B1B2B3 XXXX for second pixel

            // load xf and 1-xf
            ps01 = _mm_setr_epi32(Xf1[x], Xf1[x], Xf[x], Xf[x]);			// xfxfxf1xf1
            ps01 = _mm_mullo_epi32(ps01, w_y);                      // W0W1W2W3 for first pixel
            ps23 = _mm_setr_epi32(Xf1[x + 1], Xf1[x + 1], Xf[x + 1], Xf[x + 1]);
            ps23 = _mm_mullo_epi32(ps23, w_y);                      // W0W1W2W3 for second pixel
            ps01 = _mm_srli_epi32(ps01, 8);                 // convert to 16bit
            ps23 = _mm_srli_epi32(ps23, 8);                 // convert to 16bit
            ps01 = _mm_packus_epi32(ps01, ps01);                 // convert to 16bit
            ps23 = _mm_packus_epi32(ps23, ps23);                 // convert to 16bit

            // extend to 16bit
            pRG1 = _mm_unpacklo_epi8(p01, mm_zeros);        // R0R1R2R3 and G0G1G2G3
            p01 = _mm_srli_si128(p01, 8);             // B0B1B2B3xxxx
            p01 = _mm_unpacklo_epi32(p01, p23);       // B0B1B2B3 R0R1R2R3: ist and second
            p23 = _mm_srli_si128(p23, 4);             // G0G1G2G3 B0B1B2B3 for second pixel
            p01 = _mm_unpacklo_epi8(p01, mm_zeros);         // B0B1B2B3 R0R1R2R3
            pRG2 = _mm_unpacklo_epi8(p23, mm_zeros);        // G0G1G2G3 B0B1B2B3 for second pixel

            pRG1 = _mm_madd_epi16(pRG1, ps01);                  // (W0*R0+W1*R1), (W2*R2+W3*R3), (W0*G0+W1*G1), (W2*G2+W3*G3)
            pRG2 = _mm_madd_epi16(pRG2, ps23);                  //(W0*R0+W1*R1), (W2*R2+W3*R3), (W0*G0+W1*G1), (W2*G2+W3*G3) for seond pixel
            ps01 = _mm_unpacklo_epi64(ps01, ps23);
            p01 = _mm_madd_epi16(p01, ps01);                  //(W0*B0+W1*B1), (W2*B2+W3*B3), (W0*R0+W1*R1), (W2*R2+W3*R3) 1st and second pixel

            pRG1 = _mm_hadd_epi32(pRG1, p01);      // R0,G0, B0, R1 (32bit)
            p01 = _mm_loadl_epi64((const __m128i*) &pSrc1[Xmap[x+2]]);
            p23 = _mm_loadl_epi64((const __m128i*) &pSrc1[Xmap[x+3]]);
            ps01 = _mm_loadl_epi64((const __m128i*) &pSrc2[Xmap[x+2]]);
            ps23 = _mm_loadl_epi64((const __m128i*) &pSrc2[Xmap[x+3]]);
            pRG1 = _mm_add_epi32(pRG1, mm_round);
            // unpcklo for p01 and ps01
            p01 = _mm_unpacklo_epi8(p01, ps01);
            p01 = _mm_unpacklo_epi16(p01, _mm_srli_si128(p01, 6));     //R0R1R2R3 G0G1G2G3 B0B1B2B3 XXXX for first pixel
            p23 = _mm_unpacklo_epi8(p23, ps23);
            p23 = _mm_unpacklo_epi16(p23, _mm_srli_si128(p23, 6));      //R0R1R2R3 G0G1G2G3 B0B1B2B3 XXXX for second pixel
            // load xf and 1-xf
            ps01 = _mm_setr_epi32(Xf1[x+2], Xf1[x+2], Xf[x+2], Xf[x+2]);			// xfxfxf1xf1
            ps01 = _mm_mullo_epi32(ps01, w_y);                      // W0W1W2W3 for first pixel
            ps23 = _mm_setr_epi32(Xf1[x + 3], Xf1[x + 3], Xf[x + 3], Xf[x + 3]);
            ps23 = _mm_mullo_epi32(ps23, w_y);                      // W0W1W2W3 for second pixel
            ps01 = _mm_srli_epi32(ps01, 8);                 // convert to 16bit
            ps23 = _mm_srli_epi32(ps23, 8);                 // convert to 16bit
            ps01 = _mm_packus_epi32(ps01, ps01);                 // convert to 16bit
            ps23 = _mm_packus_epi32(ps23, ps23);                 // convert to 16bit
            // extend to 16bit
            pRG3 = _mm_unpacklo_epi8(p01, mm_zeros);        // R0R1R2R3 and G0G1G2G3
            p01 = _mm_srli_si128(p01, 8);             // B0B1B2B3xxxx
            p01 = _mm_unpacklo_epi32(p01, p23);       // B0B1B2B3 R0R1R2R3: ist and second
            p23 = _mm_srli_si128(p23, 4);             // G0G1G2G3 B0B1B2B3 for second pixel
            p01 = _mm_unpacklo_epi8(p01, mm_zeros);         // B0B1B2B3 R0R1R2R3
            p23 = _mm_unpacklo_epi8(p23, mm_zeros);        // G0G1G2G3 B0B1B2B3 for second pixel

            pRG3 = _mm_madd_epi16(pRG3, ps01);                  // (W0*R0+W1*R1), (W2*R2+W3*R3), (W0*G0+W1*G1), (W2*G2+W3*G3)
            p23 = _mm_madd_epi16(p23, ps23);                  //(W0*R0+W1*R1), (W2*R2+W3*R3), (W0*G0+W1*G1), (W2*G2+W3*G3) for seond pixel
            ps01 = _mm_unpacklo_epi64(ps01, ps23);
            p01 = _mm_madd_epi16(p01, ps01);                  //(W0*B0+W1*B1), (W2*B2+W3*B3), (W0*B0+W1*B1), (W2*B2+W3*B3) for seond pixel

            pRG2 = _mm_hadd_epi32(pRG2, pRG3);      // G1, B1, R2,G2 (32bit)
            p01 = _mm_hadd_epi32(p01, p23);      // B2,R3, G3, B3 (32bit)
            pRG2 = _mm_add_epi32(pRG2, mm_round);
            p01 = _mm_add_epi32(p01, mm_round);
            pRG1 = _mm_srli_epi32(pRG1, 8);      // /256
            pRG2 = _mm_srli_epi32(pRG2, 8);      // /256
            p01 = _mm_srli_epi32(p01, 8);      // /256

            // convert to 16bit
            pRG1 = _mm_packus_epi32(pRG1, pRG2); //R0G0B0R1G1B1R2G2
            p01 = _mm_packus_epi32(p01, p01); //B2R3B3G3
            pRG1 = _mm_packus_epi16(pRG1, mm_zeros);
            p01 = _mm_packus_epi16(p01, mm_zeros);
            _mm_storeu_si128((__m128i *)pdst, _mm_unpacklo_epi64(pRG1, p01));
            pdst += 12;
        }

        for (; x < dwidth; x++) {
            int result;
            const unsigned char *p0 = pSrc1 + Xmap[x];
            const unsigned char *p01 = p0 + 3;
            const unsigned char *p1 = pSrc2 + Xmap[x];
            const unsigned char *p11 = p1 + 3;
            if (p0 > pSrcBorder) p0 = pSrcBorder;
            if (p1 > pSrcBorder) p1 = pSrcBorder;
            if (p01 > pSrcBorder) p01 = pSrcBorder;
            if (p11 > pSrcBorder) p11 = pSrcBorder;
            result = ((Xf1[x] * fy1*p0[0]) + (Xf[x] * fy1*p01[0]) + (Xf1[x] * fy*p1[0]) + (Xf[x] * fy*p11[0]) + 0x8000) >> 16;
            *pdst++ = (unsigned char) std::max(0, std::min(result, 255));
            result = ((Xf1[x] * fy1*p0[1]) + (Xf[x] * fy1*p01[1]) + (Xf1[x] * fy*p1[1]) + (Xf[x] * fy*p11[1]) + 0x8000) >> 16;
            *pdst++ = (unsigned char)std::max(0, std::min(result, 255));
            result = ((Xf1[x] * fy1*p0[2]) + (Xf[x] * fy1*p01[2]) + (Xf1[x] * fy*p1[2]) + (Xf[x] * fy*p11[2]) + 0x8000) >> 16;
            *pdst++ = (unsigned char)std::max(0, std::min(result, 255));
        }
    }
    if (Xmap) delete[] Xmap;
}

unsigned int Benchmark::readFile(int count) {
    FILE *file;
    char *buffer;
    int totalsize = mFilenames.size();
    unsigned int filelen;
    unsigned int byteread = 0;

    for (int j=0; j < totalsize; j++) {
        file = fopen(mFilenames[j].c_str(), "r");
        if (file== NULL) {
            cout << "can't open" << mFilenames[j] << endl;
            exit(-1);
        }
        fseek(file, 0, SEEK_END);
        filelen = ftell(file);
        rewind(file);
        buffer = (char *)malloc(filelen);
        byteread+=fread(buffer, 1, filelen, file);
        fclose(file);
        if (count == 0) {
            tuple<char*, int> image(buffer, filelen);
            mFileQueue->enqueue(image);
        }
    }
    return byteread;
}

void Benchmark::decodeFile() {
    tuple<char*, int> image = mFileQueue->dequeue();
    char * byteStream = get<0>(image);
    int size = get<1>(image);
    Mat matOrig = imdecode(Mat(1, size, CV_8UC1, byteStream), CV_LOAD_IMAGE_UNCHANGED);
    if (matOrig.empty()) {
        cout << "ERROR: Image corrupted (Mat empty)" << endl;
        exit(1);
    }
    convertToTensor(matOrig);
    matOrig.release();
}

void Benchmark::convertToTensor(const Mat &matOrig) {
    int length = matOrig.cols * matOrig.rows;
    unsigned char * img;
    unsigned char *data_resize = nullptr;
    
    int width = 224; // TODO: set width and height
    int height = 224; // TODO: set width and height
    
    if ((width == matOrig.cols) && (height == matOrig.rows)) {
        // no resize required
        img = matOrig.data;
    } else {
        unsigned int aligned_size = ((length+width) * 3 + 128)&~127;
        data_resize = new unsigned char[aligned_size];
        RGB_resize(matOrig.data, data_resize, matOrig.cols, matOrig.rows, width, height);
        img = data_resize;
    }
 
    __m128i mask_B, mask_G, mask_R;
    if (reverseInputChannelOrder) {
        mask_B = _mm_setr_epi8((char)0x0, (char)0x80, (char)0x80, (char)0x80, (char)0x3, (char)0x80, (char)0x80, (char)0x80, (char)0x6, (char)0x80, (char)0x80, (char)0x80, (char)0x9, (char)0x80, (char)0x80, (char)0x80);
        mask_G = _mm_setr_epi8((char)0x1, (char)0x80, (char)0x80, (char)0x80, (char)0x4, (char)0x80, (char)0x80, (char)0x80, (char)0x7, (char)0x80, (char)0x80, (char)0x80, (char)0xA, (char)0x80, (char)0x80, (char)0x80);
        mask_R = _mm_setr_epi8((char)0x2, (char)0x80, (char)0x80, (char)0x80, (char)0x5, (char)0x80, (char)0x80, (char)0x80, (char)0x8, (char)0x80, (char)0x80, (char)0x80, (char)0xB, (char)0x80, (char)0x80, (char)0x80);
    } else {
        mask_R = _mm_setr_epi8((char)0x0, (char)0x80, (char)0x80, (char)0x80, (char)0x3, (char)0x80, (char)0x80, (char)0x80, (char)0x6, (char)0x80, (char)0x80, (char)0x80, (char)0x9, (char)0x80, (char)0x80, (char)0x80);
        mask_G = _mm_setr_epi8((char)0x1, (char)0x80, (char)0x80, (char)0x80, (char)0x4, (char)0x80, (char)0x80, (char)0x80, (char)0x7, (char)0x80, (char)0x80, (char)0x80, (char)0xA, (char)0x80, (char)0x80, (char)0x80);
        mask_B = _mm_setr_epi8((char)0x2, (char)0x80, (char)0x80, (char)0x80, (char)0x5, (char)0x80, (char)0x80, (char)0x80, (char)0x8, (char)0x80, (char)0x80, (char)0x80, (char)0xB, (char)0x80, (char)0x80, (char)0x80);
    }
    int alignedLength = (length-2)& ~3;
    if (!mUse_fp16) {
        float * buf = (float *)malloc(sizeof(float)*1*3*length);
        float * B_buf = buf;
        float * G_buf = B_buf + length;
        float * R_buf = G_buf + length;
        int i = 0;

        __m128 fR, fG, fB;

        for (; i < alignedLength; i += 4) {
            __m128i pix0 = _mm_loadu_si128((__m128i *) img);
            fB = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_B));
            fG = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_G));
            fR = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_R));
            fB = _mm_mul_ps(fB, _mm_set1_ps(preprocessMpy[0]));
            fG = _mm_mul_ps(fG, _mm_set1_ps(preprocessMpy[1]));
            fR = _mm_mul_ps(fR, _mm_set1_ps(preprocessMpy[2]));
            fB = _mm_add_ps(fB, _mm_set1_ps(preprocessAdd[0]));
            fG = _mm_add_ps(fG, _mm_set1_ps(preprocessAdd[1]));
            fR = _mm_add_ps(fR, _mm_set1_ps(preprocessAdd[2]));
            _mm_storeu_ps(B_buf, fB);
            _mm_storeu_ps(G_buf, fG);
            _mm_storeu_ps(R_buf, fR);
            B_buf += 4;
            G_buf += 4;
            R_buf += 4;
            img += 12;
        }
        for (; i < length; i++, img += 3) {
            *B_buf++ = (img[0] * preprocessMpy[0]) + preprocessAdd[0];
            *G_buf++ = (img[1] * preprocessMpy[1]) + preprocessAdd[1];
            *R_buf++ = (img[2] * preprocessMpy[2]) + preprocessAdd[2];
        }
    } else {
        unsigned short * buf = (unsigned short *)malloc(sizeof(unsigned short)*1*3*length);
        unsigned short * B_buf = (unsigned short *)buf;
        unsigned short * G_buf = B_buf + length;
        unsigned short * R_buf = G_buf + length;
        int i = 0;

        __m128 fR, fG, fB;
        __m128i hR, hG, hB;

        for (; i < alignedLength; i += 4) {
            __m128i pix0 = _mm_loadu_si128((__m128i *) img);
            fB = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_B));
            fG = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_G));
            fR = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_R));
            fB = _mm_mul_ps(fB, _mm_set1_ps(preprocessMpy[0]));
            fG = _mm_mul_ps(fG, _mm_set1_ps(preprocessMpy[1]));
            fR = _mm_mul_ps(fR, _mm_set1_ps(preprocessMpy[2]));
            fB = _mm_add_ps(fB, _mm_set1_ps(preprocessAdd[0]));
            fG = _mm_add_ps(fG, _mm_set1_ps(preprocessAdd[1]));
            fR = _mm_add_ps(fR, _mm_set1_ps(preprocessAdd[2]));
            // convert to half
            hB = _mm_cvtps_ph(fB, 0xF);
            hG = _mm_cvtps_ph(fG, 0xF);
            hR = _mm_cvtps_ph(fR, 0xF);
            _mm_storel_epi64((__m128i*)B_buf, hB);
            _mm_storel_epi64((__m128i*)G_buf, hG);
            _mm_storel_epi64((__m128i*)R_buf, hR);
            B_buf += 4;
            G_buf += 4;
            R_buf += 4;
            img += 12;
        }
        for (; i < length; i++, img += 3) {
            *B_buf++ = _cvtss_sh((float)((img[0] * preprocessMpy[0]) + preprocessAdd[0]), 1);
            *G_buf++ = _cvtss_sh((float)((img[1] * preprocessMpy[1]) + preprocessAdd[1]), 1);
            *R_buf++ = _cvtss_sh((float)((img[2] * preprocessMpy[2]) + preprocessAdd[2]), 1);
        }
    }
}
