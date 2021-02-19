#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "hist-equ.h"

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[BIN_COUNT];
    int debug_hist[BIN_COUNT];
    float start, end;

    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
    printf("\nStart of histogram creation...\n");
    start = clock();
    histogram(hist, img_in.img, img_in.h * img_in.w, BIN_COUNT);
    end = clock();
    printf("\nHistogram creation ---------- %lf s\n", (double)(end-start)/CLOCKS_PER_SEC);
//    d_histogram(debug_hist, img_in.img, img_in.h * img_in.w, BIN_COUNT);
//    for (int i = 0; i < BIN_COUNT; i++) {
//        if (debug_hist[i] != hist[i])
//        {
//            printf ("PARTA MALAKAAAAA, sto %d ekanes malakia!!\n\n\n", i);
//        }
//    }
    histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);
    return result;
}
