#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "hist-equ.h"

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d;
    float start, end;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist_in[i++];
    }
    d = img_size - min;
    start = clock();
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
//         else if (lut[i] > 255) {
//             lut[i] = 255;
//         }
    }
    end=clock();
    printf("\nBuilding the Look-Up-Table in CPU takes %lf s\n", (double)(end-start)/CLOCKS_PER_SEC);
    
    /* Get the result image */
    start = clock();
    for(i = 0; i < img_size; i ++){
        if (lut[img_in[i]] > 255) {
            img_out[i] = 255;
        }
        else {
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
    }
    end = clock();
    printf("\nBuilding the final image from LUT in CPU takes %lf s\n", (double)(end-start)/CLOCKS_PER_SEC);
}
