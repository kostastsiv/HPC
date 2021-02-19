#ifndef HIST_EQU_COLOR_H
#define HIST_EQU_COLOR_H

#define BIN_COUNT 256
#define WARP_SIZE 32
#define BLOCK_THREADS 1024

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;    



PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin);

/* CPU API for GPU kernel invocation */
//void d_histogram(int *hist_out, unsigned char *img_in, int img_size, int nbr_bin);
//void d_histogram_equalization(unsigned char *img_out, unsigned char *img_in,
//                            int *hist_in, int img_size, int nbr_bin);

//Contrast enhancement for gray-scale images
PGM_IMG contrast_enhancement_g(PGM_IMG img_in);

#endif
