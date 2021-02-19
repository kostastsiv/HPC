#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define BIN_COUNT 256	/* Size of histogram and LUT */
#define BLOCK_THREADS 256	/* Threads per block */
#define NUM_OF_CUTS 4	/* Number of blocks to "cut" the src_image in */
#define STREAMS 2	/* Number of streams */
#define NUM_OF_BLOCKS 1000	/* Number of thread blocks */

#define CEIL(x,y) (((x) + (y) - 1) / (y))

#define cudaCheckError() { \
        cudaError_t error=cudaGetLastError(); \
        if(error!=cudaSuccess) { \
            printf("ERROR IN CUDA %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error)); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    }
    

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;    


__device__ __constant__ int d_lut[BIN_COUNT];	/* LUT in device memory, fast because small size */
__device__ __constant__ int d_hist[BIN_COUNT];	/* Histogram in device memory */
    

/* Kernel for histogram creation in GPU.
 * Arguments:
 *		- d_buff_hist: intermediate buffer to store subhistograms, size = NUM_OF_BLOCKS * BIN_COUNT
 *		- d_img_in: src image located in global device memory
 * 		- img_size: total image size (widht*height)
 *		- nbr_bin: number of bins in histogram (== BIN_COUNT)
 */
__global__ void kernelHistogram(int * d_buff_hist, unsigned char * d_img_in, int img_size, int nbr_bin){
    
	/* One thread for every position of image */
    unsigned long int i = threadIdx.x + blockIdx.x * blockDim.x;
    
	/* Stride for accesses on src image */
	unsigned long int offset = blockDim.x * gridDim.x;
    
	/* Histogram in shared memory */
    __shared__ int shareHist[BIN_COUNT];
    
    /* Initialize shared memory */
    shareHist[threadIdx.x] = 0;
	
	__syncthreads();
    
    while(i < img_size){
        atomicAdd(&shareHist[d_img_in[i]],1);
        i+=offset;
    }
	__syncthreads();
    
	/* Store subhistograms in global memory, where they will be used in another kernel to form final histogram */
    d_buff_hist[blockIdx.x * nbr_bin + threadIdx.x] = shareHist[threadIdx.x];
}

/* Kernel for subhistogram merging.
 * Arguments:
 * 		- d_hist_out: final histogram
 * 		- d_hist_in: matrix of subhistograms
 *		- blocks: number of blocks (== NUM_OF_BLOCKS)
 *		- nbr_bin: number of bins (== BIN_COUNT)
 */
__global__ void AddHistograms(int *d_hist_out, int *d_hist_in, int blocks, int nbr_bin){
    
    int i = threadIdx.x;
    int sum = 0;
    int j;
    
	/* Thread-level sums for each "luminocity" of final histogram */
    for(j=0; j < blocks; j++){
        sum += d_hist_in[j*nbr_bin + i];
    }
    
    d_hist_out[i] = sum;
}

/* Kernel for calculating LookUpTable for final image creation
 * Arguments:
 * 		- lut_out: final LookUpTable
 *		- nbr_bin: number of histogram bins (== BIN_COUNT)
 *		- min: minimum value in histogram
 */
__global__ void lutCalculate(int *lut_out,int nbr_bin,int d,int min){
    
    int i = threadIdx.x;
    int j,cdf = 0;
    
    /* Calculate CDF values for each thread. Does introduce divergence, but is overall fast because of histogram size */
    for(j=0; j < i+1; j++) {
        cdf += d_hist[j];
    }
    
    lut_out[i] = (int)(((float)cdf-min)*255/d + 0.5);
    
    if(lut_out[i] < 0){
        lut_out[i] = 0;
    }
    else if(lut_out[i] > 255){
        lut_out[i] = 255;
    }
}

/* Kernel for reconstructing the image.
 * Arguments:
 *		- img_in: original image in global memory
 *		- img_out: final reconstructed image
 *		- img_size: total size of image
 */
__global__ void imageCreate(unsigned char *img_in , unsigned char *img_out, int img_size) {
    
    unsigned long int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long int offset = blockDim.x * gridDim.x;
    
     while(i < img_size){
        img_out[i] = (unsigned char)d_lut[img_in[i]];
        i+=offset;
    }
}

/* CPU API to GPU histogram kernel.
 * arguments:
 *      - img_in: source image, (memcpy host 2 device)
 * 		- img_out: final reconstructed image
 *      - img_size: total size of image (like a stripe)
 *      - nbr_bin: number of bins in histogram (here every possible grayscale value is a bin)
 */
void d_contrast_enhancement(unsigned char *img_in, unsigned char *img_out ,int img_size, int nbr_bin)
{
    unsigned char * d_img_in_stream0;
    unsigned char * d_img_in_stream1;
    unsigned char * d_img_out_stream0;
    unsigned char * d_img_out_stream1;
	int * hist_out;	/* Host memory final histogram */
    int * d_buff_hist; /* intermediate collection of subhistograms */
    int * d_hist_out; /* Final histogram */
	int * d_lut_calc;
	cudaStream_t stream0, stream1;
	
	int i;
    int new_img_size = img_size/NUM_OF_CUTS;
	int mod = img_size % NUM_OF_CUTS;
	int min=0;
    int d;

    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    
    cudaMallocHost((void**)&hist_out, BIN_COUNT*sizeof(int));
	
    /* Set dim3 variables */
    dim3 threads_per_block(BLOCK_THREADS, 1, 1);
	dim3 blocks_in_grid(NUM_OF_BLOCKS, 1, 1);
    
    dim3 threads_per_block_Add(nbr_bin,1,1);
    dim3 blocks_in_grid_Add(1,1,1);
    
	/* Allocate d_img_in and d_hist_out to pinned memory */
    cudaMallocManaged((void**)&d_img_in_stream0, sizeof(unsigned char)*new_img_size);
    cudaMallocManaged((void**)&d_img_in_stream1, sizeof(unsigned char)*(new_img_size+mod));
    cudaMallocManaged((void**)&d_buff_hist, nbr_bin*sizeof(int)*NUM_OF_BLOCKS*NUM_OF_CUTS);
    cudaMallocManaged((void**)&d_hist_out, nbr_bin*sizeof(int));
    
	cudaCheckError();
    	
    for(i=0; i < NUM_OF_CUTS; i += STREAMS) {
        
        cudaMemcpyAsync(d_img_in_stream0, &img_in[i*new_img_size], sizeof(unsigned char)*new_img_size, cudaMemcpyHostToDevice,stream0);

        if (mod != 0 && i+1 == NUM_OF_CUTS - 1) {
			cudaMemcpyAsync(d_img_in_stream1, &img_in[(i+1)*new_img_size], sizeof(unsigned char)*(new_img_size+mod), cudaMemcpyHostToDevice,stream1);
			cudaDeviceSynchronize();
			cudaCheckError();
		}
        else {
			cudaMemcpyAsync(d_img_in_stream1, &img_in[(i+1)*new_img_size], sizeof(unsigned char)*new_img_size, cudaMemcpyHostToDevice,stream1);
            cudaDeviceSynchronize();
			cudaCheckError();
        }
		
        kernelHistogram<<<blocks_in_grid, threads_per_block, 0, stream0>>>(&d_buff_hist[i*NUM_OF_BLOCKS*nbr_bin], d_img_in_stream0, new_img_size, nbr_bin);
        cudaDeviceSynchronize();
        cudaCheckError();
		if (mod != 0 && i+1 == NUM_OF_CUTS - 1) {
			kernelHistogram<<<blocks_in_grid, threads_per_block, 0, stream1>>>(&d_buff_hist[(i+1)*NUM_OF_BLOCKS*nbr_bin], d_img_in_stream1, new_img_size+mod, nbr_bin);
            cudaDeviceSynchronize();
			cudaCheckError();
		}
		else {
			kernelHistogram<<<blocks_in_grid, threads_per_block, 0, stream1>>>(&d_buff_hist[(i+1)*NUM_OF_BLOCKS*nbr_bin], d_img_in_stream1, new_img_size, nbr_bin);
            cudaDeviceSynchronize();
			cudaCheckError();
		}
    }
	
	cudaDeviceSynchronize();
	cudaCheckError();
    
    AddHistograms<<<blocks_in_grid_Add,threads_per_block_Add>>>(d_hist_out,d_buff_hist, NUM_OF_BLOCKS*NUM_OF_CUTS , nbr_bin);
    
    cudaDeviceSynchronize();
	cudaCheckError();
    
	cudaMemcpy(hist_out, d_hist_out, nbr_bin*sizeof(int), cudaMemcpyDeviceToHost);
    
	cudaCheckError();
    
    i=0;
    while(min == 0){
        min = hist_out[i++];
    }
    d = img_size - min;
    
    cudaMallocManaged((void**)&d_lut_calc, nbr_bin*sizeof(int));
    
	cudaCheckError();
	
    cudaMemcpyToSymbol(d_hist,d_hist_out,nbr_bin*sizeof(int),0,cudaMemcpyDeviceToDevice);
    
	cudaCheckError();
	
    lutCalculate<<<blocks_in_grid_Add,threads_per_block_Add>>>(d_lut_calc,nbr_bin,d,min);
    
    cudaMallocManaged((void**)&d_img_out_stream0, sizeof(unsigned char)*new_img_size);
    cudaCheckError();
	
	cudaMallocManaged((void**)&d_img_out_stream1, sizeof(unsigned char)*new_img_size);
    cudaCheckError();
	
    cudaMemcpyToSymbol(d_lut,d_lut_calc,nbr_bin*sizeof(int),0,cudaMemcpyDeviceToDevice);
    cudaCheckError();
	
     for(i=0; i < NUM_OF_CUTS; i += STREAMS) {
        cudaMemcpyAsync(d_img_in_stream0, &img_in[i*new_img_size], sizeof(unsigned char)*new_img_size, cudaMemcpyHostToDevice,stream0);
        cudaMemcpyAsync(d_img_in_stream1, &img_in[(i+1)*new_img_size], sizeof(unsigned char)*new_img_size, cudaMemcpyHostToDevice,stream1);
        
        imageCreate<<<blocks_in_grid, threads_per_block, 0, stream0>>>(d_img_in_stream0, d_img_out_stream0, new_img_size);
        imageCreate<<<blocks_in_grid, threads_per_block, 0, stream1>>>(d_img_in_stream1, d_img_out_stream1, new_img_size);
        
        cudaMemcpyAsync(&img_out[i*new_img_size], d_img_out_stream0, sizeof(unsigned char)*new_img_size, cudaMemcpyDeviceToHost,stream0);
        cudaMemcpyAsync(&img_out[(i+1)*new_img_size], d_img_out_stream1, sizeof(unsigned char)*new_img_size, cudaMemcpyDeviceToHost,stream1);
    }
    cudaDeviceSynchronize();
	cudaCheckError();
	
	/* Free all CUDA memory */
    cudaFree(d_img_in_stream0);
    cudaFree(d_img_in_stream1);
    cudaFree(d_img_out_stream0);
    cudaFree(d_img_out_stream1);
    cudaFree(d_buff_hist);
    cudaFree(d_hist_out);
    cudaFree(d_lut_calc);
	cudaFree(hist_out);
}

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    
    result.w = img_in.w;
    result.h = img_in.h;
    
    cudaMallocHost((void**)&result.img, result.w * result.h * sizeof(unsigned char));
    
	d_contrast_enhancement(img_in.img, result.img ,img_in.h * img_in.w, BIN_COUNT);
    
	return result;
}

PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    
	/* For GPU purpose */
    cudaMallocHost((void**)&result.img, result.w * result.h * sizeof(unsigned char));
    cudaCheckError();
    
    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    cudaFree(img.img);
}

void run_gpu_gray_test(PGM_IMG img_in, char *out_filename);


void run_gpu_gray_test(PGM_IMG img_in, char *out_filename)
{
    unsigned int timer = 0;
    PGM_IMG img_obuf;
    
    
    printf("Starting GPU processing...\n");
    img_obuf = contrast_enhancement_g(img_in);
    write_pgm(img_obuf, out_filename);
    free_pgm(img_obuf);
}


int main(int argc, char *argv[]){
    PGM_IMG img_ibuf_g;

	if (argc != 3) {
		printf("Run with input file name and output file name as arguments\n");
		exit(1);
	}
	
    printf("Running contrast enhancement for gray-scale images.\n");
    img_ibuf_g = read_pgm(argv[1]);
    run_gpu_gray_test(img_ibuf_g, argv[2]);
    free_pgm(img_ibuf_g);
    cudaDeviceReset();
	return 0;
}
