/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.00005 
#define TILE_H 32
#define TILE_W 32
#define FILTER_RADIUS 32
#define FILTER_LENGTH 	(2 * FILTER_RADIUS  + 1)
#define MAX_SIZE 4096   /* Maximum possible size of submatrix */
#define STREAMS 4


__device__ __constant__ double d_Filter[FILTER_LENGTH];

#define cudaCheckError() {                                                                       \
        cudaError_t error=cudaGetLastError();                                                        \
        if(error!=cudaSuccess) {                                                                     \
            printf("ERROR IN CUDA %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));        \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    }

/* Separable convolution kernel in regards to ROWS.
 * Arguments:
 * 		- d_Dst: output array, buffered to COLUMN kernel
 * 		- d_Src: source array
 * 		- padding_right: array that contains the right-most padding elements of original array
 * 		- padding_left: same as padding_right, but for the left-most padding elements 
 * 		- imageW/H: the width/height of the image matrix
 * 		- option: [0,1,2] --
 *						   |-> 0: the kernel only uses the padding_left variable to calculate the last column
 *						   |-> 1: the kernel only uses the padding_right variable to calculate the 1st column
 *						   |-> 2: the kernel uses both paddings to calculate all elements in between
 */
__global__ void convolutionRowGPU(double *d_Dst, double *d_Src,double *padding_right,double *padding_left,int imageW, int imageH,int option) {
    
    int x,y,k,d,P_x,x0;
   
    __shared__ double image[TILE_H * (TILE_W+FILTER_RADIUS*2)];  
    
    x =  blockIdx.x*blockDim.x + threadIdx.x;
    y =  blockIdx.y*blockDim.y + threadIdx.y;
    
    P_x = x - FILTER_RADIUS;
    
    /* Divergent code iff FILTER_RADIUS < 32 BUT can be customized depending on threads/block */
    if(P_x < 0) {
        if(option == 0 || option == 2) {
            image[threadIdx.x + threadIdx.y*(TILE_W+FILTER_RADIUS*2)] = padding_left[y*FILTER_RADIUS + x];
        }
        else{
            image[threadIdx.x + threadIdx.y*(TILE_W+FILTER_RADIUS*2)] = 0;
        }
        image[threadIdx.x + blockDim.x + threadIdx.y*(TILE_W+FILTER_RADIUS*2)] = d_Src[y*imageW + x + blockDim.x - FILTER_RADIUS];
        image[threadIdx.x + blockDim.x + FILTER_RADIUS + threadIdx.y*(TILE_W+FILTER_RADIUS*2)] = d_Src[y*imageW + x + blockDim.x];
    }
    else{
        if(threadIdx.x < FILTER_RADIUS) {
            image[threadIdx.x + threadIdx.y*(TILE_W+FILTER_RADIUS*2)] = d_Src[y*imageW + x - FILTER_RADIUS];
            image[threadIdx.x + blockDim.x + threadIdx.y*(TILE_W+FILTER_RADIUS*2)] = d_Src[y*imageW + x + blockDim.x - FILTER_RADIUS];
            image[threadIdx.x + blockDim.x + FILTER_RADIUS + threadIdx.y*(TILE_W+FILTER_RADIUS*2)] = d_Src[y*imageW + x + blockDim.x];
        }
        else{
            image[threadIdx.x + threadIdx.y*(TILE_W+FILTER_RADIUS*2)] = d_Src[y*imageW + x - FILTER_RADIUS];
        }
            
    }

    P_x = x + FILTER_RADIUS;
    
    if(P_x > imageW - 1) {
        if(option == 1 || option == 2) {
            image[threadIdx.x + 2*FILTER_RADIUS +  threadIdx.y*(TILE_W+FILTER_RADIUS*2)] = padding_right[y*FILTER_RADIUS + P_x-imageW];
        }
        else{
           image[threadIdx.x + 2*FILTER_RADIUS +  threadIdx.y*(TILE_W+FILTER_RADIUS*2)] = 0;
        }
        
        
        image[threadIdx.x + FILTER_RADIUS +  threadIdx.y*(TILE_W+FILTER_RADIUS*2)] = d_Src[y*imageW + x];
    }
    
    
    
    __syncthreads();
    

    
    double sum = 0;
    
    x0 = threadIdx.x + FILTER_RADIUS ;
    for(k = -FILTER_RADIUS; k <= FILTER_RADIUS; k++) {
        d = x0 + k;
        
        sum += image[threadIdx.y*(TILE_W+FILTER_RADIUS*2) + d] * d_Filter[FILTER_RADIUS - k];
    }
        

    d_Dst[y*imageW + x] = sum;
    
}

/* Separable convolution kernel in regards to COLUMNS.
 * Arguments:
 * 		- d_Dst: final output array
 * 		- d_Src: source array, buffered from d_Dst of ROW kernel
 * 		- padding_up: array that contains the up-most padding elements of original array
 * 		- padding_down: same as padding_up, but for the down-most padding elements 
 * 		- imageW/H: the width/height of the image matrix
 * 		- option: [0,1,2] --
 *						   |-> 0: the kernel only uses the padding_up variable to calculate the last row
 *						   |-> 1: the kernel only uses the padding_down variable to calculate the 1st row
 *						   |-> 2: the kernel uses both paddings to calculate all elements in between
 */
__global__ void convolutionColumnGPU(double *d_Dst, double *d_Src,double *padding_up,double *padding_down,int imageW, int imageH,int option) {
    
      
    int x,y,k,d,P_y,y0;
   
    __shared__ double image[TILE_W * (TILE_H+FILTER_RADIUS*2)];  
    
    x =  blockIdx.x*blockDim.x + threadIdx.x;
    y =  blockIdx.y*blockDim.y + threadIdx.y;
    
    
    P_y = y - FILTER_RADIUS;

	/* Same as row kernel, divergent iff FILTER_RADIUS < 32 */    
    if(P_y < 0) {
        if(option == 0 || option == 2) {
            image[threadIdx.y*(TILE_W) + threadIdx.x] = padding_up[y*imageW + x];
        }
        else{
            image[threadIdx.y*(TILE_W) + threadIdx.x] = 0;
        }
        image[(threadIdx.y + blockDim.y)*TILE_W + threadIdx.x] = d_Src[(y + blockDim.y - FILTER_RADIUS)*imageW + x ];
        image[(threadIdx.y + blockDim.y + FILTER_RADIUS)*TILE_W + threadIdx.x] = d_Src[(y + blockDim.y)*imageW + x ];
    }
    else{
        if(threadIdx.y < FILTER_RADIUS){
            
            if(y + blockDim.y > imageH - 1) {
                image[threadIdx.y*(TILE_W) + threadIdx.x] = d_Src[(y- FILTER_RADIUS)*imageW + x];
            }
            else {
                image[threadIdx.y*TILE_W + threadIdx.x] = d_Src[(y- FILTER_RADIUS)*imageW + x];
                image[(threadIdx.y + blockDim.y)*TILE_W + threadIdx.x] = d_Src[(y + blockDim.y - FILTER_RADIUS)*imageW + x ];
                image[(threadIdx.y + blockDim.y + FILTER_RADIUS)*TILE_W + threadIdx.x] = d_Src[(y + blockDim.y)*imageW + x ];
            }
                
        }
        else{
            image[threadIdx.y*(TILE_W) + threadIdx.x] = d_Src[(y- FILTER_RADIUS)*imageW + x];
        }
    }

    P_y = y + FILTER_RADIUS;
    
    if(P_y > imageH - 1) {
        if(option == 1 || option == 2) {
            image[(threadIdx.y+2*FILTER_RADIUS)*TILE_W + threadIdx.x] = padding_down[(P_y-imageH)*imageW + x];
        }
        else{
            image[(threadIdx.y+2*FILTER_RADIUS)*TILE_W + threadIdx.x] = 0;
        }
        
        image[(threadIdx.y + FILTER_RADIUS)*TILE_W + threadIdx.x] = d_Src[y*imageW + x];
    }
        
    __syncthreads();
    

 
    double sum = 0;
    
    y0 = threadIdx.y + FILTER_RADIUS ;
    for(k = -FILTER_RADIUS; k <= FILTER_RADIUS; k++) {
        d = y0 + k;
        
        sum += image[d*TILE_W + threadIdx.x] * d_Filter[FILTER_RADIUS - k];
        
    }
    d_Dst[y*imageW + x] = sum;
    
}

 

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////


void convolutionRowCPU(double *h_Dst, double *h_Src, double *h_Filter, 
                       int imageW, int imageH) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

      for (k = -FILTER_RADIUS; k <= FILTER_RADIUS; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[FILTER_RADIUS - k];
        }     
      }

      
      //printf("ROW X:%d Y:%d SUM:%f\n\n",x,y,sum);
      h_Dst[y * imageW + x] = sum;
    }
  }
        
}



////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(double *h_Dst, double *h_Src, double *h_Filter,
    			   int imageW, int imageH) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

      for (k = -FILTER_RADIUS; k <= FILTER_RADIUS; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[FILTER_RADIUS - k];
        }   
      }
      //printf("COL X:%d Y:%d SUM:%f\n\n",x,y,sum);
      h_Dst[y * imageW + x] = sum;
    }
  }
    
}


//


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    double
    *h_Filter,
    *h_Input,
    *h_input_new,
    *h_filters_row_start,
    *h_filters_row_end,
    *h_filters_col_up,
    *h_filters_col_down,
    *h_Buffer_new,
    *d_Input_stream0,
    *d_Input_stream1,
    *d_Input_stream2,
    *d_Input_stream3,
    *d_Buffer_stream0,
    *d_Buffer_stream1,
    *d_Buffer_stream2,
    *d_Buffer_stream3,
    *d_OutputGPU_stream0,
    *d_OutputGPU_stream1,
    *d_OutputGPU_stream2,
    *d_OutputGPU_stream3,
    *d_Filter_col_up_stream0,
    *d_Filter_col_up_stream1,
    *d_Filter_col_up_stream2,
    *d_Filter_col_up_stream3,
    *d_Filter_col_down_steam0,
    *d_Filter_col_down_steam1,
    *d_Filter_col_down_steam2,
    *d_Filter_col_down_steam3,
    *d_Filter_row_left_stream0,
    *d_Filter_row_left_stream1,
    *d_Filter_row_left_stream2,
    *d_Filter_row_left_stream3,
    *d_Filter_row_right_stream0,
    *d_Filter_row_right_stream1,
    *d_Filter_row_right_stream2,
    *d_Filter_row_right_stream3,
    *h_OutputGPU,
    *h_OutputGPU_new;
#ifdef _HOST
    *h_Buffer,
    *h_OutputCPU,
    *h_Buffer_debug,
#endif


    int imageW;
    int imageH;
    unsigned int i;
    int number_of_blocks;
    int array_W;
    int array_H;
    int x;
    int y;
    int j;
    
#ifdef _HOST
    double timing;
    clock_t start;
    clock_t end;
#endif
    cudaStream_t stream0, stream1 , stream2, stream3;

    

 
    
    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;
    
    /* number_of_blocks <= 256 strictly */
    printf("Enter number of blocks should be a power of 2 (this number will be squared for actual calculations, e.g. 4->real blocks = 16): ");
    scanf("%d", &number_of_blocks);
    imageH = imageW;
    

    dim3 threads(TILE_H,TILE_W);
    dim3 blocks (number_of_blocks,number_of_blocks);
    
    array_H = TILE_H*number_of_blocks;
    array_W = TILE_W*number_of_blocks;
    
    
    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays and device array...\n");
    
    h_Filter    = (double *)malloc(FILTER_LENGTH * sizeof(double));
    h_Input     = (double *)malloc(imageW * imageH * sizeof(double));
#ifdef _HOST
    h_Buffer    = (double *)malloc(imageW * imageH * sizeof(double));
    h_Buffer_debug = (double *)malloc(imageW * imageH * sizeof(double));
    h_OutputCPU = (double *)malloc(imageW * imageH * sizeof(double));
#endif
    h_OutputGPU = (double *)malloc(imageW * imageH * sizeof(double));   
    
    /* Allocating pinned memory for row elements */
    cudaMallocHost((void**)&h_input_new, imageW * imageH * sizeof(double));
    cudaCheckError();
    cudaMallocHost((void**)&h_Buffer_new, imageW * imageH * sizeof(double));
    cudaCheckError();
    cudaMallocHost((void**)&h_filters_row_start,imageH/array_H * (imageW/array_W-1)* array_H * FILTER_RADIUS *sizeof(double));
    cudaCheckError();
    cudaMallocHost((void**)& h_filters_row_end, imageH/array_H * (imageW/array_W-1)* array_H * FILTER_RADIUS *sizeof(double));
    cudaCheckError();

    
    if (h_Filter==NULL || h_Input == NULL){
        printf("Something went wrong wille malloc in CPU\n");
        exit(EXIT_FAILURE);
    }
    
    printf("Memory allocation for host arrays: COMPLETED \n");
    
    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (double)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (double)rand() / ((double)RAND_MAX / 255) + (double)rand() / (double)RAND_MAX;
    }
    
    for(i=0; i < imageH/array_H; i++) {
        for(j=0; j < imageW/array_W; j++) {
            for(y=0; y < array_H; y++) {
                for(x=0; x < array_W;x++) {
                    
                    h_input_new[y*array_W + x + (i*imageH/array_H + j)*array_H*array_W]= h_Input[(y+i*array_H)*imageW + x + j*array_W];
                    
                    
                    if(j != 0 && x < FILTER_RADIUS) {
                        h_filters_row_start[x + y*FILTER_RADIUS +(i*(imageH/array_H-1)+j-1)*FILTER_RADIUS*array_H] = h_input_new[y*array_W + x + (i*imageH/array_H + j)*array_H*array_W];
                    }
                    
                    if(j != imageW/array_W -1 && x+FILTER_RADIUS > array_W-1){
                        h_filters_row_end[x- array_W+FILTER_RADIUS + y*FILTER_RADIUS +(i*(imageH/array_H-1)+j)*FILTER_RADIUS*array_H] = h_input_new[y*array_W + x + (i*imageH/array_H + j)*array_H*array_W];

                    }
                }
            }
        }
    }
    
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    
    cudaMemcpyToSymbol(d_Filter,h_Filter,FILTER_LENGTH * sizeof(double),0,cudaMemcpyHostToDevice);
    
    cudaCheckError();
    
    cudaMallocManaged((void**)&d_Input_stream0,array_H * array_W * sizeof(double));
    cudaCheckError();
    cudaMallocManaged((void**)&d_Input_stream1,array_H * array_W * sizeof(double));
    cudaCheckError();
    cudaMallocManaged((void**)&d_Input_stream2,array_H * array_W * sizeof(double));
    cudaCheckError();
    cudaMallocManaged((void**)&d_Input_stream3,array_H * array_W * sizeof(double));
    cudaCheckError();
    
    cudaMallocManaged((void**)&d_Buffer_stream0,array_H * array_W * sizeof(double));
    cudaCheckError();
    cudaMallocManaged((void**)&d_Buffer_stream1,array_H * array_W * sizeof(double));
    cudaCheckError();
    cudaMallocManaged((void**)&d_Buffer_stream2,array_H * array_W * sizeof(double));
    cudaCheckError();
    cudaMallocManaged((void**)&d_Buffer_stream3,array_H * array_W * sizeof(double));
    cudaCheckError();
    
    cudaMallocManaged((void**)&d_Filter_row_right_stream0, array_H * FILTER_RADIUS *sizeof(double));
    cudaCheckError();
    cudaMallocManaged((void**)&d_Filter_row_left_stream0, array_H * FILTER_RADIUS *sizeof(double));
    cudaCheckError();
    cudaMallocManaged((void**)&d_Filter_row_right_stream1, array_H * FILTER_RADIUS *sizeof(double));
    cudaCheckError();
    cudaMallocManaged((void**)&d_Filter_row_left_stream1, array_H * FILTER_RADIUS *sizeof(double));
    cudaCheckError();
    cudaMallocManaged((void**)&d_Filter_row_right_stream2, array_H * FILTER_RADIUS *sizeof(double));
    cudaCheckError();
    cudaMallocManaged((void**)&d_Filter_row_left_stream2, array_H * FILTER_RADIUS *sizeof(double));
    cudaCheckError();
    cudaMallocManaged((void**)&d_Filter_row_right_stream3, array_H * FILTER_RADIUS *sizeof(double));
    cudaCheckError();
    cudaMallocManaged((void**)&d_Filter_row_left_stream3, array_H * FILTER_RADIUS *sizeof(double));
    

    
    cudaCheckError();
    
    printf("start..\n");
    
    for(i=0; i < imageH/array_H; i++) {
        for(j=0; j < imageW/array_W; j += STREAMS) {
            
            cudaMemcpyAsync(d_Input_stream0,&h_input_new[(i*imageH/array_H + j)*array_H*array_W],array_H*array_W*sizeof(double),cudaMemcpyHostToDevice,stream0);
          
            cudaMemcpyAsync(d_Filter_row_right_stream0,&h_filters_row_start[(i*(imageH/array_H-1) + j)*FILTER_RADIUS*array_H],array_H*FILTER_RADIUS*sizeof(double),cudaMemcpyHostToDevice,stream0);
             
            
            if(j != 0) {                 
                cudaMemcpyAsync(d_Filter_row_left_stream0,&h_filters_row_end[(i*(imageH/array_H-1) + j -1)*FILTER_RADIUS*array_H],array_H*FILTER_RADIUS*sizeof(double),cudaMemcpyHostToDevice,stream0);
            }
            
            cudaMemcpyAsync(d_Input_stream1,&h_input_new[(i*imageH/array_H + j + 1)*array_H*array_W],array_H*array_W*sizeof(double),cudaMemcpyHostToDevice,stream1);
             
            cudaMemcpyAsync(d_Filter_row_right_stream1,&h_filters_row_start[(i*(imageH/array_H-1) + j+1)*FILTER_RADIUS*array_H],array_H*FILTER_RADIUS*sizeof(double),cudaMemcpyHostToDevice,stream1);
            
            cudaMemcpyAsync(d_Filter_row_left_stream1,&h_filters_row_end[(i*(imageH/array_H-1) + j)*FILTER_RADIUS*array_H],array_H*FILTER_RADIUS*sizeof(double),cudaMemcpyHostToDevice,stream1);
             
            cudaMemcpyAsync(d_Input_stream2,&h_input_new[(i*imageH/array_H + j + 2)*array_H*array_W],array_H*array_W*sizeof(double),cudaMemcpyHostToDevice,stream2);
             
            cudaMemcpyAsync(d_Filter_row_right_stream2,&h_filters_row_start[(i*(imageH/array_H-1) + j+2)*FILTER_RADIUS*array_H],array_H*FILTER_RADIUS*sizeof(double),cudaMemcpyHostToDevice,stream2);
              
            cudaMemcpyAsync(d_Filter_row_left_stream2,&h_filters_row_end[(i*(imageH/array_H-1) + j + 1)*FILTER_RADIUS*array_H],array_H*FILTER_RADIUS*sizeof(double),cudaMemcpyHostToDevice,stream2);
             
            cudaMemcpyAsync(d_Input_stream3,&h_input_new[(i*imageH/array_H + j + 3)*array_H*array_W],array_H*array_W*sizeof(double),cudaMemcpyHostToDevice,stream3);
             
            cudaMemcpyAsync(d_Filter_row_left_stream3,&h_filters_row_end[(i*(imageH/array_H-1) + j + 2)*FILTER_RADIUS*array_H],array_H*FILTER_RADIUS*sizeof(double),cudaMemcpyHostToDevice,stream3);
             
            if(j != (imageW/array_W) - STREAMS) {
                cudaMemcpyAsync(d_Filter_row_right_stream3,&h_filters_row_start[(i*(imageH/array_H-1) + j+3)*FILTER_RADIUS*array_H],array_H*FILTER_RADIUS*sizeof(double),cudaMemcpyHostToDevice,stream3);
            }
            
            if(j == 0) {
                convolutionRowGPU<<<blocks,threads,0,stream0>>>(d_Buffer_stream0,d_Input_stream0,d_Filter_row_right_stream0,d_Filter_row_right_stream0,array_W,array_H,1);
            }
            
            else {
                convolutionRowGPU<<<blocks,threads,0,stream0>>>(d_Buffer_stream0,d_Input_stream0,d_Filter_row_right_stream0,d_Filter_row_left_stream0,array_W,array_H,2);
            }
            
            convolutionRowGPU<<<blocks,threads,0,stream1>>>(d_Buffer_stream1,d_Input_stream1,d_Filter_row_right_stream1,d_Filter_row_left_stream1,array_W,array_H,2);
            
            convolutionRowGPU<<<blocks,threads,0,stream2>>>(d_Buffer_stream2,d_Input_stream2,d_Filter_row_right_stream2,d_Filter_row_left_stream2,array_W,array_H,2);
            
            if(j == (imageW/array_W) - STREAMS) {
                convolutionRowGPU<<<blocks,threads,0,stream3>>>(d_Buffer_stream3,d_Input_stream3,d_Filter_row_left_stream3,d_Filter_row_left_stream3,array_W,array_H,0);
            }
            
            else {
                convolutionRowGPU<<<blocks,threads,0,stream3>>>(d_Buffer_stream3,d_Input_stream3,d_Filter_row_right_stream3,d_Filter_row_left_stream3,array_W,array_H,2);
            }
            
            
            cudaMemcpyAsync(&h_Buffer_new[(i*imageH/array_H + j)*array_H*array_W],d_Buffer_stream0,array_H * array_W * sizeof(double),cudaMemcpyDeviceToHost,stream0);
            
            cudaMemcpyAsync(&h_Buffer_new[(i*imageH/array_H + j+1)*array_H*array_W],d_Buffer_stream1,array_H * array_W * sizeof(double),cudaMemcpyDeviceToHost,stream1);
            
            cudaMemcpyAsync(&h_Buffer_new[(i*imageH/array_H + j+2)*array_H*array_W],d_Buffer_stream2,array_H * array_W * sizeof(double),cudaMemcpyDeviceToHost,stream2);
            
            cudaMemcpyAsync(&h_Buffer_new[(i*imageH/array_H + j+3)*array_H*array_W],d_Buffer_stream3,array_H * array_W * sizeof(double),cudaMemcpyDeviceToHost,stream3);
            
        }
    }
    cudaDeviceSynchronize();
    cudaCheckError();
    
    cudaFreeHost(h_input_new);
    cudaFreeHost(h_filters_row_start);
    cudaFreeHost(h_filters_row_end);
    
    cudaCheckError();
    
    cudaFree(d_Input_stream0);
    cudaFree(d_Input_stream1);
    cudaFree(d_Input_stream2);
    cudaFree(d_Input_stream3);
    
    cudaFree(d_Filter_row_right_stream0);
    cudaFree(d_Filter_row_right_stream1);
    cudaFree(d_Filter_row_right_stream2);
    cudaFree(d_Filter_row_right_stream3);
    
    cudaFree(d_Filter_row_left_stream0);
    cudaFree(d_Filter_row_left_stream1);
    cudaFree(d_Filter_row_left_stream2);
    cudaFree(d_Filter_row_left_stream3);
    
    cudaCheckError();
    
    
    cudaMallocHost((void**)&h_OutputGPU_new, imageW * imageH * sizeof(double));
    cudaCheckError();
    cudaMallocHost((void**)& h_filters_col_up,(imageH/array_H-1) * imageW/array_W * array_W * FILTER_RADIUS *sizeof(double));
    cudaCheckError();
    cudaMallocHost((void**)&  h_filters_col_down,(imageH/array_H-1) * imageW/array_W * array_W * FILTER_RADIUS *sizeof(double));
    cudaCheckError();
    
    cudaMallocManaged((void**)&d_Filter_col_up_stream0, array_W * FILTER_RADIUS *sizeof(double));
    cudaCheckError();
    cudaMallocManaged((void**)&d_Filter_col_up_stream1, array_W * FILTER_RADIUS *sizeof(double));
    cudaCheckError();
    cudaMallocManaged((void**)&d_Filter_col_up_stream2, array_W * FILTER_RADIUS *sizeof(double));
    cudaCheckError();
    cudaMallocManaged((void**)&d_Filter_col_up_stream3, array_W * FILTER_RADIUS *sizeof(double));
    cudaCheckError();
    
    cudaMallocManaged((void**)&d_Filter_col_down_steam0, array_W * FILTER_RADIUS *sizeof(double));
    cudaCheckError();
    cudaMallocManaged((void**)&d_Filter_col_down_steam1, array_W * FILTER_RADIUS *sizeof(double));
    cudaCheckError();
    cudaMallocManaged((void**)&d_Filter_col_down_steam2, array_W * FILTER_RADIUS *sizeof(double));
    cudaCheckError();
    cudaMallocManaged((void**)&d_Filter_col_down_steam3, array_W * FILTER_RADIUS *sizeof(double));
    cudaCheckError();
    
    cudaMallocManaged((void**)&d_OutputGPU_stream0,array_H * array_W * sizeof(double));
    cudaCheckError();
    cudaMallocManaged((void**)&d_OutputGPU_stream1,array_H * array_W * sizeof(double));
    cudaCheckError();
    cudaMallocManaged((void**)&d_OutputGPU_stream2,array_H * array_W * sizeof(double));
    cudaCheckError();
    cudaMallocManaged((void**)&d_OutputGPU_stream3,array_H * array_W * sizeof(double));
    cudaCheckError();
    
    for(i=0; i < imageH/array_H; i++) {
        for(j=0; j < imageW/array_W; j++) {
            for(y=0; y < array_H; y++) {
                for(x=0; x < array_W;x++) {
                    
                    if(i != 0 && y < FILTER_RADIUS) {
                        h_filters_col_up[x+ y*array_W+((i-1)*(imageH/array_H)+j)*FILTER_RADIUS*array_W] = h_Buffer_new[y*array_W + x + (i*imageH/array_H + j)*array_H*array_W];
                    }
                    
                    if(i != imageH/array_H-1 && y + FILTER_RADIUS > array_H-1) {
                        h_filters_col_down[x + (y-array_H+FILTER_RADIUS)*array_W + (i*(imageH/array_H)+j)*FILTER_RADIUS*array_W] =  h_Buffer_new[y*array_W + x + (i*imageH/array_H + j)*array_H*array_W];
                    }
                    
                }
            }
        }
    }
    
    

    for(i=0; i < imageH/array_H; i += STREAMS) {
        for(j=0; j < imageW/array_W; j++ ) {
            
            cudaMemcpyAsync(d_Buffer_stream0,&h_Buffer_new[(i*imageH/array_H + j)*array_H*array_W],array_H*array_W*sizeof(double),cudaMemcpyHostToDevice,stream0);
            
            cudaMemcpyAsync(d_Filter_col_down_steam0,&h_filters_col_up[(i*(imageH/array_H) + j)*FILTER_RADIUS*array_W],array_W*FILTER_RADIUS*sizeof(double),cudaMemcpyHostToDevice,stream0);
            
            if(i != 0) {
                cudaMemcpyAsync(d_Filter_col_up_stream0,&h_filters_col_down[((i-1)*(imageH/array_H) + j)*FILTER_RADIUS*array_W],array_W*FILTER_RADIUS*sizeof(double),cudaMemcpyHostToDevice,stream0);
            }
            
            cudaMemcpyAsync(d_Buffer_stream1,&h_Buffer_new[((i+1)*imageH/array_H + j)*array_H*array_W],array_H*array_W*sizeof(double),cudaMemcpyHostToDevice,stream1);
            
            cudaMemcpyAsync(d_Filter_col_down_steam1,&h_filters_col_up[((i+1)*(imageH/array_H) + j)*FILTER_RADIUS*array_W],array_W*FILTER_RADIUS*sizeof(double),cudaMemcpyHostToDevice,stream1);
            
            cudaMemcpyAsync(d_Filter_col_up_stream1,&h_filters_col_down[(i*(imageH/array_H) + j)*FILTER_RADIUS*array_W],array_W*FILTER_RADIUS*sizeof(double),cudaMemcpyHostToDevice,stream1);
            
            cudaMemcpyAsync(d_Buffer_stream2,&h_Buffer_new[((i+2)*imageH/array_H + j)*array_H*array_W],array_H*array_W*sizeof(double),cudaMemcpyHostToDevice,stream2);
            
            cudaMemcpyAsync(d_Filter_col_down_steam2,&h_filters_col_up[((i+2)*(imageH/array_H) + j)*FILTER_RADIUS*array_W],array_W*FILTER_RADIUS*sizeof(double),cudaMemcpyHostToDevice,stream2);
            
            cudaMemcpyAsync(d_Filter_col_up_stream2,&h_filters_col_down[((i+1)*(imageH/array_H) + j)*FILTER_RADIUS*array_W],array_W*FILTER_RADIUS*sizeof(double),cudaMemcpyHostToDevice,stream2);
            
            cudaMemcpyAsync(d_Buffer_stream3,&h_Buffer_new[((i+3)*imageH/array_H + j)*array_H*array_W],array_H*array_W*sizeof(double),cudaMemcpyHostToDevice,stream3);
            
            cudaMemcpyAsync(d_Filter_col_up_stream3,&h_filters_col_down[((i+2)*(imageH/array_H) + j)*FILTER_RADIUS*array_W],array_W*FILTER_RADIUS*sizeof(double),cudaMemcpyHostToDevice,stream3);
            
            if(i != imageH/array_H - STREAMS) {
                cudaMemcpyAsync(d_Filter_col_down_steam3,&h_filters_col_up[((i+3)*(imageH/array_H) + j)*FILTER_RADIUS*array_W],array_W*FILTER_RADIUS*sizeof(double),cudaMemcpyHostToDevice,stream3);
            }
                
            
            
            if(i == 0) {
                convolutionColumnGPU<<<blocks,threads,0,stream0>>>(d_OutputGPU_stream0,d_Buffer_stream0,d_Filter_col_down_steam0,d_Filter_col_down_steam0,array_W,array_H,1);
            }
            else{
                convolutionColumnGPU<<<blocks,threads,0,stream0>>>(d_OutputGPU_stream0,d_Buffer_stream0,d_Filter_col_up_stream0,d_Filter_col_down_steam0,array_W,array_H,2);
            }
            
            convolutionColumnGPU<<<blocks,threads,0,stream1>>>(d_OutputGPU_stream1,d_Buffer_stream1,d_Filter_col_up_stream1,d_Filter_col_down_steam1,array_W,array_H,2);
                
            convolutionColumnGPU<<<blocks,threads,0,stream2>>>(d_OutputGPU_stream2,d_Buffer_stream2,d_Filter_col_up_stream2,d_Filter_col_down_steam2,array_W,array_H,2);
            
            
            if(i == imageH/array_H -1) {
                convolutionColumnGPU<<<blocks,threads,0,stream3>>>(d_OutputGPU_stream3,d_Buffer_stream3,d_Filter_col_up_stream3,d_Filter_col_up_stream3,array_W,array_H,0);
            }
            else{
                convolutionColumnGPU<<<blocks,threads,0,stream3>>>(d_OutputGPU_stream3,d_Buffer_stream3,d_Filter_col_up_stream3,d_Filter_col_down_steam3,array_W,array_H,2);
            }
            
            cudaMemcpyAsync(&h_OutputGPU_new[(i*imageH/array_H + j)*array_H*array_W],d_OutputGPU_stream0,array_H * array_W * sizeof(double),cudaMemcpyDeviceToHost,stream0);
            
            cudaMemcpyAsync(&h_OutputGPU_new[((i+1)*imageH/array_H + j)*array_H*array_W],d_OutputGPU_stream1,array_H * array_W * sizeof(double),cudaMemcpyDeviceToHost,stream1);
            
            cudaMemcpyAsync(&h_OutputGPU_new[((i+2)*imageH/array_H + j)*array_H*array_W],d_OutputGPU_stream2,array_H * array_W * sizeof(double),cudaMemcpyDeviceToHost,stream2);
            
            cudaMemcpyAsync(&h_OutputGPU_new[((i+3)*imageH/array_H + j)*array_H*array_W],d_OutputGPU_stream3,array_H * array_W * sizeof(double),cudaMemcpyDeviceToHost,stream3);
           
            
        }
    }
    
    
    cudaDeviceSynchronize();
    cudaCheckError();
    
    cudaFree(d_OutputGPU_stream0);
    cudaFree(d_OutputGPU_stream1);
    cudaFree(d_OutputGPU_stream2);
    cudaFree(d_OutputGPU_stream3);
    
    cudaFree(d_Buffer_stream0);
    cudaFree(d_Buffer_stream1);
    cudaFree(d_Buffer_stream2);
    cudaFree(d_Buffer_stream3);
    
    cudaFree(d_Filter_col_down_steam0);
    cudaFree(d_Filter_col_down_steam1);
    cudaFree(d_Filter_col_down_steam2);
    cudaFree(d_Filter_col_down_steam3);
    
    cudaFree(d_Filter_col_up_stream0);
    cudaFree(d_Filter_col_up_stream1);
    cudaFree(d_Filter_col_up_stream2);
    cudaFree(d_Filter_col_up_stream3);
    
    
    for(i=0; i < imageH/array_H; i++) {
        for(j=0; j < imageW/array_W; j++) {
            for(y=0; y < array_H; y++) {
                for(x=0; x < array_W;x++) {
                    h_OutputGPU[(y+i*array_H)*imageW + x + j*array_W] = h_OutputGPU_new[y*array_W + x + (i*imageH/array_H + j)*array_H*array_W];
                }
            }
        }
    }
    
        
    cudaFreeHost(h_OutputGPU_new);
    cudaFreeHost(h_Buffer_new);
    cudaFreeHost(h_filters_col_down);
    cudaFreeHost(h_filters_col_up);
    
    
#ifdef _HOST    
    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");
    
    start = clock();
    
    convolutionRowCPU(h_Buffer_debug, h_Input, h_Filter, imageW, imageH); // convolution kata grammes
    
    convolutionColumnCPU(h_OutputCPU, h_Buffer_debug, h_Filter, imageW, imageH); // convolution kata sthles
    
    end = clock();
    
    timing = ((double) (end - start)) / CLOCKS_PER_SEC;
    
    printf("CPU computation : COMPLETED in time:%10.5f\n",timing);

    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas 
    
    
    printf("\nCPU computations == GPU computation?\n");
    for (i = 0; i < imageW * imageH; i++) {
        if(h_OutputGPU[i] > h_OutputCPU[i] + accuracy || h_OutputGPU[i] < h_OutputCPU[i] - accuracy){
            printf("CPU computations == GPU computation : FALSE line:%d difrence:%f \nExitting program...\n GPU: %lf \n",i,h_OutputGPU[i]-h_OutputCPU[i],h_OutputGPU[i]);
           
            
            cudaDeviceReset();
            return(1);
        }
            
    }
    printf("CPU computations == GPU computation : TRUE \nExitting program after Memmory Free...\n");
#endif

    cudaDeviceReset();
    return 0;
}
