/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_XY 32
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.00005 
#define TILE_H 32
#define TILE_W 32
#define FILTER_RADIUS 32
#define FILTER_LENGTH 	(2 * FILTER_RADIUS  + 1)


__device__ __constant__ double d_Filter[FILTER_LENGTH];

#define cudaCheckError() {                                                                       \
        cudaError_t error=cudaGetLastError();                                                        \
        if(error!=cudaSuccess) {                                                                     \
            printf("ERROR IN CUDA %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));        \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    }


__global__ void convolutionRowGPU(double *d_Dst, double *d_Src,int imageW, int imageH) {
    
    int x,y,k,d,P_x,x0,number_of_writes = 1;
   
    __shared__ double image[TILE_H * (TILE_W+FILTER_RADIUS*2)];  
    
    x =  blockIdx.x*blockDim.x + threadIdx.x;
    y =  blockIdx.y*blockDim.y + threadIdx.y;
   
    if(FILTER_RADIUS > TILE_W){
        number_of_writes = FILTER_RADIUS/TILE_W + 1;
    }
    
    P_x = x - FILTER_RADIUS;
    

    
    if(P_x < 0) {
        image[threadIdx.x + threadIdx.y*(TILE_W+FILTER_RADIUS*2)] = 0;
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
        image[threadIdx.x + 2*FILTER_RADIUS +  threadIdx.y*(TILE_W+FILTER_RADIUS*2)] = 0;
        
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


__global__ void convolutionColumnGPU(double *d_Dst, double *d_Src,int imageW, int imageH) {
    
      
    int x,y,k,d,P_y,y0;
   
    __shared__ double image[TILE_W * (TILE_H+FILTER_RADIUS*2)];  
    
    x =  blockIdx.x*blockDim.x + threadIdx.x;
    y =  blockIdx.y*blockDim.y + threadIdx.y;
    
    
    P_y = y - FILTER_RADIUS;
    
    if(P_y < 0) {
        image[threadIdx.y*(TILE_W) + threadIdx.x] = 0;
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
        
        image[(threadIdx.y+2*FILTER_RADIUS)*TILE_W + threadIdx.x] = 0;
        
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
    *h_Buffer,
    *h_OutputCPU,
    *d_Input,
    *d_Buffer,
    *d_OutputGPU,
    *h_OutputGPU;


    int imageW;
    int imageH;
    //int i=MAX_XY;
    //int count=0;
    unsigned int i;
    
    double timing;
    clock_t start;
    clock_t end;
    

    
    
// Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;
    
 //   while(1){
 //       if(imageW % i == 0) {
 //           dim3 threads(i,i);
 //           dim3 blocks(imageW/i,imageW/i);
 //           break;
 //       }
 //       i--;
 //   }
    
    dim3 threads(MAX_XY,MAX_XY);
    dim3 blocks (imageH/MAX_XY,imageW/MAX_XY); /* this is wrong, fix later */
    
    if(imageH < MAX_XY && imageW < MAX_XY){
        threads.x = imageH;
	threads.y = imageH;
	blocks.x = 1;
	blocks.y = 1;
    }
    
    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays and device array...\n");
    
    
        
    
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (double *)malloc(FILTER_LENGTH * sizeof(double));
    h_Input     = (double *)malloc(imageW * imageH * sizeof(double));
    h_Buffer    = (double *)malloc(imageW * imageH * sizeof(double));
    h_OutputCPU = (double *)malloc(imageW * imageH * sizeof(double));
    h_OutputGPU = (double *)malloc(imageW * imageH * sizeof(double));
    
    if (h_Filter==NULL || h_Input == NULL || h_Buffer == NULL || h_OutputCPU == NULL){
        printf("Something went wrong wille malloc in CPU\n");
    }
    
    printf("Memmory allocation for host arrays: COMPLETED \n");
    
    
    cudaMallocManaged((void**)&d_Input,imageH * imageW * sizeof(double));
    cudaMallocManaged((void**)&d_Buffer,imageH * imageW * sizeof(double));
    cudaMallocManaged((void**)&d_OutputGPU,imageH * imageW * sizeof(double));
    
    cudaCheckError();
    
    printf("Memmory allocation for device arrays: COMPLETED \n");
    
    
    
    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.
    
    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (double)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (double)rand() / ((double)RAND_MAX / 255) + (double)rand() / (double)RAND_MAX;
    }
    
    printf("initialization of host arrays: COMPLETED \n");

    /* START OF HOST2DEVICE TRANSFER */    
    start=clock();
    cudaMemcpyToSymbol(d_Filter,h_Filter,FILTER_LENGTH * sizeof(double),0,cudaMemcpyHostToDevice);
    cudaMemcpy(d_Input, h_Input,imageH * imageW * sizeof(double),cudaMemcpyHostToDevice);
    
    cudaCheckError();
    end=clock();
    printf("memcpy host2device execution time: %g s\n", (double(end-start))/CLOCKS_PER_SEC);
    printf("initialization of device arrays: COMPLETED \n\n");
    
    printf("GPU computation...\n");
    
    /* START OF ROW KERNEL */
    start=clock();
    convolutionRowGPU<<<blocks,threads>>>(d_Buffer,d_Input,imageW,imageH);
    cudaCheckError();
    
    cudaDeviceSynchronize();
    end=clock();
    printf("row kernel execution time: %g s\n", (double(end-start))/CLOCKS_PER_SEC);
    /* END OF ROW KERNEL */

    /* START OF COL KERNEL */
    start=clock();
    convolutionColumnGPU<<<blocks,threads>>>(d_OutputGPU,d_Buffer,imageW,imageH);
    
    cudaCheckError();
    cudaDeviceSynchronize();
    end=clock();
    printf("col kernel execution time: %g s\n", (double(end-start))/CLOCKS_PER_SEC);
    /* END OF COL KERNEL */
    printf("GPU computation : COMPLETED\n\n");
    
    /* START OF DEVICE2HOST TRANSFER */
    start=clock();
    cudaMemcpy(h_OutputGPU,d_OutputGPU,imageH * imageW * sizeof(double),cudaMemcpyDeviceToHost);
    end=clock();
    printf("memcpy device2host execution time: %g s\n", (double(end-start))/CLOCKS_PER_SEC);
    /* END OF DEVICE2HOST TRANSFER */   
    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");
    
    start = clock();
    
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH); // convolution kata sthles
    
    end = clock();
    
    timing = ((double) (end - start)) / CLOCKS_PER_SEC;
    
    printf("CPU computation : COMPLETED in time:%10.5f\n",timing);

    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas 
    
    
    printf("\nCPU computations == GPU computation?\n");
    for (i = 0; i < imageW * imageH; i++) {
        if(h_OutputGPU[i] > h_OutputCPU[i] + accuracy || h_OutputGPU[i] < h_OutputCPU[i] - accuracy){
            printf("CPU computations == GPU computation : FALSE line:%d difrence:%f \nExitting program...\n GPU: %lf \n",i,h_OutputGPU[i]-h_OutputCPU[i],h_OutputGPU[i]);
           
            
            free(h_OutputCPU);
            free(h_OutputGPU);
            free(h_Buffer);
            free(h_Input);
            free(h_Filter);
            
            
            // free all the allocated memory GPU
            cudaFree(d_OutputGPU);
            cudaFree(d_Buffer);
            cudaFree(d_Input);
            
            cudaCheckError();
            
            cudaDeviceReset();
            
           return(1);
        }
            
    }
    printf("CPU computations == GPU computation : TRUE \nExitting program after Memmory Free...\n");



    // free all the allocated memory CPU
    free(h_OutputCPU);
    free(h_OutputGPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);
    
    // free all the allocated memory GPU
    cudaFree(d_OutputGPU);
    cudaFree(d_Buffer);
    cudaFree(d_Input);
      
    cudaDeviceReset();


    return 0;
}
