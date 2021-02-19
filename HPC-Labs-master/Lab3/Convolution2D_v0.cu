/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

unsigned int filter_radius;

typedef float myDataType;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))


#define cudaCheckError() {                                                                       \
        cudaError_t error=cudaGetLastError();                                                        \
        if(error!=cudaSuccess) {                                                                     \
            printf("ERROR IN CUDA %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));        \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    }


__global__ void convolutionRowGPU(myDataType *d_Dst, myDataType *d_Src, myDataType *d_Filter,int imageW, int imageH, int filterR) {
    int x,y,k,d;
   
    
    x =  threadIdx.x;
    y =  threadIdx.y;
   
  
    
    myDataType sum = 0;
    
    for(k = -filterR; k <= filterR; k++) {
        d = x + k;
        
        if(d >= 0 && d < imageW) {
            sum += d_Src[y * imageW + d] * d_Filter[filterR -k];
        }
    }
    //printf("ROW X:%d Y:%d SUM:%f\n\n",threadIdx.x,threadIdx.y,sum);
    d_Dst[y*imageW + x] = sum;
    
}

__global__ void convolutionColumnGPU(myDataType *d_Dst, myDataType *d_Src, myDataType *d_Filter, 
                       int imageW, int imageH, int filterR) {
    int x,y,k,d;
    
    x =  threadIdx.x;
    y =  threadIdx.y;
    
    
    myDataType sum = 0;
    
    for(k = -filterR; k <= filterR; k++) {
        d = y + k;
        
        if(d >= 0 && d < imageH) {
            sum += d_Src[d * imageW + x] * d_Filter[filterR -k];
            //printf("X:%d Y:%d SUM:%f\n\n",threadIdx.x,threadIdx.y,sum);
        }
    }
    
    //printf("COL X:%d Y:%d SUM:%f\n\n",threadIdx.x,threadIdx.y,sum);
    d_Dst[y * imageW + x] = sum;
    
}
 

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////


void convolutionRowCPU(myDataType *h_Dst, myDataType *h_Src, myDataType *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      myDataType sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
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
void convolutionColumnCPU(myDataType *h_Dst, myDataType *h_Src, myDataType *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      myDataType sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }   
      }
      //printf("COL X:%d Y:%d SUM:%f\n\n",x,y,sum);
      h_Dst[y * imageW + x] = sum;
    }
  }
    
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    myDataType
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *d_Filter,
    *d_Input,
    *d_Buffer,
    *d_OutputGPU,
    *h_OutputGPU;


    int imageW;
    int imageH;
    //int count=0;
    unsigned int i;
    
    double accuracy;
    double timing;
    
    clock_t start;
    clock_t end;
    
    
    

	printf("Enter filter radius : ");
	scanf(" %d", &filter_radius);

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf(" %d", &imageW);
    imageH = imageW;
    
    printf("Enter Accuracy:");
    scanf(" %lf", &accuracy);

    dim3 threads(imageH,imageW);
    
    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays and device array...\n");
    
    
        
    
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (myDataType *)malloc(FILTER_LENGTH * sizeof(myDataType));
    h_Input     = (myDataType *)malloc(imageW * imageH * sizeof(myDataType));
    h_Buffer    = (myDataType *)malloc(imageW * imageH * sizeof(myDataType));
    h_OutputCPU = (myDataType *)malloc(imageW * imageH * sizeof(myDataType));
    h_OutputGPU = (myDataType *)malloc(imageW * imageH * sizeof(myDataType));
    
    if (h_Filter==NULL || h_Input == NULL || h_Buffer == NULL || h_OutputCPU == NULL){
        printf("Something went wrong wille malloc in CPU\n");
    }
    
    printf("Memmory allocation for host arrays: COMPLETED \n");
    
    cudaMallocManaged((void**)&d_Filter,FILTER_LENGTH * sizeof(myDataType));
    cudaMallocManaged((void**)&d_Input,imageH * imageW * sizeof(myDataType));
    cudaMallocManaged((void**)&d_Buffer,imageH * imageW * sizeof(myDataType));
    cudaMallocManaged((void**)&d_OutputGPU,imageH * imageW * sizeof(myDataType));
    
    cudaCheckError();
    
    printf("Memmory allocation for device arrays: COMPLETED \n");
    
    
    
    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.
    
    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (myDataType)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (myDataType)rand() / ((myDataType)RAND_MAX / 255) + (myDataType)rand() / (myDataType)RAND_MAX;
    }
    
    printf("initialization of host arrays: COMPLETED \n");
    
    cudaMemcpy(d_Filter, h_Filter,FILTER_LENGTH * sizeof(myDataType),cudaMemcpyHostToDevice);
    cudaMemcpy(d_Input, h_Input,imageH * imageW * sizeof(myDataType),cudaMemcpyHostToDevice);
    
    cudaCheckError();
    
    printf("initialization of device arrays: COMPLETED \n\n");
    
    printf("GPU computation...\n");
    
    convolutionRowGPU<<<1,threads>>>(d_Buffer,d_Input,d_Filter,imageW,imageH,filter_radius);
    
    cudaCheckError();
    
    cudaDeviceSynchronize();
    
    convolutionColumnGPU<<<1,threads>>>(d_OutputGPU,d_Buffer,d_Filter,imageW,imageH,filter_radius);
    
    cudaCheckError();
    
    printf("GPU computation : COMPLETED\n\n");
    
    cudaMemcpy(h_OutputGPU,d_OutputGPU,imageH * imageW * sizeof(myDataType),cudaMemcpyDeviceToHost);
    
    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");
    
    start = clock();
    
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
    
    end = clock();
    
    timing = ((double) (end - start)) / CLOCKS_PER_SEC;
    
    printf("CPU computation : COMPLETED in time:%10.8f\n",timing);

    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas 
    
    
    printf("\nCPU computations == GPU computation?\n");
    for (i = 0; i < imageW * imageH; i++) {
        if(h_OutputGPU[i] > h_OutputCPU[i] + accuracy || h_OutputGPU[i] < h_OutputCPU[i] - accuracy){
            printf("CPU computations == GPU computation : FALSE line:%d difrence:%f \nExitting program after Memmory Free...\n",i,h_OutputGPU[i]-h_OutputCPU[i]);
            //count++;
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
            cudaFree(d_Filter);
            
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
    cudaFree(d_Filter);
    
    cudaDeviceReset();


    return 0;
}
