// This will apply the sobel filter and return the PSNR between the golden sobel and the produced sobel
// sobelized image
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>

#define SIZE	4096
#define INPUT_FILE	"../Data/input.grey"
#define OUTPUT_FILE	"../Data/output_sobel.grey"
#define GOLDEN_FILE	"../Data/golden.grey"
#define UNROLLING_TIMES 1//Amount to add in for loop When unrolling

/* The horizontal and vertical operators to be used in the sobel filter */
const char horiz_operator[3][3] = {{-1, 0, 1}, 
                             {-2, 0, 2}, 
                             {-1, 0, 1}};
const char vert_operator[3][3] = {{1, 2, 1}, 
                            {0, 0, 0}, 
                            {-1, -2, -1}};

double sobel(unsigned char *input, unsigned char *output, unsigned char *golden);
int convolution2D(int posy, int posx, const unsigned char *input, const char operator[][3]);

/* The arrays holding the input image, the output image and the output used *
 * as golden standard. The luminosity (intensity) of each pixel in the      *
 * grayscale image is represented by a value between 0 and 255 (an unsigned *
 * character). The arrays (and the files) contain these values in row-major *
 * order (element after element within each row and row after row. 			*/
unsigned char input[SIZE*SIZE], output[SIZE*SIZE], golden[SIZE*SIZE];


/* Implement a 2D convolution of the matrix with the operator */
/* posy and posx correspond to the vertical and horizontal disposition of the *
 * pixel we process in the original image, input is the input image and       *
 * operator the operator we apply (horizontal or vertical). The function ret. *
 * value is the convolution of the operator with the neighboring pixels of the*
 * pixel we process.														  */


/* The main computational function of the program. The input, output and *
 * golden arguments are pointers to the arrays used to store the input   *
 * image, the output produced by the algorithm and the output used as    *
 * golden standard for the comparisons.									 */
double sobel(unsigned char *input, unsigned char *output, unsigned char *golden)
{
	double PSNR = 0, t;
    	int conv_horiz,conv_vert;
    	register int array_index;
    	register int i,j,k;
	unsigned int p;
	int res;
	struct timespec  tv1, tv2;
	FILE *f_in, *f_out, *f_golden;

	/* The first and last row of the output array, as well as the first  *
	* and last element of each column are not going to be filled by the *
	* algorithm, therefore make sure to initialize them with 0s.		 */
	memset(output, 0, SIZE*sizeof(unsigned char));
	memset(&output[SIZE*(SIZE-1)], 0, SIZE*sizeof(unsigned char));
	for (i = 1; i < SIZE-1; i++) {
		output[i*SIZE] = 0;
		output[i*SIZE + SIZE - 1] = 0;
	}

	/* Open the input, output, golden files, read the input and golden    *
     * and store them to the corresponding arrays.						  */
	f_in = fopen(INPUT_FILE, "r");
	if (f_in == NULL) {
		printf("File " INPUT_FILE " not found\n");
		exit(1);
	}
  
	f_out = fopen(OUTPUT_FILE, "wb");
	if (f_out == NULL) {
		printf("File " OUTPUT_FILE " could not be created\n");
		fclose(f_in);
		exit(1);
	}  
  
	f_golden = fopen(GOLDEN_FILE, "r");
	if (f_golden == NULL) {
		printf("File " GOLDEN_FILE " not found\n");
		fclose(f_in);
		fclose(f_out);
		exit(1);
	}    

	fread(input, sizeof(unsigned char), SIZE*SIZE, f_in);
	fread(golden, sizeof(unsigned char), SIZE*SIZE, f_golden);
	fclose(f_in);
	fclose(f_golden);
  
	/* This is the main computation. Get the starting time. */
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
	/* For each pixel of the output image */
	
    
    k = SIZE;
    // Loop interchange #1 START -> performance speedup
    	for (i=1; i<SIZE-1; i+=1) {
		//Loop unrolling in this loop doesnt improve performance
        	//in contrast loop unrolling is x2.5 times slower if UNROLLING_TIMES >= 2
		for (j=1; j<SIZE-1; j+=1 ) {
           	//Common subexpression Elimination #1
            	//Strength reduction we removed i*SIZE with k
            		array_index = k + j;
			conv_horiz=0;
        		conv_vert=0;
			/* Apply the sobel filter and calculate the magnitude *
			 * of the derivative.								  */
			/* Function inlining the Conv2D function */
			conv_horiz += input[array_index - SIZE - 1] * horiz_operator[0][0];
			conv_horiz += input[array_index - SIZE] * horiz_operator[0][1];
			conv_horiz += input[array_index - SIZE + 1] * horiz_operator[0][2];
		
			conv_horiz += input[array_index - 1] * horiz_operator[1][0];
			conv_horiz += input[array_index] * horiz_operator[1][1];
			conv_horiz += input[array_index + 1] * horiz_operator[1][2];
		
			conv_horiz += input[array_index + SIZE - 1] * horiz_operator[2][0];
			conv_horiz += input[array_index + SIZE] * horiz_operator[2][1];
			conv_horiz += input[array_index + SIZE + 1] * horiz_operator[2][2];
			//
			conv_vert += input[array_index - SIZE - 1] * vert_operator[0][0];
			conv_vert += input[array_index - SIZE] * vert_operator[0][1];
			conv_vert += input[array_index - SIZE + 1] * vert_operator[0][2];
		
			conv_vert += input[array_index - 1] * vert_operator[1][0];
			conv_vert += input[array_index] * vert_operator[1][1];
			conv_vert += input[array_index + 1] * vert_operator[1][2];
		
			conv_vert += input[array_index + SIZE - 1] * vert_operator[2][0];
			conv_vert += input[array_index + SIZE] * vert_operator[2][1];
			conv_vert += input[array_index + SIZE + 1] * vert_operator[2][2];
            		p = conv_horiz*conv_horiz + conv_vert*conv_vert; //Function inlining replacing pow() #1
			res = (int)sqrt(p);
			/* If the resulting value is greater than 255, clip it *
			 * to 255.											   */
			if (res > 255)
				output[array_index] = 255;
			else
				output[array_index] = (unsigned char)res;
		}
		k += SIZE;
	}
	// Loop interchange #1 END

	/* Now run through the output and the golden output to calculate *
	 * the MSE and then the PSNR.									 */

    /* Loop interchange #2 performed worse, so commented
    for (j=1; j<SIZE-1; j++) {
		for ( i=1; i<SIZE-1; i++ ) {
			t = pow((output[i*SIZE+j] - golden[i*SIZE+j]),2);
			PSNR += t;
		}
	}
	*/
    
    	k = SIZE;
    	for (i=1; i<SIZE-1; i++) {
		for ( j=1; j<SIZE-1; j+=UNROLLING_TIMES) {
            	//Strength reduction we removed i*SIZE with k
            	//Function inlining replacing pow() #2
			t = (output[k+j] - golden[k+j])*(output[k+j] - golden[k+j]);
			PSNR += t;
		}
		k+=SIZE;
	}
    	PSNR /= (double)SIZE*SIZE;
	PSNR = 10*log10(65536/PSNR);

	/* This is the end of the main computation. Take the end time,  *
	 * calculate the duration of the computation and report it. 	*/
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

	printf ("Total time = %10g seconds\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec));

  
	/* Write the output file */
	fwrite(output, sizeof(unsigned char), SIZE*SIZE, f_out);
	fclose(f_out);
  
	return PSNR;
}


int main(int argc, char* argv[])
{
	double PSNR;
	PSNR = sobel(input, output, golden);
	printf("PSNR of original Sobel and computed Sobel image: %g\n", PSNR);
	printf("A visualization of the sobel filter can be found at " OUTPUT_FILE ", or you can run 'make image' to get the jpg\n");

	return 0;
}

