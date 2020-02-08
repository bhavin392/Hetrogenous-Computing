/* ECGR 6090 Heterogeneous Computing Homework0
Problem 2- 1D stencil using GPU and shared memory
Written by Bhavin Thakar - 801151488
*/

// To execute the program type: ./1dstencilsharedmemory 


#include<stdio.h>
#include <sys/time.h>

struct timeval stop, start,start1,stop1;

#define R 4 // Defining radius as 4
#define B 128 // Defining Thread Block Size as 128
#define N 1000000 // Defining Number of Elements as 1M


// Kernel Function
__global__ void stencil1d(int *in, int *out){
    __shared__ int temp[B + 2 * R]; // Declaring a shared integer array 
	int gindex = threadIdx.x + blockIdx.x * blockDim.x; 
	int lindex = threadIdx.x + R;
	temp[lindex] = in[gindex]; //storing in shared memory
	
	if (threadIdx.x < R) 
	{
	temp[lindex - R] = in[gindex - R];
	temp[lindex + B] = in[gindex + B];
	}
	__syncthreads();
	int result = 0;
	for (int offset = -R ; offset <= R ; offset++)
	{
		result += temp[lindex + offset];
	}
// Store the result
	out[gindex] = result;
}

// random function to generate random numbers
void random(int *a, int n ){
    int i;
    for (i = 0; i <=n+1; ++i)
     a[i] = rand()%100;
     
 }

int main(void){
    int n;
    int *c_in, *c_out; // integer aray for CPU
    int size= N*sizeof(int);
    n=N+2*R;
    // Allocating memory for CPU integer array 
    c_in=(int*)malloc(n*size);
    c_out=(int*)malloc(N*size);

    random(c_in,n); // Calling random function
    

    int *d_in,*d_out; //integer array for GPU
    //Allocating memory for GPU integer array
    cudaMalloc(&d_in,n*size);
    cudaMalloc(&d_out,N*size);

    // Copying input from CPU to GPU
    cudaMemcpy(d_in,c_in,n*size,cudaMemcpyHostToDevice);

    gettimeofday(&start, NULL);
    stencil1d<<<(N/B-1)/B,B>>>(d_in,d_out); //Calling Kernel Function
    gettimeofday(&stop, NULL);
    cudaDeviceSynchronize(); // Check if streams are completed
    printf("Execution time of kernel: %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
    // Copying back the results from GPU to CPU
    cudaMemcpy(c_out,d_out,n*size,cudaMemcpyDeviceToHost);

    // Free resources
    free(c_in);
    free(c_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;




}