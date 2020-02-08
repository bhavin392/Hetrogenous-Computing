/* ECGR 6090 Heterogeneous Computing Homework0
Problem 2- 1D stencil using GPU
Written by Bhavin Thakar - 801151488
*/


//  To execute the program type: ./1DstencilGPU


#include<stdio.h>
#include <sys/time.h>
#include<stdlib.h>

struct timeval stop, start,start1,stop1,start2, stop2;

#define R 16 // Define Radius
#define B 128 // Define Thread block size
#define N 10000 // Define number of elements in array


// Kernel Function
__global__ void stencil1d(int *in, int *out){
    int gindex=threadIdx.x+(blockIdx.x*blockDim.x) + R;
    int result=0;
    for (int offset = -R; offset <= R ; offset++){
        result += in[gindex + offset];
    }
    out[gindex-R]=result;
}

// Function to generate random numbers and adding it to an integer array
void random(int *a, int n ){
    int i;
    for (i = 0; i <=n+1; ++i)
     a[i] = rand()%100; // generating integer values from 0 to 100
     
 }

int main(void){
    int n;
    int *c_in, *c_out; // Declaring integer array for CPU
    int size= N*sizeof(int); // SIZE = N*4bytes(int)
    n=N+2*R;

    // Allocating memory for  CPU integer array
    c_in=(int*)malloc(n*size);
    c_out=(int*)malloc(N*size);

    gettimeofday(&start1, NULL);
    random(c_in,n); // Calling random function
    

    int *d_in,*d_out; // Declaring integer array for GPU

    // Allocating memory for GPU integer array
    cudaMalloc(&d_in,n*size);
    cudaMalloc(&d_out,N*size);

    // Copying inputs from CPU to GPU
    cudaMemcpy(d_in,c_in,n*size,cudaMemcpyHostToDevice);
    gettimeofday(&stop1, NULL);

    gettimeofday(&start, NULL);
    stencil1d<<<(N/B-1)/B,B>>>(d_in,d_out); // Calling kernel function
    gettimeofday(&stop, NULL);
    cudaDeviceSynchronize(); // Checking if all the streams is completed successfully
    printf("Execution time of kernel: %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
    gettimeofday(&start2, NULL);

    // Copying results from GPU to CPU
    cudaMemcpy(c_out,d_out,n*size,cudaMemcpyDeviceToHost);
    gettimeofday(&stop2, NULL);

    // Calculating the Overhead data transfer execution time
    unsigned int i;
    i=(stop1.tv_sec - start1.tv_sec )* 1000000 + (stop1.tv_usec - start1.tv_usec);
    i=i+((stop2.tv_sec - start2.tv_sec )* 1000000 + (stop2.tv_usec - start2.tv_usec));
    printf("Execution time for data transfer: %lu us\n", i);


    // Free resources
    free(c_in);
    free(c_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;




}