/* ECGR 6090 Heterogeneous Computing Homework0
Problem 1 - Vector add on gpu
Written by Bhavin Thakar - 801151488
*/

// To execute the program type: ./vectorAddGPU


#include<stdio.h>
#include <sys/time.h>
struct timeval stop, start,start1,stop1,start2,stop2;
#define N (10000*1024) //Defining N
#define THREADS_PER_BLOCK 1024 

//Kernel Function 
__global__ void add(int *a, int *b, int *c, int n){ 
    int index =threadIdx.x + blockIdx.x * blockDim.x;
    if(index<n){
        c[index]=a[index]+b[index];
    }
}

// function to generate random number and adding it to an array
void random(int *a, int n ){
   int i;
   for (i = 0; i < n; ++i)
    a[i] = rand()%100; // Generate random integer values from 0 to 100
    
}

int main(){
    int *a, *b, *c; // Declaring the integer array
    int size= N*sizeof(int); // Declaring size as size of N * 4 bytes(int)

    //Allocating memory in CPU
    a=(int*)malloc(size);
    b=(int*)malloc(size);
    c=(int*)malloc(size);

    // Declaring variables for GPU
    int *d_a, *d_b, *d_c;

    //  Allocating memory in GPU
    cudaMalloc((void ** )&d_a,size);
    cudaMalloc((void **)&d_b,size);
    cudaMalloc((void **)&d_c,size);  
    
    
    gettimeofday(&start1, NULL);
    // Calling the random function
    random(a,N); 
    random(b,N);

    // Copying the CPU array to GPU device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    gettimeofday(&stop1, NULL); 

    // Launch add() kernel on GPU
    gettimeofday(&start, NULL);
    add<<<(N + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c,N);
    gettimeofday(&stop, NULL);
    cudaDeviceSynchronize(); // To ensure that every stream is finished
    printf("Execution time for kernel:  %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
    // Copy result back to host
    gettimeofday(&start2, NULL);
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    gettimeofday(&stop2, NULL);

    // Calculating execution time for Host to Device and Device to Host Data Transfer
    unsigned int i;
    i=(stop1.tv_sec - start1.tv_sec )* 1000000 + (stop1.tv_usec - start1.tv_usec);
    i=i+((stop2.tv_sec - start2.tv_sec )* 1000000 + (stop2.tv_usec - start2.tv_usec));
    printf("Execution time for data transfer: %lu us\n", i);

    // Freeing up the resources
    free(a); 
    free(b); 
    free(c);
    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_c);
    return 0;
}
