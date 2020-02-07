#include<stdio.h>
#include <sys/time.h>
struct timeval stop, start,start1,stop1;
#define N (2048*2048)
#define THREADS_PER_BLOCK 1024


__global__ void add(int *a, int *b, int *c, int n){
    int index =threadIdx.x + blockIdx.x * blockDim.x;
    if(index<n){
        c[index]=a[index]+b[index];
    }
}
void random(int *a, int n ){
   int i;
   for (i = 0; i < n; ++i)
    a[i] = rand()%100;
    
}

int main(){
    int *a, *b, *c;
    int size= N*sizeof(int);
    a=(int*)malloc(size);
    b=(int*)malloc(size);
    c=(int*)malloc(size);

    int *d_a, *d_b, *d_c;

    cudaMalloc((void ** )&d_a,size);
    cudaMalloc((void **)&d_b,size);
    cudaMalloc((void **)&d_c,size);  

    random(a,N);
    random(b,N);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    // Launch add() kernel on GPU
    gettimeofday(&start, NULL);
    add<<<(N + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c,N);
    gettimeofday(&stop, NULL);
    printf("took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
    cudaDeviceSynchronize();
    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    // Cleanup
    free(a); 
    free(b); 
    free(c);
    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_c);
    return 0;
}
