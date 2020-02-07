#include<stdio.h>
#include <sys/time.h>

struct timeval stop, start,start1,stop1;

#define R 2
#define B 128
#define N 1000

__global__ void stencil1d(int *in, int *out){
    __shared__ int temp[B + 2 * R];
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

void random(int *a, int n ){
    int i;
    for (i = 0; i <=n+1; ++i)
     a[i] = rand()%100;
     
 }

int main(void){
    int n;
    int *c_in, *c_out;
    int size= N*sizeof(int);
    n=N+2*R;
    c_in=(int*)malloc(n*size);
    c_out=(int*)malloc(N*size);
    random(c_in,n);
    

    int *d_in,*d_out;
    cudaMalloc(&d_in,n*size);
    cudaMalloc(&d_out,N*size);

    cudaMemcpy(d_in,c_in,n*size,cudaMemcpyHostToDevice);

    gettimeofday(&start, NULL);
    stencil1d<<<(N/B-1)/B,B>>>(d_in,d_out);
    gettimeofday(&stop, NULL);
    printf("took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
    cudaDeviceSynchronize();
    cudaMemcpy(c_out,d_out,n*size,cudaMemcpyDeviceToHost);

    free(c_in);
    free(c_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;




}