#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>

#define A 10000
#define B 10000
#define k 10000

struct timeval  start, stop,start1,stop1,start2,stop2;

__global__ void matrix_mul(int *a, int *b, int *c){

    int row = (blockIdx.y*blockDim.y + threadIdx.y);
    int col = (blockIdx.x*blockDim.x + threadIdx.x);
    int temp=0;
    for(int i = 0;i < k;i++){
		temp += a[row * k + i] * b[i * k + col]; //add and multiply
	}
	c[row * k + col] = temp;
}

void random_int(int *a, int b){
    for(int i=0;i<b;i++){
        a[i] = rand()%100;
    }
}

int main(void){
    int *a,*b,*c;
    int *d_a,*d_b,*d_c;
    int matrix_size= A* B * sizeof(int);

    

    cudaMalloc((void **)&d_a,matrix_size);
    cudaMalloc((void **)&d_b,matrix_size);
    cudaMalloc((void **)&d_c,matrix_size);

    a=(int*)malloc(matrix_size);
    random_int(a,A);
    b=(int*)malloc(matrix_size);
    random_int(b,B);
    c=(int*)malloc(matrix_size);
    gettimeofday(&start1,NULL);
    cudaMemcpy(d_a,a,matrix_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,matrix_size,cudaMemcpyHostToDevice);
    gettimeofday(&stop1,NULL);

    dim3 threadBlocks = dim3((int) std::ceil( (double) k/16 ),(int) std::ceil ( (double) k/16),1);
    dim3 threadsPerBlock = dim3(16,16,1);

   
    gettimeofday(&start,NULL);
    matrix_mul<<<threadBlocks,threadsPerBlock>>>(d_a,d_b,d_c);
    gettimeofday(&stop,NULL);

    printf("GPU Execution time of kernel without memory Transfer: %lu us\n", (stop.tv_sec-start.tv_sec)+ (stop.tv_usec-start.tv_usec) * 1e-6);
    float kernelTime=(stop.tv_sec-start.tv_sec)+ (stop.tv_usec-start.tv_usec) * 1e-6;
	

	gettimeofday(&start2,NULL);
	cudaMemcpy(c, d_c, matrix_size, cudaMemcpyDeviceToHost);
    gettimeofday(&stop2,NULL);

    float htod=(stop1.tv_sec-start1.tv_sec)+ (stop1.tv_usec-start1.tv_usec) * 1e-6;
    float dtoh=(stop2.tv_sec-start2.tv_sec)+ (stop2.tv_usec-start2.tv_usec) * 1e-6;

    float totaltime=htod+dtoh+kernelTime;
    printf("GPU Execution Time of kernel with memory Transfer: %lu us\n",totaltime);
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	return 0;

}