#include <stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

#define N 10000
#define M 10000
#define K 10000

__global__ void matrix_mul_coal(float *a, float *b, float *c)	{

	int row = blockIdx.y* blockDim.y+ threadIdx.y;		
	int col = blockIdx.x* blockDim.x+ threadIdx.x;		
	float temp = 0.0; //calculate sum
	for (int k = 0; k < K; k++)
		{
			temp += a[row * K + k] + b[k * K + col]; //add and multiply
		}
		
	c[row * K + col] = temp; //final c matrix
}

//Function to initialize matrices with random values
void randomInit (float *data, int size)	
{
	for (int i = 0; i <  size; i++) 
		for (int j = 0; j < size; j++) 
			*(data + i * size + j) = rand() % 1024; 
}



int main(void)	
{
	
	float *a, *b, *c, *bt; //CPU copies
	float *d_a, *d_b, *d_c;  //GPU copies 
	int matrix_size = N * M * sizeof(float);
	
	cudaEvent_t start, stop,start1,stop1,start2,stop2;
	float time,time1,time2;

	cudaEventCreate(&start);
	cudaEventCreate(&start1);
	cudaEventCreate(&start2);
	cudaEventCreate(&stop);
	cudaEventCreate(&stop1);
	cudaEventCreate(&stop2);

	//Allocate CPU memory
	a = (float *) malloc(matrix_size);	randomInit(a, N);
	b = (float *) malloc(matrix_size);	randomInit(b, M);
	bt = (float *) malloc(matrix_size);
	c = (float *) malloc(matrix_size);

	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			*(bt + i * M + j) = *(b + j * M + i);


	//Allocate GPU memory 
	cudaMalloc((void **) &d_a, matrix_size);
	cudaMalloc((void **) &d_b, matrix_size);
	cudaMalloc((void **) &d_c, matrix_size);

	cudaEventRecord( start1, 0 );
	//Copy from CPU memory to GPU memory
	cudaMemcpy( d_a, a, matrix_size, cudaMemcpyHostToDevice);
	cudaMemcpy( d_b, bt, matrix_size, cudaMemcpyHostToDevice);
	cudaEventRecord( stop1, 0 );
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime( &time1, start1, stop1 );
	cudaEventDestroy( start1);
	cudaEventDestroy( stop1);
	
	//Set thread and grid dimensions
	dim3 tBlock(16, 16);
	dim3 Grid((N + 16 - 1)/tBlock.x, (M + 16 - 1)/tBlock.y);

	cudaEventRecord( start, 0 );

	//Call kernels
	matrix_mul_coal<<< Grid, tBlock >>> (d_a, d_b, d_c);

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &time, start, stop );
	printf("GPU Execution Time without memory transfer= %f\n",time);
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	
	//Copy from device to host
	cudaEventRecord( start2, 0 );
	cudaMemcpy( c, d_c, matrix_size, cudaMemcpyDeviceToHost);
	cudaEventRecord( stop2, 0 );
	cudaEventElapsedTime( &time2, start2, stop2 );
	cudaEventDestroy( start2 );
	cudaEventDestroy( stop2 );


	float tTime=time+time1+time2;
	printf("GPU Execution Time with memory transfer: %f\n",tTime);
	//free cpu and gpu memory
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	return 0;

}