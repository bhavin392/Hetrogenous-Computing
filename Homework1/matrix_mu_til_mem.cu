#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#define N 10000
#define M 10000
#define K 10000
#define tile_size 16


__global__ void matrix_mul_shared_coal(int *a, int *b, int *c) {
	
	__shared__ int a_tile[tile_size][tile_size]; 		//define shared memory tile for matrix a
	__shared__ int b_tile[tile_size][tile_size];		//define shared memory tile for matrix b

    int row = blockIdx.y * tile_size + threadIdx.y;	
	int col = blockIdx.x * tile_size + threadIdx.x;	

	int temp = 0;
    int tileIdx;

	//Load one tile into shared memory
	for (int s = 0; s < gridDim.x; s++) {
		tileIdx = row * K + s * tile_size + threadIdx.x;

		if(tileIdx >= K*K)
			a_tile[threadIdx.y][threadIdx.x] = 0;	//check if K is divisible by tile size for a_tile
		else
			a_tile[threadIdx.y][threadIdx.x] = a[tileIdx];

		tileIdx = (s * tile_size + threadIdx.y) * K + col;

		if(tileIdx >= K*K)
			b_tile[threadIdx.y][threadIdx.x] = 0; 	//check if K is divisible by tile size for b_tile
		else
			b_tile[threadIdx.y][threadIdx.x] = b[tileIdx];
			
		__syncthreads(); 
		for (int j = 0; j < tile_size; j++)
			temp += a_tile[threadIdx.y][j] * b_tile[j][threadIdx.x]; //perform addition and multiply

		__syncthreads();  
		
	}
	
	if(row < K && col < K) 	
		c[row * K + col] = temp; //store result 
    	
}

//Function to initialize matrices with random values
void randomInit (int *data, int size)	
{
	for (int i = 0; i <  size; i++) 
		for (int j = 0; j < size; j++) 
			*(data + i * size + j) = rand() % 1024; 
}

int main()	{
	
	int *a, *b, *c, *bt; //CPU copies
	int *d_a, *d_b, *d_c;  //GPU copies 
	int matrix_size = N * M * sizeof(int);
	
	cudaEvent_t start, stop,start1,stop1, start2,stop2;
	int time,time1,time2;

	//Start the cuda timer
	cudaEventCreate(&start);
	cudaEventCreate(&start1);
	cudaEventCreate(&start2);
	cudaEventCreate(&stop);
	cudaEventCreate(&stop1);
	cudaEventCreate(&stop2);

	//Allocate CPU memory
	a = (int *) malloc(matrix_size);	randomInit(a, N);
	b = (int *) malloc(matrix_size);	randomInit(b, M);
	bt = (int *) malloc(matrix_size);
	c = (int *) malloc(matrix_size);
	
	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			*(bt + i * M + j) = *(b + j * M + i);
	

	//Allocate GPU memory 
	cudaMalloc((void **) &d_a, matrix_size);
	cudaMalloc((void **) &d_b, matrix_size);
	cudaMalloc((void **) &d_c, matrix_size);

	//Copy from CPU memory to GPU memory
	cudaEventRecord( start1, 0 );

	cudaMemcpy( d_a, a, matrix_size, cudaMemcpyHostToDevice);
	cudaMemcpy( d_b, bt, matrix_size, cudaMemcpyHostToDevice);
	cudaEventRecord( stop1, 0 );
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime( &time1, start1, stop1 );
	cudaEventDestroy( start1 );
	cudaEventDestroy( stop1 );

	//Set thread and grid dimensions
	dim3 tBlock(16, 16);
	dim3 Grid((N + 16 - 1)/tBlock.x, (M + 16 - 1)/tBlock.y);

	cudaEventRecord( start, 0 );

	//Call kernels
	matrix_mul_shared_coal<<< Grid,tBlock >>> (d_a,d_b,d_c);

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf("GPU Execution Time without Memory Transfer= %f\n",time);

	//Copy from device to host
	cudaEventRecord( start2, 0 );
	cudaMemcpy( c, d_c, matrix_size, cudaMemcpyDeviceToHost);
	cudaEventRecord( stop2, 0 );
	cudaEventSynchronize(stop2);
	cudaEventElapsedTime( &time2, start2, stop2 );
	cudaEventDestroy( start2 );
	cudaEventDestroy( stop2 );

	float tTime=time+time1+time2;
	printf("GPU Execution Time with Memory Transfer: %f\n",tTime);
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	return 0;

}