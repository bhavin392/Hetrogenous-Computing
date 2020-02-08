/* ECGR 6090 Heterogeneous Computing Homework0
Problem 2- 1D stencil using CPU
Written by Bhavin Thakar - 801151488
*/

// To execute this program type: ./1dstencil R N (R = radius  and N = Number of Elements in array) 
// (N=1 for 1K ;10 for 10K; 100 for 100K; 1000 for 1M; 10000 for 10M)
// For example: ./1dstencil 2 10

#include<stdio.h>
#include<sys/time.h>
#include<math.h>
#include<stdlib.h>

struct timeval  start, stop, start1, stop1;

// Kernel Function
void stencil(int *c, int *d, int r, int n){ 
    int offset, k;
    int index=0;
    for (k=0;k<n+r;k++){ // Looping through values to find the index and adding the neighbouring elements 
        for(offset=-r;offset<=r;offset++){
            c[k-r]=c[k-r]+d[index+offset+r];
        }
    }
}

int main(int argc, char *argv[]){
    if(argc<2){ // Checking for the valid number of arguments
        printf("Please enter the value of N(number of elements)\n");
    }
    else{
        int R = atoi(argv[1]); // Convert the string argument into integer
        int N = atoi(argv[2]);
        N=N*1024;
        int i,j, *a,*b;
        // Allocating memory to integer array
        a=(int *)malloc(N * sizeof(int));
        b=(int *)malloc(N * sizeof(int));

        // Storing the random value into input integer array
        for(i=0;i<N;i++){
            a[i]=rand()%1000;
        }
        for (j = 0; j<N;j++)
	    {
		    b[j] = 0;		
	    }
        gettimeofday(&start, NULL);
        stencil(b,a,R,N); // Calling the Kernel Function 
        gettimeofday(&stop, NULL);
        printf("Execution time of kernel: %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
    }
    return 0;

}