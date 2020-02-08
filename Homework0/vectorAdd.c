/* ECGR 6090 Heterogeneous Computing Homework0
Problem 1 - Vector add on cpu
Written by Bhavin Thakar - 801151488
*/


//To execute this program type : ./vectorAdd N (N=1 for 1K ;10 for 10K; 100 for 100K; 1000 for 1M; 10000 for 10M)
//  example: ./vectorAdd 10


#include<stdio.h>
#include<stdlib.h>
#include <sys/time.h>
struct timeval stop, start,start1,stop1;
void add(int *a, int *b, int *c, int N);
int *a, *b ,*c; // global integer array a,b,c

int main(int argc,char* argv[]){
    if(argc<2){
        printf("Please enter the value of N(number of elements)\n"); // print if the shell cant find an agument
    }
    else
    {
        int i,N=0;
        N=atoi(argv[1]); // convert the argument string to integer
        N=N*1024; // Multiply N by 1024

        // Allocating memory to integer arrays
        a=(int *)malloc(N * sizeof(int)); 
        b=(int *)malloc(N * sizeof(int));
        c=(int *)malloc(N * sizeof(int));

        gettimeofday(&start1, NULL); //start the timer
        for(i=0;i<N;i++){
            a[i]=rand()%1000; // put a random value from 0 to 1000 in integer array
            b[i]=rand()%1000;
        }
        gettimeofday(&stop1, NULL); // stop the timer
        printf("Execution time for data transfer: %lu us\n", (stop1.tv_sec - start1.tv_sec) * 1000000 + stop1.tv_usec - start1.tv_usec); //Get time in usec
        gettimeofday(&start, NULL);
        add(a,b,c,N); //Calling the kernel function add
        gettimeofday(&stop, NULL);
        printf("Execution time of kernel: %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
        
        // Free the memory 
        free(a);
        free(b);
        free(c);
    }
    return 0;
}

void add(int *a, int *b,int *c, int N){
    int i;
    for(i=0;i<=N;i++){
        c[i]=a[i]+b[i];
        // printf("%d + %d = %d\n",a[i],b[i],c[i]);
    }
}