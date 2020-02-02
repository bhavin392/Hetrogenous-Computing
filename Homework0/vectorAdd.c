#include<stdio.h>
#include<stdlib.h>
#include <sys/time.h>
struct timeval stop, start,start1,stop1;
void add(int a[], int b[],int N);

int main(int argc, char* argv[]){

    gettimeofday(&start1, NULL);

    if(argc<2){
        printf("Please enter the value of N(number of elements)\n");
    }else{
        int i,N;
        N=atoi(argv[1]);
        N=N*1024;   
        int a[N],b[N];
        for(i=0;i<=N;i++){
            a[i]=rand()%1000;
            b[i]=rand()%1000;
        }
        gettimeofday(&start, NULL);
        add(a,b,N);
        gettimeofday(&stop, NULL);
        printf("took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
    }
    gettimeofday(&stop1, NULL);
    printf("Total took %lu us\n", (stop1.tv_sec - start1.tv_sec) * 1000000 + stop1.tv_usec - start1.tv_usec);
    return 0;
}

void add(int a[], int b[], int N){
    int c,i;
    for(i=0;i<=N;i++){
        c=a[i]+b[i];
        // printf("%d + %d = %d\n",a[i],b[i],c);
    }
}