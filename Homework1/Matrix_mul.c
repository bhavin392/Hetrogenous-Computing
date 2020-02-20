#include <stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>

struct timeval  start, stop;

#define A 100
#define B 100
#define k 100

void random_int(int *a, int b){
    for(int i=0;i<b;i++){
        a[i] = rand()%100;
    }
}

void matrix_mul(int *a,int *b,int *c){
    int offset,*m;
    int index =0;
    for(int i=0;i<k;i++){
        for(int j=0;j<k;j++){
            c[i]=c[i]+a[i]*b[j];
        }
    }

}

int main(void){
    int *a,*b,*c;
    int matrix_size= A*B*sizeof(int);

    
    

    a=(int*)malloc(matrix_size);
    random_int(a,A);
    b=(int*)malloc(matrix_size);
    random_int(b,B);
    c=(int*)malloc(matrix_size);

    gettimeofday(&start,NULL);
    matrix_mul(a,b,c);
	gettimeofday(&stop,NULL);
    printf("CPU Execution time of kernel: %lu ms\n", (stop.tv_sec-start.tv_sec)+ (stop.tv_usec-start.tv_usec) * 1e-6);
    free(a);
    free(b);
    free(c);


    return 0;

}
