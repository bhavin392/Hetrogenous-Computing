#define N (1024*1024)
// #define THREADS_PER_BLOCK 512


__global__ void add(int *a, int *b, int *c, int n){
    int index =threadIdx.x + blockIdx.x * blockDim.x;
    if(intex<n){
        c[index]=a[index]+b[index];
    }
}

int main(int argc, char * argv[]){
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size= N*sizeof(int);

    cudaMalloc((void ** )&d_a,size);
    cudaMalloc((void **)&d_b,size);
    cudaMalloc((void **)&d_c,size);

    a=(int *)malloc(size); random_ints(a,N);
    b=(int *)malloc(size); random_ints(b,N);
    c=(int *)malloc(size);
}

