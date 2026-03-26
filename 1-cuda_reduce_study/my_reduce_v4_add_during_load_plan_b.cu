#include<cstdio>
#include<cuda.h>
#include<stdlib.h>
#include<cuda_runtime.h>

// 保持 block 数量不变，减少 每个block中的线程数
#define THREAD_PRE_BLOCK (256/2)
__global__ void reduce(float *d_input, float *d_output) {
    // 先把数据读到 共享内存里面，这样一个block里面都是共享的
    __shared__ float shared[THREAD_PRE_BLOCK];
    // 1.获取每个block要处理数据的起始数组
    float *input_start = d_input + blockIdx.x * blockDim.x * 2 ;
    shared[threadIdx.x] = input_start[threadIdx.x] + input_start[threadIdx.x + blockDim.x];

    // 同步：确保所有线程都完成数据加载后才继续执行
    __syncthreads();
    // 2. 每个线程处理两个元素
    for(int i = blockDim.x / 2; i > 0; i >>= 1) {
        // 解决 bank conflict
        if(threadIdx.x < i) {
            shared[threadIdx.x] += shared[threadIdx.x + i];
        }
        // if(threadIdx.x % (i * 2) == 0) {
        //     shared[threadIdx.x] += shared[threadIdx.x + i];
        // }
        __syncthreads();
    }
    // 3. 一个线程写回
    if(threadIdx.x == 0) {
        d_output[blockIdx.x] = shared[0];
    }
}

bool check(const float * a, const float *b, int N) {

    for(int i = 0; i < N; i++) {
        
        if(abs(a[i] - b[i]) > 0.005) {
            return false;
        }
    }
    return true;
}

int main() {
    printf("my_reduce_v4_add_during_load_plan_b\n");
    const int N = 32 * 1024 * 1024;
    // cpu
    int block_num = N / THREAD_PRE_BLOCK  / 2;
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_result = (float*)malloc(block_num * sizeof(float));
    float *h_gpu_result = (float*)malloc(block_num * sizeof(float));
    // init data
    for(int i = 0; i < N; ++i) {
        h_input[i] = 2.0 * (float)drand48() - 1.0;
    }
    // cpu calc
    for(int i = 0; i < block_num; i++) {
        float cur = 0.0f;
        for(int j = 0; j < THREAD_PRE_BLOCK * 2; ++j) {
            cur += h_input[i * THREAD_PRE_BLOCK * 2 + j];
        }
        h_result[i] = cur;
    }
    // gpu
    float *d_input, *d_result;
    cudaMalloc((void **)&d_input,N * sizeof(float));
    cudaMalloc((void **)&d_result,block_num * sizeof(float));

    // 从 cpu 拷贝到 gpu
    cudaMemcpy(d_input,h_input,N * sizeof(float),cudaMemcpyHostToDevice);

    // 配置 线程
    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PRE_BLOCK,1);

    // for(int i = 0; i < 50; i++) {
    //     reduce<<<Grid, Block>>>(d_input,d_result);
    // }
    reduce<<<Grid, Block>>>(d_input,d_result);
    // 拷贝回cpu
    cudaMemcpy(h_gpu_result, d_result, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    if(check(h_result, h_gpu_result,block_num)){
        printf("The ans is right.\n");
    } else {
        printf("The ans is wrong.\n");
        for(int i = 0; i < block_num; i++) {
            printf("%lf ", h_result[i]);
        }
        printf("\n");
    }

    // 释放内存
    cudaFree(d_input);
    cudaFree(d_result);
    free(h_input);
    free(h_gpu_result);
    free(h_result);

    return 0;
}