#include<cstdio>
#include<cuda.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define THREAD_PER_BLOCK 256


__device__ void warpReduce(volatile float* v_shared, unsigned int tid) {
     // 第一步： 32 折叠 为 16
        v_shared[tid] += v_shared[tid + 32];
        __syncwarp(); // 32 对齐进度

        // 第二步： 16 折叠 为 8
        v_shared[tid] += v_shared[tid + 16];
        __syncwarp();

        // 第三步： 8 折叠为 4
        v_shared[tid] += v_shared[tid + 8];
        __syncwarp();

        // 第四步： 4 折叠为 2
        v_shared[tid] += v_shared[tid + 4];
        __syncwarp();

        // 第五步： 2 折叠为 1
        v_shared[tid] += v_shared[tid + 2];
        __syncwarp();

        // 第六步： 总和
        v_shared[tid] += v_shared[tid + 1];
        __syncwarp();    
}



__global__ void reduce(float *d_input, float *d_output,unsigned int n) {
    // 先把数据读到 共享内存里面，这样一个block里面都是共享的
    int tid = threadIdx.x;
    unsigned int global_i = tid + blockIdx.x * blockDim.x;
    // __shared__ float shared[THREAD_PER_BLOCK];
    // 1.获取每个block要处理数据的起始数组
    float *input_start = d_input + blockIdx.x * blockDim.x;
    unsigned int block_step = blockDim.x * gridDim.x;
    
    float my_sum = 0.0f;

    

    // 1.跨网格滚动循环
    while (global_i < n)
    {
        my_sum += input_start[tid];

        // 分块指针和全局索引同时跳跃一整个网格长度
        input_start += block_step;
        global_i += block_step;
    }
    // 2.第一级规约：全快所有的Warp各自独立进行寄存器洗牌
    // 这一块不需要任何同步。执行完后，每个Warp的0号线程 手里拿着它自己那个warp的局部总和
    my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 16);
    my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 8);
    my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 4);
    my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 2);
    my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 1);

    // 3. 跨warp通信 
    // 现代 GPU最大支持1024 线程/Block（即最多 32 个 warp）。因此分配 32绝对安全
    __shared__ float warpLevelSums[32];
    int laneId = tid % 32;
    int warpId = tid / 32;
    
    // 每个Warp 的老大（即laneId == 0) 把刚刚算出来的Warp总和给中转站
    if(laneId == 0) {
        warpLevelSums[warpId] = my_sum;
    }
    
    // 同步：确保所有Warp 老大把数据完全写完了
    __syncthreads();

    // 4. 第二季规约：由 0 号 Warp，对中转站收尾
    if(warpId == 0) {
        // 【妙招：用运行时表达式代替模板】
        // 256 线程对应 8 个 Warp。这里 laneId 从 0~7 的线程会去读中转站的数据
        // 而 8~31 号线程会自动补 0.0f。全员无分支脱节，齐步走进入下方的洗牌
        my_sum = (laneId < (blockDim.x / 32)) ? warpLevelSums[laneId] : 0.0f;
        
        // Warp 0 在寄存器层面执行最后一轮高潮规约，一瞬间把这 8 个数压缩成 1 个终极答案
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 16);
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 8);
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 4);
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 2);
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 1);
    }

    // 5. 单线程写回
    if(tid == 0) {
        d_output[blockIdx.x] = my_sum;
    }
}

bool check(const float * a, const float *b, int N) {

    for(int i = 0; i < N; i++) {
        
        if(fabs(a[i] - b[i]) > 0.005) {
            return false;
        }
    }
    return true;
}

int main() {
    printf("my_reduce_v9_anynum\n");
    const int N = 64 * 1024 * 1024 ;
    // cpu
    constexpr int block_num = 1024;
   

    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_result = (float*)malloc(block_num * sizeof(float));
    float *h_gpu_result = (float*)malloc(block_num * sizeof(float));

    // init data
    for(int i = 0; i < N; ++i) {
        h_input[i] = 2.0 * (float)drand48() - 1.0;
    }
    // cpu calc
for (int b = 0; b < block_num; b++) {
        float block_sum = 0.0f;
        // 模拟当前 block 内部 256 个线程的行为
        for (int t = 0; t < THREAD_PER_BLOCK; t++) {
            unsigned int global_i = b * THREAD_PER_BLOCK + t;
            unsigned int block_step = THREAD_PER_BLOCK * block_num;
            while (global_i < N) {
                block_sum += h_input[global_i];
                global_i += block_step;
            }
        }
        h_result[b] = block_sum;
    }
    // gpu
    float *d_input, *d_result;
    cudaMalloc((void **)&d_input,N * sizeof(float));
    cudaMalloc((void **)&d_result,block_num * sizeof(float));

    // 从 cpu 拷贝到 gpu
    cudaMemcpy(d_input,h_input,N * sizeof(float),cudaMemcpyHostToDevice);

    // 配置 线程
    dim3 Grid(block_num,1);
    dim3 Block(THREAD_PER_BLOCK,1);

    // for(int i = 0; i < 50; i++) {
    //     reduce<<<Grid, Block>>>(d_input,d_result);
    // }
    reduce<<<Grid, Block>>>(d_input,d_result,N);
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