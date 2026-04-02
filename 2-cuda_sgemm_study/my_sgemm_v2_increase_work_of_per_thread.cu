#include<stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// 设为 1 开启高性能外积法，设为 0 退回内积法
#define USE_OUTER_PRODUCT 1

void rand_matrix(float *matrix, int row, int col) {
   

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            // 将 2D 坐标 (i, j) 映射到 1D 线性空间
            matrix[i * col + j] = (float)rand() / (float)RAND_MAX;
        }
    }
}
void cpu_sgemm(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K){

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                sum += A_ptr[m * K + k] * B_ptr[k * N + n];
            }
            C_ptr[m * N + n] = sum;
        }
    }
}
template<unsigned int BLOCK_SIZE, unsigned int STRIDE>
__global__ void cuda_sgemm(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K){

    constexpr int STEP = BLOCK_SIZE * STRIDE; // 整个block线程 需要计算 STEP * STEP 个数

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 获取block开始位置
    float *A_ptr_start = A_ptr + blockIdx.y * STEP * K;
    float *B_ptr_start = B_ptr + blockIdx.x * STEP;

    __shared__ float a_shared[STEP][STEP];
    __shared__ float b_shared[STEP][STEP];
    // 每个线程需要计算
    float temp[STRIDE][STRIDE] = {0.0f};


    for (int s = 0; s < K; s += STEP) {

        for (int i = 0; i < STRIDE; i++) {
            for (int j = 0; j < STRIDE; j++) {
                
                // ---------------- 搬运 A ----------------
                // 保安：算出 A 矩阵的【全局绝对坐标】
                int global_a_row = blockIdx.y * STEP + ty + i * BLOCK_SIZE;
                int global_a_col = s + tx + j * BLOCK_SIZE;
                
                // 如果没越界，搬运工用【相对指针】干活
                if (global_a_row < M && global_a_col < K) {
                    // 横着滑动
                    a_shared[ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = 
                        A_ptr_start[(ty + i * BLOCK_SIZE) * K + (tx + j * BLOCK_SIZE) + s];
                } else {
                    a_shared[ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = 0.0f;
                }

                // ---------------- 搬运 B ----------------
                // 保安：算出 B 矩阵的【全局绝对坐标】
                int global_b_row = s + ty + i * BLOCK_SIZE;
                int global_b_col = blockIdx.x * STEP + tx + j * BLOCK_SIZE;

                // 如果没越界，搬运工用【相对指针】干活
                if (global_b_row < K && global_b_col < N) {
                    // 竖着滑动
                    b_shared[ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = 
                        B_ptr_start[(ty + i * BLOCK_SIZE + s) * N + (tx + j * BLOCK_SIZE)];
                } else {
                    b_shared[ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = 0.0f;
                }
            }
        }

        // 所有线程都在这里集合！
        __syncthreads();

        #if USE_OUTER_PRODUCT
        // 外积法
        float a_reg[STRIDE];
        float b_reg[STRIDE];

        for (int k = 0; k < STEP; k++) {
            // 把1个k需要用到的A元素，一次性拿入口袋里面
            for (int i = 0; i < STRIDE; i++) {
                a_reg[i] = a_shared[ty + i * BLOCK_SIZE][k];
            }
            // 把1个k需要用到的B元素，一次性拿入口袋里面
            for (int j = 0; j < STRIDE; j++) {
                b_reg[j] = b_shared[k][tx + j * BLOCK_SIZE];
            }
            // 进行外积运算
            for (int i = 0; i < STRIDE; i++) {
                for (int j = 0; j < STRIDE; j++) {
                    temp[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        
        #else
        // 内积法
        // 计算环节，单线程计算
        for (int i = 0; i < STRIDE; i++) {
            for (int j = 0; j < STRIDE; j++) {
                for (int k = 0; k < STEP; k++){
                    temp[i][j] += a_shared[ty + i * BLOCK_SIZE][k] * b_shared[k][tx + j * BLOCK_SIZE];
                }
            }
        }
        #endif
        // 所有线程都在这里集合！
        __syncthreads();
    }

    // 结果写回去, 总共负责的块大小为 STEP
    float *C_ptr_start = C_ptr + blockIdx.y * STEP * N + blockIdx.x * STEP;
    for (int i = 0; i < STRIDE; i++) {
        for (int j = 0; j < STRIDE; j++) {
            // 保安：算出 C 矩阵当前要写的点的全局坐标
            int global_c_row = blockIdx.y * STEP + ty + i * BLOCK_SIZE;
            int global_c_col = blockIdx.x * STEP + tx + j * BLOCK_SIZE;

            // 没越界才能写回！
            if (global_c_row < M && global_c_col < N) {
                C_ptr_start[(ty + i * BLOCK_SIZE) * N + (tx + j * BLOCK_SIZE)] = temp[i][j];
            }
        }
    }
}

float compare_matrics(float *A, float *B, int M, int N){

    float max_err = 0.0f;

    for (int m = 0; m < M; m++) {

        for (int n = 0; n < N; n++) {

            float err = fabs(A[m * N + n] - B[m * N + n]);
            if(err > max_err) {
                max_err = err;
            }
        }
    }
    return max_err;
}
int main()
{

    printf("my_sgemm_v2_increase_work_of_per_thread\n");
     // 设置随机数种子，确保每次运行生成的结果不同
    // 如果调试时想要结果可复现，可以把这行注掉
    // srand(time(NULL)); 
    
    int m = 1024;
    int n = 1024;
    int k = 1024;
    const size_t mem_size_A = m * k * sizeof(float);
    const size_t mem_size_B = k * n * sizeof(float);
    const size_t mem_size_C = m * n * sizeof(float);

    // cpu相关变量
    float *matrix_host_A = (float *)malloc(mem_size_A);
    float *matrix_host_B = (float *)malloc(mem_size_B);

    float *matrix_cpu_cacl_C = (float *)malloc(mem_size_C);
    float *matrix_gpu_cacl_C = (float *)malloc(mem_size_C);
    // 初始化
    rand_matrix(matrix_host_A, m, k);
    rand_matrix(matrix_host_B, k, n);
    memset(matrix_cpu_cacl_C, 0, mem_size_C);
    memset(matrix_gpu_cacl_C, 0, mem_size_C);
    // cpu 进行 sgemm
    cpu_sgemm(matrix_host_A, matrix_host_B, matrix_cpu_cacl_C, m, n, k);

    // gpu相关变量
    float *matrix_device_A, *matrix_device_B, *matrix_device_C;
    cudaMalloc((void **)&matrix_device_A, mem_size_A);
    cudaMalloc((void **)&matrix_device_B, mem_size_B);
    cudaMalloc((void **)&matrix_device_C, mem_size_C);
    // 初始化
    cudaMemcpy(matrix_device_A, matrix_host_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_device_B, matrix_host_B, mem_size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_device_C, matrix_gpu_cacl_C, mem_size_C, cudaMemcpyHostToDevice);

    // gpu计算
    constexpr int BLOCK = 16;
    constexpr int STRIDE = 2;
    constexpr int STEP = BLOCK * STRIDE;
    dim3 block(BLOCK, BLOCK);
    // 把代表列的 n 放在第一个参数(x)，代表行的 m 放在第二个参数(y)
    dim3 grid((n - 1 + STEP) / STEP, (m - 1 + STEP) / STEP);
    cuda_sgemm<BLOCK,STRIDE><<<grid, block>>>(matrix_device_A, matrix_device_B, matrix_device_C, m, n, k);

    // 进行比较
    cudaMemcpy(matrix_gpu_cacl_C, matrix_device_C, mem_size_C,cudaMemcpyDeviceToHost);
    float diff = compare_matrics(matrix_cpu_cacl_C, matrix_gpu_cacl_C, m, n);
    if(diff < 1e-4) {
        printf("Result Check: PASS! Max Error = %f\n", diff);
    } else {
        printf("Result Check: FAILED! Max Error = %f\n", diff);
    }
    // 释放相关内存
    free(matrix_host_A);
    free(matrix_host_B);
    free(matrix_cpu_cacl_C);
    free(matrix_gpu_cacl_C);
    cudaFree(matrix_device_A);
    cudaFree(matrix_device_B);
    cudaFree(matrix_device_C);
    return 0;
}