# 学习cuda编程
## 1-cuda_reduce_study
## 2-cuda_sgemm_study
### my_sgemm_v1_shared_memory
在 CUDA 中，__syncthreads() 的严格定义是：同一个 Block 内的所有线程，必须全部到达这个同步点，才能继续往下执行。
假设你的矩阵是 $100 \times 100$，而你的 Block 大小是 $32 \times 32$。当处理到矩阵边缘的 Block 时，有些线程的坐标可能超出了矩阵（比如 $x=120, y=120$）。那些超出边缘的线程，因为 if(x < N && y < M) 为假，它们直接跳过了整个 if 块，执行结束了。而那些在矩阵内部的线程，进入了 if 块，并在 __syncthreads() 这里停下来，苦苦等待 Block 里的其他人。
结果就是：等的人永远等不到，走的人也不会再回来。整个 Block 永久卡死，程序崩溃！
### my_sgemm_v3_using_float4
在 CUDA中，内存合并访问，只要一个warp中存在某几个线程读取内存的时候总和为128KB的时候，就可以合并内存访问
flaot4 就是 可以一次性读取4个float的数据
### my_sgemm_v4_using_float4_and_register
在编写和优化 CUDA 矩阵乘法时，核心逻辑可以归纳为宏观上的“输出导向分块”与微观上的“外积与寄存器复用”：

1. 宏观视角：以结果为导向的分块策略（Output-Stationary / Block Tiling）

核心准则： 矩阵乘法的任务划分必须**“以结果矩阵 C 为基准”**。

执行逻辑： 每个 Thread Block 负责计算并写回 C 矩阵的一个独立子块（例如 32x32 的 Block）。在编写核函数时，必须时刻清晰当前 Block 和当前 Thread 在 C 矩阵坐标系中的“责任田”。

数据流向： 明确了 C 的目标区域后，再反向推导，驱动代码去 Global Memory 中分步（分 Phase）搬运对应 A 的行块和 B 的列块到 Shared Memory 中。

2. 微观视角：从内积到外积的思维转换（Outer Product Accumulation）

痛点分析： 如果让一个线程只计算 C 的一个点（内积思想），会导致 A 和 B 的同一组数据在 Shared Memory 中被同 Block 内的线程反复读取，极大地浪费了访存带宽。

外积破局： 为了突破访存瓶颈，必须扩大单个线程的计算职责（例如让一个线程计算 1x4 或 4x4 的 C 矩阵块）。此时，计算模式在底层就转换为了**外积（Outer Product）**的累加。

数据复用： 在外积模式下，A 的 1 个元素可以与 B 的 4 个元素组合，产生 4 次乘加运算。这种“一对多”的关系创造了极佳的数据复用机会。

3. 终极优化：寄存器级缓存（Register Tiling）

存储降维打击： 既然在外积计算中，同一个数据会被连续复用多次，就绝不能让它留在 Shared Memory 里挨个被读取。必须利用 CUDA 中速度最快、延迟最低的存储层级——寄存器（Register）。

操作手法： 将高频复用的数据（如 A 的单个元素和 B 的向量元素）提前从 Shared Memory 读入局部寄存器变量中。接下来的多次 FMA（融合乘加）运算全部在寄存器之间进行。

收益： 这样不仅大幅减少了对 Shared Memory 的读取指令，还消除了潜在的 Bank Conflict，从而真正让 GPU 的计算单元（ALU）达到满载跑满的状态。
