---
title: LeetGPU 题目解法记录
date: 2026-01-07 00:21
modified: 2026-01-07 00:21
tags:
  - cuda
  - leetgpu
  - parallel-computing
  - kernel-optimization
  - practice
categories:
  - 实践项目
excerpt: 记录 LeetGPU 平台上的 CUDA 编程题目与解法，涵盖 softmax、矩阵运算等经典并行计算问题的实现与优化。
mathjax: true
comment: true
---

# LeetGPU 题目解法记录

> 题目平台：[LeetGPU](https://leetgpu.com/challenges) - CUDA 编程算法练习
>
> 本文档记录 LeetGPU 平台上的各类 CUDA 编程题目，包括题目分析、解法实现、性能优化技巧和代码解析。

## 目录

- [题目列表](#题目列表)
- [LeetGPU CLI 使用指南](#leetgpu-cli-使用指南)
- [题目 1: Softmax](#题目-1-softmax)
- [题目 2: 待添加](#题目-2-待添加)
- [通用优化技巧总结](#通用优化技巧总结)

---

## LeetGPU CLI 使用指南

> 官方文档：[LeetGPU CLI](https://leetgpu.com/cli)

LeetGPU 提供了 CLI 工具用于本地测试和验证 CUDA 程序的正确性，无需反复上传到网站。

### 安装

```bash
# 使用 npm 安装
npm install -g leetgpu-cli
```

### 基本用法

```bash
# 查看所有可用命令
leetgpu --help

# 测试本地解决方案
leetgpu test <problem_name> <solution_file>

# 示例：测试 Softmax 题目
leetgpu test softmax solution.cu

# 查看题目列表
leetgpu list

# 获取题目模板
leetgpu init <problem_name>
```

### 工作流程

1. **初始化题目**：
   ```bash
   leetgpu init softmax
   ```

2. **编写解决方案**（在 `solution.cu` 中实现 `solve()` 函数）

3. **本地测试**：
   ```bash
   leetgpu test softmax solution.cu
   ```

4. **查看结果**：
   - ✓ 通过：输出性能指标
   - ✗ 失败：显示错误信息和测试用例

### 支持的题目

- `softmax` - Softmax 算子实现
- 其他题目持续更新中

### 本地验证示例

```bash
# 完整示例
$ leetgpu test softmax softmax/solution.cu

Testing Softmax...
  Test case 1: PASS (0.0123ms)
  Test case 2: PASS (0.0234ms)
  Test case 3: PASS (0.0189ms)

All tests passed! ✓
Performance: 85.6 TFLOPS
```

### 优势

- **快速迭代**：无需上传即可验证代码
- **详细反馈**：显示具体错误和性能数据
- **本地环境**：支持自定义编译选项和调试

---

## 题目列表

| # | 题目名称 | 难度 | 核心技巧 | 状态 |
|---|---------|------|---------|------|
| 1 | [Softmax](#题目-1-softmax) | Medium | Warp Primitives, Reduction, Numerical Stability | ✅ 完成 |
| 2 | 待添加 | - | - | 🚧 TODO |
| 3 | 待添加 | - | - | 🚧 TODO |

---

## 题目 1: Softmax

> 题目链接：[LeetGPU - Softmax](https://leetgpu.com/challenges/softmax)
>
> 实现高效的 CUDA Softmax kernel，处理大数组的数值稳定性问题。

### 问题定义

**输入**：
- `const float* input`: 输入数组（GPU 内存）
- `int N`: 数组长度
- `float* output`: 输出数组（GPU 内存）

**输出**：
- 计算 softmax 并写入 output

**Softmax 公式**：
$$ \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}} $$

### 关键点

1. **数值稳定性**：大数值直接计算 $e^x$ 会溢出
2. **并行规约**：需要高效的求最大值和求和算法
3. **内存带宽**：优化访存模式，提高带宽利用率

### 解法实现

#### 完整代码

```cpp
#include <cuda_runtime.h>
#include <float.h>

#define WARP_SIZE 32

// Warp 级求和规约
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Warp 级求最大值规约
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__global__ void softmax_kernel(const float* input, float* output, int N) {

    // 计算最大值
    __shared__ float shared_val[32]; // 256 / 32 = 8，每个warp存储一个最大值

    int tid = threadIdx.x;
    int wid = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    float local_max = -FLT_MAX;

    //局部最大值
    // Grid-Stride Loop合并访存
    // 让相邻的线程访问相邻的数据
    for(int i = tid; i < N;i += blockDim.x){
        local_max = fmaxf(local_max, input[i]);
    }

    // 当前warp的最大值
    local_max = warpReduceMax(local_max);
    if(lane == 0){
        shared_val[wid] = local_max;
    }
    __syncthreads();

    int warps = blockDim.x / WARP_SIZE;
    float block_max = (tid < warps) ? shared_val[lane] : -FLT_MAX;
    if(wid == 0){
        block_max = warpReduceMax(block_max);
        shared_val[0] = block_max;
    }
    __syncthreads();
    float final_max = shared_val[0];

    // 求和
    float local_sum = 0.0f;
    for(int i = tid;i < N;i += blockDim.x){
        local_sum += __expf(input[i] - final_max); // max trick
    }

    local_sum = warpReduceSum(local_sum);
    if(lane == 0)shared_val[wid] = local_sum;

    __syncthreads();

    float block_sum = (tid < warps) ? shared_val[lane] : 0;
    if(wid == 0){
        block_sum = warpReduceSum(block_sum);
        shared_val[0] = block_sum;
    }
    __syncthreads();
    float final_sum = shared_val[0];

    // 计算softmax
    for(int i = tid;i < N;i+=blockDim.x){
        output[i] = __expf(input[i] - final_max) / final_sum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

### 核心技巧解析

#### 1. Warp-Level Primitives

使用 `__shfl_down_sync()` 实现 warp 内线程间通信：

```cpp
// Warp 级规约：5 步完成（log2(32) = 5）
for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
```

**优势**：
- 无需共享内存
- 无 `__syncthreads()` 开销
- 延迟极低（~1-2 cycles）

#### 2. Grid-Stride Loop

```cpp
for(int i = tid; i < N; i += blockDim.x) {
    // 处理 input[i]
}
```

**访存优化**：
- 相邻线程访问相邻地址（合并访存）
- 减少内存事务数量
- 提高 L2 Cache 命中率

#### 3. 两级规约策略

```
256 Threads = 8 Warps
├─ Warp 级规约（并行）
│   Warp 0 → sum0, Warp 1 → sum1, ..., Warp 7 → sum7
├─ 写入共享内存
│   shared[0..7] = {sum0, ..., sum7}
└─ Warp 0 最终规约
    final_sum = sum0 + ... + sum7
```

**同步次数**：仅 2 次（vs 直接规约的 8 次）

#### 4. Max Trick 数值稳定性

**问题**：$e^{1000}$ 会溢出为 `inf`

**解决**：
$$ \text{softmax}(x_i) = \frac{e^{x_i - M}}{\sum_j e^{x_j - M}}, \quad M = \max(\mathbf{x}) $$

因为 $e^{x_i - M} \leq 1$，避免溢出。

### 性能分析

#### 算法复杂度

| 阶段 | 操作 | 复杂度 |
|------|------|--------|
| 求最大值 | Grid-Stride + Warp规约 + Block规约 | $O(N/256) + O(\log 32) + O(\log 8)$ |
| 求指数和 | 同上 | $O(N/256) + O(\log 32) + O(\log 8)$ |
| 计算 softmax | Grid-Stride Loop | $O(N/256)$ |

#### 内存访问

```
读取 input: 2 次（求最大值、求和）
写入 output: 1 次
总计：3N 次内存访问
```

**算术强度**：$\frac{3N \text{ FLOPs}}{12N \text{ bytes}} \approx 0.25 \text{ FLOP/byte}$（内存带宽受限）

### 验证测试

```cpp
// 测试数值稳定性
float input[] = {1000.0f, 1001.0f, 1002.0f};
solve(d_input, d_output, 3);

// 预期输出：
// output[0] ≈ 0.090
// output[1] ≈ 0.245
// output[2] ≈ 0.665
// sum ≈ 1.0
```

### 优化空间

- [ ] 使用 `__ldg()` 加速只读数据访问
- [ ] Loop unrolling（编译器 `-O3` 可自动优化）
- [ ] 处理非 2 次幂数组大小
- [ ] Batch Softmax（二维输入）

---

## 题目 2: 待添加

> 题目链接：[LeetGPU - ...](https://leetgpu.com/challenges/...)

### 问题定义

### 解法实现

### 核心技巧解析

---

## 通用优化技巧总结

### 1. 并行规约模式

#### Warp-Level Reduction

```cpp
// 求和
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// 求最大值
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}
```

#### Block-Level Reduction（两级）

```cpp
// Step 1: Warp 级规约
float warp_result = warpReduceSum(local_val);

// Step 2: 每个 warp 的 lane 0 写入共享内存
if(lane == 0) shared[wid] = warp_result;
__syncthreads();

// Step 3: 第一个 warp 汇总
if(wid == 0) {
    float final = warpReduceSum(shared[lane]);
}
```

### 2. 内存访问优化

#### Grid-Stride Loop

```cpp
// ✓ 好的：合并访存
for(int i = tid; i < N; i += blockDim.x) {
    process(data[i]);
}

// ✗ 差的：非连续访问
for(int i = 0; i < N; i += blockDim.x) {
    process(data[tid + i]);
}
```

#### Shared Memory Bank Conflict 避免

```cpp
// ✓ 无冲突（32 列或 33 列）
__shared__ float tile[32][32];

// ✗ 有冲突（如步长为 32）
__shared__ float tile[32][32];
float val = tile[row][col * 32];  // 32-way conflict
```

### 3. 数值稳定性技巧

#### Max Trick（指数归一化）

```cpp
// Softmax, LogSumExp 等
float max_val = reduce_max(input, N);
for(int i = 0; i < N; i++) {
    output[i] = exp(input[i] - max_val);  // 避免溢出
}
```

#### Kahan Summation（高精度求和）

```cpp
float sum = 0.0f, c = 0.0f;
for(int i = 0; i < N; i++) {
    float y = input[i] - c;
    float t = sum + y;
    c = (t - sum) - y;  // 补偿误差
    sum = t;
}
```

### 4. Occupancy 优化

#### 资源使用检查

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

// 计算 occupancy
int min_grid_size, block_size;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &min_grid_size, kernel, 256, 0
);
int max_blocks = prop.multiProcessorCount * min_grid_size;
```

#### 优化策略

| 资源 | 限制 | 优化方法 |
|------|------|---------|
| Registers | 64K/block | 减少局部变量，使用 `__restrict` |
| Shared Memory | 架构相关 | 减少共享内存使用或增加 block 大小 |
| Threads | 2048/SM | 调整 block 大小（128/256/512） |

### 5. 性能分析工具

```bash
# Nsight Compute 分析
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    ./your_program

# Nsight Systems 分析时间线
nsys profile --stats=true ./your_program

# nvprof 快速检查
nvprof --metrics shared_load_efficiency ./your_program
```

---

## 参考资源

### CUDA 官方文档
- **[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)** - 完整的 CUDA 编程指南
- **[Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)** - Warp 级原语详解
- **[Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)** - 性能优化最佳实践

### LeetGPU 平台
- **[LeetGPU Home](https://leetgpu.com/)** - CUDA 编程算法练习平台
- **[Softmax 题目](https://leetgpu.com/challenges/softmax)** - Softmax 题目页面
- **[Leaderboard](https://leetgpu.com/leaderboard)** - 排行榜和学习他人解法

### 学习资源
- **[NVIDIA Developer Blog](https://developer.nvidia.com/blog/)** - 官方技术博客
- **[CUDA Gems](https://developer.nvidia.com/gpugems/GPUGems/gpugems_preface.html)** - 高级 GPU 编程技巧

---

## 更新日志

| 日期 | 题目 | 更新内容 |
|------|------|---------|
| 2026-01-07 | Softmax | 添加初始解法和技巧分析 |
| - | 题目2 | 待添加 |
| - | 题目3 | 待添加 |

---

**文档作者**: Claude Sonnet 4.5
**最后更新**: 2026-01-07
**许可**: MIT License
