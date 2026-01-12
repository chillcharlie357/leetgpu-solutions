#include <cuda_runtime.h>

__device__ float warpReduceMax(float local_max){
    // warp内规约求最大值，使用shuffle指令在线程间交换数据
    for (size_t i = 16; i > 0; i /= 2)
    {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff,local_max, i));
    }
    return local_max;
}

__device__ float warpReduceSum(float local_sum){
    // warp内规约求和，使用shuffle指令在线程间交换数据
    for (size_t i = 16; i > 0; i /= 2)
    {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, i);
    }
    return local_sum;
}

__global__ void softmax_kernel(const float* input, float* output, int N) {
    // 计算线程ID、warp ID、lane ID（线程在warp内的位置）
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;
    int num_warps = blockDim.x / 32;

    extern __shared__ float shared_data[];
    // 共享内存分区：前半部分存储每个warp的最大值，后半部分存储每个warp的和
    float* s_max = shared_data;
    float* s_sum = &shared_data[num_warps];

    // ===== 第一阶段：求最大值（用于数值稳定性的max trick）=====

    float local_max = -INFINITY;
    // Grid-Stride Loop：每个线程处理多个元素（stride=blockDim.x），实现合并访存
    for (size_t i = tid; i < N; i += blockDim.x)
    {
        local_max  = fmaxf(local_max, input[i]);
    }
    // warp内规约：获取当前warp的最大值
    float warp_max = warpReduceMax(local_max);

    // 每个warp的lane 0将该warp的最大值写入共享内存
    if(lane == 0){
        s_max[wid] = warp_max;
    }
    __syncthreads();

    float global_max = -INFINITY;
    // block内规约：汇总所有warp的最大值，得到全局最大值
    for (size_t i = 0; i < num_warps; i++)
    {
        global_max = fmaxf(global_max, s_max[i]);
    }
    __syncthreads();


    // ===== 第二阶段：求和（使用max trick避免溢出）=====

    float local_sum = 0.0f;
    // 使用max trick：exp(x - max)，避免大数溢出
    for (size_t i = tid; i < N; i += blockDim.x)
    {
        local_sum += expf(input[i] - global_max);
    }
    // warp内规约：获取当前warp的和
    float warp_sum = warpReduceSum(local_sum);

    // 每个warp的lane 0将该warp的和写入共享内存
    if(lane == 0){
        s_sum[wid] = warp_sum;
    }
    __syncthreads();

    float global_sum = 0.0f;
    // block内规约：汇总所有warp的和，得到全局和
    for (size_t i = 0; i < num_warps; i++)
    {
        global_sum += s_sum[i];
    }


    // ===== 第三阶段：计算softmax输出 =====
    for (size_t i = tid; i < N; i += blockDim.x)
    {
        // softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
        output[i] = expf(input[i] - global_max) / global_sum;
    }

}

extern "C" void solve(const float* input, float* output, int N) {
    // 配置kernel启动参数
    int threadsPerBlock = 256;  // 每个block 256个线程，即8个warp
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 计算共享内存大小：每个warp存储一个max和一个sum
    int num_warps = threadsPerBlock / 32;
    int shared_data_size = 2 * num_warps * sizeof(float);

    // 启动softmax kernel
    softmax_kernel<<<blocksPerGrid, threadsPerBlock, shared_data_size>>>(input, output, N);
    // 等待GPU执行完成
    cudaDeviceSynchronize();
}
