#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define NUM_MAX_FIFO_SLOTS 32768
#define FINISHED_SUM_TAG 1024

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

constexpr int kNumRanks = 4;  // 假设有 4 个 ranks
constexpr int kNumExperts = 8;  // 假设有 8 个 experts
constexpr int kNumTokens = 16;  // 假设有 16 个 tokens
constexpr int kNumChannels = 2;  // 假设有 2 个 channels
constexpr int kNumThreads = 128;  // 每个 block 的线程数

template <typename dtype_t>
__host__ __device__ dtype_t cell_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ void memory_fence() {
    asm volatile("fence.acq_rel.sys;":: : "memory");
}

__forceinline__ __device__ void get_channel_task_range(int num_tokens, int num_sms, int sm_id,
                                                       int& token_start_idx, int& token_end_idx) {
    int num_tokens_per_sm = cell_div(num_tokens, num_sms);
    token_start_idx = min(num_tokens_per_sm * sm_id, num_tokens);
    token_end_idx = min(token_start_idx + num_tokens_per_sm, num_tokens);
}

template <int kNumRanks>
__forceinline__ __device__ void
barrier_device(int **task_fifo_ptrs, int head, int rank, int tag = 0) {
    auto thread_id = static_cast<int>(threadIdx.x);
    // EP_DEVICE_ASSERT(kNumRanks <= 32);

    if (thread_id < kNumRanks) {
        atomicAdd_system(task_fifo_ptrs[rank] + head + thread_id, FINISHED_SUM_TAG);
        memory_fence();
        atomicSub_system(task_fifo_ptrs[thread_id] + head + rank, FINISHED_SUM_TAG);
    }
    // timeout_check<kNumRanks>(task_fifo_ptrs, head, rank, 0, tag);
}

template <int kNumRanks>
__forceinline__ __device__ void move_fifo_slots(int &head) {
    head = (head + kNumRanks) % NUM_MAX_FIFO_SLOTS;
}

// 内核函数声明
template<int kNumRanks>
__global__ void notify_dispatch(
    const int* num_tokens_per_rank, int* moe_recv_counter_mapped,
    const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
    int num_tokens, int num_channels, const bool* is_token_in_rank, int* channel_prefix_matrix,
    int* rank_prefix_matrix_copy, int num_memset_int, int expert_alignment,
    void** buffer_ptrs, int** task_fifo_ptrs, int head, int rank) {

    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
    auto lane_id = thread_id % 32, warp_id = thread_id / 32, num_warps = num_threads / 32;

    if (sm_id == 0) {
        // Barrier first
        barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
        move_fifo_slots<kNumRanks>(head);
        __syncthreads();

        int *per_rank_buffer, *per_expert_buffer;
        if (thread_id < kNumRanks) {
            per_rank_buffer = reinterpret_cast<int*>(buffer_ptrs[thread_id]);
            per_expert_buffer = per_rank_buffer + kNumRanks * kNumRanks;
        }

        // After this loop:
        //  - `per_rank_buffer[rank][i, j]` means the number of tokens from rank i to rank j
        //  - `per_expert_buffer[rank][i, j]` means the number of tokens from rank i to local expert j
        int num_experts_per_rank = num_experts / kNumRanks;
        if (thread_id < kNumRanks) {
            #pragma unroll
            for (int i = 0; i < kNumRanks; ++ i)
                per_rank_buffer[rank * kNumRanks + i] = num_tokens_per_rank[i];
            #pragma unroll
            for (int i = 0; i < num_experts_per_rank; ++ i)
                per_expert_buffer[rank * num_experts_per_rank + i] = num_tokens_per_expert[thread_id * num_experts_per_rank + i];
        }
        __syncthreads();

        // Wait for all ranks to be finished
        barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
        move_fifo_slots<kNumRanks>(head);
        __syncthreads();

        // Sum per-rank counts and return to CPU
        // Also pre-compute the prefix sum for data sending
        auto local_per_rank_buffer = reinterpret_cast<int*>(buffer_ptrs[rank]);
        if (thread_id < kNumRanks) {
            #pragma unroll
            for (int i = 1; i < kNumRanks; ++ i)
                local_per_rank_buffer[i * kNumRanks + thread_id] += local_per_rank_buffer[(i - 1) * kNumRanks + thread_id];
            if (thread_id == rank)
                *moe_recv_counter_mapped = local_per_rank_buffer[(kNumRanks - 1) * kNumRanks + rank];
        }

        // Sum per-experts counts and return to CPU
        auto local_per_expert_buffer = local_per_rank_buffer + kNumRanks * kNumRanks;
        if (thread_id < num_experts_per_rank) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumRanks; ++ i)
                sum += local_per_expert_buffer[i * num_experts_per_rank + thread_id];
            sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
            moe_recv_expert_counter_mapped[thread_id] = sum;
        }
        __syncthreads();

        // Copy rank size prefix matrix to another tensor
        #pragma unroll
        for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)
            rank_prefix_matrix_copy[i] = local_per_rank_buffer[i];

        // Extra memset for later communication queue
        #pragma unroll
        for (int i = thread_id; i < num_memset_int; i += num_threads)
            local_per_expert_buffer[i] = 0;

        // Barrier
        memory_fence();
        __syncthreads();
        barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
    } else {
        int dst_rank = sm_id - 1;
        for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

            // Iterate over tokens
            int count = 0;
            for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += 32)
                count += is_token_in_rank[i * kNumRanks + dst_rank];
            count = warp_reduce_sum(count);
            if (lane_id == 0)
                channel_prefix_matrix[dst_rank * num_channels + channel_id] = count;
        }
        __syncthreads();

        // Pre-compute prefix sum for all channels
        if (thread_id == 0) {
            #pragma unroll
            for (int i = 1; i < num_channels; ++ i)
                channel_prefix_matrix[dst_rank * num_channels + i] += channel_prefix_matrix[dst_rank * num_channels + i - 1];
        }
    }
}

int main() {
    // 1. 初始化输入数据
    std::vector<int> h_num_tokens_per_rank(kNumRanks, kNumTokens / kNumRanks);  // 每个 rank 平均分配 tokens
    std::vector<int> h_num_tokens_per_expert(kNumExperts, kNumTokens / kNumExperts);  // 每个 expert 平均分配 tokens
    std::vector<bool> h_is_token_in_rank(kNumTokens * kNumRanks, true);  // 假设所有 tokens 都属于所有 ranks
    std::vector<int> h_channel_prefix_matrix(kNumRanks * kNumChannels, 0);  // 初始化为 0
    std::vector<int> h_rank_prefix_matrix_copy(kNumRanks * kNumRanks, 0);  // 初始化为 0

    int h_moe_recv_counter_mapped = 0;
    std::vector<int> h_moe_recv_expert_counter_mapped(kNumExperts, 0);

    // 2. 分配设备内存并拷贝数据
    int *d_num_tokens_per_rank, *d_num_tokens_per_expert, *d_channel_prefix_matrix, *d_rank_prefix_matrix_copy;
    int *d_moe_recv_counter_mapped, *d_moe_recv_expert_counter_mapped;
    bool *d_is_token_in_rank;

    CUDA_CHECK(cudaMalloc(&d_num_tokens_per_rank, kNumRanks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_num_tokens_per_expert, kNumExperts * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_is_token_in_rank, kNumTokens * kNumRanks * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_channel_prefix_matrix, kNumRanks * kNumChannels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rank_prefix_matrix_copy, kNumRanks * kNumRanks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_moe_recv_counter_mapped, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_moe_recv_expert_counter_mapped, kNumExperts * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_num_tokens_per_rank, h_num_tokens_per_rank.data(), kNumRanks * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_num_tokens_per_expert, h_num_tokens_per_expert.data(), kNumExperts * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_is_token_in_rank, h_is_token_in_rank.data(), kNumTokens * kNumRanks * sizeof(bool), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_channel_prefix_matrix, h_channel_prefix_matrix.data(), kNumRanks * kNumChannels * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rank_prefix_matrix_copy, h_rank_prefix_matrix_copy.data(), kNumRanks * kNumRanks * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_moe_recv_counter_mapped, &h_moe_recv_counter_mapped, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_moe_recv_expert_counter_mapped, h_moe_recv_expert_counter_mapped.data(), kNumExperts * sizeof(int), cudaMemcpyHostToDevice));

    // 3. 设置其他参数
    int num_memset_int = kNumRanks * kNumRanks;  // 假设需要清理的整数数量
    int expert_alignment = 1;  // 假设对齐为 1
    int head = 0;  // FIFO 队列头部
    int rank = 0;  // 当前 rank

    // 模拟 buffer_ptrs 和 task_fifo_ptrs
    void* d_buffer_ptrs[kNumRanks];
    int* d_task_fifo_ptrs[kNumRanks];
    CUDA_CHECK(cudaMalloc(&d_buffer_ptrs[0], kNumRanks * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&d_task_fifo_ptrs[0], kNumRanks * sizeof(int*)));

    // 4. 启动内核
    dim3 grid(1);  // 一个 block
    dim3 block(kNumThreads);  // 每个 block 的线程数

    notify_dispatch<kNumRanks><<<grid, block>>>(
        d_num_tokens_per_rank, d_moe_recv_counter_mapped,
        d_num_tokens_per_expert, d_moe_recv_expert_counter_mapped, kNumExperts,
        kNumTokens, kNumChannels, d_is_token_in_rank, d_channel_prefix_matrix,
        d_rank_prefix_matrix_copy, num_memset_int, expert_alignment,
        d_buffer_ptrs, d_task_fifo_ptrs, head, rank);

    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. 拷贝结果回主机并验证
    CUDA_CHECK(cudaMemcpy(&h_moe_recv_counter_mapped, d_moe_recv_counter_mapped, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_moe_recv_expert_counter_mapped.data(), d_moe_recv_expert_counter_mapped.data(), kNumExperts * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_channel_prefix_matrix.data(), d_channel_prefix_matrix, kNumRanks * kNumChannels * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rank_prefix_matrix_copy.data(), d_rank_prefix_matrix_copy, kNumRanks * kNumRanks * sizeof(int), cudaMemcpyDeviceToHost));

    // 打印结果
    std::cout << "moe_recv_counter_mapped: " << h_moe_recv_counter_mapped << std::endl;
    std::cout << "moe_recv_expert_counter_mapped: ";
    for (int val : h_moe_recv_expert_counter_mapped) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // 6. 释放设备内存
    CUDA_CHECK(cudaFree(d_num_tokens_per_rank));
    CUDA_CHECK(cudaFree(d_num_tokens_per_expert));
    CUDA_CHECK(cudaFree(d_is_token_in_rank));
    CUDA_CHECK(cudaFree(d_channel_prefix_matrix));
    CUDA_CHECK(cudaFree(d_rank_prefix_matrix_copy));
    CUDA_CHECK(cudaFree(d_moe_recv_counter_mapped));
    CUDA_CHECK(cudaFree(d_moe_recv_expert_counter_mapped));
    CUDA_CHECK(cudaFree(d_buffer_ptrs[0]));
    CUDA_CHECK(cudaFree(d_task_fifo_ptrs[0]));

    return 0;
}
