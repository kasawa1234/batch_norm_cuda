#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#include "constants.h"


/*
Considering following accelerations:
    - calculate mean and std in the same loop
    - change (x_i - (\sum x_i) / N) / std_eps to (N * x_i - \sum x_i) / (N * std_eps)
*/


template <typename scalar_t>
__global__ void sum_std_sram_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> sum,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> batch_norm_output
){
    const int N = input_data.size(0);

    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_BATCH][BLOCK_SIZE_FEATURE][2];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int thread_id_n = threadIdx.x;
    const int thread_id_c = threadIdx.y;

    // if the loc cover our data, load in shared memory
    if (n < input_data.size(0) && c < input_data.size(1)){
        shared_memory[thread_id_n][thread_id_c][0] = input_data[n][c];
        shared_memory[thread_id_n][thread_id_c][1] = input_data[n][c] * input_data[n][c];
    } else {
        shared_memory[thread_id_n][thread_id_c][0] = static_cast<scalar_t>(0);
        shared_memory[thread_id_n][thread_id_c][1] = static_cast<scalar_t>(0);
    }
    __syncthreads();            // need to fully load all items into shared_memory

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (thread_id_n < offset) {
            shared_memory[thread_id_n][thread_id_c][0] += shared_memory[thread_id_n + offset][thread_id_c][0];
            shared_memory[thread_id_n][thread_id_c][1] += shared_memory[thread_id_n + offset][thread_id_c][1];
        }
        __syncthreads();        // wait, till all threads in this block reach
    }
    
    // calculate N * mean and N * std_eps
    if (thread_id_n == 0) {
        sum[c] = shared_memory[0][thread_id_c][0];
        batch_norm_output[N][c] = sqrt(N * shared_memory[0][thread_id_c][1] - shared_memory[0][thread_id_c][0] * shared_memory[0][thread_id_c][0] + N * N * EPSILON);
    }
}

template <typename scalar_t>
__global__ void bn_forward_mlp_sram_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> sum,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> gamma,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> beta,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output_data
){
    // batch size
    const int N = input_data.size(0);

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (n >= input_data.size(0) || c >= input_data.size(1)) return;

    output_data[n][c] = gamma[c] * (N * input_data[n][c] - sum[c]) / output_data[N][c] + beta[c];
}


torch::Tensor bn_forward_mlp_sram_cuda(
    const torch::Tensor X,
    const torch::Tensor gamma,
    const torch::Tensor beta
){
    // X: (n, c), n is parallel
    const int N = X.size(0);
    const int C = X.size(1);
    std::cout << N << ", " << C << std::endl;

    // calculate sum and std
    torch::Tensor sum = torch::zeros({C}, X.options());
    // batch_norm_out: bn_forward + N * std_eps
    torch::Tensor batch_norm_out = torch::zeros({N + 1, C}, X.options());
    
    const dim3 threads_sum_std(BLOCK_SIZE_BATCH, BLOCK_SIZE_FEATURE);
    const dim3 blocks_sum_std((N + threads_sum_std.x - 1) / threads_sum_std.x, (C + threads_sum_std.y - 1) / threads_sum_std.y);

    std::cout << "blocks sum_std: " << blocks_sum_std.x << ", " << blocks_sum_std.y << std::endl;

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(X.type(), "sum_std_kernel",
    ([&] {
        sum_std_sram_kernel<scalar_t><<<blocks_sum_std, threads_sum_std>>>(
            X.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sum.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            batch_norm_out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));


    // batch norm will use a even dispatched block size
    const dim3 threads_batch_norm(BLOCK_SIZE_BN_X, BLOCK_SIZE_BN_Y);
    const dim3 blocks_batch_norm((N + threads_batch_norm.x - 1) / threads_batch_norm.x, (C + threads_batch_norm.y - 1) / threads_batch_norm.y);

    std::cout << "blocks batch norm: " << blocks_batch_norm.x << ", " << blocks_batch_norm.y << std::endl;

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(X.type(), "bn_forward_mlp_kernel",
    ([&] {
        bn_forward_mlp_sram_kernel<scalar_t><<<blocks_batch_norm, threads_batch_norm>>>(
            X.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sum.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            gamma.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            beta.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            batch_norm_out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));

    // batch_norm_out contains all things that we need to save in PyTorch
    // gamma will be saved outside in PyTorch, here only save bn_out and std_eps
    /*
        0: [x1, x2, ..., xn]
        1: [x1, x2, ..., xn]
        ... ...
        n - 1: [x1, x2, ..., xn]

        n: n * [std1, std2, ..., stdn]
    */
    return batch_norm_out;
}


/*                          Parallel Conv Forward                          */


template <typename scalar_t>
__global__ void partial_sum_sum2_conv_sram_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> partial_sum,
    const int valid_height,
    const int valid_width,
    const int block_num_width
){
    // partial sum
    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_HW][2];

    const int n = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z * blockDim.z + threadIdx.z;
    const int tid_hw = threadIdx.x;
    const int tid_h = tid_hw / BLOCK_SIZE_W;
    const int tid_w = tid_hw - tid_h * BLOCK_SIZE_W;
    const int hw = blockIdx.x;
    const int row = hw / block_num_width;
    const int col = hw - row * block_num_width;
    const int h = row * BLOCK_SIZE_H + tid_h;
    const int w = col * BLOCK_SIZE_W + tid_w;

    // if the loc cover our data, load in shared memory
    if (n < input_data.size(0) && c < input_data.size(1) && h < valid_height && w < valid_width){
        shared_memory[tid_hw][0] = input_data[n][c][h][w];
        shared_memory[tid_hw][1] = input_data[n][c][h][w] * input_data[n][c][h][w];
    } else {
        shared_memory[tid_hw][0] = static_cast<scalar_t>(0);
        shared_memory[tid_hw][1] = static_cast<scalar_t>(0);
    }
    __syncthreads();
    
    for (int offset = BLOCK_SIZE_HW >> 1; offset > 0; offset >>= 1) {
        if (tid_hw < offset) {
            shared_memory[tid_hw][0] += shared_memory[tid_hw + offset][0];
            shared_memory[tid_hw][1] += shared_memory[tid_hw + offset][1];
        }
        __syncthreads();        // wait, till all threads in this block reach
    }
    
    // after this for loop, all should be set, so dump the data and calculate the partial sum
    if (tid_hw == 0) {
        partial_sum[n][c][row][col][0] = shared_memory[0][0];
        partial_sum[n][c][row][col][1] = shared_memory[0][1];
    }
}

template <typename scalar_t>
__global__ void partial_sum_conv_sram_kernel(
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> partial_sum,
    const int valid_height,
    const int valid_width,
    const int block_num_width
){
    // partial sum
    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_HW][2];

    const int n = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z * blockDim.z + threadIdx.z;
    const int tid_hw = threadIdx.x;
    const int tid_h = tid_hw / BLOCK_SIZE_W;
    const int tid_w = tid_hw - tid_h * BLOCK_SIZE_W;
    const int hw = blockIdx.x;
    const int row = hw / block_num_width;
    const int col = hw - row * block_num_width;
    const int h = row * BLOCK_SIZE_H + tid_h;
    const int w = col * BLOCK_SIZE_W + tid_w;

    // if the loc cover our data, load in shared memory
    if (n < input_data.size(0) && c < input_data.size(1) && h < valid_height && w < valid_width){
        shared_memory[tid_hw][0] = input_data[n][c][h][w][0];
        shared_memory[tid_hw][1] = input_data[n][c][h][w][1];
    } else {
        shared_memory[tid_hw][0] = static_cast<scalar_t>(0);
        shared_memory[tid_hw][1] = static_cast<scalar_t>(0);
    }
    __syncthreads();
    
    for (int offset = BLOCK_SIZE_HW >> 1; offset > 0; offset >>= 1) {
        if (tid_hw < offset) {
            shared_memory[tid_hw][0] += shared_memory[tid_hw + offset][0];
            shared_memory[tid_hw][1] += shared_memory[tid_hw + offset][1];
        }
        __syncthreads();        // wait, till all threads in this block reach
    }
    
    // after this for loop, all should be set, so dump the data and calculate the partial sum
    if (tid_hw == 0) {
        partial_sum[n][c][row][col][0] = shared_memory[0][0];
        partial_sum[n][c][row][col][1] = shared_memory[0][1];
    }
}

template <typename scalar_t>
__global__ void sum_std_conv_sram_kernel(
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> sum,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> batch_norm_output,
    const int h,
    const int w
){
    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_BATCH][BLOCK_SIZE_FEATURE][2];
    const int N = input_data.size(0);
    const int C = input_data.size(1);
    const int Nhw = N * h * w;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int thread_id_n = threadIdx.x;
    const int thread_id_c = threadIdx.y;
    
    // if the loc cover our data, load in shared memory
    if (n < N && c < C){
        // input_data: [batch][channel][h][w][sum / sum2]
        shared_memory[thread_id_n][thread_id_c][0] = input_data[n][c][0][0][0];
        shared_memory[thread_id_n][thread_id_c][1] = input_data[n][c][0][0][1];
    } else {
        shared_memory[thread_id_n][thread_id_c][0] = static_cast<scalar_t>(0);
        shared_memory[thread_id_n][thread_id_c][1] = static_cast<scalar_t>(0);
    }
    __syncthreads();   
    
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (thread_id_n < offset) {
            shared_memory[thread_id_n][thread_id_c][0] += shared_memory[thread_id_n + offset][thread_id_c][0];
            shared_memory[thread_id_n][thread_id_c][1] += shared_memory[thread_id_n + offset][thread_id_c][1];
        }
        __syncthreads();        // wait, till all threads in this block reach
    }
    
    // sum and N * std
    if (thread_id_n == 0) {
        sum[c] = shared_memory[0][thread_id_c][0];
        batch_norm_output[N][c][0][0] = sqrt(Nhw * shared_memory[0][thread_id_c][1] - shared_memory[0][thread_id_c][0] * shared_memory[0][thread_id_c][0] + Nhw * Nhw * EPSILON);
    }
}

template <typename scalar_t>
__global__ void bn_forward_conv_sram_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> sum,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> gamma,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> beta,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output_data
){
    // batch size
    const int N = input_data.size(0);
    const int h = input_data.size(2);   // height
    const int w = input_data.size(3);   // width

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (n >= input_data.size(0) || c >= input_data.size(1)) return;

    for(int i = 0 ; i < h; i++) {
        for(int j = 0; j < w; j++) {
            output_data[n][c][i][j] = gamma[c] * (N * h * w * input_data[n][c][i][j] - sum[c]) / output_data[N][c][0][0] + beta[c];
        }
    }
}


torch::Tensor bn_forward_conv_sram_cuda(
    const torch::Tensor X,
    const torch::Tensor gamma,
    const torch::Tensor beta
){
    // X: (n, c, h, w)
    const int N = X.size(0);
    const int C = X.size(1);
    const int H = X.size(2);
    const int W = X.size(3);

    // using partial sum
    // dim3: (h, w), n, c
    const int num_h = (H + BLOCK_SIZE_H - 1) / BLOCK_SIZE_H;
    const int num_w = (W + BLOCK_SIZE_W - 1) / BLOCK_SIZE_W;
    const int num_hw = num_h * num_w;
    torch::Tensor partial_sum = torch::zeros({N, C, num_h, num_w, 2}, X.options());

    const dim3 threads_partial_sum(BLOCK_SIZE_HW, 1, 1);
    const dim3 blocks_partial_sum(num_hw, N, C);

    std::cout << "blocks partial sum & sum2: " << blocks_partial_sum.x << ", " << blocks_partial_sum.y << ", " << blocks_partial_sum.z << std::endl;

    AT_DISPATCH_FLOATING_TYPES(X.type(), "partial_sum_sum2_conv_sram_kernel",
    ([&] {
        partial_sum_sum2_conv_sram_kernel<scalar_t><<<blocks_partial_sum, threads_partial_sum>>>(
            X.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            partial_sum.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
            H,
            W,
            num_w
        );
    }));
    
    // loop till all (h, w) sum to (1, 1)
    int valid_h = num_h;
    int valid_w = num_w;
    while (valid_h > 1 || valid_w > 1) {
        // xxx_loop == xxx with:
        //      1. type const
        //      2. only exists in one iteration
        const int valid_h_loop = valid_h;
        const int valid_w_loop = valid_w;
        const int num_h_loop = (valid_h_loop + BLOCK_SIZE_H - 1) / BLOCK_SIZE_H;
        const int num_w_loop = (valid_w_loop + BLOCK_SIZE_W - 1) / BLOCK_SIZE_W;
        const int num_hw_loop = num_h_loop * num_w_loop;
        const dim3 blocks_partial_sum_loop(num_hw_loop, N, C);

        std::cout << "blocks partial sum in loop: " << blocks_partial_sum_loop.x << ", " << blocks_partial_sum_loop.y << ", " << blocks_partial_sum_loop.z << std::endl;

        AT_DISPATCH_FLOATING_TYPES(X.type(), "partial_sum_conv_sram_kernel",
        ([&] {
            partial_sum_conv_sram_kernel<scalar_t><<<blocks_partial_sum_loop, threads_partial_sum>>>(
                partial_sum.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                partial_sum.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                valid_h_loop,
                valid_w_loop,
                num_w_loop
            );
        }));

        valid_h = num_h_loop;
        valid_w = num_w_loop;
    }

    // batch_norm_out: bn_forward + std_eps
    torch::Tensor batch_norm_out = torch::zeros({N + 1, C, H, W}, X.options());
    torch::Tensor sum = torch::zeros({C}, X.options());

    const dim3 threads_sum_std(BLOCK_SIZE_BATCH, BLOCK_SIZE_FEATURE);
    const dim3 blocks_sum_std((N + threads_sum_std.x - 1) / threads_sum_std.x, (C + threads_sum_std.y - 1) / threads_sum_std.y);

    std::cout << "blocks sum & std: " << blocks_sum_std.x << ", " << blocks_sum_std.y << std::endl;

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(X.type(), "sum_std_conv_sram_kernel",
    ([&] {
        sum_std_conv_sram_kernel<scalar_t><<<blocks_sum_std, threads_sum_std>>>(
            partial_sum.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
            sum.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            batch_norm_out.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            H,
            W
        );
    }));

    // batch norm will use a even dispatched block size
    const dim3 threads_batch_norm(BLOCK_SIZE_BN_X, BLOCK_SIZE_BN_Y);
    const dim3 blocks_batch_norm((N + threads_batch_norm.x - 1) / threads_batch_norm.x, (C + threads_batch_norm.y - 1) / threads_batch_norm.y);

    std::cout << "blocks batch norm: " << blocks_batch_norm.x << ", " << blocks_batch_norm.y << std::endl;

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(X.type(), "bn_forward_conv_sram_kernel",
    ([&] {
        bn_forward_conv_sram_kernel<scalar_t><<<blocks_batch_norm, threads_batch_norm>>>(
            X.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            sum.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            gamma.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            beta.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            batch_norm_out.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()
        );
    }));

    return batch_norm_out;
}
