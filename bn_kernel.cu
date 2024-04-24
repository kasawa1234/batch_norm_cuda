#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#include "constants.h"


template <typename scalar_t>
__global__ void mean_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> output_data
){
    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_BATCH][BLOCK_SIZE_FEATURE];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int thread_id_n = threadIdx.x;
    const int thread_id_c = threadIdx.y;

    // if the loc cover our data, load in shared memory
    if (n < input_data.size(0) && c < input_data.size(1)){
        shared_memory[thread_id_n][thread_id_c] = input_data[n][c];
    } else {
        shared_memory[thread_id_n][thread_id_c] = static_cast<scalar_t>(0);
    }
    __syncthreads();            // need to fully load all items into shared_memory

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (thread_id_n < offset) {
            shared_memory[thread_id_n][thread_id_c] += shared_memory[thread_id_n + offset][thread_id_c];
        }
        __syncthreads();        // wait, till all threads in this block reach
    }
    
    // after this for loop, all should be set, so dump the data and calculate the mean
    if (thread_id_n == 0) {
        output_data[c] = shared_memory[0][thread_id_c] / static_cast<scalar_t>(input_data.size(0));
    }
}

template <typename scalar_t>
__global__ void var_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> mean,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> var
){
    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_BATCH][BLOCK_SIZE_FEATURE];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int thread_id_n = threadIdx.x;
    const int thread_id_c = threadIdx.y;

    // if the loc cover our data, load in shared memory
    if (n < input_data.size(0) && c < input_data.size(1)){
        shared_memory[thread_id_n][thread_id_c] = (input_data[n][c] - mean[c]) * (input_data[n][c] - mean[c]);
    } else {
        shared_memory[thread_id_n][thread_id_c] = static_cast<scalar_t>(0);
    }
    __syncthreads();            // need to fully load all items into shared_memory

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (thread_id_n < offset) {
            shared_memory[thread_id_n][thread_id_c] += shared_memory[thread_id_n + offset][thread_id_c];
        }
        __syncthreads();        // wait, till all threads in this block reach
    }
    
    // after this for loop, all should be set, so dump the data and calculate the var
    if (thread_id_n == 0) {
        var[c] = shared_memory[0][thread_id_c] / static_cast<scalar_t>(input_data.size(0));
    }
}

template <typename scalar_t>
__global__ void batch_norm_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> mean,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> var,
    const float gamma,
    const float beta,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output_data
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (n >= input_data.size(0) || c >= input_data.size(1)) return;

    output_data[n][c] = gamma * (input_data[n][c] - mean[c]) / sqrt(var[c] + EPSILON) + beta;
}


torch::Tensor bn_forward_mlp_cuda(
    const torch::Tensor X,
    const float gamma,
    const float beta
){
    // X: (n, c), n is parallel
    const int N = X.size(0);
    const int C = X.size(1);
    std::cout << N << ", " << C << std::endl;

    torch::Tensor mean = torch::zeros({C}, X.options());
    
    const dim3 threads_mean(BLOCK_SIZE_BATCH, BLOCK_SIZE_FEATURE);
    const dim3 blocks_mean((N + threads_mean.x - 1) / threads_mean.x, (C + threads_mean.y - 1) / threads_mean.y);

    std::cout << "blocks mean: " << blocks_mean.x << ", " << blocks_mean.y << std::endl;

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(X.type(), "mean_kernel",
    ([&] {
        mean_kernel<scalar_t><<<blocks_mean, threads_mean>>>(
            X.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            mean.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()
        );
    }));

    // calculate variance
    torch::Tensor var = torch::zeros({C}, X.options());

    // variance share the same block size with mean
    std::cout << "blocks var: " << blocks_mean.x << ", " << blocks_mean.y << std::endl;

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(X.type(), "var_kernel",
    ([&] {
        var_kernel<scalar_t><<<blocks_mean, threads_mean>>>(
            X.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            mean.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            var.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()
        );
    }));

    // batch norm
    torch::Tensor batch_norm_out = torch::zeros({N, C}, X.options());

    // batch norm will use a even dispatched block size
    const dim3 threads_batch_norm(BLOCK_SIZE_BN_X, BLOCK_SIZE_BN_Y);
    const dim3 blocks_batch_norm((N + threads_batch_norm.x - 1) / threads_batch_norm.x, (C + threads_batch_norm.y - 1) / threads_batch_norm.y);

    std::cout << "blocks batch norm: " << blocks_batch_norm.x << ", " << blocks_batch_norm.y << std::endl;

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(X.type(), "batch_norm_kernel",
    ([&] {
        batch_norm_kernel<scalar_t><<<blocks_batch_norm, threads_batch_norm>>>(
            X.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            mean.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            var.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            gamma,
            beta,
            batch_norm_out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));

    return batch_norm_out;
}

// torch::Tensor bn_forward_cuda_conv(
//     const torch::Tensor X,
//     const int a,
//     const int b
// ){
//     // X: (n, c, h, w), n, h, w are parallel
//     return 0;
// }
