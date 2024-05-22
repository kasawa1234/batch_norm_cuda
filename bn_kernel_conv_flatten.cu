#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#include "constants.h"


// ————————————————————————————————————————————————————————————————————————
/*                          Flatten Conv Forward                          */
// ————————————————————————————————————————————————————————————————————————


// FOR MEAN
template <typename scalar_t>
__global__ void partial_sum_conv_flatten_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output_data,
    const int valid_hw
){
    // partial sum
    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_HW];

    const int n = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z * blockDim.z + threadIdx.z;
    const int hw = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid_hw = threadIdx.x;
    const int index = blockIdx.x;

    // if the loc cover our data, load in shared memory
    if (n < input_data.size(0) && c < input_data.size(1) && hw < valid_hw){
        shared_memory[tid_hw] = input_data[n][c][hw];
    } else {
        shared_memory[tid_hw] = static_cast<scalar_t>(0);
    }
    __syncthreads();
    
    for (int offset = BLOCK_SIZE_HW >> 1; offset > 0; offset >>= 1) {
        if (tid_hw < offset) {
            shared_memory[tid_hw] += shared_memory[tid_hw + offset];
        }
        __syncthreads();        // wait, till all threads in this block reach
    }
    
    // after this for loop, all should be set, so dump the data and calculate the partial sum
    if (tid_hw == 0) {
        output_data[n][c][index] = shared_memory[0];
    }
}

template <typename scalar_t>
__global__ void mean_conv_flatten_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input_data,
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
        shared_memory[thread_id_n][thread_id_c] = input_data[n][c][0];
    } else {
        shared_memory[thread_id_n][thread_id_c] = static_cast<scalar_t>(0);
    }
    __syncthreads();   
    
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (thread_id_n < offset) {
            shared_memory[thread_id_n][thread_id_c] += shared_memory[thread_id_n + offset][thread_id_c];
        }
        __syncthreads();        // wait, till all threads in this block reach
    }
    
    // after this for loop, all should be set, so dump the data and calculate the mean
    if (thread_id_n == 0) {
        output_data[c] = shared_memory[0][thread_id_c] / static_cast<scalar_t>(input_data.size(0) * input_data.size(2));
    }
}



// FOR STD_EPS
template <typename scalar_t>
__global__ void partial_sum2_conv_flatten_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output_data,
    const int valid_hw
){
    // partial sum
    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_HW];

    const int n = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z * blockDim.z + threadIdx.z;
    const int hw = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid_hw = threadIdx.x;
    const int index = blockIdx.x;

    // if the loc cover our data, load in shared memory
    if (n < input_data.size(0) && c < input_data.size(1) && hw < valid_hw){
        shared_memory[tid_hw] = input_data[n][c][hw] * input_data[n][c][hw];
    } else {
        shared_memory[tid_hw] = static_cast<scalar_t>(0);
    }
    __syncthreads();
    
    for (int offset = BLOCK_SIZE_HW >> 1; offset > 0; offset >>= 1) {
        if (tid_hw < offset) {
            shared_memory[tid_hw] += shared_memory[tid_hw + offset];
        }
        __syncthreads();        // wait, till all threads in this block reach
    }
    
    // after this for loop, all should be set, so dump the data and calculate the partial sum
    if (tid_hw == 0) {
        output_data[n][c][index] = shared_memory[0];
    }
}

template <typename scalar_t>
__global__ void std_conv_flatten_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> mean,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> batch_norm_output
){
    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_BATCH][BLOCK_SIZE_FEATURE];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int thread_id_n = threadIdx.x;
    const int thread_id_c = threadIdx.y;
    
    // if the loc cover our data, load in shared memory
    if (n < input_data.size(0) && c < input_data.size(1)){
        shared_memory[thread_id_n][thread_id_c] = input_data[n][c][0];
    } else {
        shared_memory[thread_id_n][thread_id_c] = static_cast<scalar_t>(0);
    }
    __syncthreads();   
    
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (thread_id_n < offset) {
            shared_memory[thread_id_n][thread_id_c] += shared_memory[thread_id_n + offset][thread_id_c];
        }
        __syncthreads();        // wait, till all threads in this block reach
    }
    
    // after this for loop, all should be set, so dump the data and calculate the mean
    if (thread_id_n == 0) {
        const int N = input_data.size(0);
        batch_norm_output[N][c][0][0] = sqrt(shared_memory[0][thread_id_c] / static_cast<scalar_t>(input_data.size(0) * input_data.size(2)) - mean[c] * mean[c] + EPSILON);
    }
}

// FORWARD
template <typename scalar_t>
__global__ void bn_forward_conv_flatten_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> mean,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> gamma,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> beta,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output_data,
    const int block_num_width
){
    const int N = input_data.size(0);

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.z * blockDim.z + threadIdx.z;
    const int h = blockIdx.y / block_num_width;
    const int w = (blockIdx.y - h * block_num_width) * blockDim.y + threadIdx.y;

    if (n >= input_data.size(0) || c >= input_data.size(1) || h >= input_data.size(2) || w >= input_data.size(3)) return;

    output_data[n][c][h][w] = gamma[c] * (input_data[n][c][h][w] - mean[c]) / output_data[N][c][0][0] + beta[c];
}


torch::Tensor bn_forward_conv_flatten_cuda(
    const torch::Tensor X,
    const torch::Tensor gamma,
    const torch::Tensor beta
){
    // X: (n, c, h, w)
    const int N = X.size(0);
    const int C = X.size(1);
    const int H = X.size(2);
    const int W = X.size(3);
    const int HW = H * W;

    std::vector<int64_t> new_shape = {N, C, H * W};
    torch::Tensor X_flatten = X.view(new_shape);
    torch::Tensor partial_sum = torch::zeros(new_shape, X.options());

    const dim3 threads_partial_sum(BLOCK_SIZE_HW, 1, 1);
    const int valid_hw_first = (HW + threads_partial_sum.x - 1) / threads_partial_sum.x;
    const dim3 blocks_partial_sum(valid_hw_first, N, C);

    AT_DISPATCH_FLOATING_TYPES(X.type(), "partial_sum_conv_flatten_kernel",
    ([&] {
        partial_sum_conv_flatten_kernel<scalar_t><<<blocks_partial_sum, threads_partial_sum>>>(
            X_flatten.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            partial_sum.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            HW
        );
    }));
    
    // loop till valid_hw -> 1
    int valid_hw = valid_hw_first;
    while (valid_hw > 1) {
        // xxx_loop == xxx with:
        //      1. type const
        //      2. only exists in one iteration
        const int valid_hw_loop = valid_hw;
        const int num_hw_loop = (valid_hw + BLOCK_SIZE_HW - 1) / BLOCK_SIZE_HW;
        const dim3 blocks_partial_sum_loop(num_hw_loop, N, C);

        AT_DISPATCH_FLOATING_TYPES(X.type(), "partial_sum_conv_flatten_kernel",
        ([&] {
            partial_sum_conv_flatten_kernel<scalar_t><<<blocks_partial_sum_loop, threads_partial_sum>>>(
                partial_sum.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                partial_sum.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                valid_hw_loop
            );
        }));

        valid_hw = num_hw_loop;
    }

    // final sum
    torch::Tensor mean = torch::zeros({C}, X.options());

    const dim3 threads_mean(BLOCK_SIZE_BATCH, BLOCK_SIZE_FEATURE);
    const dim3 blocks_mean((N + threads_mean.x - 1) / threads_mean.x, (C + threads_mean.y - 1) / threads_mean.y);

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(X.type(), "mean_conv_flatten_kernel",
    ([&] {
        mean_conv_flatten_kernel<scalar_t><<<blocks_mean, threads_mean>>>(
            partial_sum.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            mean.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()
        );
    }));
    
    // calculate std
    AT_DISPATCH_FLOATING_TYPES(X.type(), "partial_sum2_conv_flatten_kernel",
    ([&] {
        partial_sum2_conv_flatten_kernel<scalar_t><<<blocks_partial_sum, threads_partial_sum>>>(
            X_flatten.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            partial_sum.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            HW
        );
    }));
    
    // same with partial sum
    // loop till all (h, w) sum to (1, 1)
    valid_hw = valid_hw_first;
    while (valid_hw > 1) {
        // xxx_loop == xxx with:
        //      1. type const
        //      2. only exists in one iteration
        const int valid_hw_loop = valid_hw;
        const int num_hw_loop = (valid_hw + BLOCK_SIZE_HW - 1) / BLOCK_SIZE_HW;
        const dim3 blocks_partial_sum_loop(num_hw_loop, N, C);

        AT_DISPATCH_FLOATING_TYPES(X.type(), "partial_sum_conv_flatten_kernel",
        ([&] {
            partial_sum_conv_flatten_kernel<scalar_t><<<blocks_partial_sum_loop, threads_partial_sum>>>(
                partial_sum.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                partial_sum.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                valid_hw_loop
            );
        }));

        valid_hw = num_hw_loop;
    }

    // batch_norm_out: bn_forward + std_eps
    torch::Tensor batch_norm_out = torch::zeros({N + 1, C, H, W}, X.options());

    const dim3 threads_std(BLOCK_SIZE_BATCH, BLOCK_SIZE_FEATURE);
    const dim3 blocks_std((N + threads_std.x - 1) / threads_std.x, (C + threads_std.y - 1) / threads_std.y);

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(X.type(), "std_conv_flatten_kernel",
    ([&] {
        std_conv_flatten_kernel<scalar_t><<<blocks_std, threads_std>>>(
            partial_sum.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            mean.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            batch_norm_out.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()
        );
    }));

    // batch norm will use a even dispatched block size
    const dim3 threads_batch_norm(BLOCK_SIZE_BN_BATCH, BLOCK_SIZE_BN_HW, 1);
    const int num_width = (W + threads_batch_norm.y - 1) / threads_batch_norm.y;
    const dim3 blocks_batch_norm((N + threads_batch_norm.x - 1) / threads_batch_norm.x, num_width * H, C);

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(X.type(), "bn_forward_conv_flatten_kernel",
    ([&] {
        bn_forward_conv_flatten_kernel<scalar_t><<<blocks_batch_norm, threads_batch_norm>>>(
            X.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            mean.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            gamma.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            beta.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            batch_norm_out.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            num_width
        );
    }));

    return batch_norm_out;
}