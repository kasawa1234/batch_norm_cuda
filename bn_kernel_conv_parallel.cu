#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#include "constants.h"


// ————————————————————————————————————————————————————————————————————————
/*                          Parallel Conv Forward                          */
// ————————————————————————————————————————————————————————————————————————


template <typename scalar_t>
__global__ void partial_sum_conv_parallel_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> partial_sum,
    const int valid_height,
    const int valid_width,
    const int block_num_width
){
    // partial sum
    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_HW];

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
        shared_memory[tid_hw] = input_data[n][c][h][w];
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
        partial_sum[n][c][row][col] = shared_memory[0];
    }
}

template <typename scalar_t>
__global__ void mean_conv_parallel_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> output_data,
    const int h,
    const int w
){
    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_BATCH][BLOCK_SIZE_FEATURE];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int thread_id_n = threadIdx.x;
    const int thread_id_c = threadIdx.y;
    
    // if the loc cover our data, load in shared memory
    if (n < input_data.size(0) && c < input_data.size(1)){
        shared_memory[thread_id_n][thread_id_c] = input_data[n][c][0][0];
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
        output_data[c] = shared_memory[0][thread_id_c] / static_cast<scalar_t>(input_data.size(0) * h * w);
    }
}

template <typename scalar_t>
__global__ void partial_sum2_conv_parallel_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> partial_sum2,
    const int valid_height,
    const int valid_width,
    const int block_num_width
){
    // calculate \sum x_i^2 for the first time
    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_HW];

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
        shared_memory[tid_hw] = input_data[n][c][h][w] * input_data[n][c][h][w];
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
        partial_sum2[n][c][row][col] = shared_memory[0];
    }
}

template <typename scalar_t>
__global__ void std_conv_parallel_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> mean,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> batch_norm_output,
    const int h,
    const int w
){
    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_BATCH][BLOCK_SIZE_FEATURE];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int thread_id_n = threadIdx.x;
    const int thread_id_c = threadIdx.y;
    
    // if the loc cover our data, load in shared memory
    if (n < input_data.size(0) && c < input_data.size(1)){
        shared_memory[thread_id_n][thread_id_c] = input_data[n][c][0][0];
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
        batch_norm_output[N][c][0][0] = sqrt(shared_memory[0][thread_id_c] / static_cast<scalar_t>(N * h * w) - mean[c] * mean[c] + EPSILON);
    }
}

template <typename scalar_t>
__global__ void bn_forward_conv_parallel_kernel(
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

    if (n >= input_data.size(0) || c >= input_data.size(1)) return;

    output_data[n][c][h][w] = gamma[c] * (input_data[n][c][h][w] - mean[c]) / output_data[N][c][0][0] + beta[c];
}


torch::Tensor bn_forward_conv_parallel_cuda(
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
    torch::Tensor partial_sum = torch::zeros({N, C, num_h, num_w}, X.options());

    const dim3 threads_partial_sum(BLOCK_SIZE_HW, 1, 1);
    const dim3 blocks_partial_sum(num_hw, N, C);

    // std::cout << "blocks partial sum: " << blocks_partial_sum.x << ", " << blocks_partial_sum.y << ", " << blocks_partial_sum.z << std::endl;

    AT_DISPATCH_FLOATING_TYPES(X.type(), "partial_sum_conv_parallel_kernel",
    ([&] {
        partial_sum_conv_parallel_kernel<scalar_t><<<blocks_partial_sum, threads_partial_sum>>>(
            X.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            partial_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
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

        // std::cout << "blocks partial sum in loop: " << blocks_partial_sum_loop.x << ", " << blocks_partial_sum_loop.y << ", " << blocks_partial_sum_loop.z << std::endl;

        AT_DISPATCH_FLOATING_TYPES(X.type(), "partial_sum_conv_parallel_kernel",
        ([&] {
            partial_sum_conv_parallel_kernel<scalar_t><<<blocks_partial_sum_loop, threads_partial_sum>>>(
                partial_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                partial_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                valid_h_loop,
                valid_w_loop,
                num_w_loop
            );
        }));

        valid_h = num_h_loop;
        valid_w = num_w_loop;
    }

    // sum in partial_sum[n][c][0][0]
    // final sum
    torch::Tensor mean = torch::zeros({C}, X.options());

    const dim3 threads_mean(BLOCK_SIZE_BATCH, BLOCK_SIZE_FEATURE);
    const dim3 blocks_mean((N + threads_mean.x - 1) / threads_mean.x, (C + threads_mean.y - 1) / threads_mean.y);

    // std::cout << "blocks mean: " << blocks_mean.x << ", " << blocks_mean.y << std::endl;

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(X.type(), "mean_conv_parallel_kernel",
    ([&] {
        mean_conv_parallel_kernel<scalar_t><<<blocks_mean, threads_mean>>>(
            partial_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            mean.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            H,
            W
        );
    }));
    
    // calculate std

    // using partial sum for std too
    // dim3: (h, w), n, c
    torch::Tensor partial_sum2 = torch::zeros({N, C, num_h, num_w}, X.options());

    const dim3 threads_partial_sum2(BLOCK_SIZE_HW, 1, 1);
    const dim3 blocks_partial_sum2(num_hw, N, C);

    // std::cout << "blocks partial sum2: " << blocks_partial_sum2.x << ", " << blocks_partial_sum2.y << ", " << blocks_partial_sum2.z << std::endl;

    AT_DISPATCH_FLOATING_TYPES(X.type(), "partial_sum2_conv_parallel_kernel",
    ([&] {
        partial_sum2_conv_parallel_kernel<scalar_t><<<blocks_partial_sum, threads_partial_sum>>>(
            X.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            partial_sum2.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            H,
            W,
            num_w
        );
    }));
    
    // same with partial sum
    // loop till all (h, w) sum to (1, 1)
    valid_h = num_h;
    valid_w = num_w;
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

        // std::cout << "blocks partial sum in loop: " << blocks_partial_sum_loop.x << ", " << blocks_partial_sum_loop.y << ", " << blocks_partial_sum_loop.z << std::endl;

        AT_DISPATCH_FLOATING_TYPES(X.type(), "partial_sum_conv_parallel_kernel",
        ([&] {
            partial_sum_conv_parallel_kernel<scalar_t><<<blocks_partial_sum_loop, threads_partial_sum>>>(
                partial_sum2.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                partial_sum2.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                valid_h_loop,
                valid_w_loop,
                num_w_loop
            );
        }));

        valid_h = num_h_loop;
        valid_w = num_w_loop;
    }

    // sum2 in partial_sum[n][c][0][0]
    // batch_norm_out: bn_forward + std_eps
    torch::Tensor batch_norm_out = torch::zeros({N + 1, C, H, W}, X.options());

    const dim3 threads_std(BLOCK_SIZE_BATCH, BLOCK_SIZE_FEATURE);
    const dim3 blocks_std((N + threads_std.x - 1) / threads_std.x, (C + threads_std.y - 1) / threads_std.y);

    // std::cout << "blocks std: " << blocks_std.x << ", " << blocks_std.y << std::endl;

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(X.type(), "std_conv_parallel_kernel",
    ([&] {
        std_conv_parallel_kernel<scalar_t><<<blocks_std, threads_std>>>(
            partial_sum2.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            mean.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            batch_norm_out.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            H,
            W
        );
    }));

    // batch norm will use a even dispatched block size
    const dim3 threads_batch_norm(BLOCK_SIZE_BN_BATCH, BLOCK_SIZE_BN_HW, 1);
    const int num_width = (W + threads_batch_norm.y - 1) / threads_batch_norm.y;
    const dim3 blocks_batch_norm((N + threads_batch_norm.x - 1) / threads_batch_norm.x, num_width * H, C);

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(X.type(), "bn_forward_conv_parallel_kernel",
    ([&] {
        bn_forward_conv_parallel_kernel<scalar_t><<<blocks_batch_norm, threads_batch_norm>>>(
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



// —————————————————————————————————————————————————————————————————————————
/*                          Parallel Conv Backward                          */
// —————————————————————————————————————————————————————————————————————————


template <typename scalar_t>
__global__ void dx_sum_conv_parallel_hw_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dout,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> gamma_1d,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> partial_sum,
    const int valid_height,
    const int valid_width,
    const int block_num_width
){
    // partial sum
    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_HW];

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
    if (n < dL_dout.size(0) && c < dL_dout.size(1) && h < valid_height && w < valid_width){
        shared_memory[tid_hw] = dL_dout[n][c][h][w] * gamma_1d[c];
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
        partial_sum[n][c][row][col] = shared_memory[0];
    }
}

// loop can use partial_sum

template <typename scalar_t>
__global__ void dx_sum_conv_parallel_n_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> dx_sum
){
    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_BATCH][BLOCK_SIZE_FEATURE];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int thread_id_n = threadIdx.x;
    const int thread_id_c = threadIdx.y;
    
    // if the loc cover our data, load in shared memory
    if (n < input_data.size(0) && c < input_data.size(1)){
        shared_memory[thread_id_n][thread_id_c] = input_data[n][c][0][0];
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
        dx_sum[c] = shared_memory[0][thread_id_c];
    }
}

template <typename scalar_t>
__global__ void dx_norm_sum_conv_parallel_hw_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dout,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> gamma_1d,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> normalized,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> partial_sum,
    const int valid_height,
    const int valid_width,
    const int block_num_width
){
    // partial sum
    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_HW];

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
    if (n < dL_dout.size(0) && c < dL_dout.size(1) && h < valid_height && w < valid_width){
        shared_memory[tid_hw] = dL_dout[n][c][h][w] * gamma_1d[c] * normalized[n][c][h][w];
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
        partial_sum[n][c][row][col] = shared_memory[0];
    }
}

// loop can use partial_sum

// dx_norm_sum_conv_parallel_n == dx_sum_conv_parallel_n

template <typename scalar_t>
__global__ void grad_gamma_conv_parallel_hw_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dout,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> normalized,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> partial_sum,
    const int valid_height,
    const int valid_width,
    const int block_num_width
){
    // partial sum
    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_HW];

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
    if (n < dL_dout.size(0) && c < dL_dout.size(1) && h < valid_height && w < valid_width){
        shared_memory[tid_hw] = dL_dout[n][c][h][w] * normalized[n][c][h][w];
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
        partial_sum[n][c][row][col] = shared_memory[0];
    }
}

// loop can use partial_sum

template <typename scalar_t>
__global__ void grad_gamma_conv_parallel_n_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output
){
    const int N = input_data.size(0);

    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_BATCH][BLOCK_SIZE_FEATURE];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int thread_id_n = threadIdx.x;
    const int thread_id_c = threadIdx.y;
    
    // if the loc cover our data, load in shared memory
    if (n < input_data.size(0) && c < input_data.size(1)){
        shared_memory[thread_id_n][thread_id_c] = input_data[n][c][0][0];
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
        output[N][c][0][0] = shared_memory[0][thread_id_c];
    }
}


template <typename scalar_t>
__global__ void grad_beta_conv_parallel_hw_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dout,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> partial_sum,
    const int valid_height,
    const int valid_width,
    const int block_num_width
){
    // partial sum
    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_HW];

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
    if (n < dL_dout.size(0) && c < dL_dout.size(1) && h < valid_height && w < valid_width){
        shared_memory[tid_hw] = dL_dout[n][c][h][w];
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
        partial_sum[n][c][row][col] = shared_memory[0];
    }
}

// loop can use partial_sum

template <typename scalar_t>
__global__ void grad_beta_conv_parallel_n_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output
){
    const int N = input_data.size(0);

    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_BATCH][BLOCK_SIZE_FEATURE];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int thread_id_n = threadIdx.x;
    const int thread_id_c = threadIdx.y;
    
    // if the loc cover our data, load in shared memory
    if (n < input_data.size(0) && c < input_data.size(1)){
        shared_memory[thread_id_n][thread_id_c] = input_data[n][c][0][0];
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
        output[N + 1][c][0][0] = shared_memory[0][thread_id_c];
    }
}

template <typename scalar_t>
__global__ void bn_backward_input_conv_parallel_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dout,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> gamma,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> dx_sum,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> dx_norm_sum,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> normalized,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> std_eps,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dinput,
    const int block_num_width
){
    const int N = normalized.size(0);

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.z * blockDim.z + threadIdx.z;
    const int h = blockIdx.y / block_num_width;
    const int w = (blockIdx.y - h * block_num_width) * blockDim.y + threadIdx.y;

    if (n >= normalized.size(0) || c >= normalized.size(1)) return;

    dL_dinput[n][c][h][w] = (N * h * w * dL_dout[n][c][h][w] * gamma[c] - dx_sum[c] - normalized[n][c][h][w] * dx_norm_sum[c]) / (N * h * w * std_eps[c]);
}


torch::Tensor bn_backward_conv_parallel_cuda(
    const torch::Tensor dL_dout,
    const torch::Tensor normalized,
    const torch::Tensor gamma,
    const torch::Tensor std_eps
){
    // dL_dout: (n, c, h, w)
    const int N = dL_dout.size(0);
    const int C = dL_dout.size(1);
    const int H = dL_dout.size(2);
    const int W = dL_dout.size(3);

    // dx_sum
    // using partial sum
    // dim3: (h, w), n, c
    const int num_h = (H + BLOCK_SIZE_H - 1) / BLOCK_SIZE_H;
    const int num_w = (W + BLOCK_SIZE_W - 1) / BLOCK_SIZE_W;
    const int num_hw = num_h * num_w;
    torch::Tensor partial_sum = torch::zeros({N, C, num_h, num_w}, dL_dout.options());

    const dim3 threads_partial_sum(BLOCK_SIZE_HW, 1, 1);
    const dim3 blocks_partial_sum(num_hw, N, C);

    // std::cout << "blocks partial dx_sum: " << blocks_partial_sum.x << ", " << blocks_partial_sum.y << ", " << blocks_partial_sum.z << std::endl;

    AT_DISPATCH_FLOATING_TYPES(dL_dout.type(), "dx_sum_conv_parallel_hw_kernel",
    ([&] {
        dx_sum_conv_parallel_hw_kernel<scalar_t><<<blocks_partial_sum, threads_partial_sum>>>(
            dL_dout.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            gamma.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            partial_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
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

        // std::cout << "blocks partial dx_sum in loop: " << blocks_partial_sum_loop.x << ", " << blocks_partial_sum_loop.y << ", " << blocks_partial_sum_loop.z << std::endl;

        AT_DISPATCH_FLOATING_TYPES(dL_dout.type(), "partial_sum_conv_parallel_kernel",
        ([&] {
            partial_sum_conv_parallel_kernel<scalar_t><<<blocks_partial_sum_loop, threads_partial_sum>>>(
                partial_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                partial_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                valid_h_loop,
                valid_w_loop,
                num_w_loop
            );
        }));

        valid_h = num_h_loop;
        valid_w = num_w_loop;
    }

    torch::Tensor dx_sum = torch::zeros({C}, dL_dout.options());

    const dim3 threads_mean(BLOCK_SIZE_BATCH, BLOCK_SIZE_FEATURE);
    const dim3 blocks_mean((N + threads_mean.x - 1) / threads_mean.x, (C + threads_mean.y - 1) / threads_mean.y);

    // std::cout << "blocks dx_sum final: " << blocks_mean.x << ", " << blocks_mean.y << std::endl;

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(dL_dout.type(), "dx_sum_conv_parallel_n_kernel",
    ([&] {
        dx_sum_conv_parallel_n_kernel<scalar_t><<<blocks_mean, threads_mean>>>(
            partial_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            dx_sum.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()
        );
    }));


    // dx_norm_sum

    // std::cout << "blocks partial dx_norm_sum: " << blocks_partial_sum.x << ", " << blocks_partial_sum.y << ", " << blocks_partial_sum.z << std::endl;

    AT_DISPATCH_FLOATING_TYPES(dL_dout.type(), "dx_norm_sum_conv_parallel_hw_kernel",
    ([&] {
        dx_norm_sum_conv_parallel_hw_kernel<scalar_t><<<blocks_partial_sum, threads_partial_sum>>>(
            dL_dout.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            gamma.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            normalized.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            partial_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            H,
            W,
            num_w
        );
    }));
    
    // same with partial sum
    // loop till all (h, w) sum to (1, 1)
    valid_h = num_h;
    valid_w = num_w;
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

        // std::cout << "blocks partial sum in loop: " << blocks_partial_sum_loop.x << ", " << blocks_partial_sum_loop.y << ", " << blocks_partial_sum_loop.z << std::endl;

        AT_DISPATCH_FLOATING_TYPES(dL_dout.type(), "partial_sum_conv_parallel_kernel",
        ([&] {
            partial_sum_conv_parallel_kernel<scalar_t><<<blocks_partial_sum_loop, threads_partial_sum>>>(
                partial_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                partial_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                valid_h_loop,
                valid_w_loop,
                num_w_loop
            );
        }));

        valid_h = num_h_loop;
        valid_w = num_w_loop;
    }

    torch::Tensor dx_norm_sum = torch::zeros({C}, dL_dout.options());

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(dL_dout.type(), "dx_sum_conv_parallel_n_kernel",
    ([&] {
        dx_sum_conv_parallel_n_kernel<scalar_t><<<blocks_mean, threads_mean>>>(
            partial_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            dx_norm_sum.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()
        );
    }));

    // grad_gamma

    AT_DISPATCH_FLOATING_TYPES(dL_dout.type(), "grad_gamma_conv_parallel_hw_kernel",
    ([&] {
        grad_gamma_conv_parallel_hw_kernel<scalar_t><<<blocks_partial_sum, threads_partial_sum>>>(
            dL_dout.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            normalized.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            partial_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            H,
            W,
            num_w
        );
    }));
    
    // same with partial sum
    // loop till all (h, w) sum to (1, 1)
    valid_h = num_h;
    valid_w = num_w;
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

        // std::cout << "blocks partial sum in loop: " << blocks_partial_sum_loop.x << ", " << blocks_partial_sum_loop.y << ", " << blocks_partial_sum_loop.z << std::endl;

        AT_DISPATCH_FLOATING_TYPES(dL_dout.type(), "partial_sum_conv_parallel_kernel",
        ([&] {
            partial_sum_conv_parallel_kernel<scalar_t><<<blocks_partial_sum_loop, threads_partial_sum>>>(
                partial_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                partial_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                valid_h_loop,
                valid_w_loop,
                num_w_loop
            );
        }));

        valid_h = num_h_loop;
        valid_w = num_w_loop;
    }

    torch::Tensor bn_backward_output = torch::zeros({N + 2, C, H, W}, dL_dout.options());

    // std::cout << "blocks grad_gamma final: " << blocks_mean.x << ", " << blocks_mean.y << std::endl;

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(dL_dout.type(), "grad_gamma_conv_parallel_n_kernel",
    ([&] {
        grad_gamma_conv_parallel_n_kernel<scalar_t><<<blocks_mean, threads_mean>>>(
            partial_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            bn_backward_output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()
        );
    }));


    // grad_beta
    // std::cout << "blocks partial grad_beta: " << blocks_partial_sum.x << ", " << blocks_partial_sum.y << ", " << blocks_partial_sum.z << std::endl;

    AT_DISPATCH_FLOATING_TYPES(dL_dout.type(), "grad_beta_conv_parallel_hw_kernel",
    ([&] {
        grad_beta_conv_parallel_hw_kernel<scalar_t><<<blocks_partial_sum, threads_partial_sum>>>(
            dL_dout.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            partial_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            H,
            W,
            num_w
        );
    }));
    
    // same with partial sum
    // loop till all (h, w) sum to (1, 1)
    valid_h = num_h;
    valid_w = num_w;
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

        // std::cout << "blocks partial sum in loop: " << blocks_partial_sum_loop.x << ", " << blocks_partial_sum_loop.y << ", " << blocks_partial_sum_loop.z << std::endl;

        AT_DISPATCH_FLOATING_TYPES(dL_dout.type(), "partial_sum_conv_parallel_kernel",
        ([&] {
            partial_sum_conv_parallel_kernel<scalar_t><<<blocks_partial_sum_loop, threads_partial_sum>>>(
                partial_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                partial_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                valid_h_loop,
                valid_w_loop,
                num_w_loop
            );
        }));

        valid_h = num_h_loop;
        valid_w = num_w_loop;
    }

    // std::cout << "blocks grad_gamma final: " << blocks_mean.x << ", " << blocks_mean.y << std::endl;

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(dL_dout.type(), "grad_beta_conv_parallel_n_kernel",
    ([&] {
        grad_beta_conv_parallel_n_kernel<scalar_t><<<blocks_mean, threads_mean>>>(
            partial_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            bn_backward_output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()
        );
    }));


    // same as forward_conv_parallel
    const dim3 threads_batch_norm(BLOCK_SIZE_BN_BATCH, BLOCK_SIZE_BN_HW, 1);
    const int num_width = (W + threads_batch_norm.y - 1) / threads_batch_norm.y;
    const dim3 blocks_batch_norm((N + threads_batch_norm.x - 1) / threads_batch_norm.x, num_width * H, C);

    // std::cout << "blocks batch norm backwards: " << blocks_batch_norm.x << ", " << blocks_batch_norm.y << std::endl;

    AT_DISPATCH_FLOATING_TYPES(normalized.type(), "bn_backward_input_conv_parallel_kernel",
    ([&] {
        bn_backward_input_conv_parallel_kernel<scalar_t><<<blocks_batch_norm, threads_batch_norm>>>(
            dL_dout.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            gamma.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            dx_sum.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            dx_norm_sum.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            normalized.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            std_eps.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            bn_backward_output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            num_width
        );
    }));

    return bn_backward_output;
}
