#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#include "constants.h"


// ———————————————————————————————————————————————————————————————
/*                          Conv Forward                          */
// ———————————————————————————————————————————————————————————————


template <typename scalar_t>
__global__ void mean_conv_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> output_data
){
    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_BATCH][BLOCK_SIZE_FEATURE];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int thread_id_n = threadIdx.x;
    const int thread_id_c = threadIdx.y;
    const int h = input_data.size(2);   // height
    const int w = input_data.size(3);   // width

    
    // if the loc cover our data, load in shared memory
    if (n < input_data.size(0) && c < input_data.size(1)){
        shared_memory[thread_id_n][thread_id_c] = static_cast<scalar_t>(0);
        for(int i = 0 ; i < h; i++){
            for(int j = 0; j < w; j++){
                shared_memory[thread_id_n][thread_id_c] += input_data[n][c][i][j];
            }
        }
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
__global__ void std_conv_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> mean,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> batch_norm_output
){
    const int N = input_data.size(0);
    const int h = input_data.size(2);   //height
    const int w = input_data.size(3);   //width

    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_BATCH][BLOCK_SIZE_FEATURE];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int thread_id_n = threadIdx.x;
    const int thread_id_c = threadIdx.y;


    // if the loc cover our data, load in shared memory
    if (n < input_data.size(0) && c < input_data.size(1)){
        shared_memory[thread_id_n][thread_id_c] = static_cast<scalar_t>(0);
        for(int i = 0 ; i < h; i++){
            for(int j = 0; j < w; j++){
                shared_memory[thread_id_n][thread_id_c] += (input_data[n][c][i][j] - mean[c]) * (input_data[n][c][i][j] - mean[c]);
            }
        }
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
    
    // after this for loop, all should be set, so dump the data and calculate the std
    if (thread_id_n == 0) {
        batch_norm_output[N][c][0][0] = sqrt(shared_memory[0][thread_id_c] / static_cast<scalar_t>(input_data.size(0) * h * w + EPSILON));
    }
}

template <typename scalar_t>
__global__ void bn_forward_conv_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> mean,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> gamma,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> beta,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output_data
){
    // batch size
    const int N = input_data.size(0);
    const int h = input_data.size(2);   //height
    const int w = input_data.size(3);   //width

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (n >= input_data.size(0) || c >= input_data.size(1)) return;

    for(int i = 0 ; i < h; i++){
        for(int j = 0; j < w; j++){
            output_data[n][c][i][j] = gamma[c] * (input_data[n][c][i][j] - mean[c]) / output_data[N][c][0][0] + beta[c];
        }
    }
}

torch::Tensor bn_forward_conv_cuda(
    const torch::Tensor X,
    const torch::Tensor gamma,
    const torch::Tensor beta
){
    // X: (n, c), n is parallel
    const int N = X.size(0);
    const int C = X.size(1);
    const int H = X.size(2);
    const int W = X.size(3);
    // std::cout << N << ", " << C << std::endl;

    torch::Tensor mean = torch::zeros({C}, X.options());

    const dim3 threads_mean(BLOCK_SIZE_BATCH, BLOCK_SIZE_FEATURE);
    const dim3 blocks_mean((N + threads_mean.x - 1) / threads_mean.x, (C + threads_mean.y - 1) / threads_mean.y);

    // std::cout << "blocks mean: " << blocks_mean.x << ", " << blocks_mean.y << std::endl;

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(X.type(), "mean_conv_kernel",
    ([&] {
        mean_conv_kernel<scalar_t><<<blocks_mean, threads_mean>>>(
            X.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            mean.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()
        );
    }));
    
    // calculate std
    // batch_norm_out: bn_forward + std_eps
    torch::Tensor batch_norm_out = torch::zeros({N + 1, C, H, W}, X.options());

    // standard share the same block size with mean
    // std::cout << "blocks std: " << blocks_mean.x << ", " << blocks_mean.y << std::endl;

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(X.type(), "std_conv_kernel",
    ([&] {
        std_conv_kernel<scalar_t><<<blocks_mean, threads_mean>>>(
            X.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            mean.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            batch_norm_out.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()
        );
    }));

    // batch norm will use a even dispatched block size
    const dim3 threads_batch_norm(BLOCK_SIZE_BN_X, BLOCK_SIZE_BN_Y);
    const dim3 blocks_batch_norm((N + threads_batch_norm.x - 1) / threads_batch_norm.x, (C + threads_batch_norm.y - 1) / threads_batch_norm.y);

    // std::cout << "blocks batch norm: " << blocks_batch_norm.x << ", " << blocks_batch_norm.y << std::endl;

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(X.type(), "bn_forward_conv_kernel",
    ([&] {
        bn_forward_conv_kernel<scalar_t><<<blocks_batch_norm, threads_batch_norm>>>(
            X.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            mean.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            gamma.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            beta.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            batch_norm_out.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()
        );
    }));
    return batch_norm_out;
}

/*                          Conv Backward                          */

template <typename scalar_t>
__global__ void dx_sum_conv_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dout,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> gamma_1d,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> dx_sum
){
    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_BATCH][BLOCK_SIZE_FEATURE];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int thread_id_n = threadIdx.x;
    const int thread_id_c = threadIdx.y;

    const int h = dL_dout.size(2);   //height
    const int w = dL_dout.size(3);   //width
    

    // if the loc cover our data, load in shared memory
    if (n < dL_dout.size(0) && c < dL_dout.size(1)){
        shared_memory[thread_id_n][thread_id_c] = static_cast<scalar_t>(0);
        for(int i = 0; i < h; i++){
            for(int j = 0; j < w; j++){
                shared_memory[thread_id_n][thread_id_c] += dL_dout[n][c][i][j] * gamma_1d[c];
            }
        }
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
        dx_sum[c] = shared_memory[0][thread_id_c];
    }
}

template <typename scalar_t>
__global__ void dx_norm_sum_conv_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dout,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> gamma,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> normalized,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> dx_norm_sum
){
    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_BATCH][BLOCK_SIZE_FEATURE];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int thread_id_n = threadIdx.x;
    const int thread_id_c = threadIdx.y;

    const int h = dL_dout.size(2);   //height
    const int w = dL_dout.size(3);   //width

    // if the loc cover our data, load in shared memory
    if (n < dL_dout.size(0) && c < dL_dout.size(1)){
        shared_memory[thread_id_n][thread_id_c] = static_cast<scalar_t>(0);
        for(int i = 0; i < h; i++){
            for(int j = 0; j < w; j++){
                shared_memory[thread_id_n][thread_id_c] += dL_dout[n][c][i][j] * gamma[c] * normalized[n][c][i][j];
            }
        }
    
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
        dx_norm_sum[c] = shared_memory[0][thread_id_c];
    }
}

template <typename scalar_t>
__global__ void grad_gamma_conv_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dout,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> normalized,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output
){
    const int N = dL_dout.size(0);

    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_BATCH][BLOCK_SIZE_FEATURE];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int thread_id_n = threadIdx.x;
    const int thread_id_c = threadIdx.y;

    const int h = dL_dout.size(2);   // height
    const int w = dL_dout.size(3);   // width

    // if the loc cover our data, load in shared memory
    if (n < dL_dout.size(0) && c < dL_dout.size(1)){
        shared_memory[thread_id_n][thread_id_c] = static_cast<scalar_t>(0);
        for(int i = 0; i < h; i++){
            for(int j = 0; j < w; j++){
                shared_memory[thread_id_n][thread_id_c] += dL_dout[n][c][i][j] * normalized[n][c][i][j];
            }
        }
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
        output[N][c][0][0] = shared_memory[0][thread_id_c];
    }
}

template <typename scalar_t>
__global__ void grad_beta_conv_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dout,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output
){
    const int N = dL_dout.size(0);

    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_BATCH][BLOCK_SIZE_FEATURE];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int thread_id_n = threadIdx.x;
    const int thread_id_c = threadIdx.y;

    const int h = dL_dout.size(2);   // height
    const int w = dL_dout.size(3);   // width

    // if the loc cover our data, load in shared memory
    if (n < dL_dout.size(0) && c < dL_dout.size(1)){
        shared_memory[thread_id_n][thread_id_c] = static_cast<scalar_t>(0);
        for(int i = 0; i < h; i++){
            for(int j = 0; j < w; j++){
                shared_memory[thread_id_n][thread_id_c] += dL_dout[n][c][i][j];
            }
        }

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
        output[N + 1][c][0][0] = shared_memory[0][thread_id_c];
    }
}

template <typename scalar_t>
__global__ void bn_backward_input_conv_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dout,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> gamma,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> dx_sum,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> dx_norm_sum,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> normalized,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> std_eps,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dinput
){
    const int N = normalized.size(0);
    const int h = dL_dout.size(2);   // height  
    const int w = dL_dout.size(3);   // width

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (n >= normalized.size(0) || c >= normalized.size(1)) return;

    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            dL_dinput[n][c][i][j] = (N * h * w * dL_dout[n][c][i][j] * gamma[c] - dx_sum[c] - normalized[n][c][i][j] * dx_norm_sum[c]) / (N * h * w * std_eps[c]);
        }
    }
}

torch::Tensor bn_backward_conv_cuda(
    const torch::Tensor dL_dout,
    const torch::Tensor normalized,
    const torch::Tensor gamma,
    const torch::Tensor std_eps
){
    const int N = normalized.size(0);
    const int C = normalized.size(1);
    const int H = normalized.size(2);
    const int W = normalized.size(3);
    // std::cout << N << ", " << C << std::endl;

    torch::Tensor dx_sum = torch::zeros({C}, normalized.options());

    // using the same block size as mean
    const dim3 threads_sum(BLOCK_SIZE_BATCH, BLOCK_SIZE_FEATURE);
    const dim3 blocks_sum((N + threads_sum.x - 1) / threads_sum.x, (C + threads_sum.y - 1) / threads_sum.y);

    // std::cout << "blocks dx_sum: " << blocks_sum.x << ", " << blocks_sum.y << std::endl;

    AT_DISPATCH_FLOATING_TYPES(normalized.type(), "dx_sum_conv_kernel",
    ([&] {
        dx_sum_conv_kernel<scalar_t><<<blocks_sum, threads_sum>>>(
            dL_dout.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            gamma.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            dx_sum.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()
        );
    }));

    torch::Tensor dx_norm_sum = torch::zeros({C}, normalized.options());

    // std::cout << "blocks dx_norm_sum: " << blocks_sum.x << ", " << blocks_sum.y << std::endl;

    AT_DISPATCH_FLOATING_TYPES(normalized.type(), "dx_norm_sum_conv_kernel",
    ([&] {
        dx_norm_sum_conv_kernel<scalar_t><<<blocks_sum, threads_sum>>>(
            dL_dout.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            gamma.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            normalized.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            dx_norm_sum.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()
        );
    }));

    // bn_backward_output: grad_input + grad_gamma + grad_beta
    torch::Tensor bn_backward_output = torch::zeros({N + 2, C, H, W}, normalized.options());

    // std::cout << "blocks grad_gamma: " << blocks_sum.x << ", " << blocks_sum.y << std::endl;

    AT_DISPATCH_FLOATING_TYPES(normalized.type(), "grad_gamma_conv_kernel",
    ([&] {
        grad_gamma_conv_kernel<scalar_t><<<blocks_sum, threads_sum>>>(
            dL_dout.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            normalized.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            bn_backward_output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()
        );
    }));

    // std::cout << "blocks grad_beta: " << blocks_sum.x << ", " << blocks_sum.y << std::endl;

    AT_DISPATCH_FLOATING_TYPES(normalized.type(), "grad_beta_conv_kernel",
    ([&] {
        grad_beta_conv_kernel<scalar_t><<<blocks_sum, threads_sum>>>(
            dL_dout.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            bn_backward_output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()
        );
    }));

    // batch norm will use a even dispatched block size
    const dim3 threads_batch_norm(BLOCK_SIZE_BN_X, BLOCK_SIZE_BN_Y);
    const dim3 blocks_batch_norm((N + threads_batch_norm.x - 1) / threads_batch_norm.x, (C + threads_batch_norm.y - 1) / threads_batch_norm.y);

    // std::cout << "blocks batch norm backwards: " << blocks_batch_norm.x << ", " << blocks_batch_norm.y << std::endl;

    AT_DISPATCH_FLOATING_TYPES(normalized.type(), "bn_backward_input_conv_kernel",
    ([&] {
        bn_backward_input_conv_kernel<scalar_t><<<blocks_batch_norm, threads_batch_norm>>>(
            dL_dout.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            gamma.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            dx_sum.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            dx_norm_sum.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            normalized.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            std_eps.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            bn_backward_output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()
        );
    }));

    return bn_backward_output;
}
