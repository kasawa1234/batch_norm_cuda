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
__global__ void bn_forward_sum_std_sram_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> gamma,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> beta,
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

    if (n >= input_data.size(0) || c >= input_data.size(1)) return;

    const scalar_t std_eps_N = sqrt(N * shared_memory[0][thread_id_c][1] - shared_memory[0][thread_id_c][0] * shared_memory[0][thread_id_c][0] + N * N * EPSILON);

    batch_norm_output[n][c] = gamma[c] * (N * input_data[n][c] - shared_memory[0][thread_id_c][0]) / std_eps_N + beta[c];
    
    // calculate N * mean and N * std_eps
    if (thread_id_n == 0) {
        // sum[c] = shared_memory[0][thread_id_c][0];
        // batch_norm_output[N][c] = sqrt(N * shared_memory[0][thread_id_c][1] - shared_memory[0][thread_id_c][0] * shared_memory[0][thread_id_c][0] + N * N * EPSILON);
        batch_norm_output[N][c] = std_eps_N;
    }
}

// template <typename scalar_t>
// __global__ void bn_forward_mlp_sram_kernel(
//     torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input_data,
//     torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> sum,
//     torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> gamma,
//     torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> beta,
//     torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output_data
// ){
//     // batch size
//     const int N = input_data.size(0);

//     const int n = blockIdx.x * blockDim.x + threadIdx.x;
//     const int c = blockIdx.y * blockDim.y + threadIdx.y;

//     if (n >= input_data.size(0) || c >= input_data.size(1)) return;

//     output_data[n][c] = gamma[c] * (N * input_data[n][c] - sum[c]) / output_data[N][c] + beta[c];
// }


torch::Tensor bn_forward_mlp_sram_cuda(
    const torch::Tensor X,
    const torch::Tensor gamma,
    const torch::Tensor beta
){
    // X: (n, c), n is parallel
    const int N = X.size(0);
    const int C = X.size(1);
    // std::cout << N << ", " << C << std::endl;

    // calculate sum and std
    // torch::Tensor sum = torch::zeros({C}, X.options());
    // batch_norm_out: bn_forward + N * std_eps
    torch::Tensor batch_norm_out = torch::zeros({N + 1, C}, X.options());
    
    const dim3 threads_sum_std(BLOCK_SIZE_BATCH, BLOCK_SIZE_FEATURE);
    const dim3 blocks_sum_std((N + threads_sum_std.x - 1) / threads_sum_std.x, (C + threads_sum_std.y - 1) / threads_sum_std.y);

    // std::cout << "blocks sum_std: " << blocks_sum_std.x << ", " << blocks_sum_std.y << std::endl;

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(X.type(), "bn_forward_sum_std_sram_kernel",
    ([&] {
        bn_forward_sum_std_sram_kernel<scalar_t><<<blocks_sum_std, threads_sum_std>>>(
            X.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            gamma.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            beta.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            batch_norm_out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));


    // // batch norm will use a even dispatched block size
    // const dim3 threads_batch_norm(BLOCK_SIZE_BN_X, BLOCK_SIZE_BN_Y);
    // const dim3 blocks_batch_norm((N + threads_batch_norm.x - 1) / threads_batch_norm.x, (C + threads_batch_norm.y - 1) / threads_batch_norm.y);

    // // std::cout << "blocks batch norm: " << blocks_batch_norm.x << ", " << blocks_batch_norm.y << std::endl;

    // // launch the kernel
    // AT_DISPATCH_FLOATING_TYPES(X.type(), "bn_forward_mlp_kernel",
    // ([&] {
    //     bn_forward_mlp_sram_kernel<scalar_t><<<blocks_batch_norm, threads_batch_norm>>>(
    //         X.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
    //         sum.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
    //         gamma.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
    //         beta.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
    //         batch_norm_out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
    //     );
    // }));

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


// ———————————————————————————————————————————————————————————————
/*                          MLP Backward                          */
// ———————————————————————————————————————————————————————————————



template <typename scalar_t>
__global__ void grad_gamma_beta_input_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> dL_dout,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> normalized,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> gamma,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> std_eps_N,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> dL_dinput
){
    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_BATCH][BLOCK_SIZE_FEATURE][2];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int thread_id_n = threadIdx.x;
    const int thread_id_c = threadIdx.y;

    // if the loc cover our data, load in shared memory
    if (n < normalized.size(0) && c < normalized.size(1)){
        shared_memory[thread_id_n][thread_id_c][0] = dL_dout[n][c] * normalized[n][c];
        shared_memory[thread_id_n][thread_id_c][1] = dL_dout[n][c];
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
    
    // after this for loop, all should be set, so dump the data and calculate the mean
    const int N = normalized.size(0);
    if (n >= normalized.size(0) || c >= normalized.size(1)) return;
    dL_dinput[n][c] = (N * dL_dout[n][c] * gamma[c] - shared_memory[0][thread_id_c][1] * gamma[c] - normalized[n][c] * shared_memory[0][thread_id_c][0] * gamma[c]) / std_eps_N[c];
    if (thread_id_n == 0) {
        dL_dinput[N][c] = shared_memory[0][thread_id_c][0];                 // grad gamma
        dL_dinput[N + 1][c] = shared_memory[0][thread_id_c][1];             // grad beta
    }
}

torch::Tensor bn_backward_mlp_sram_cuda(
    const torch::Tensor dL_dout,
    const torch::Tensor normalized,
    const torch::Tensor gamma,
    const torch::Tensor std_eps_N
){
    const int N = normalized.size(0);
    const int C = normalized.size(1);

    // bn_backward_output: grad_input + grad_gamma + grad_beta
    torch::Tensor bn_backward_output = torch::zeros({N + 2, C}, normalized.options());

    const dim3 threads(BLOCK_SIZE_BATCH, BLOCK_SIZE_FEATURE);
    const dim3 blocks((N + threads.x - 1) / threads.x, (C + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(normalized.type(), "grad_gamma_beta_input_kernel",
    ([&] {
        grad_gamma_beta_input_kernel<scalar_t><<<blocks, threads>>>(
            dL_dout.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            normalized.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            gamma.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            std_eps_N.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            bn_backward_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));

    return bn_backward_output;
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
    const int H,
    const int W
){
    // declare a shared memory space as same as one block
    __shared__ scalar_t shared_memory[BLOCK_SIZE_BATCH][BLOCK_SIZE_FEATURE][2];
    const int N = input_data.size(0);
    const int C = input_data.size(1);
    const int NHW = N * H * W;
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
        batch_norm_output[N][c][0][0] = sqrt(NHW * shared_memory[0][thread_id_c][1] - shared_memory[0][thread_id_c][0] * shared_memory[0][thread_id_c][0] + NHW * NHW * EPSILON);
    }
}

template <typename scalar_t>
__global__ void bn_forward_conv_sram_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> sum,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> gamma,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> beta,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output_data,
    const int block_num_width
){
    const int N = input_data.size(0);
    const int H = input_data.size(2);   // height
    const int W = input_data.size(3);   // width

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.z * blockDim.z + threadIdx.z;
    const int h = blockIdx.y / block_num_width;
    const int w = (blockIdx.y - h * block_num_width) * blockDim.y + threadIdx.y;

    if (n >= input_data.size(0) || c >= input_data.size(1) || h >= input_data.size(2) || w >= input_data.size(3)) return;

    // __shared__ scalar_t shared_gamma[MAX_C];
    // __shared__ scalar_t shared_beta[MAX_C];
    // __shared__ scalar_t shared_sum[MAX_C];
    // __shared__ scalar_t shared_std_eps_N[MAX_C];

    // if (threadIdx.x == 0 && threadIdx.y == 0) {
    //     shared_gamma[threadIdx.z] = gamma[c];
    //     shared_beta[threadIdx.z] = beta[c];
    //     shared_sum[threadIdx.z] = sum[c];
    //     shared_std_eps_N[threadIdx.z] = output_data[N][c][0][0];
    // }
    // __syncthreads();

    // const scalar_t gamma_val = shared_gamma[threadIdx.z];
    // const scalar_t beta_val = shared_beta[threadIdx.z];
    // const scalar_t sum_val = shared_sum[threadIdx.z];
    // const scalar_t std_eps_N_val = shared_std_eps_N[threadIdx.z];

    output_data[n][c][h][w] = gamma[c] * (N * H * W * input_data[n][c][h][w] - sum[c]) / output_data[N][c][0][0] + beta[c];
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

    // std::cout << "blocks partial sum & sum2: " << blocks_partial_sum.x << ", " << blocks_partial_sum.y << ", " << blocks_partial_sum.z << std::endl;

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

        // std::cout << "blocks partial sum in loop: " << blocks_partial_sum_loop.x << ", " << blocks_partial_sum_loop.y << ", " << blocks_partial_sum_loop.z << std::endl;

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

    // std::cout << "blocks sum & std: " << blocks_sum_std.x << ", " << blocks_sum_std.y << std::endl;

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
    const dim3 threads_batch_norm(BLOCK_SIZE_BN_BATCH, BLOCK_SIZE_BN_HW, 1);
    const int num_width = (W + threads_batch_norm.y - 1) / threads_batch_norm.y;
    const dim3 blocks_batch_norm((N + threads_batch_norm.x - 1) / threads_batch_norm.x, num_width * H, C);

    // std::cout << "blocks batch norm: " << blocks_batch_norm.x << ", " << blocks_batch_norm.y << std::endl;

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(X.type(), "bn_forward_conv_sram_kernel",
    ([&] {
        bn_forward_conv_sram_kernel<scalar_t><<<blocks_batch_norm, threads_batch_norm>>>(
            X.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            sum.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            gamma.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            beta.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            batch_norm_out.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            num_width
        );
    }));
    return batch_norm_out;
}


// —————————————————————————————————————————————————————————————————————————
/*                          Conv Parallel Backward                          */
// —————————————————————————————————————————————————————————————————————————

template <typename scalar_t>
__global__ void grad_gamma_beta_conv_sram_hw_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dout,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> normalized,
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
    if (n < dL_dout.size(0) && c < dL_dout.size(1) && h < valid_height && w < valid_width){
        shared_memory[tid_hw][0] = dL_dout[n][c][h][w] * normalized[n][c][h][w];
        shared_memory[tid_hw][1] = dL_dout[n][c][h][w];
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


// partial_sum_conv_sram_kernel


template <typename scalar_t>
__global__ void grad_gamma_beta_conv_sram_n_kernel(
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> input_data,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output
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

    if (thread_id_n == 0) {
        output[N][c][0][0] = shared_memory[0][thread_id_c][0];
        output[N + 1][c][0][0] = shared_memory[0][thread_id_c][1];
    }
}

template <typename scalar_t>
__global__ void bn_backward_input_conv_sram_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dout,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> gamma,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> normalized,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> std_eps_N,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dinput,
    const int block_num_width
){
    const int N = normalized.size(0);
    const int H = normalized.size(2);      // height  
    const int W = normalized.size(3);      // width

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.z * blockDim.z + threadIdx.z;
    const int h = blockIdx.y / block_num_width;
    const int w = (blockIdx.y - h * block_num_width) * blockDim.y + threadIdx.y;

    if (n >= normalized.size(0) || c >= normalized.size(1) || h >= normalized.size(2) || w >= normalized.size(3)) return;

    dL_dinput[n][c][h][w] = (N * H * W * dL_dout[n][c][h][w] * gamma[c] - dL_dinput[N + 1][c][0][0] * gamma[c] - normalized[n][c][h][w] * dL_dinput[N][c][0][0]) / (std_eps_N[c]);
}


torch::Tensor bn_backward_conv_sram_cuda(
    const torch::Tensor dL_dout,
    const torch::Tensor normalized,
    const torch::Tensor gamma,
    const torch::Tensor std_eps_N
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
    torch::Tensor partial_sum = torch::zeros({N, C, num_h, num_w, 2}, dL_dout.options());

    const dim3 threads_partial_sum(BLOCK_SIZE_HW, 1, 1);
    const dim3 blocks_partial_sum(num_hw, N, C);

    // std::cout << "blocks partial dx_sum: " << blocks_partial_sum.x << ", " << blocks_partial_sum.y << ", " << blocks_partial_sum.z << std::endl;

    AT_DISPATCH_FLOATING_TYPES(dL_dout.type(), "grad_gamma_beta_conv_sram_hw_kernel",
    ([&] {
        grad_gamma_beta_conv_sram_hw_kernel<scalar_t><<<blocks_partial_sum, threads_partial_sum>>>(
            dL_dout.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            normalized.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
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

        // std::cout << "blocks partial dx_sum in loop: " << blocks_partial_sum_loop.x << ", " << blocks_partial_sum_loop.y << ", " << blocks_partial_sum_loop.z << std::endl;

        AT_DISPATCH_FLOATING_TYPES(dL_dout.type(), "partial_sum_conv_sram_kernel",
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

    const dim3 threads_mean(BLOCK_SIZE_BATCH, BLOCK_SIZE_FEATURE);
    const dim3 blocks_mean((N + threads_mean.x - 1) / threads_mean.x, (C + threads_mean.y - 1) / threads_mean.y);
    torch::Tensor bn_backward_output = torch::zeros({N + 2, C, H, W}, dL_dout.options());

    // launch the kernel
    AT_DISPATCH_FLOATING_TYPES(dL_dout.type(), "grad_gamma_beta_conv_sram_n_kernel",
    ([&] {
        grad_gamma_beta_conv_sram_n_kernel<scalar_t><<<blocks_mean, threads_mean>>>(
            partial_sum.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
            bn_backward_output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()
        );
    }));

    // same as forward_conv_parallel
    const dim3 threads_batch_norm(BLOCK_SIZE_BN_BATCH, BLOCK_SIZE_BN_HW, 1);
    const int num_width = (W + threads_batch_norm.y - 1) / threads_batch_norm.y;
    const dim3 blocks_batch_norm((N + threads_batch_norm.x - 1) / threads_batch_norm.x, num_width * H, C);

    // std::cout << "blocks batch norm backwards: " << blocks_batch_norm.x << ", " << blocks_batch_norm.y << std::endl;

    AT_DISPATCH_FLOATING_TYPES(normalized.type(), "bn_backward_input_conv_sram_kernel",
    ([&] {
        bn_backward_input_conv_sram_kernel<scalar_t><<<blocks_batch_norm, threads_batch_norm>>>(
            dL_dout.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            gamma.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            normalized.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            std_eps_N.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            bn_backward_output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            num_width
        );
    }));

    return bn_backward_output;
}


