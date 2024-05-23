#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor bn_forward_mlp_cuda(
    const torch::Tensor X,
    const torch::Tensor gamma,
    const torch::Tensor beta
);

torch::Tensor bn_forward_mlp_sram_cuda(
    const torch::Tensor X,
    const torch::Tensor gamma,
    const torch::Tensor beta
);

torch::Tensor bn_backward_mlp_cuda(
    const torch::Tensor dL_dout,
    const torch::Tensor normalized,
    const torch::Tensor gamma,
    const torch::Tensor std_eps
);

torch::Tensor bn_backward_mlp_sram_cuda(
    const torch::Tensor dL_dout,
    const torch::Tensor normalized,
    const torch::Tensor gamma,
    const torch::Tensor std_eps_N
);

torch::Tensor bn_forward_conv_cuda(
    const torch::Tensor X,
    const torch::Tensor gamma,
    const torch::Tensor beta
);

torch::Tensor bn_forward_conv_parallel_cuda(
    const torch::Tensor X,
    const torch::Tensor gamma,
    const torch::Tensor beta
);

torch::Tensor bn_forward_conv_flatten_cuda(
    const torch::Tensor X,
    const torch::Tensor gamma,
    const torch::Tensor beta
);

torch::Tensor bn_forward_conv_sram_cuda(
    const torch::Tensor X,
    const torch::Tensor gamma,
    const torch::Tensor beta
);

torch::Tensor bn_backward_conv_cuda(
    const torch::Tensor dL_dout,
    const torch::Tensor normalized,
    const torch::Tensor gamma,
    const torch::Tensor std_eps
);

torch::Tensor bn_backward_conv_parallel_cuda(
    const torch::Tensor dL_dout,
    const torch::Tensor normalized,
    const torch::Tensor gamma,
    const torch::Tensor std_eps
);

torch::Tensor bn_backward_conv_sram_cuda(
    const torch::Tensor dL_dout,
    const torch::Tensor normalized,
    const torch::Tensor gamma,
    const torch::Tensor std_eps_N
);
