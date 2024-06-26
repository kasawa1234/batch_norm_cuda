#include <torch/extension.h>
#include "utils.h"


torch::Tensor bn_forward_mlp(
    const torch::Tensor X,
    const torch::Tensor gamma,
    const torch::Tensor beta
){
    CHECK_INPUT(X);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    
    return bn_forward_mlp_cuda(X, gamma, beta);
}

torch::Tensor bn_forward_mlp_sram(
    const torch::Tensor X,
    const torch::Tensor gamma,
    const torch::Tensor beta
){
    CHECK_INPUT(X);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    
    return bn_forward_mlp_sram_cuda(X, gamma, beta);
}

torch::Tensor bn_backward_mlp(
    const torch::Tensor dL_dout,
    const torch::Tensor normalized,
    const torch::Tensor gamma,
    const torch::Tensor std_eps
){
    CHECK_INPUT(dL_dout);
    CHECK_INPUT(normalized);
    CHECK_INPUT(gamma);
    CHECK_INPUT(std_eps);

    return bn_backward_mlp_cuda(dL_dout, normalized, gamma, std_eps);
}

torch::Tensor bn_backward_mlp_sram(
    const torch::Tensor dL_dout,
    const torch::Tensor normalized,
    const torch::Tensor gamma,
    const torch::Tensor std_eps_N
){
    CHECK_INPUT(dL_dout);
    CHECK_INPUT(normalized);
    CHECK_INPUT(gamma);
    CHECK_INPUT(std_eps_N);

    return bn_backward_mlp_sram_cuda(dL_dout, normalized, gamma, std_eps_N);
}

torch::Tensor bn_forward_conv(
    const torch::Tensor X,
    const torch::Tensor gamma,
    const torch::Tensor beta
){
    CHECK_INPUT(X);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    
    return bn_forward_conv_cuda(X, gamma, beta);
}

torch::Tensor bn_forward_conv_parallel(
    const torch::Tensor X,
    const torch::Tensor gamma,
    const torch::Tensor beta
){
    CHECK_INPUT(X);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    
    return bn_forward_conv_parallel_cuda(X, gamma, beta);
}

torch::Tensor bn_forward_conv_flatten(
    const torch::Tensor X,
    const torch::Tensor gamma,
    const torch::Tensor beta
){
    CHECK_INPUT(X);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    
    return bn_forward_conv_flatten_cuda(X, gamma, beta);
}

torch::Tensor bn_forward_conv_sram(
    const torch::Tensor X,
    const torch::Tensor gamma,
    const torch::Tensor beta
){
    CHECK_INPUT(X);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    
    return bn_forward_conv_sram_cuda(X, gamma, beta);
}

torch::Tensor bn_backward_conv(
    const torch::Tensor dL_dout,
    const torch::Tensor normalized,
    const torch::Tensor gamma,
    const torch::Tensor std_eps
){
    CHECK_INPUT(dL_dout);
    CHECK_INPUT(normalized);
    CHECK_INPUT(gamma);
    CHECK_INPUT(std_eps);

    return bn_backward_conv_cuda(dL_dout, normalized, gamma, std_eps);
}

torch::Tensor bn_backward_conv_parallel(
    const torch::Tensor dL_dout,
    const torch::Tensor normalized,
    const torch::Tensor gamma,
    const torch::Tensor std_eps
){
    CHECK_INPUT(dL_dout);
    CHECK_INPUT(normalized);
    CHECK_INPUT(gamma);
    CHECK_INPUT(std_eps);

    return bn_backward_conv_parallel_cuda(dL_dout, normalized, gamma, std_eps);
}

torch::Tensor bn_backward_conv_sram(
    const torch::Tensor dL_dout,
    const torch::Tensor normalized,
    const torch::Tensor gamma,
    const torch::Tensor std_eps_N
){
    CHECK_INPUT(dL_dout);
    CHECK_INPUT(normalized);
    CHECK_INPUT(gamma);
    CHECK_INPUT(std_eps_N);

    return bn_backward_conv_sram_cuda(dL_dout, normalized, gamma, std_eps_N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("bn_forward_mlp", &bn_forward_mlp, "");
    m.def("bn_forward_mlp_sram", &bn_forward_mlp_sram, "");
    m.def("bn_backward_mlp", &bn_backward_mlp, "");
    m.def("bn_backward_mlp_sram", &bn_backward_mlp_sram, "");
    m.def("bn_forward_conv", &bn_forward_conv, "");
    m.def("bn_forward_conv_flatten", &bn_forward_conv_flatten, "");
    m.def("bn_forward_conv_parallel", &bn_forward_conv_parallel, "");
    m.def("bn_forward_conv_sram", &bn_forward_conv_sram, "");
    m.def("bn_backward_conv", &bn_backward_conv, "");
    m.def("bn_backward_conv_parallel", &bn_backward_conv_parallel, "");
    m.def("bn_backward_conv_sram", &bn_backward_conv_sram, "");
}
