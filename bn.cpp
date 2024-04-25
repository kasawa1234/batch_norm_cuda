#include <torch/extension.h>
#include "utils.h"


torch::Tensor bn_forward_mlp(
    const torch::Tensor X,
    const torch::Tensor gamma,
    const torch::Tensor beta
){
    CHECK_INPUT(X);
    
    return bn_forward_mlp_cuda(X, gamma, beta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("bn_forward_mlp", &bn_forward_mlp, "");
}
