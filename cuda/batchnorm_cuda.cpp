#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> batchnorm2d_cuda_forward(
  torch::Tensor input,
  torch::Tensor gamma,
  torch::Tensor beta);

std::vector<torch::Tensor> batchnorm2d_cuda_backward(
  torch::Tensor input,
  torch::Tensor d_output,
  torch::Tensor mu,
  torch::Tensor sigma2,
  torch::Tensor normalized_input,
  torch::Tensor gamma,
  torch::Tensor beta);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> batchnorm2d_forward(
  torch::Tensor input,
  torch::Tensor gamma,
  torch::Tensor beta) {
  CHECK_INPUT(input);
  CHECK_INPUT(gamma);
  CHECK_INPUT(beta);

  return batchnorm2d_cuda_forward(input, gamma, beta);
}

std::vector<torch::Tensor> batchnorm2d_backward(
  torch::Tensor d_output,
  torch::Tensor input,
  torch::Tensor mu,
  torch::Tensor sigma2,
  torch::Tensor normalized_input,
  torch::Tensor gamma,
  torch::Tensor beta) {
  CHECK_INPUT(input);
  CHECK_INPUT(d_output);
  CHECK_INPUT(mu);
  CHECK_INPUT(sigma2);
  CHECK_INPUT(normalized_input);
  CHECK_INPUT(gamma);
  CHECK_INPUT(beta);

  return batchnorm2d_cuda_backward(
    input,
    d_output,
    mu,
    sigma2,
    normalized_input,
    gamma,
    beta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batchnorm2d_forward", &batchnorm2d_forward, "batchnorm2d forward (CUDA)");
  m.def("batchnorm2d_backward", &batchnorm2d_backward, "batchnorm2d backward (CUDA)");
}
