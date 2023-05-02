#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>


#include <vector>

namespace {
  template <typename scalar_t>
  __device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
    return 1.0 / (1.0 + exp(-z));
  }

  template <typename scalar_t>
  __device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
    const auto s = sigmoid(z);
    return (1.0 - s) * s;
  }

  template <typename scalar_t>
  __device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
    const auto t = tanh(z);
    return 1 - (t * t);
  }

  template <typename scalar_t>
  __device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
    return fmaxf(0.0, z) + fminf(0.0, alpha * (exp(z) - 1.0));
  }

  template <typename scalar_t>
  __device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
    const auto e = exp(z);
    const auto d_relu = z < 0.0 ? 0.0 : 1.0;
    return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
  }

  template <typename scalar_t>
  __global__ void batchnorm2d_cuda_forward_kernel(
    const size_t C, const size_t batch_size,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> input,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> mu,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> sigma2,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> normalized_input,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> gamma,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> beta,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> output) {
    //batch index
    const int n = blockIdx.z;
    //row index
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    //column index
    const int c = blockIdx.y * blockDim.y + threadIdx.y;

    cooperative_groups::grid_group g = cooperative_groups::this_grid();

    for (size_t i = 0; i < C; i++) {
      mu[i][r][c] = mu[i][r][c] + input[n][i][r][c] / batch_size;
    }

    g.sync(); // synchronize across all the batches

    for (size_t i = 0; i < C; i++) {
      sigma2[i][r][c] = sigma2[i][r][c] + pow(input[n][i][r][c] - mu[i][r][c], 2.0) / batch_size;
    }

    g.sync(); // synchronize across all the batches

    for (size_t i = 0; i < C; i++) {
      normalized_input[n][i][r][c] = (input[n][i][r][c] - mu[i][r][c]) * rsqrt(sigma2[i][r][c] + 1e-5);
      output[n][i][r][c] = gamma[i][r][c] * normalized_input[n][i][r][c] + beta[i][r][c];
    }
  }

  template <typename scalar_t>
  __global__ void batchnorm2d_cuda_backward_kernel(
    const size_t C, const size_t batch_size,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_input,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> d_gamma,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> d_beta,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> input,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_output,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> mu,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> sigma2,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> normalized_input,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> gamma,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> beta) {
    //batch index
    const int n = blockIdx.z;
    //row index
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    //column index
    const int c = blockIdx.y * blockDim.y + threadIdx.y;

    cooperative_groups::grid_group g = cooperative_groups::this_grid();

    auto d_normalized_input = torch::empty(C); // actually d_normalizad_input[n][][r][c] for this thread
    auto d_sigma2 = torch::zeros(C); // actually d_sigma[][r][c] for this thread
    auto d_mu = torch::zero(C); // actually d_mu[][r][c] for this thread

    for (size_t i = 0; i < C; i++) {
      d_normalized_input[i] = d_output[n][i][r][c] * gamma[i][r][c];
      d_gamma[i][r][c] = d_gamma[i][r][c] + d_output[n][i][r][c] * normalized_input[n][i][r][c];
      d_beta[i][r][c] = d_beta[i][r][c] + d_output[n][i][r][c];
      d_sigma2[i] = d_sigma2[i] + d_output[n][i][r][c] * (input[n][i][r][c] - mu[i][r][c]) * (-0.5 * gamma[i][r][c] * pow(sigma2[i][r][c] + 1e-5, -1.5));
    }

    g.sync(); // synchronize across all the batches

    for (size_t i = 0; i < C; i++) {
      d_mu[i] = d_mu[i] + d_output[n][i][r][c] * (-1) * gamma[i][r][c] * rsqrt(sigma2[i][r][c] + 1e-5);
      d_mu[i] = d_mu[i] + d_sigma2[i] * (1 / batch_size) * (-2) * (input[n][i][r][c] - mu[i][r][c]);
    }

    g.sync(); // synchronize across all the batches

    for (size_t i = 0; i < C; i++) {
      d_input[n][i][r][c] = d_normalized_input[i] * rsqrt(sigma2[i][r][c] + 1e-5) + d_sigma2[i] * (2 / m) * (input[n][i][r][c] - mu[i][r][c]) + d_mu[i] * (1 / m);
    }


  }
} // namespace

std::vector<torch::Tensor> batchnorm_cuda_forward(
  torch::Tensor input,
  torch::Tensor gamma,
  torch::Tensor beta) {
  const auto batch_size = input.size(0);
  const auto C = input.size(1);
  const auto H = input.size(2);
  const auto W = input.size(3);

  auto mu = torch::zeros_like(gamma);
  auto sigma2 = torch::zeros_like(gamma);
  auto normalized_input = torch::zeros_like(input);
  auto output = torch::zeros_like(input);

  const dim3 threads(32, 32);
  const dim3 blocks((H + 32 - 1) / 32, (W + 32 - 1) / 32, batch_size);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "batchnorm2d_forward_cuda", ([&] {
    batchnorm2d_cuda_forward_kernel<scalar_t> << <blocks, threads >> > (
      C, batch_size,
      input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
      mu.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
      sigma2.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
      normalized_input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
      gamma.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
      beta.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
      output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());
                                                                        }));

  return { input, mu, sigma2, normalized_input, gamma, beta, output };
}

std::vector<torch::Tensor> batchnorm_cuda_backward(
  torch::Tensor input,
  torch::Tensor d_output,
  torch::Tensor mu,
  torch::Tensor sigma2,
  torch::Tensor normalized_input,
  torch::Tensor gamma,
  torch::Tensor beta) {
  const auto batch_size = input.size(0);
  const auto C = input.size(1);
  const auto H = input.size(2);
  const auto W = input.size(3);

  auto d_input = torch::empty_like(input);
  auto d_gamma = torch::zeros_like(gamma);
  auto d_beta = torch::zeros_like(beta);

  const dim3 threads(32, 32);
  const dim3 blocks((H + 32 - 1) / 32, (W + 32 - 1) / 32, batch_size);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "batchnorm2d_backward_cuda", ([&] {
    batchnorm2d_cuda_backward_kernel<scalar_t> << <blocks, threads >> > (
      C, batch_size,
      d_input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
      d_gamma.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
      d_beta.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
      input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
      d_output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
      mu.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
      sigma2.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
      normalized_input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
      gamma.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
      beta.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>());
                                                                         }));

  return { d_input, d_gamma, d_beta };
}
