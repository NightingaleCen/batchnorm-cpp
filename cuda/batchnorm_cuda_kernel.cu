#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

#include <vector>

namespace {

  template <typename scalar_t>
  __global__ void batchnorm2d_cuda_forward_kernel(
    const size_t C, const size_t H, const size_t W, const size_t batch_size,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> mu,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> sigma2,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> normalized_input,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> gamma,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> beta,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output) {

    cooperative_groups::grid_group g = cooperative_groups::this_grid();

    const size_t slice_size = batch_size * H * W;
    //batch index
    const int n = threadIdx.z;
    //column index
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    //row index
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    //channel index
    const int i = blockIdx.z;

    if (r < H && c < W) {
      mu[i] = mu[i] + input[n][i][r][c] / slice_size;
      //printf("Thread(%d, %d, %d): %d, %d, %d\nBlock(%d, %d, %d): %d, %d, %d\nslice size: %d\nn:%d, i:0, r:%d, c:%d: %f\nmu[0]: %d\n", blockDim.x, blockDim.y, blockDim.z, threadIdx.x, threadIdx.y, threadIdx.z, gridDim.x, gridDim.y, gridDim.z, blockIdx.x, blockIdx.y, blockIdx.z, slice_size, n, r, c, input[n][0][r][c], mu[0]);
      g.sync(); // synchronize across all the batches

      sigma2[i] = sigma2[i] + (pow(input[n][i][r][c] - mu[i], 2.0) / slice_size);

      g.sync(); // synchronize across all the batches
      normalized_input[n][i][r][c] = (input[n][i][r][c] - mu[i]) * rsqrt(sigma2[i] + 1e-5);
      output[n][i][r][c] = gamma[i] * normalized_input[n][i][r][c] + beta[i];
    }

  }

  template <typename scalar_t>
  __global__ void batchnorm2d_cuda_backward_kernel(
    const size_t C, const size_t H, const size_t W, const size_t batch_size,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> d_input,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_gamma,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_beta,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> d_normalized_input,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_sigma2,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_mu,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> d_output,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> mu,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> sigma2,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> normalized_input,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> gamma,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> beta) {

    cooperative_groups::grid_group g = cooperative_groups::this_grid();

    const size_t slice_size = batch_size * H * W;
    //batch index
    const int n = threadIdx.z;
    //column index
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    //row index
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    //channel index
    const int i = blockIdx.z;
    if (r < H && c < W) {
      d_normalized_input[n][i][r][c] = d_output[n][i][r][c] * gamma[i];
      d_gamma[i] = d_gamma[i] + d_output[n][i][r][c] * normalized_input[n][i][r][c];
      d_beta[i] = d_beta[i] + d_output[n][i][r][c];
      d_sigma2[i] = d_sigma2[i] + d_output[n][i][r][c] * (input[n][i][r][c] - mu[i]) * (-0.5 * gamma[i] * pow(sigma2[i] + 1e-5, -1.5));

      //__syncthreads();
      g.sync(); // synchronize across all the batches

      d_mu[i] = d_mu[i] + d_output[n][i][r][c] * (-1) * gamma[i] * rsqrt(sigma2[i] + 1e-5);
      d_mu[i] = d_mu[i] + d_sigma2[i] * ((input[n][i][r][c] - mu[i]) * (-2 / slice_size));

      //__syncthreads();
      g.sync(); // synchronize across all the batches

      d_input[n][i][r][c] = d_normalized_input[n][i][r][c] * rsqrt(sigma2[i] + 1e-5) + d_sigma2[i] * (2 / slice_size) * (input[n][i][r][c] - mu[i]) + d_mu[i] * (1 / slice_size);
    }


  }
} // namespace

std::vector<torch::Tensor> batchnorm2d_cuda_forward(
  torch::Tensor input,
  torch::Tensor gamma,
  torch::Tensor beta) {
  const auto batch_size = input.size(0);
  const auto C = input.size(1);
  const auto H = input.size(2);
  const auto W = input.size(3);

  const int dim_size = int(floor(sqrt(1024 / batch_size)));

  auto mu = torch::zeros_like(gamma);
  auto sigma2 = torch::zeros_like(gamma);
  auto normalized_input = torch::zeros_like(input);
  auto output = torch::zeros_like(input);

  const dim3 threads(dim_size, dim_size, batch_size);
  const dim3 blocks((H + dim_size - 1) / dim_size, (W + dim_size - 1) / dim_size, C);

  printf("Dim: %d, %d, %d, %d\n", batch_size, C, H, W);
  printf("DimSize: %d\n", dim_size);
  printf("blockDim: %d, %d, %d\n", threads.x, threads.y, threads.z);
  printf("gridDim: %d, %d, %d\n", blocks.x, blocks.y, blocks.z);

  /*int dev = 0;
  int supportsCoopLaunch = 0;
  cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
  std::printf("%d", supportsCoopLaunch);*/

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batchnorm2d_forward_cuda", ([&] {
    auto a_input = input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
    auto a_mu = mu.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>();
    auto a_sigma2 = sigma2.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>();
    auto a_normalized_input = normalized_input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
    auto a_gamma = gamma.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>();
    auto a_beta = beta.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>();
    auto a_output = output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
    void* args[] = { (void*)&C, (void*)&H,(void*)&W, (void*)&batch_size,(void*)&a_input,(void*)&a_mu, (void*)&a_sigma2, (void*)&a_normalized_input, (void*)&a_gamma, (void*)&a_beta, (void*)&a_output };
    cudaLaunchCooperativeKernel((void*)batchnorm2d_cuda_forward_kernel<scalar_t>, blocks, threads, args);}));

  return { input, mu, sigma2, normalized_input, gamma, beta, output };
}

std::vector<torch::Tensor> batchnorm2d_cuda_backward(
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

  const int dim_size = int(floor(sqrt(1024 / batch_size)));

  auto d_input = torch::empty_like(input);
  auto d_gamma = torch::zeros_like(gamma);
  auto d_beta = torch::zeros_like(beta);

  auto d_normalized_input = torch::empty_like(normalized_input);
  auto d_sigma2 = torch::zeros_like(sigma2);
  auto d_mu = torch::zeros_like(mu);

  const dim3 threads(dim_size, dim_size, batch_size);
  const dim3 blocks((H + dim_size - 1) / dim_size, (W + dim_size - 1) / dim_size, C);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batchnorm2d_backward_cuda", ([&] {
    auto a_d_input = d_input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
    auto a_d_gamma = d_gamma.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>();
    auto a_d_beta = d_beta.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>();
    auto a_d_normalized_input = d_normalized_input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
    auto a_d_sigma2 = d_sigma2.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>();
    auto a_d_mu = d_mu.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>();
    auto a_input = input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
    auto a_d_output = d_output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
    auto a_mu = mu.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>();
    auto a_sigma2 = sigma2.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>();
    auto a_normalized_input = normalized_input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
    auto a_gamma = gamma.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>();
    auto a_beta = beta.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>();
    void* args[] = { (void*)&C, (void*)&H,(void*)&W, (void*)&batch_size,
      (void*)&a_d_input,
      (void*)&a_d_gamma,
      (void*)&a_d_beta,
      (void*)&a_d_normalized_input,
      (void*)&a_d_sigma2,
      (void*)&a_d_mu,
      (void*)&a_input,
      (void*)&a_d_output,
      (void*)&a_mu,
      (void*)&a_sigma2,
      (void*)&a_normalized_input,
      (void*)&a_gamma,
      (void*)&a_beta };
    cudaLaunchCooperativeKernel((void*)batchnorm2d_cuda_backward_kernel<scalar_t>, blocks, threads, args);}));

  return { d_input, d_gamma, d_beta };
}
