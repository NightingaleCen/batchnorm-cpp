#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include "sharedmem.cuh"

#include <vector>

namespace {

  template <typename scalar_t>
  __global__ void batchnorm1d_cuda_forward_mean_kernel(
    const size_t C, const size_t batch_size,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> mu_cache) {

    //batch index
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    //channel index
    const int i = threadIdx.y;

    if (n < batch_size) {
      mu_cache[n][i] = input[n][i] / batch_size;
    }
  }
  template <typename scalar_t>
  __global__ void batchnorm1d_cuda_forward_sigma2_kernel(
    const size_t C, const size_t batch_size,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> mu,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sigma2_cache
  ) {

    //batch index
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    //channel index
    const int i = threadIdx.y;

    if (n < batch_size) {
      sigma2_cache[n][i] = (input[n][i] - mu[i]) * (input[n][i] - mu[i]) / batch_size;
    }
  }
  template <typename scalar_t>
  __global__ void batchnorm1d_cuda_forward_output_kernel(
    const size_t C, const size_t batch_size,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> mu,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> sigma2,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> normalized_input,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> gamma,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> beta,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output) {

    //batch index
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    //channel index
    const int i = threadIdx.y;

    if (n < batch_size) {
      normalized_input[n][i] = (input[n][i] - mu[i]) * rsqrt(sigma2[i] + 1e-5);
      output[n][i] = gamma[i] * normalized_input[n][i] + beta[i];
    }
  }

  template <typename scalar_t>
  __global__ void batchnorm1d_cuda_backward_d_normalized_input_kernel(
    const size_t C, const size_t batch_size,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> gamma,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_normalized_input
  ) {

    //batch index
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    //channel index
    const int i = threadIdx.y;

    if (n < batch_size) {
      d_normalized_input[n][i] = d_output[n][i] * gamma[i];
      //////printf("channel: %d\nd_normalized_input: %f,d_output: %f, gamma: %f\n", i, d_normalized_input[n][i][r][c], d_output[n][i][r][c], gamma[i]);
    }
  }
  template <typename scalar_t>
  __global__ void batchnorm1d_cuda_backward_d_gamma_kernel(
    const size_t C, const size_t batch_size,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> normalized_input,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_gamma_cache
  ) {

    //batch index
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    //channel index
    const int i = threadIdx.y;

    if (n < batch_size) {
      d_gamma_cache[n][i] = d_output[n][i] * normalized_input[n][i];
      //////printf("d_gamma_cache: %f\n", d_gamma_cache[n][i][r][c]);

    }
  }
  template <typename scalar_t>
  __global__ void batchnorm1d_cuda_backward_d_beta_kernel(
    const size_t C, const size_t batch_size,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_beta_cache
  ) {

    //batch index
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    //channel index
    const int i = threadIdx.y;

    if (n < batch_size) {
      d_beta_cache[n][i] = d_output[n][i];
      //printf("d_beta_cache: %f\n", d_beta_cache[n][i][r][c]);
    }
  }
  template <typename scalar_t>
  __global__ void batchnorm1d_cuda_backward_d_sigma2_kernel(
    const size_t C, const size_t batch_size,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> mu,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> sigma2,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> gamma,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_sigma2_cache) {

    //batch index
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    //channel index
    const int i = threadIdx.y;

    if (n < batch_size) {
      d_sigma2_cache[n][i] = d_output[n][i] * (input[n][i] - mu[i]) * ((gamma[i] / (-2)) * rsqrt(sigma2[i] + 1e-5) * rsqrt(sigma2[i] + 1e-5) * rsqrt(sigma2[i] + 1e-5));
      //////printf("d_sigma2_cache: %f,d_output: %f,input: %f,mu: %f,sigma2: %f,gamma: %f\n", d_sigma2_cache[n][i][r][c], d_output[n][i][r][c], input[n][i][r][c], mu[i], sigma2[i], gamma[i]);
    }
  }

  template <typename scalar_t>
  __global__ void batchnorm1d_cuda_backward_d_mu_kernel(
    const size_t C, const size_t batch_size,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> mu,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> sigma2,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> gamma,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_output,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_sigma2,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_mu_cache) {

    //batch index
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    //channel index
    const int i = threadIdx.y;

    if (n < batch_size) {
      d_mu_cache[n][i] = (((-2) * d_sigma2[i] * (input[n][i] - mu[i])) / batch_size) + (d_output[n][i] * (-1) * gamma[i] * rsqrt(sigma2[i] + 1e-5));
      //////printf("d_mu_cache: %f,d_sigma2: %f,d_output: %f,input: %f,mu: %f,sigma2: %f,gamma: %f,slice_size: %u, rsqrt: %f\n", d_mu_cache[n][i][r][c], d_sigma2[i], d_output[n][i][r][c], input[n][i][r][c], mu[i], sigma2[i], gamma[i], slice_size, rsqrt(sigma2[i] + 1e-5));
    }
  }

  template <typename scalar_t>
  __global__ void batchnorm1d_cuda_backward_d_input_kernel(
    const size_t C, const size_t batch_size,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> mu,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> sigma2,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> normalized_input,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_normalized_input,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_sigma2,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_mu,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_input) {

    //batch index
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    //channel index
    const int i = threadIdx.y;

    if (n < batch_size) {
      d_input[n][i] = (d_normalized_input[n][i] * rsqrt(sigma2[i] + 1e-5)) + (d_sigma2[i] * ((2 * (input[n][i] - mu[i])) / batch_size) + (d_mu[i] / batch_size));
      //////printf("d_input: %f,d_mu: %f,d_sigma2: %f,input: %f,d_normalized_input: %f,mu: %f,sigma2: %f,slice_size: %u\n", d_input[n][i][r][c], d_mu[i], d_sigma2[i], input[n][i][r][c], d_normalized_input[n][i][r][c], mu[i], sigma2[i], slice_size);
    }
  }
} // namespace

template <typename scalar_t>
__global__ void reduction_sum1d_kernel(scalar_t* array, scalar_t* sum_output, const int64_t total_num) {

  SharedMemory<scalar_t> shared_memory;
  scalar_t* shared_data = shared_memory.getPointer();
  const auto tid = threadIdx.x;
  const auto i = threadIdx.x + blockIdx.x * blockDim.x;


  if (i < total_num) {
    shared_data[tid] = array[i];
    //printf("%f ,", array[i]);
  }
  else {
    shared_data[tid] = 0;
  }

  __syncthreads();

  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_data[tid] += shared_data[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    //printf("\n");
    sum_output[blockIdx.x] = shared_data[0];
  }
}

template <typename scalar_t>
scalar_t reduction_sum(
  torch::Tensor array
) {

  const auto total_num = array.numel();
  const auto thread_num = 1024;
  const auto block_num = (total_num + 1024 - 1) / 1024;

  scalar_t* block_sums; // block_sum[-1] represents the total sum of array
  cudaMalloc((void**)&block_sums, sizeof(scalar_t) * (1 + block_num));
  //printf("Block num: %d\n", block_num);
  //printf("Redution total num: %d\n", total_num);

  reduction_sum1d_kernel<scalar_t> << < block_num, thread_num, thread_num * sizeof(scalar_t) >> > (array.data_ptr<scalar_t>(), block_sums, total_num);
  cudaDeviceSynchronize();
  reduction_sum1d_kernel<scalar_t> << <1, thread_num, thread_num * sizeof(scalar_t) >> > (block_sums, block_sums + block_num, block_num);
  cudaDeviceSynchronize();


  scalar_t total_sum;
  cudaMemcpy(&total_sum, block_sums + block_num, sizeof(scalar_t), cudaMemcpyDeviceToHost);
  cudaFree(block_sums);

  //printf("Redution total sum: %f\n", total_sum);

  return total_sum;

}

std::vector<torch::Tensor> batchnorm1d_cuda_forward(
  torch::Tensor input,
  torch::Tensor gamma,
  torch::Tensor beta) {
  const auto batch_size = input.size(0);
  const auto C = input.size(1);

  const int thread_batch_num = int(floor(sqrt(1024 / C)));

  auto mu = torch::zeros_like(gamma);
  auto sigma2 = torch::zeros_like(gamma);
  auto normalized_input = torch::zeros_like(input);
  auto output = torch::zeros_like(input);
  auto cache = torch::empty_like(input);

  const dim3 threads(thread_batch_num, C, 1);
  const dim3 blocks((batch_size + thread_batch_num - 1) / thread_batch_num, 1, 1);

  //printf("Forward:\n");
  //printf("Dim: %d, %d, %d, %d\n", batch_size, C, H, W);
  //printf("DimSize: %d\n", dim_size);
  //printf("blockDim: %d, %d, %d\n", threads.x, threads.y, threads.z);
  //printf("gridDim: %d, %d, %d\n", blocks.x, blocks.y, blocks.z);


  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batchnorm1d_forward_cuda", ([&] {
    batchnorm1d_cuda_forward_mean_kernel<scalar_t> << <blocks, threads >> > (C, batch_size,
    input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
    cache.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
  cudaDeviceSynchronize();

  for (int i = 0; i < C; i++) {
    mu.index_put_({ i }, reduction_sum<scalar_t>(cache.index({ torch::indexing::Slice(), i }).clone()));
  }
  cudaDeviceSynchronize();

  batchnorm1d_cuda_forward_sigma2_kernel<scalar_t> << <blocks, threads >> > (C, batch_size,
                                                                             input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                             mu.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                             cache.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
  cudaDeviceSynchronize();

  for (int i = 0; i < C; i++) {
    sigma2.index_put_({ i }, reduction_sum<scalar_t>(cache.index({ torch::indexing::Slice(), i }).clone()));
  }
  cudaDeviceSynchronize();

  batchnorm1d_cuda_forward_output_kernel<scalar_t> << <blocks, threads >> > (C, batch_size,
                                                                             input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                             mu.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                             sigma2.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                             normalized_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                             gamma.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                             beta.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                             output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
  cudaDeviceSynchronize();
                                                                               }));
  return { input, mu, sigma2, normalized_input, gamma, beta, output };
}

std::vector<torch::Tensor> batchnorm1d_cuda_backward(
  torch::Tensor d_output,
  torch::Tensor input,
  torch::Tensor mu,
  torch::Tensor sigma2,
  torch::Tensor normalized_input,
  torch::Tensor gamma,
  torch::Tensor beta) {
  const auto batch_size = input.size(0);
  const auto C = input.size(1);

  const int thread_batch_num = int(floor(sqrt(1024 / C)));

  auto cache = torch::empty_like(input);

  auto d_input = torch::empty_like(input);
  auto d_gamma = torch::empty_like(gamma);
  auto d_beta = torch::empty_like(beta);

  auto d_normalized_input = torch::empty_like(normalized_input);
  auto d_sigma2 = torch::empty_like(sigma2);
  auto d_mu = torch::empty_like(mu);

  const dim3 threads(thread_batch_num, C, 1);
  const dim3 blocks((batch_size + thread_batch_num - 1) / thread_batch_num, 1, 1);

  //printf("Backward:\n");
  //printf("Dim: %d, %d, %d, %d\n", batch_size, C, H, W);
  //printf("DimSize: %d\n", dim_size);
  //printf("blockDim: %d, %d, %d\n", threads.x, threads.y, threads.z);
  //printf("gridDim: %d, %d, %d\n", blocks.x, blocks.y, blocks.z);


  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batchnorm1d_backward_cuda", ([&] {
    batchnorm1d_cuda_backward_d_normalized_input_kernel<scalar_t> << <blocks, threads >> > (C, batch_size,
    gamma.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
    d_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
    d_normalized_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
    );
  cudaDeviceSynchronize();

  batchnorm1d_cuda_backward_d_gamma_kernel<scalar_t> << <blocks, threads >> > (C, batch_size,
                                                                               normalized_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                               d_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                               cache.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
                                                                               );
  cudaDeviceSynchronize();

  for (int i = 0; i < C; i++) {
    d_gamma.index_put_({ i }, reduction_sum<scalar_t>(cache.index({ torch::indexing::Slice(), i }).clone()));
  }
  cudaDeviceSynchronize();

  batchnorm1d_cuda_backward_d_beta_kernel<scalar_t> << <blocks, threads >> > (C, batch_size,
                                                                              d_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                              cache.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
                                                                              );
  cudaDeviceSynchronize();

  for (int i = 0; i < C; i++) {
    d_beta.index_put_({ i }, reduction_sum<scalar_t>(cache.index({ torch::indexing::Slice(), i }).clone()));
  }
  cudaDeviceSynchronize();

  batchnorm1d_cuda_backward_d_sigma2_kernel<scalar_t> << <blocks, threads >> > (C, batch_size,
                                                                                input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                mu.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                                sigma2.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                                gamma.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                                d_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                cache.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
                                                                                );
  cudaDeviceSynchronize();

  for (int i = 0; i < C; i++) {
    d_sigma2.index_put_({ i }, reduction_sum<scalar_t>(cache.index({ torch::indexing::Slice(), i }).clone()));
  }
  cudaDeviceSynchronize();

  batchnorm1d_cuda_backward_d_mu_kernel<scalar_t> << <blocks, threads >> > (C, batch_size,
                                                                            input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                            mu.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                            sigma2.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                            gamma.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                            d_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                            d_sigma2.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                            cache.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
                                                                            );
  cudaDeviceSynchronize();

  for (int i = 0; i < C; i++) {
    d_mu.index_put_({ i }, reduction_sum<scalar_t>(cache.index({ torch::indexing::Slice(), i }).clone()));
  }
  cudaDeviceSynchronize();

  batchnorm1d_cuda_backward_d_input_kernel<scalar_t> << <blocks, threads >> > (C, batch_size,
                                                                               input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                               mu.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                               sigma2.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                               normalized_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                               d_normalized_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                               d_sigma2.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                               d_mu.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                               d_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
                                                                               );
  cudaDeviceSynchronize();}));

  return { d_input, d_gamma, d_beta };
}

