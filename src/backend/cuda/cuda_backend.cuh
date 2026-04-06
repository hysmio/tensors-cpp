#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__host__ void launch_cuda_sgemm(uint32_t m, uint32_t n, uint32_t k, float alpha, float *a, float *b,
                                float beta, float *c);

__global__ void cuda_sgemm(uint32_t m, uint32_t n, uint32_t k, float alpha, float *a, float *b,
                           float beta, float *c);

__global__ void scalar_divide(float *a, float scalar, uint32_t size);
__host__ void launch_scalar_divide(float *a, float scalar, uint32_t size);

__global__ void scalar_multiply(float *a, float scalar, uint32_t size);
__host__ void launch_scalar_multiply(float *a, float scalar, uint32_t size);

__global__ void vec_divide(float *a, float *b, uint32_t size);
__host__ void launch_vec_divide(float *a, float *b, uint32_t size);

__global__ void linspace(float *a, float start, float end, uint32_t size);
__host__ void launch_linspace(float *a, float start, float end, uint32_t size);

__global__ void transpose(float *a, uint32_t rows, uint32_t cols);
__host__ void launch_transpose(float *a, uint32_t rows, uint32_t cols);

__global__ void tanh(float *a, uint32_t size);
__host__ void launch_tanh(float *a, uint32_t size);

__global__ void square_error(float *a, float *b, float *c, uint32_t size);
__host__ void launch_square_error(float *a, float *b, float *c, uint32_t size);
