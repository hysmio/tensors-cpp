#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define NV_CHECK(call)                                                                             \
    do {                                                                                           \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                       \
                    cudaGetErrorString(err));                                                       \
            abort();                                                                               \
        }                                                                                          \
    } while (0)

__host__ void launch_cuda_sgemm(uint32_t m, uint32_t n, uint32_t k, float alpha, float *a, float *b,
                                float beta, float *c);

__global__ void cuda_sgemm(uint32_t m, uint32_t n, uint32_t k, float alpha, float *a, float *b,
                           float beta, float *c);

__global__ void dot(const float *a, const float *b, float *out, uint32_t size);
__host__ void launch_dot(const float *a, const float *b, float *out, uint32_t size);

__global__ void scalar_divide(const float *a, float scalar, float *out, uint32_t size);
__host__ void launch_scalar_divide(const float *a, float scalar, float *out, uint32_t size);

__global__ void scalar_multiply(const float *a, const float scalar, float *out, uint32_t size);
__host__ void launch_scalar_multiply(const float *a, const float scalar, float *out, uint32_t size);

__global__ void scalar_add(const float *a, const float scalar, float *out, uint32_t size);
__host__ void launch_scalar_add(const float *a, const float scalar, float *out, uint32_t size);

__global__ void scalar_addp(const float *a, const float *scalar, float *out, uint32_t size);
__host__ void launch_scalar_addp(const float *a, const float *scalar, float *out, uint32_t size);

__global__ void scalar_subtract(const float *a, const float scalar, float *out, uint32_t size);
__host__ void launch_scalar_subtract(const float *a, const float scalar, float *out, uint32_t size);

__global__ void vec_divide(const float *a, const float *b, float *out, uint32_t size);
__host__ void launch_vec_divide(const float *a, const float *b, float *out, uint32_t size);

__global__ void vec_subtract(const float *a, const float *b, float *out, uint32_t size);
__host__ void launch_vec_subtract(const float *a, const float *b, float *out, uint32_t size);

__global__ void vec_multiply(const float *a, const float *b, float *out, uint32_t size);
__host__ void launch_vec_multiply(const float *a, const float *b, float *out, uint32_t size);

__global__ void vec_add(const float *a, const float *b, float *out, uint32_t size);
__host__ void launch_vec_add(const float *a, const float *b, float *out, uint32_t size);

__global__ void linspace(float *a, float start, float end, uint32_t size);
__host__ void launch_linspace(float *a, float start, float end, uint32_t size);

__global__ void transpose(float *a, uint32_t rows, uint32_t cols);
__host__ void launch_transpose(float *a, uint32_t rows, uint32_t cols);

__global__ void tanh(float *a, uint32_t size);
__host__ void launch_tanh(float *a, uint32_t size);

__global__ void square_error(float *a, float *b, float *c, uint32_t size);
__host__ void launch_square_error(float *a, float *b, float *c, uint32_t size);

__global__ void fill_value(float *a, float value, uint32_t size);
__host__ void launch_fill_value(float *a, float value, uint32_t size);

__global__ void negate(const float *a, float *out, uint32_t size);
__host__ void launch_negate(const float *a, float *out, uint32_t size);

__global__ void transpose_copy(const float *in, float *out, uint32_t rows, uint32_t cols);
__host__ void launch_transpose_copy(const float *in, float *out, uint32_t rows, uint32_t cols);

__global__ void tanh_forward(const float *in, float *out, uint32_t size);
__host__ void launch_tanh_forward(const float *in, float *out, uint32_t size);

__host__ void launch_reduce_sum(const float *in, float *out, uint32_t size);
