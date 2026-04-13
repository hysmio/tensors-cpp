#include "cuda_backend.cuh"

// #define CEIL_DIV (x, y) (1 + ((x - 1) / y))

__global__ void cuda_sgemm(uint32_t m, uint32_t n, uint32_t k, float alpha, float *a, float *b,
                           float beta, float *c) {
    /// a = float[m * n] = float[4, 6]
    /// b = float[n * k] = float[6, 5]
    /// c = float[m * k] = float[4, 5]
    ///
    /// loop cidx 0..19
    ///   loop i 0..5
    ///     ~> aidx = cidx / n + i = (0 / 6 + 0) = 0
    ///        bidx = cidx / k + i = (0 / 4 + 0) = 0
    ///     ~> aidx = 0 / 6 + 1 = 1
    ///        bidx = 0 / 4 + 1 = 1
    ///     ...
    ///   ~> cidx = 14 = 3rd row, 5th col
    ///     ~> aidx = 14 % 6 + 0 * 6 = (0 + 2)
    ///     ~> bidx = 14 % 5 + 0 = (4 + 0)
    ///     ...
    ///     ~> aidx = 14 % 6 + 3 * 6 = (2 + 18)

    // go through each row of a
    int cRow = blockIdx.x * blockDim.x + threadIdx.x;
    int cCol = blockIdx.y * blockDim.y + threadIdx.y;
    if (cRow < m && cCol < k) {
        float tmp = 0.0;
        for (uint32_t i = 0; i < n; i++) {
            uint32_t aIdx = (cRow * n) + i; // cRow * aCols = aRow + i = aIdx
            uint32_t bIdx = (i * k) + cCol; // bCols * i = cRow + cCol = bIdx
            float result = a[aIdx] * b[bIdx];
            tmp += result;
        }

        // this allows for a single function to do `ab` & `ab + c` eg. `mx + b`
        float result = alpha * tmp;
        if (beta != 0.0f)
            result += beta * c[(cRow * k) + cCol];
        c[(cRow * k) + cCol] = result;
    }
}

__host__ void launch_cuda_sgemm(uint32_t m, uint32_t n, uint32_t k, float alpha, float *a, float *b,
                                float beta, float *c) {
    dim3 blockDim(32, 32);
    dim3 gridDim((m + 31) / 32, (k + 31) / 32);
    cuda_sgemm<<<gridDim, blockDim>>>(m, n, k, alpha, a, b, beta, c);
}

__global__ void scalar_divide(const float *a, float scalar, float *out, uint32_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] / scalar;
    }
}

__host__ void launch_scalar_divide(const float *a, float scalar, float *out, uint32_t size) {
    scalar_divide<<<(size + 255) / 256, 256>>>(a, scalar, out, size);
}

__global__ void scalar_multiply(const float *a, float scalar, float *out, uint32_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * scalar;
    }
}

__host__ void launch_scalar_multiply(const float *a, const float scalar, float *out,
                                     uint32_t size) {
    scalar_multiply<<<(size + 255) / 256, 256>>>(a, scalar, out, size);
}

__global__ void scalar_add(const float *a, float scalar, float *out, uint32_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + scalar;
    }
}

__host__ void launch_scalar_add(const float *a, const float scalar, float *out, uint32_t size) {
    scalar_add<<<(size + 255) / 256, 256>>>(a, scalar, out, size);
}

__global__ void scalar_addp(const float *a, const float *scalar, float *out, uint32_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + scalar[0];
    }
}

__host__ void launch_scalar_addp(const float *a, const float *scalar, float *out, uint32_t size) {
    scalar_addp<<<(size + 255) / 256, 256>>>(a, scalar, out, size);
}

__global__ void scalar_subtract(const float *a, float scalar, float *out, uint32_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - scalar;
    }
}

__host__ void launch_scalar_subtract(const float *a, const float scalar, float *out,
                                     uint32_t size) {
    scalar_subtract<<<(size + 255) / 256, 256>>>(a, scalar, out, size);
}

__global__ void vec_divide(const float *a, const float *b, float *out, uint32_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] / b[idx];
    }
}

__host__ void launch_vec_divide(const float *a, const float *b, float *out, uint32_t size) {
    vec_divide<<<(size + 255) / 256, 256>>>(a, b, out, size);
}

__global__ void vec_subtract(const float *a, const float *b, float *out, uint32_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

__host__ void launch_vec_subtract(const float *a, const float *b, float *out, uint32_t size) {
    vec_subtract<<<(size + 255) / 256, 256>>>(a, b, out, size);
}

__global__ void vec_multiply(const float *a, const float *b, float *out, uint32_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

__host__ void launch_vec_multiply(const float *a, const float *b, float *out, uint32_t size) {
    vec_multiply<<<(size + 255) / 256, 256>>>(a, b, out, size);
}

__global__ void vec_add(const float *a, const float *b, float *out, uint32_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

__host__ void launch_vec_add(const float *a, const float *b, float *out, uint32_t size) {
    vec_add<<<(size + 255) / 256, 256>>>(a, b, out, size);
}

__global__ void linspace(float *a, float start, float end, uint32_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] = start + idx * (end - start) / (size - 1);
    }
}

__host__ void launch_linspace(float *a, float start, float end, uint32_t size) {
    linspace<<<(size + 255) / 256, 256>>>(a, start, end, size);
}

__global__ void transpose(float *a, uint32_t rows, uint32_t cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int row = idx / cols;
        int col = idx % cols;
        int new_idx = col * rows + row;
        float temp = a[idx];
        a[idx] = a[new_idx];
        a[new_idx] = temp;
    }
}

__host__ void launch_transpose(float *a, uint32_t rows, uint32_t cols) {
    transpose<<<(rows * cols + 255) / 256, 256>>>(a, rows, cols);
}

__global__ void tanh(float *a, uint32_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] = tanh(a[idx]);
    }
}

__host__ void launch_tanh(float *a, uint32_t size) { tanh<<<(size + 255) / 256, 256>>>(a, size); }

__global__ void square_error(float *a, float *b, float *c, uint32_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = (a[idx] - b[idx]) * (a[idx] - b[idx]);
    }
}

__host__ void launch_square_error(float *a, float *b, float *c, uint32_t size) {
    square_error<<<(size + 255) / 256, 256>>>(a, b, c, size);
}

__global__ void fill_value(float *a, float value, uint32_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] = value;
    }
}

__host__ void launch_fill_value(float *a, float value, uint32_t size) {
    fill_value<<<(size + 255) / 256, 256>>>(a, value, size);
}

__global__ void negate(const float *a, float *out, uint32_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = -a[idx];
    }
}

__host__ void launch_negate(const float *a, float *out, uint32_t size) {
    negate<<<(size + 255) / 256, 256>>>(a, out, size);
}

__global__ void transpose_copy(const float *in, float *out, uint32_t rows, uint32_t cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int row = idx / cols;
        int col = idx % cols;
        out[col * rows + row] = in[idx];
    }
}

__host__ void launch_transpose_copy(const float *in, float *out, uint32_t rows, uint32_t cols) {
    uint32_t total = rows * cols;
    transpose_copy<<<(total + 255) / 256, 256>>>(in, out, rows, cols);
}

__global__ void tanh_forward(const float *in, float *out, uint32_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = tanhf(in[idx]);
    }
}

__host__ void launch_tanh_forward(const float *in, float *out, uint32_t size) {
    tanh_forward<<<(size + 255) / 256, 256>>>(in, out, size);
}

__global__ void reduce_sum(const float *in, float *out, uint32_t size) {
    extern __shared__ float sdata[];
    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < size) ? in[idx] : 0.0f;
    __syncthreads();

    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

__host__ void launch_reduce_sum(const float *in, float *out, uint32_t size) {
    const uint32_t blockSize = 256;
    uint32_t gridSize = (size + blockSize - 1) / blockSize;
    // Zero the output first since we use atomicAdd
    cudaMemsetAsync(out, 0, sizeof(float));
    reduce_sum<<<gridSize, blockSize, blockSize * sizeof(float)>>>(in, out, size);
}

__global__ void accumulate_sq_norm(const float *in, float *out, uint32_t size) {
    extern __shared__ float sdata[];
    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (idx < size) ? in[idx] : 0.0f;
    sdata[tid] = val * val;
    __syncthreads();

    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(out, sdata[0]);
}

__host__ void launch_accumulate_sq_norm(const float *in, float *out, uint32_t size) {
    const uint32_t bs = 256;
    accumulate_sq_norm<<<(size + bs - 1) / bs, bs, bs * sizeof(float)>>>(in, out, size);
}

__global__ void sgd_update(float *param, const float *grad, const float *total_sq_norm, float lr,
                           float max_norm, uint32_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float norm = sqrtf(*total_sq_norm);
        float clip = (norm > max_norm) ? (max_norm / norm) : 1.0f;
        param[idx] -= grad[idx] * clip * lr;
    }
}

__host__ void launch_sgd_update(float *param, const float *grad, const float *total_sq_norm,
                                float lr, float max_norm, uint32_t size) {
    sgd_update<<<(size + 255) / 256, 256>>>(param, grad, total_sq_norm, lr, max_norm, size);
}

// Tensor sin(Tensor &in) {
//     Tensor out(in.shape, in.requires_grad);
//     for (uint32_t i = 0; i < in.size; i++) {
//         out.data()[i] = std::sin(in.data()[i]);
//     }
//     return out;
// }

// Tensor cos(Tensor &in) {
//     Tensor out(in.shape, in.requires_grad);
//     for (uint32_t i = 0; i < in.size; i++) {
//         out.data()[i] = std::cos(in.data()[i]);
//     }
//     return out;
// }

// Tensor relu(Tensor &in) {
//     constexpr float leak = 0.01f; // LeakyReLU
//     Tensor out(in.shape, in.requires_grad);
//     for (uint32_t i = 0; i < in.size; i++) {
//         out.data()[i] = in.data()[i] > 0 ? in.data()[i] : leak * in.data()[i];
//     }
//     if (in.requires_grad) {
//         out.grad_fn = std::make_shared<ReluBackward>(std::make_shared<Tensor>(in));
//     }
//     return out;
// }

// Tensor tanh(Tensor &in) {
//     Tensor out(in.shape, in.requires_grad);
//     for (uint32_t i = 0; i < in.size; i++) {
//         out.data()[i] = std::tanh(in.data()[i]);
//     }
//     if (in.requires_grad) {
//         out.grad_fn = std::make_shared<TanhBackward>(
//             std::make_shared<Tensor>(in),
//             std::make_shared<Tensor>(out) // Store output for backward
//         );
//     }
//     return out;
// }

// // (1/n) * sum(y - y_pred)^2
// Tensor mse(Tensor &y, Tensor &y_pred) {
//     Tensor error = y_pred - y;
//     Tensor squared = error * error;
//     return squared.mean();
// }
