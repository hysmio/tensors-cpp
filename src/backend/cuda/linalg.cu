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
        c[(cRow * k) + cCol] = (alpha * tmp) + (beta * c[(cRow * k) + cCol]);
    }
}

__host__ void launch_cuda_sgemm(uint32_t m, uint32_t n, uint32_t k, float alpha, float *a, float *b,
                                float beta, float *c) {
    cuda_sgemm<<<(m + 31) / 32, (k + 31) / 32>>>(m, n, k, alpha, a, b, beta, c);
}

__global__ void scalar_divide(float *a, float scalar, uint32_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] /= scalar;
    }
}

__host__ void launch_scalar_divide(float *a, float scalar, uint32_t size) {
    scalar_divide<<<(size + 255) / 256, 256>>>(a, scalar, size);
}

__global__ void scalar_multiply(float *a, float scalar, uint32_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] *= scalar;
    }
}

__host__ void launch_scalar_multiply(float *a, float scalar, uint32_t size) {
    scalar_multiply<<<(size + 255) / 256, 256>>>(a, scalar, size);
}

__global__ void vec_divide(float *a, float *b, uint32_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] /= b[idx];
    }
}

__host__ void launch_vec_divide(float *a, float *b, uint32_t size) {
    vec_divide<<<(size + 255) / 256, 256>>>(a, b, size);
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
