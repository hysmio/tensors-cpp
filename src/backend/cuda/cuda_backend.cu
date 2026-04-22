#include "cuda_backend.cuh"

// Global cuBLAS handle and workspace - initialized lazily
static cublasLtHandle_t g_cublasLtHandle = nullptr;
static void *g_workspace = nullptr;
static const size_t g_workspaceSize = 4 * 1024 * 1024; // 4MB workspace

static void ensureCublasInitialized() {
    if (g_cublasLtHandle == nullptr) {
        NVCUBLT_CHECK(cublasLtCreate(&g_cublasLtHandle));
        NV_CHECK(cudaMalloc(&g_workspace, g_workspaceSize));
    }
}

// cuBLAS Lt single-precision GEMM wrapper
void LtSgemm(cublasOperation_t transa,
             cublasOperation_t transb,
             int m,
             int n,
             int k,
             const float *alpha,
             const float *A,
             int lda,
             const float *B,
             int ldb,
             const float *beta,
             float *C,
             int ldc) {
    ensureCublasInitialized();
    cublasLtHandle_t ltHandle = g_cublasLtHandle;
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // Create operation descriptor
    NVCUBLT_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    NVCUBLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    NVCUBLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Create matrix descriptors
    NVCUBLT_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    NVCUBLT_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    NVCUBLT_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));

    // Create preference handle
    NVCUBLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    NVCUBLT_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_workspaceSize, sizeof(g_workspaceSize)));

    // Get best algorithm
    NVCUBLT_CHECK(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        NVCUBLT_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    // Execute matmul
    NVCUBLT_CHECK(cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, &heuristicResult.algo, g_workspace, g_workspaceSize, 0));

    // Cleanup
    if (preference) NVCUBLT_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc) NVCUBLT_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) NVCUBLT_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) NVCUBLT_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) NVCUBLT_CHECK(cublasLtMatmulDescDestroy(operationDesc));
}

__host__ void launch_cuda_sgemm(uint32_t m, uint32_t n, uint32_t k, float alpha, float *a, float *b,
                                float beta, float *c) {
    ensureCublasInitialized();

    // Row-major A[m,n] @ B[n,k] -> C[m,k]
    // cuBLAS uses column-major, so we use the identity: C = A*B <=> C^T = B^T * A^T
    // When row-major data is viewed as column-major, it's effectively transposed.
    // So we call: LtSgemm(B, A) with swapped dimensions to get C in row-major.
    //
    // cuBLAS computes: C_col = op(A_col) * op(B_col)
    // For row-major C[m,k] = A[m,n] * B[n,k]:
    //   - Pass B as first matrix (col-major view = B^T[k,n]), with CUBLAS_OP_T -> B[n,k]
    //   - Pass A as second matrix (col-major view = A^T[n,m]), with CUBLAS_OP_T -> A[m,n]
    //   - Result dimensions: m_cublas=k, n_cublas=m, k_cublas=n
    //   - Output C (col-major view = C^T[k,m]), read as row-major = C[m,k]
    LtSgemm(CUBLAS_OP_N,  // transa: B is already in correct orientation when viewed col-major
            CUBLAS_OP_N,  // transb: A is already in correct orientation when viewed col-major
            k,            // m: rows of op(B) and C (= cols of our result)
            m,            // n: cols of op(A) and C (= rows of our result)
            n,            // k: inner dimension
            &alpha,
            b, k,         // B with leading dim = k (its row stride in row-major = num cols)
            a, n,         // A with leading dim = n (its row stride in row-major = num cols)
            &beta,
            c, k);        // C with leading dim = k
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
