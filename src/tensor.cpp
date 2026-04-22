#include "tensor.hpp"
#include "autograd/grad_node.hpp"
#include "device.hpp"
#include "linalg.hpp"
#include <cassert>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "backend/cuda/cuda_backend.cuh"

Tensor Tensor::linspace(float start, float end, uint32_t num_points, Device device) {
    Tensor result({num_points}, false, device);
    switch (device) {
    case Device::CPU: {
        float step = (end - start) / (num_points - 1);
        for (uint32_t i = 0; i < num_points; ++i) {
            result.data()[i] = start + i * step;
        }
        break;
    }
    case Device::CUDA: {
        launch_linspace(result.data(), start, end, num_points);
        break;
    }
    }
    return result;
}

Tensor Tensor::zeros(std::vector<uint32_t> shape, bool requires_grad, Device device) {
    Tensor result(shape, requires_grad, device);
    result.zero();
    return result;
}

Tensor Tensor::ones_like(const Tensor &other) {
    Tensor result(other.shape, false, other.device);
    result.ones();
    return result;
}

Tensor::Tensor(std::vector<uint32_t> shape, bool requires_grad, Device device)
    : storage(std::make_shared<TensorData>(compute_size(shape), device)), offset(0),
      size(compute_size(shape)), shape(shape), strides(compute_strides(shape)),
      requires_grad(requires_grad), grad_fn(nullptr),
      grad(std::make_shared<std::shared_ptr<Tensor>>(nullptr)), device(device) {}

Tensor::Tensor(const Tensor &other)
    : storage(std::make_shared<TensorData>(other.size, other.device)), offset(0), size(other.size),
      shape(other.shape), strides(compute_strides(other.shape)), requires_grad(other.requires_grad),
      grad_fn(other.grad_fn), grad(other.grad), device(other.device) {
    // if (other.is_contiguous()) {
    //     switch (other.device) {
    //     case Device::CPU:
    //         std::copy(other.data(), other.data() + other.size, this->data());
    //         break;
    //     case Device::CUDA:
    //         cudaMemcpy(this->data(), other.data(), other.size * sizeof(float),
    //                    cudaMemcpyDeviceToDevice);
    //         break;
    //     if
    //     }
    // } else {
    //     // Handle non-contiguous copy (iterate element by element)
    //     for (uint32_t i = 0; i < size; ++i) {
    //         // Convert linear index to multi-dimensional indices
    //         std::vector<uint32_t> indices(shape.size());
    //         uint32_t remaining = i;
    //         for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
    //             indices[d] = remaining % shape[d];
    //             remaining /= shape[d];
    //         }
    //         this->data()[i] = other.at(indices);
    //     }
    // }
    if (this->device != other.device) {
        // move to cuda from other cpu
        if (this->device == Device::CUDA) {
            cudaMemcpy(this->data(), other.data(), other.size * sizeof(float),
                       cudaMemcpyHostToDevice);
        } else {
            // move from other cuda to cpu
            cudaMemcpy(this->data(), other.data(), other.size * sizeof(float),
                       cudaMemcpyDeviceToHost);
        }
    } else {
        switch (other.device) {
        case Device::CPU:
            std::copy(other.data(), other.data() + other.size, this->data());
            break;
        case Device::CUDA:
            cudaMemcpy(this->data(), other.data(), other.size * sizeof(float),
                       cudaMemcpyDeviceToDevice);
            break;
        }
    }
}

Tensor &Tensor::operator=(const Tensor &other) {
    if (this != &other) {
        storage = std::make_shared<TensorData>(other.size, other.device);
        offset = 0;
        size = other.size;
        shape = other.shape;
        strides = compute_strides(other.shape);
        requires_grad = other.requires_grad;
        grad_fn = other.grad_fn;
        grad = other.grad;
        device = other.device;

        if (other.is_contiguous()) {
            switch (other.device) {
            case Device::CPU:
                std::copy(other.data(), other.data() + other.size, this->data());
                break;
            case Device::CUDA:
                cudaMemcpy(this->data(), other.data(), other.size * sizeof(float),
                           cudaMemcpyDeviceToDevice);
                break;
            }
        } else {
            // Non-contiguous copy only supported on CPU
            assert(other.device == Device::CPU);
            for (uint32_t i = 0; i < size; ++i) {
                std::vector<uint32_t> indices(shape.size());
                uint32_t remaining = i;
                for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
                    indices[d] = remaining % shape[d];
                    remaining /= shape[d];
                }
                this->data()[i] = other.at(indices);
            }
        }
    }
    return *this;
}

Tensor::Tensor(std::shared_ptr<TensorData> storage, size_t offset, std::vector<uint32_t> shape,
               std::vector<uint32_t> strides, bool requires_grad, Device device)
    : storage(std::move(storage)), offset(offset), size(compute_size(shape)),
      shape(std::move(shape)), strides(std::move(strides)), requires_grad(requires_grad),
      grad_fn(nullptr), grad(std::make_shared<std::shared_ptr<Tensor>>(nullptr)), device(device) {}

void Tensor::zero_grad() {
    if (*grad) {
        switch (device) {
        case Device::CPU:
            std::fill_n((*grad)->data(), (*grad)->size, 0.0f);
            break;
        case Device::CUDA:
            cudaMemsetAsync((*grad)->data(), 0, (*grad)->size * sizeof(float), 0);
            break;
        }
    }
}

// Data access
float *Tensor::data() { return storage->ptr() + offset; }

const float *Tensor::data() const { return storage->ptr() + offset; }

std::shared_ptr<Tensor> Tensor::shared_copy() const {
    auto t = std::make_shared<Tensor>(storage, offset, shape, strides, requires_grad, device);
    t->grad_fn = grad_fn;
    t->grad = grad;
    return t;
}

// Element access with proper stride handling
float &Tensor::at(const std::vector<uint32_t> &indices) {
    size_t idx = compute_linear_index(indices, strides, offset);
    return storage->data[idx];
}

float Tensor::at(const std::vector<uint32_t> &indices) const {
    size_t idx = compute_linear_index(indices, strides, offset);
    return storage->data[idx];
}

bool Tensor::is_contiguous() const { return strides == compute_strides(shape); }

void Tensor::zero() {
    switch (this->device) {
    case Device::CPU:
        std::fill(this->data(), this->data() + this->size, 0.0f);
        break;
    case Device::CUDA:
        cudaMemsetAsync(this->data(), 0, this->size * sizeof(float), 0);
        break;
    }
}

void Tensor::ones() {
    switch (this->device) {
    case Device::CPU:
        std::fill(this->data(), this->data() + this->size, 1.0f);
        break;
    case Device::CUDA:
        launch_fill_value(this->data(), 1.0f, this->size);
        break;
    }
}

void Tensor::random() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (uint32_t i = 0; i < this->size; i++) {
        this->data()[i] = dis(gen);
    }
}

void Tensor::xavier_uniform(uint32_t fan_in, uint32_t fan_out) {
    std::random_device rd;
    std::mt19937 gen(rd());
    float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    std::uniform_real_distribution<float> dis(-limit, limit);

    float *data;
    if (this->device == Device::CPU) {
        data = this->data();
    } else if (this->device == Device::CUDA) {
        data = new float[this->size];
    }

    for (uint32_t i = 0; i < this->size; i++) {
        data[i] = dis(gen);
    }

    if (this->device == Device::CUDA) {
        cudaMemcpy(this->data(), data, this->size * sizeof(float), cudaMemcpyHostToDevice);
        delete[] data;
    }
}

void Tensor::kaiming_uniform(uint32_t fan_in) {
    std::random_device rd;
    std::mt19937 gen(rd());
    // PyTorch default: kaiming_uniform with a=sqrt(5), leaky_relu mode
    // bound = sqrt(3) * gain / sqrt(fan_in) = 1 / sqrt(fan_in)
    float limit = 1.0f / std::sqrt(static_cast<float>(fan_in));
    std::uniform_real_distribution<float> dis(-limit, limit);

    float *data;
    if (this->device == Device::CPU) {
        data = this->data();
    } else if (this->device == Device::CUDA) {
        data = new float[this->size];
    }

    for (uint32_t i = 0; i < this->size; i++) {
        data[i] = dis(gen);
    }

    if (this->device == Device::CUDA) {
        cudaMemcpy(this->data(), data, this->size * sizeof(float), cudaMemcpyHostToDevice);
        delete[] data;
    }
}

void Tensor::tanh() {
    float* data = this->data();
    switch (this->device) {
    case Device::CPU:
        for (uint32_t i = 0; i < this->size; i++) {
            data[i] = std::tanh(data[i]);
        }
        break;
    case Device::CUDA:
        launch_tanh_forward(data, data, this->size);
        break;
    }
    if (this->requires_grad) {
        this->grad_fn = std::make_shared<TanhBackward>(
            this->shared_copy(),
            this->shared_copy() // Store output for backward
        );
    }
}

// Indexing - returns a view that shares storage
Tensor Tensor::operator[](uint32_t index) {
    assert(this->device != Device::CUDA);
    assert(!this->shape.empty());
    assert(index < this->shape[0]);

    // New shape is shape[1:]
    std::vector<uint32_t> new_shape(this->shape.begin() + 1, this->shape.end());
    // New strides is strides[1:]
    std::vector<uint32_t> new_strides(this->strides.begin() + 1, this->strides.end());
    // New offset is current offset + index * strides[0]
    size_t new_offset = this->offset + index * this->strides[0];

    return Tensor(this->storage, new_offset, new_shape, new_strides, this->requires_grad,
                  this->device);
}

Tensor Tensor::operator[](uint32_t index) const {
    assert(this->device != Device::CUDA);
    assert(!this->shape.empty());
    assert(index < this->shape[0]);

    std::vector<uint32_t> new_shape(this->shape.begin() + 1, this->shape.end());
    std::vector<uint32_t> new_strides(this->strides.begin() + 1, this->strides.end());
    size_t new_offset = this->offset + index * this->strides[0];

    // Note: const version still returns non-const view sharing storage
    // The constness here refers to the source tensor, not the view
    return Tensor(this->storage, new_offset, new_shape, new_strides, this->requires_grad,
                  this->device);
}

Tensor Tensor::operator-() const {
    Tensor result(this->shape, this->requires_grad, this->device);
    switch (this->device) {
    case Device::CPU:
        for (uint32_t i = 0; i < this->size; ++i) {
            result.data()[i] = -this->data()[i];
        }
        break;
    case Device::CUDA:
        launch_negate(this->data(), result.data(), this->size);
        break;
    }
    return result;
}

Tensor Tensor::operator+(Tensor &other) {
    Tensor result(this->shape, this->requires_grad || other.requires_grad, this->device);

    switch (this->device) {
    case Device::CPU:
        for (uint32_t i = 0; i < this->size; ++i) {
            result.data()[i] = this->data()[i] + other.data()[i];
        }
        break;
    case Device::CUDA:
        launch_vec_add(this->data(), other.data(), result.data(), this->size);
        break;
    }

    if (this->requires_grad || other.requires_grad) {
        result.grad_fn = std::make_shared<AddBackward>(this->shared_copy(), other.shared_copy());
    }

    return result;
}

Tensor &Tensor::operator+=(Tensor &other) {
    switch (this->device) {
    case Device::CPU:
        for (uint32_t i = 0; i < this->size; ++i) {
            this->data()[i] += other.data()[i];
        }
        break;
    case Device::CUDA:
        launch_vec_add(this->data(), other.data(), this->data(), this->size);
        break;
    }
    return *this;
}

Tensor &Tensor::operator+=(const Tensor &other) {
    switch (this->device) {
    case Device::CPU:
        for (uint32_t i = 0; i < this->size; ++i) {
            this->data()[i] += other.data()[i];
        }
        break;
    case Device::CUDA:
        launch_vec_add(this->data(), other.data(), this->data(), this->size);
        break;
    }
    return *this;
}

Tensor &Tensor::operator-=(const Tensor &other) {
    switch (this->device) {
    case Device::CPU:
        for (uint32_t i = 0; i < this->size; ++i) {
            this->data()[i] -= other.data()[i];
        }
        break;
    case Device::CUDA:
        launch_vec_subtract(this->data(), other.data(), this->data(), this->size);
        break;
    }
    return *this;
}

Tensor Tensor::operator-(Tensor &other) {
    Tensor result(this->shape, this->requires_grad || other.requires_grad, this->device);
    switch (this->device) {
    case Device::CPU:
        for (uint32_t i = 0; i < this->size; ++i) {
            result.data()[i] = this->data()[i] - other.data()[i];
        }
        break;
    case Device::CUDA:
        launch_vec_subtract(this->data(), other.data(), result.data(), this->size);
        break;
    }

    if (this->requires_grad || other.requires_grad) {
        result.grad_fn = std::make_shared<SubBackward>(this->shared_copy(), other.shared_copy());
    }

    return result;
}

Tensor Tensor::operator*(float scalar) {
    Tensor result(this->shape, this->requires_grad, this->device);
    switch (this->device) {
    case Device::CPU:
        for (uint32_t i = 0; i < this->size; ++i) {
            result.data()[i] = this->data()[i] * scalar;
        }
        break;
    case Device::CUDA:
        launch_scalar_multiply(this->data(), scalar, result.data(), this->size);
        break;
    }

    if (this->requires_grad) {
        auto other_tensor = std::make_shared<Tensor>(std::vector<uint32_t>{1}, false, this->device);
        other_tensor->data()[0] = scalar;
        result.grad_fn = std::make_shared<MulBackward>(this->shared_copy(), other_tensor);
    }

    return result;
}

Tensor Tensor::operator*(Tensor &other) {
    Tensor result(this->shape, this->requires_grad || other.requires_grad, this->device);
    switch (this->device) {
    case Device::CPU:
        for (uint32_t i = 0; i < this->size; ++i) {
            result.data()[i] = this->data()[i] * other.data()[i];
        }
        break;
    case Device::CUDA:
        launch_vec_multiply(this->data(), other.data(), result.data(), this->size);
        break;
    }

    if (this->requires_grad || other.requires_grad) {
        result.grad_fn = std::make_shared<MulBackward>(this->shared_copy(), other.shared_copy());
    }

    return result;
}

Tensor Tensor::operator+(float other) {
    Tensor result(this->shape, this->requires_grad, this->device);
    switch (this->device) {
    case Device::CPU:
        for (uint32_t i = 0; i < this->size; ++i) {
            result.data()[i] = this->data()[i] + other;
        }
        break;
    case Device::CUDA:
        launch_scalar_add(this->data(), other, result.data(), this->size);
        break;
    }

    if (this->requires_grad) {
        auto scalar_tensor = std::make_shared<Tensor>(std::vector<uint32_t>{1}, false, Device::CPU);
        scalar_tensor->data()[0] = other;
        result.grad_fn = std::make_shared<AddBackward>(this->shared_copy(), scalar_tensor);
    }

    return result;
}

Tensor Tensor::operator-(float other) {
    Tensor result(this->shape, this->requires_grad, this->device);
    switch (this->device) {
    case Device::CPU:
        for (uint32_t i = 0; i < this->size; ++i) {
            result.data()[i] = this->data()[i] - other;
        }
        break;
    case Device::CUDA:
        launch_scalar_subtract(this->data(), other, result.data(), this->size);
        break;
    }

    if (this->requires_grad) {
        auto scalar_tensor = std::make_shared<Tensor>(std::vector<uint32_t>{1}, false, Device::CPU);
        scalar_tensor->data()[0] = other;
        result.grad_fn = std::make_shared<MulBackward>(this->shared_copy(), scalar_tensor);
    }

    return result;
}

Tensor Tensor::operator/(float other) {
    Tensor result(this->shape, this->requires_grad, this->device);

    switch (this->device) {
    case Device::CPU:
        for (uint32_t i = 0; i < this->size; ++i) {
            result.data()[i] = this->data()[i] / other;
        }
        break;
    case Device::CUDA:
        launch_scalar_divide(this->data(), other, result.data(), this->size);
        break;
    }

    if (this->requires_grad) {
        result.grad_fn =
            std::make_shared<DivScalarBackward>(this->shared_copy(), other);
    }

    return result;
}

Tensor Tensor::operator/(Tensor &other) {
    assert(this->shape == other.shape);

    Tensor result(this->shape, this->requires_grad || other.requires_grad, this->device);

    switch (this->device) {
    case Device::CPU:
        for (uint32_t i = 0; i < this->size; ++i) {
            result.data()[i] = this->data()[i] / other.data()[i];
        }
        break;
    case Device::CUDA:
        launch_vec_divide(this->data(), other.data(), result.data(), this->size);
        break;
    }

    if (this->requires_grad) {
        result.grad_fn = std::make_shared<DivBackward>(this->shared_copy(), other.shared_copy());
    }

    return result;
}

Tensor Tensor::matmul(Tensor &other) {
    if (this->device != other.device) {
        throw std::runtime_error("Matmul: tensors must be on the same device");
    }

    // // Create local copies if we need to reshape (don't modify original references)
    // Tensor a_view = *this;
    // Tensor b_view = other;

    // if (a_view.shape.size() == 1) {
    //     // Reshape 1D to row vector: (n,) -> (1, n)
    //     a_view = Tensor(a_view.storage, a_view.offset, {1, a_view.shape[0]},
    //                     {a_view.strides[0], a_view.strides[0]}, a_view.requires_grad,
    //                     a_view.storage->device);
    // }
    // if (b_view.shape.size() == 1) {
    //     // Reshape 1D to column vector: (n,) -> (n, 1)
    //     b_view = Tensor(b_view.storage, b_view.offset, {b_view.shape[0], 1}, {b_view.strides[0], 0},
    //                     b_view.requires_grad, b_view.storage->device);
    // }

    assert(this->shape.size() == 2);
    assert(other.shape.size() == 2);

    // Check if we need A @ B^T (shapes [M,K] @ [N,K]) or standard A @ B ([M,K] @ [K,N])
    bool transpose_b = (this->shape[1] == other.shape[1]);

    // Output shape: [M, N] where N = other.shape[0] if transposing, else other.shape[1]
    uint32_t M = this->shape[0];
    uint32_t K = this->shape[1];
    uint32_t N = transpose_b ? other.shape[0] : other.shape[1];

    std::vector<uint32_t> new_shape({M, N});
    Tensor result(new_shape, this->requires_grad || other.requires_grad, this->device);

    switch (this->device) {
    case Device::CPU: {
        Tensor b = other;
        if (transpose_b) {
            b = other.transpose();
        }
        sgemm(M, K, N, 1.0F, this->data(), b.data(), 0.0F, result.data());
        break;
    }
    case Device::CUDA: {
        float alpha = 1.0f;
        float beta = 0.0f;

        if (transpose_b) {
            // A[M,K] @ B[N,K]^T -> C[M,N]
            // cuBLAS computes: C^T[N,M] = B[N,K] @ A^T[K,M]
            // B row-major [N,K] viewed col-major = [K,N], need OP_T to get [N,K]
            // A row-major [M,K] viewed col-major = [K,M], OP_N gives [K,M]
            LtSgemm(CUBLAS_OP_T,              // transa: transpose B
                CUBLAS_OP_N,                  // transb: A^T is what we want
                N,                            // m: rows of result (in col-major view)
                M,                            // n: cols of result (in col-major view)
                K,                            // k: inner dimension
                &alpha,
                other.data(), K,              // B[N,K] with leading dim = K
                this->data(), K,              // A[M,K] with leading dim = K
                &beta,
                result.data(), N);            // C[M,N] with leading dim = N
        } else {
            // Standard A[M,K] @ B[K,N] -> C[M,N]
            // cuBLAS computes: C^T[N,M] = B^T[N,K] @ A^T[K,M]
            // Both are already in correct orientation when viewed col-major
            LtSgemm(CUBLAS_OP_N,              // transa: B is already transposed when viewed col-major
                CUBLAS_OP_N,                  // transb: A is already transposed when viewed col-major
                N,                            // m: rows of op(B) and C
                M,                            // n: cols of op(A) and C
                K,                            // k: inner dimension
                &alpha,
                other.data(), N,              // B[K,N] with leading dim = N
                this->data(), K,              // A[M,K] with leading dim = K
                &beta,
                result.data(), N);            // C[M,N] with leading dim = N
        }
        break;
    }
    }

    if (this->requires_grad || other.requires_grad) {
        result.grad_fn = std::make_shared<MatmulBackward>(this->shared_copy(),
                                                          other.shared_copy(),
                                                          false, transpose_b);
    }

    return result;
}

/***
 * Creates a copy of the Tensor on the new device
 */
Tensor Tensor::to(Device device) {
    if (this->device == device) {
        return *this;
    }
    Tensor result(this->shape, this->requires_grad, device);
    switch (device) {
    case Device::CPU:
        cudaMemcpy(result.data(), this->data(), this->size * sizeof(float), cudaMemcpyDeviceToHost);
        break;
    case Device::CUDA:
        cudaMemcpy(result.data(), this->data(), this->size * sizeof(float), cudaMemcpyHostToDevice);
        break;
    }
    result.device = device;
    result.storage->device = device;
    return result;
}

Tensor matmul(Tensor &a, Tensor &b) { return a.matmul(b); }

Tensor matmul(Tensor &a, Tensor &b, bool transpose_a, bool transpose_b) {
    return a.matmul(b, transpose_a, transpose_b);
}

Tensor Tensor::matmul(Tensor &other, bool transpose_a, bool transpose_b) {
    if (this->device != other.device) {
        throw std::runtime_error("Matmul: tensors must be on the same device");
    }

    assert(this->shape.size() == 2);
    assert(other.shape.size() == 2);

    // Compute effective dimensions after transpose
    // A is [M,K], A^T is [K,M]
    // B is [K,N], B^T is [N,K]
    uint32_t A_rows = transpose_a ? this->shape[1] : this->shape[0];
    uint32_t A_cols = transpose_a ? this->shape[0] : this->shape[1];
    uint32_t B_rows = transpose_b ? other.shape[1] : other.shape[0];
    uint32_t B_cols = transpose_b ? other.shape[0] : other.shape[1];

    assert(A_cols == B_rows && "Inner dimensions must match");

    uint32_t M = A_rows;
    uint32_t K = A_cols;
    uint32_t N = B_cols;

    std::vector<uint32_t> new_shape({M, N});
    Tensor result(new_shape, this->requires_grad || other.requires_grad, this->device);

    switch (this->device) {
    case Device::CPU: {
        Tensor a_eff = transpose_a ? this->transpose() : *this;
        Tensor b_eff = transpose_b ? other.transpose() : other;
        sgemm(M, K, N, 1.0F, a_eff.data(), b_eff.data(), 0.0F, result.data());
        break;
    }
    case Device::CUDA: {
        float alpha = 1.0f;
        float beta = 0.0f;

        // Row-major to col-major trick: C = A @ B becomes C^T = B^T @ A^T
        // When row-major data is viewed as col-major, it's transposed
        // So we swap operand order and adjust ops accordingly

        // A stored as [this->shape[0], this->shape[1]], ld = shape[1]
        // B stored as [other.shape[0], other.shape[1]], ld = other.shape[1]

        cublasOperation_t op_a_cublas, op_b_cublas;

        // In cuBLAS call order: LtSgemm(op_B, op_A, ...)
        // Row-major trick: C = A @ B → C^T = B^T @ A^T
        // Row-major X viewed col-major = X^T (implicit transpose)
        //
        // For A: A_col = A^T
        //   - If transpose_a=false: we want A^T in product, A_col already is A^T → OP_N
        //   - If transpose_a=true: we want A in product, need to un-transpose → OP_T
        // For B: B_col = B^T
        //   - If transpose_b=false: we want B^T in product, B_col already is B^T → OP_N
        //   - If transpose_b=true: we want B in product, need to un-transpose → OP_T
        op_a_cublas = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
        op_b_cublas = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

        LtSgemm(op_b_cublas,           // op for B (first in cuBLAS due to swap)
                op_a_cublas,           // op for A (second in cuBLAS due to swap)
                N,                     // m: rows of C^T
                M,                     // n: cols of C^T
                K,                     // k: inner dimension
                &alpha,
                other.data(), other.shape[1],  // B with its row stride
                this->data(), this->shape[1],  // A with its row stride
                &beta,
                result.data(), N);     // C[M,N] with ld=N
        break;
    }
    }

    return result;
}

// Autograd utility methods
Tensor Tensor::transpose() {
    assert(this->shape.size() == 2);
    const std::vector new_shape({this->shape[1], this->shape[0]});
    Tensor result(new_shape, this->requires_grad, this->device);

    switch (this->device) {
    case Device::CPU:
        for (uint32_t i = 0; i < this->shape[0]; i++) {
            for (uint32_t j = 0; j < this->shape[1]; j++) {
                result.data()[j * this->shape[0] + i] = this->data()[i * this->shape[1] + j];
            }
        }
        break;
    case Device::CUDA:
        launch_transpose_copy(this->data(), result.data(), this->shape[0], this->shape[1]);
        break;
    }

    return result;
}

Tensor Tensor::sum_to_shape(std::vector<uint32_t> &target_shape) {
    if (this->shape == target_shape) {
        return *this;
    }

    if (target_shape.size() < this->shape.size()) {
        if (target_shape.size() == 1 && this->shape.size() == 2 &&
            target_shape[0] == this->shape[1]) {

            Tensor result(target_shape, this->requires_grad, this->device);
            result.zero();

            for (uint32_t i = 0; i < this->shape[0]; i++) {
                for (uint32_t j = 0; j < this->shape[1]; j++) {
                    result.data()[j] += this->data()[i * this->shape[1] + j];
                }
            }
            return result;
        }
    }

    return *this;
}

bool Tensor::is_leaf() { return this->grad_fn == nullptr; }

void Tensor::backward() {
    *this->grad = std::make_shared<Tensor>(this->shape, false, this->device);
    (*this->grad)->ones();
    this->backward(**this->grad);
}

void Tensor::backward(Tensor &grad_output) {
    if (this->requires_grad && this->is_leaf()) {
        if (!*this->grad) {
            *this->grad = std::make_shared<Tensor>(this->shape, false, this->device);
            (*this->grad)->zero();
        }
        (**this->grad) += grad_output;
    }

    if (this->grad_fn) {
        this->grad_fn->backward(grad_output);
    }
}

Tensor Tensor::sum() {
    Tensor result({1}, this->requires_grad, this->device);

    switch (this->device) {
    case Device::CPU:
        result.data()[0] = 0.0f;
        for (uint32_t i = 0; i < this->size; i++) {
            result.data()[0] += this->data()[i];
        }
        break;
    case Device::CUDA:
        launch_reduce_sum(this->data(), result.data(), this->size);
        break;
    }

    if (this->requires_grad) {
        result.grad_fn = std::make_shared<SumBackward>(this->shared_copy());
    }

    return result;
}

Tensor Tensor::mean() {
    Tensor sum_result = this->sum();
    return sum_result / static_cast<float>(this->size);
}
