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
    Tensor result(other.shape, false);
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

        if (other.is_contiguous()) {
            std::copy(other.data(), other.data() + other.size, this->data());
        } else {
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

// Data access
float *Tensor::data() { return storage->ptr() + offset; }

const float *Tensor::data() const { return storage->ptr() + offset; }

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
        cudaMemset(this->data(), 0, this->size * sizeof(float));
        break;
    }
}

void Tensor::ones() { std::fill(this->data(), this->data() + this->size, 1.0f); }

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

// Indexing - returns a view that shares storage
Tensor Tensor::operator[](uint32_t index) {
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

Tensor Tensor::operator+(Tensor &other) {
    Tensor result(this->shape, this->requires_grad || other.requires_grad);
    for (uint32_t i = 0; i < this->size; ++i) {
        result.data()[i] = this->data()[i] + other.data()[i];
    }

    if (this->requires_grad || other.requires_grad) {
        result.grad_fn = std::make_shared<AddBackward>(std::make_shared<Tensor>(*this),
                                                       std::make_shared<Tensor>(other));
    }

    return result;
}

Tensor &Tensor::operator+=(const Tensor &other) {
    for (uint32_t i = 0; i < this->size; ++i) {
        this->data()[i] += other.data()[i];
    }
    return *this;
}

Tensor &Tensor::operator-=(const Tensor &other) {
    for (uint32_t i = 0; i < this->size; ++i) {
        this->data()[i] -= other.data()[i];
    }
    return *this;
}

Tensor Tensor::operator-(Tensor &other) {
    Tensor result(this->shape, this->requires_grad || other.requires_grad);
    for (uint32_t i = 0; i < this->size; ++i) {
        result.data()[i] = this->data()[i] - other.data()[i];
    }

    if (this->requires_grad || other.requires_grad) {
        result.grad_fn = std::make_shared<SubBackward>(std::make_shared<Tensor>(*this),
                                                       std::make_shared<Tensor>(other));
    }

    return result;
}

Tensor Tensor::operator*(Tensor &other) {
    Tensor result(this->shape, this->requires_grad || other.requires_grad);
    for (uint32_t i = 0; i < this->size; ++i) {
        result.data()[i] = this->data()[i] * other.data()[i];
    }

    if (this->requires_grad || other.requires_grad) {
        result.grad_fn = std::make_shared<MulBackward>(std::make_shared<Tensor>(*this),
                                                       std::make_shared<Tensor>(other));
    }

    return result;
}

Tensor Tensor::operator+(float other) {
    Tensor result(this->shape, this->requires_grad);
    for (uint32_t i = 0; i < this->size; ++i) {
        result.data()[i] = this->data()[i] + other;
    }
    return result;
}

Tensor Tensor::operator-(float other) {
    Tensor result(this->shape, this->requires_grad);
    for (uint32_t i = 0; i < this->size; ++i) {
        result.data()[i] = this->data()[i] - other;
    }
    return result;
}

Tensor Tensor::operator*(float other) {
    Tensor result(this->shape, this->requires_grad);
    switch (this->device) {
    case Device::CPU:
        for (uint32_t i = 0; i < this->size; ++i) {
            result.data()[i] = this->data()[i] * other;
        }
        break;
    case Device::CUDA:
        launch_scalar_multiply(result.data(), other, this->size);
        break;
    }

    if (this->requires_grad) {
        auto other_tensor = std::make_shared<Tensor>(std::vector<uint32_t>{1}, false);
        other_tensor->data()[0] = other;
        other_tensor->to(this->device);
        result.grad_fn =
            std::make_shared<MulBackward>(std::make_shared<Tensor>(*this), other_tensor);
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
        launch_scalar_divide(result.data(), other, this->size);
        break;
    }

    if (this->requires_grad) {
        result.grad_fn =
            std::make_shared<DivScalarBackward>(std::make_shared<Tensor>(*this), other);
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
        launch_vec_divide(result.data(), other.data(), this->size);
        break;
    }
    return result;
}

Tensor Tensor::matmul(Tensor &other) {
    if (this->device != other.device) {
        throw std::runtime_error("Matmul: tensors must be on the same device");
    }

    // Create local copies if we need to reshape (don't modify original references)
    Tensor a_view = *this;
    Tensor b_view = other;

    if (a_view.shape.size() == 1) {
        // Reshape 1D to row vector: (n,) -> (1, n)
        a_view = Tensor(a_view.storage, a_view.offset, {1, a_view.shape[0]},
                        {a_view.strides[0], a_view.strides[0]}, a_view.requires_grad,
                        a_view.storage->device);
    }
    if (b_view.shape.size() == 1) {
        // Reshape 1D to column vector: (n,) -> (n, 1)
        b_view = Tensor(b_view.storage, b_view.offset, {b_view.shape[0], 1}, {b_view.strides[0], 0},
                        b_view.requires_grad, b_view.storage->device);
    }

    assert(a_view.shape.size() == 2);
    assert(b_view.shape.size() == 2);
    assert(a_view.shape[1] == b_view.shape[0]);

    std::vector<uint32_t> new_shape({a_view.shape[0], b_view.shape[1]});
    Tensor result(new_shape, a_view.requires_grad || b_view.requires_grad, a_view.device);

    switch (a_view.device) {
    case Device::CPU:
        sgemm(a_view.shape[0], a_view.shape[1], b_view.shape[1], 1.0F, a_view.data(), b_view.data(),
              0.0F, result.data());
        break;
    case Device::CUDA:
        launch_cuda_sgemm(a_view.shape[0], a_view.shape[1], b_view.shape[1], 1.0F, a_view.data(),
                          b_view.data(), 0.0F, result.data());
        break;
    }

    if (a_view.requires_grad || b_view.requires_grad) {
        result.grad_fn = std::make_shared<MatmulBackward>(std::make_shared<Tensor>(a_view),
                                                          std::make_shared<Tensor>(b_view));
    }

    return result;
}

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
    this->device = device;
    this->storage->device = device;
    return result;
}

Tensor matmul(Tensor &a, Tensor &b) { return a.matmul(b); }

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
        launch_transpose(result.data(), this->shape[0], this->shape[1]);
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
    Tensor result({1}, this->requires_grad);
    result.data()[0] = 0.0f;
    float *hostBuffer;
    switch (this->device) {
    case Device::CPU:
        hostBuffer = this->data();
        break;
    case Device::CUDA:
        hostBuffer = new float[this->size];
        cudaMemcpy(hostBuffer, this->data(), this->size * sizeof(float), cudaMemcpyDeviceToHost);
        break;
    }

    for (uint32_t i = 0; i < this->size; i++) {
        result.data()[0] += hostBuffer[i];
    }

    result.to(this->device);

    if (this->requires_grad) {
        result.grad_fn = std::make_shared<SumBackward>(std::make_shared<Tensor>(*this));
    }

    return result;
}

Tensor Tensor::mean() {
    Tensor sum_result = this->sum();
    return sum_result / static_cast<float>(this->size);
}
