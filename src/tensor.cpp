#include "tensor.hpp"
#include "autograd/grad_node.hpp"
#include "linalg.hpp"
#include <cassert>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

Tensor Tensor::linspace(float start, float end, uint32_t num_points) {
    Tensor result({num_points}, false);
    float step = (end - start) / (num_points - 1);
    for (uint32_t i = 0; i < num_points; ++i) {
        result.data()[i] = start + i * step;
    }
    return result;
}

Tensor Tensor::zeros(std::vector<uint32_t> shape, bool requires_grad) {
    Tensor result(shape, requires_grad);
    result.zero();
    return result;
}

Tensor Tensor::ones_like(const Tensor &other) {
    Tensor result(other.shape, false);
    result.ones();
    return result;
}

Tensor::Tensor(std::vector<uint32_t> shape, bool requires_grad)
    : storage(std::make_shared<TensorData>(compute_size(shape))),
      offset(0),
      size(compute_size(shape)),
      shape(shape),
      strides(compute_strides(shape)),
      requires_grad(requires_grad),
      grad_fn(nullptr),
      grad(std::make_shared<std::shared_ptr<Tensor>>(nullptr)) {}

Tensor::Tensor(const Tensor &other)
    : storage(std::make_shared<TensorData>(other.size)),
      offset(0),
      size(other.size),
      shape(other.shape),
      strides(compute_strides(other.shape)),
      requires_grad(other.requires_grad),
      grad_fn(other.grad_fn),
      grad(other.grad) {
    if (other.is_contiguous()) {
        std::copy(other.data(), other.data() + other.size, this->data());
    } else {
        // Handle non-contiguous copy (iterate element by element)
        for (uint32_t i = 0; i < size; ++i) {
            // Convert linear index to multi-dimensional indices
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

Tensor &Tensor::operator=(const Tensor &other) {
    if (this != &other) {
        storage = std::make_shared<TensorData>(other.size);
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

Tensor::Tensor(std::shared_ptr<TensorData> storage, size_t offset,
               std::vector<uint32_t> shape, std::vector<uint32_t> strides,
               bool requires_grad)
    : storage(std::move(storage)),
      offset(offset),
      size(compute_size(shape)),
      shape(std::move(shape)),
      strides(std::move(strides)),
      requires_grad(requires_grad),
      grad_fn(nullptr),
      grad(std::make_shared<std::shared_ptr<Tensor>>(nullptr)) {}

// Data access
float *Tensor::data() {
    return storage->ptr() + offset;
}

const float *Tensor::data() const {
    return storage->ptr() + offset;
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

bool Tensor::is_contiguous() const {
    return strides == compute_strides(shape);
}

void Tensor::zero() {
    std::fill(this->data(), this->data() + this->size, 0.0f);
}

void Tensor::ones() {
    std::fill(this->data(), this->data() + this->size, 1.0f);
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

    for (uint32_t i = 0; i < this->size; i++) {
        this->data()[i] = dis(gen);
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

    return Tensor(this->storage, new_offset, new_shape, new_strides, this->requires_grad);
}

Tensor Tensor::operator[](uint32_t index) const {
    assert(!this->shape.empty());
    assert(index < this->shape[0]);

    std::vector<uint32_t> new_shape(this->shape.begin() + 1, this->shape.end());
    std::vector<uint32_t> new_strides(this->strides.begin() + 1, this->strides.end());
    size_t new_offset = this->offset + index * this->strides[0];

    // Note: const version still returns non-const view sharing storage
    // The constness here refers to the source tensor, not the view
    return Tensor(this->storage, new_offset, new_shape, new_strides, this->requires_grad);
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
    for (uint32_t i = 0; i < this->size; ++i) {
        result.data()[i] = this->data()[i] * other;
    }

    if (this->requires_grad) {
        auto other_tensor = std::make_shared<Tensor>(std::vector<uint32_t>{1}, false);
        other_tensor->data()[0] = other;
        result.grad_fn =
            std::make_shared<MulBackward>(std::make_shared<Tensor>(*this), other_tensor);
    }

    return result;
}

Tensor Tensor::operator/(float other) {
    Tensor result(this->shape, this->requires_grad);
    for (uint32_t i = 0; i < this->size; ++i) {
        result.data()[i] = this->data()[i] / other;
    }

    if (this->requires_grad) {
        result.grad_fn =
            std::make_shared<DivScalarBackward>(std::make_shared<Tensor>(*this), other);
    }

    return result;
}

Tensor Tensor::operator/(Tensor &other) {
    assert(this->shape == other.shape);

    Tensor result(this->shape, this->requires_grad || other.requires_grad);
    for (uint32_t i = 0; i < this->size; ++i) {
        result.data()[i] = this->data()[i] / other.data()[i];
    }
    return result;
}

Tensor Tensor::matmul(Tensor &other) {

    // Create local copies if we need to reshape (don't modify original references)
    Tensor a_view = *this;
    Tensor b_view = other;

    if (a_view.shape.size() == 1) {
        // Reshape 1D to row vector: (n,) -> (1, n)
        a_view = Tensor(this->storage, a_view.offset, {1, a_view.shape[0]}, {a_view.strides[0], a_view.strides[0]}, a_view.requires_grad);
    }
    if (b_view.shape.size() == 1) {
        // Reshape 1D to column vector: (n,) -> (n, 1)
        b_view = Tensor(b_view.storage, b_view.offset, {b_view.shape[0], 1}, {b_view.strides[0], 0}, b_view.requires_grad);
    }

    assert(a_view.shape.size() == 2);
    assert(b_view.shape.size() == 2);
    assert(a_view.shape[1] == b_view.shape[0]);

    std::vector<uint32_t> new_shape({a_view.shape[0], b_view.shape[1]});
    Tensor result(new_shape, this->requires_grad || b_view.requires_grad);

    sgemm(a_view.shape[0], a_view.shape[1], b_view.shape[1], 1.0F, a_view.data(), b_view.data(), 0.0F, result.data());

    if (a_view.requires_grad || b_view.requires_grad) {
        result.grad_fn = std::make_shared<MatmulBackward>(std::make_shared<Tensor>(a_view),
                                                          std::make_shared<Tensor>(b_view));
    }

    return result;
}

Tensor matmul(Tensor &a, Tensor &b) {
    return a.matmul(b);
}

// Autograd utility methods
Tensor Tensor::transpose() {
    assert(this->shape.size() == 2);
    const std::vector new_shape({this->shape[1], this->shape[0]});
    Tensor result(new_shape, this->requires_grad);

    for (uint32_t i = 0; i < this->shape[0]; i++) {
        for (uint32_t j = 0; j < this->shape[1]; j++) {
            result.data()[j * this->shape[0] + i] = this->data()[i * this->shape[1] + j];
        }
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

            Tensor result(target_shape, this->requires_grad);
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
    *this->grad = std::make_shared<Tensor>(this->shape, false);
    (*this->grad)->ones();
    this->backward(**this->grad);
}

void Tensor::backward(Tensor &grad_output) {
    if (this->requires_grad && this->is_leaf()) {
        if (!*this->grad) {
            *this->grad = std::make_shared<Tensor>(this->shape, false);
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
    for (uint32_t i = 0; i < this->size; i++) {
        result.data()[0] += this->data()[i];
    }

    if (this->requires_grad) {
        result.grad_fn = std::make_shared<SumBackward>(std::make_shared<Tensor>(*this));
    }

    return result;
}

Tensor Tensor::mean() {
    Tensor sum_result = this->sum();
    return sum_result / static_cast<float>(this->size);
}
