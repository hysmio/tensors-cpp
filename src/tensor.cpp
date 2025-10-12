#include "tensor.hpp"
#include "autograd/grad_node.hpp"
#include <numeric>
#include <random>

#include "linalg.hpp"

Tensor::Tensor(const Tensor &other) : allocated(true), shape(other.shape), requires_grad(other.requires_grad), grad_fn(nullptr), grad(nullptr) {
    uint32_t const size =
        std::accumulate(shape.begin(), shape.end(), uint32_t(1), std::multiplies<>());

    this->size = size;
    this->data = new float[size];
    std::copy(other.data, other.data + size, this->data);
}

Tensor::Tensor(std::vector<uint32_t> shape, bool requires_grad) : allocated(true), size(0), shape(std::move(shape)), data(nullptr), requires_grad(requires_grad), grad_fn(nullptr), grad(nullptr) {

    uint32_t const size =
        std::accumulate(this->shape.begin(), this->shape.end(), uint32_t(1), std::multiplies<>());

    this->size = size;
    this->data = new float[size];
}

Tensor::Tensor(const std::vector<uint32_t> &shape, float *data)
    : allocated(false), shape(shape), data(data), requires_grad(false), grad_fn(nullptr), grad(nullptr) {

    uint32_t const size =
        std::accumulate(shape.begin(), shape.end(), uint32_t(1), std::multiplies<>());

    this->size = size;
}

Tensor::~Tensor() {
    if (this->allocated) {
        delete[] this->data;
    }
    if (this->grad) {
        delete this->grad;
    }
}

void Tensor::zero() {
    std::fill(this->data, this->data + this->size, 0.0f);
}

void Tensor::ones() {
    std::fill(this->data, this->data + this->size, 1.0f);
}

void Tensor::random() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (uint32_t i = 0; i < this->size; i++) {
        this->data[i] = dis(gen);
    }
}

void Tensor::dealloc() {
    if (this->allocated) {
        delete[] this->data;
        this->data = nullptr;
        this->allocated = false;
    }
}

// const Tensor Tensor::operator[](uint32_t index) {
//     assert(index < this->shape[0]);

//     uint32_t size = std::accumulate(this->shape.begin() + 1,
//     this->shape.end(), uint32_t(1), std::multiplies<uint32_t>()); Tensor
//     result({this->shape.begin() + 1, this->shape.end()}, this->data + index *
//     size); return result;
// }

Tensor Tensor::operator[](uint32_t index) const {
    assert(!this->shape.empty());
    assert(index < this->shape[0]);

    uint32_t const size = std::accumulate(this->shape.begin() + 1, this->shape.end(), uint32_t(1),
                                          std::multiplies<>());
    Tensor const result({this->shape.begin() + 1, this->shape.end()},
                        this->data + size_t(index * size));
    return result;
}

// std::ostream& Tensor::operator<<(std::ostream &stream) {
//     for (uint32_t i = 0; i < this->shape.size(); i++) {
//         stream << "[";
//         for (uint32_t j = 0; j < this->shape[i]; j++) {
//             stream << this->data[i * this->shape[i] + j];
//             if (j < this->shape[i] - 1) {
//                 stream << ", ";
//             }
//         }
//         stream << "]" << std::endl;
//     }
//     return stream;
// }

Tensor Tensor::operator+(const Tensor &other) const {
    Tensor result(this->shape, this->requires_grad || other.requires_grad);
    for (uint32_t i = 0; i < this->size; ++i) {
        result.data[i] = this->data[i] + other.data[i];
    }
    
    // Set up gradient function if needed
    if (this->requires_grad || other.requires_grad) {
        result.grad_fn = std::make_shared<AddBackward>(
            this->shape, other.shape, this->requires_grad, other.requires_grad
        );
    }
    
    return result;
}

Tensor& Tensor::operator+=(const Tensor &other) {
    for (uint32_t i = 0; i < this->size; ++i) {
        this->data[i] += other.data[i];
    }
    return *this;
}

Tensor Tensor::operator*(const Tensor &other) const {
    if (this->shape.size() == 2 && other.shape.size() == 2) {
        return this->matmul(other);
    }

    Tensor result(this->shape, this->requires_grad || other.requires_grad);
    for (uint32_t i = 0; i < this->size; ++i) {
        result.data[i] = this->data[i] * other.data[i];
    }
    
    // Set up gradient function if needed
    if (this->requires_grad || other.requires_grad) {
        result.grad_fn = std::make_shared<MulBackward>(*this, other);
    }
    
    return result;
}

Tensor Tensor::operator/(const Tensor &other) const {
    assert(this->shape == other.shape);

    const Tensor result(this->shape, this->requires_grad || other.requires_grad);
    for (uint32_t i = 0; i < this->size; ++i) {
        result.data[i] = this->data[i] / other.data[i];
    }
    return result;
}

Tensor Tensor::matmul(const Tensor &other) const {
    assert(this->shape.size() == 2);
    assert(other.shape.size() == 2);
    assert(this->shape[1] == other.shape[0]);

    Tensor result({this->shape[0], other.shape[1]}, this->requires_grad || other.requires_grad);

    sgemm(this->shape[0], this->shape[1], other.shape[1], 1.0F, this->data, other.data, 0.0F,
          result.data);
    
    // Set up gradient function if needed
    if (this->requires_grad || other.requires_grad) {
        result.grad_fn = std::make_shared<MatmulBackward>(*this, other);
    }
    
    return result;
}

// Autograd utility methods
Tensor Tensor::transpose() const {
    assert(this->shape.size() == 2);
    Tensor result({this->shape[1], this->shape[0]}, this->requires_grad);
    
    for (uint32_t i = 0; i < this->shape[0]; i++) {
        for (uint32_t j = 0; j < this->shape[1]; j++) {
            result.data[j * this->shape[0] + i] = this->data[i * this->shape[1] + j];
        }
    }
    
    return result;
}

Tensor Tensor::sum_to_shape(const std::vector<uint32_t>& target_shape) const {
    // For now, implement simple case where shapes are identical or can be summed directly
    if (this->shape == target_shape) {
        return Tensor(*this);
    }
    
    // Simple broadcasting: if target is smaller, sum over extra dimensions
    if (target_shape.size() < this->shape.size()) {
        // For now, just handle the case where we need to sum the first dimension
        if (target_shape.size() == 1 && this->shape.size() == 2 && 
            target_shape[0] == this->shape[1]) {
            
            Tensor result(target_shape, this->requires_grad);
            result.zero();
            
            for (uint32_t i = 0; i < this->shape[0]; i++) {
                for (uint32_t j = 0; j < this->shape[1]; j++) {
                    result.data[j] += this->data[i * this->shape[1] + j];
                }
            }
            return result;
        }
    }
    
    // If shapes don't match and we can't handle the case, just return a copy
    // In a full implementation, this would handle all broadcasting rules
    return Tensor(*this);
}

bool Tensor::is_leaf() const {
    return this->grad_fn == nullptr;
}

void Tensor::backward() {
    // Create ones tensor with same shape as gradient starter
    Tensor ones(this->shape, false);
    ones.ones();
    this->backward(ones);
}

void Tensor::backward(const Tensor& grad_output) {
    // If this is a leaf tensor that requires gradients, accumulate the gradient
    if (this->requires_grad && this->is_leaf()) {
        if (!this->grad) {
            this->grad = new Tensor(this->shape, false);
            this->grad->zero();
        }
        *this->grad += grad_output;
    }
    
    // If this tensor has a gradient function, compute gradients for inputs
    if (this->grad_fn) {
        auto input_grads = this->grad_fn->backward(grad_output);
        // For now, we'll just store the input gradients to demonstrate they're computed
        // In a full implementation, we'd need to track input tensors to call backward on them
        std::cout << "Computed " << input_grads.size() << " input gradients" << std::endl;
        for (size_t i = 0; i < input_grads.size(); i++) {
            std::cout << "Input grad " << i << " size: " << input_grads[i].size << std::endl;
        }
    }
}
