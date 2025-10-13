#include "tensor.hpp"
#include "autograd/grad_node.hpp"
#include "linalg.hpp"
#include <numeric>
#include <random>
#include <vector>

Tensor *Tensor::linspace(float start, float end, uint32_t num_points) {
    float step = (end - start) / (num_points - 1);
    float *data = new float[num_points];
    for (uint32_t i = 0; i < num_points; ++i) {
        data[i] = start + i * step;
    }
    return new Tensor({num_points}, data);
}

Tensor::Tensor(const Tensor &other)
    : allocated(true), shape(other.shape), requires_grad(other.requires_grad),
      grad_fn(other.grad_fn), grad(other.grad ? other.grad : nullptr) {
    uint32_t size = std::accumulate(shape.begin(), shape.end(), uint32_t(1), std::multiplies<>());

    this->size = size;
    this->data = new float[size];
    std::copy(other.data, other.data + size, this->data);
}

Tensor::Tensor(std::vector<uint32_t> shape, bool requires_grad)
    : allocated(true), size(0), shape(shape), data(nullptr), requires_grad(requires_grad),
      grad_fn(nullptr), grad(nullptr) {

    uint32_t size =
        std::accumulate(this->shape.begin(), this->shape.end(), uint32_t(1), std::multiplies<>());

    this->size = size;
    this->data = new float[size];
}

Tensor::Tensor(std::vector<uint32_t> shape, float *data)
    : allocated(false), shape(shape), data(data), requires_grad(false), grad_fn(nullptr),
      grad(nullptr) {

    uint32_t size = std::accumulate(shape.begin(), shape.end(), uint32_t(1), std::multiplies<>());

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

void Tensor::zero() { std::fill(this->data, this->data + this->size, 0.0f); }

void Tensor::ones() { std::fill(this->data, this->data + this->size, 1.0f); }

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

//  Tensor Tensor::operator[](uint32_t index) {
//     assert(index < this->shape[0]);

//     uint32_t size = std::accumulate(this->shape.begin() + 1,
//     this->shape.end(), uint32_t(1), std::multiplies<uint32_t>()); Tensor
//     result({this->shape.begin() + 1, this->shape.end()}, this->data + index *
//     size); return result;
// }

Tensor *Tensor::operator[](uint32_t index) {
    assert(!this->shape.empty());
    assert(index < this->shape[0]);

    uint32_t size = std::accumulate(this->shape.begin() + 1, this->shape.end(), uint32_t(1),
                                    std::multiplies<>());
    std::vector<uint32_t> new_shape(this->shape.begin() + 1, this->shape.end());
    Tensor *result = new Tensor(new_shape, (float *)(this->data + size_t(index * size)));
    return result;
}

Tensor *Tensor::operator[](uint32_t index) const {
    assert(!this->shape.empty());
    assert(index < this->shape[0]);

    uint32_t size = std::accumulate(this->shape.begin() + 1, this->shape.end(), uint32_t(1),
                                    std::multiplies<>());
    std::vector<uint32_t> new_shape(this->shape.begin() + 1, this->shape.end());
    Tensor *result = new Tensor(new_shape, this->data + size_t(index * size));
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

Tensor *Tensor::operator+(Tensor *other) {
    Tensor *result = new Tensor(this->shape, this->requires_grad || other->requires_grad);
    for (uint32_t i = 0; i < this->size; ++i) {
        result->data[i] = this->data[i] + other->data[i];
    }

    // Set up gradient function if needed
    if (this->requires_grad || other->requires_grad) {
    }

    return result;
}

Tensor *Tensor::operator+=(Tensor *other) {
    for (uint32_t i = 0; i < this->size; ++i) {
        this->data[i] += other->data[i];
    }
    return this;
}

Tensor *Tensor::operator-(Tensor *other) {
    Tensor *result = new Tensor(this->shape, this->requires_grad || other->requires_grad);
    for (uint32_t i = 0; i < this->size; ++i) {
        result->data[i] = this->data[i] - other->data[i];
    }

    // Set up gradient function if needed
    if (this->requires_grad || other->requires_grad) {
        result->grad_fn = std::make_shared<AddBackward>(this, other);
    }

    return result;
}

Tensor *Tensor::operator*(Tensor *other) {
    Tensor *result = new Tensor(this->shape, this->requires_grad || other->requires_grad);
    for (uint32_t i = 0; i < this->size; ++i) {
        result->data[i] = this->data[i] * other->data[i];
    }

    // Set up gradient function if needed
    if (this->requires_grad || other->requires_grad) {
        result->grad_fn = std::make_shared<MulBackward>(this, other);
    }

    return result;
}

Tensor *Tensor::operator*(float other) {
    Tensor *result = new Tensor(this->shape, this->requires_grad);
    for (uint32_t i = 0; i < this->size; ++i) {
        result->data[i] = this->data[i] * other;
    }

    // Set up gradient function if needed
    if (this->requires_grad) {
        Tensor *other_tensor = new Tensor({1}, false);
        other_tensor->data[0] = other;
        result->grad_fn = std::make_shared<MulBackward>(this, other_tensor);
    }

    return result;
}

Tensor *Tensor::operator/(Tensor *other) {
    assert(this->shape == other->shape);

    Tensor *result = new Tensor(this->shape, this->requires_grad || other->requires_grad);
    for (uint32_t i = 0; i < this->size; ++i) {
        result->data[i] = this->data[i] / other->data[i];
    }
    return result;
}

Tensor *matmul(Tensor *a, Tensor *b) {
    assert(a->shape.size() == 2);
    assert(b->shape.size() == 2);
    assert(a->shape[1] == b->shape[0]);

    std::vector<uint32_t> new_shape({a->shape[0], b->shape[1]});
    Tensor *result = new Tensor(new_shape, a->requires_grad || b->requires_grad);

    sgemm(a->shape[0], a->shape[1], b->shape[1], 1.0F, a->data, b->data, 0.0F, result->data);

    // Set up gradient function if needed
    if (a->requires_grad || b->requires_grad) {
        result->grad_fn = std::make_shared<MatmulBackward>(a, b);
    }

    return result;
}

// Autograd utility methods
Tensor *Tensor::transpose() {
    assert(this->shape.size() == 2);
    std::vector<uint32_t> new_shape({this->shape[1], this->shape[0]});
    Tensor *result = new Tensor(new_shape, this->requires_grad);

    for (uint32_t i = 0; i < this->shape[0]; i++) {
        for (uint32_t j = 0; j < this->shape[1]; j++) {
            result->data[j * this->shape[0] + i] = this->data[i * this->shape[1] + j];
        }
    }

    return result;
}

Tensor *Tensor::sum_to_shape(std::vector<uint32_t> &target_shape) {
    // For now, implement simple case where shapes are identical or can be summed directly
    if (this->shape == target_shape) {
        return new Tensor(*this);
    }

    // Simple broadcasting: if target is smaller, sum over extra dimensions
    if (target_shape.size() < this->shape.size()) {
        // For now, just handle the case where we need to sum the first dimension
        if (target_shape.size() == 1 && this->shape.size() == 2 &&
            target_shape[0] == this->shape[1]) {

            Tensor *result = new Tensor(target_shape, this->requires_grad);
            result->zero();

            for (uint32_t i = 0; i < this->shape[0]; i++) {
                for (uint32_t j = 0; j < this->shape[1]; j++) {
                    result->data[j] += this->data[i * this->shape[1] + j];
                }
            }
            return result;
        }
    }

    // If shapes don't match and we can't handle the case, just return a copy
    // In a full implementation, this would handle all broadcasting rules
    return this;
}

bool Tensor::is_leaf() { return this->grad_fn == nullptr; }

void Tensor::backward() {
    // Create ones tensor with same shape as gradient starter
    this->grad = new Tensor(this->shape, false); // For scalar tensors, grad is just ones
    this->grad->ones();
    this->backward(this->grad);
}

void Tensor::backward(Tensor *grad_output) {
    // If this is a leaf tensor that requires gradients, accumulate the gradient
    if (this->requires_grad && this->is_leaf()) {
        if (!this->grad) {
            this->grad = new Tensor(this->shape, false);
            this->grad->zero();
        }
        (*this->grad) += grad_output;
    }

    // If this tensor has a gradient function, apply gradients to input tensors
    if (this->grad_fn) {
        this->grad_fn->backward(grad_output);
    }
}
