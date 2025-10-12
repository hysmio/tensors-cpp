#include "tensor.hpp"
#include <numeric>
#include <random>

#include "linalg.hpp"

Tensor::Tensor(const Tensor &other) : allocated(true), shape(other.shape) {
    uint32_t const size =
        std::accumulate(shape.begin(), shape.end(), uint32_t(1), std::multiplies<>());

    this->size = size;
    this->data = new float[size];
    std::copy(other.data, other.data + size, this->data);
}

Tensor::Tensor(std::vector<uint32_t> shape, bool requires_grad) : allocated(true), size(0), shape(std::move(shape)), data(nullptr), requires_grad(requires_grad) {

    uint32_t const size =
        std::accumulate(this->shape.begin(), this->shape.end(), uint32_t(1), std::multiplies<>());

    this->size = size;
    this->data = new float[size];
}

Tensor::Tensor(const std::vector<uint32_t> &shape, float *data)
    : allocated(false), shape(shape), data(data) {

    uint32_t const size =
        std::accumulate(shape.begin(), shape.end(), uint32_t(1), std::multiplies<>());

    this->size = size;
}

Tensor::~Tensor() {
    if (this->allocated) {
        delete[] this->data;
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
    Tensor const result(this->shape, this->requires_grad || other.requires_grad);
    for (uint32_t i = 0; i < this->size; ++i) {
        result.data[i] = this->data[i] + other.data[i];
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

    Tensor const result(this->shape, this->requires_grad || other.requires_grad);
    for (uint32_t i = 0; i < this->size; ++i) {
        result.data[i] = this->data[i] * other.data[i];
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

    const Tensor result({this->shape[0], other.shape[1]}, this->requires_grad || other.requires_grad);

    sgemm(this->shape[0], this->shape[1], other.shape[1], 1.0F, this->data, other.data, 0.0F,
          result.data);
    return result;
}
