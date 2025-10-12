#pragma once

#include "pch.hpp"

class Tensor {
    bool allocated;

  public:
    uint32_t size;
    std::vector<uint32_t> shape;
    float *data;
    bool requires_grad;

    Tensor(const Tensor &other);
    Tensor(std::vector<uint32_t> shape, bool requires_grad);
    Tensor(const std::vector<uint32_t> &shape, float *data);
    ~Tensor();

    void zero();
    void ones();
    void random();

    void dealloc();

    Tensor operator+(const Tensor &other) const;
    Tensor& operator+=(const Tensor &other);
    Tensor operator*(const Tensor &other) const;
    Tensor operator/(const Tensor &other) const;
    // std::ostream& operator<<(std::ostream &stream);
    Tensor operator[](uint32_t index) const;
    Tensor matmul(const Tensor &other) const;
};
