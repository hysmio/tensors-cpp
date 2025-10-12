#pragma once

#include "pch.hpp"
#include <memory>

// Forward declarations
struct GradNode;

class Tensor {
    bool allocated;

  public:
    uint32_t size;
    std::vector<uint32_t> shape;
    float *data;
    bool requires_grad;

    // Autograd support
    std::shared_ptr<GradNode> grad_fn;
    Tensor *grad;

    Tensor(const Tensor &other);
    Tensor(std::vector<uint32_t> shape, bool requires_grad);
    Tensor(std::vector<uint32_t> &shape, float *data);
    ~Tensor();

    void zero();
    void ones();
    void random();

    void dealloc();

    Tensor operator+(Tensor &other);
    Tensor &operator+=(Tensor &other);
    Tensor operator*(Tensor &other);
    Tensor operator/(Tensor &other);
    // std::ostream& operator<<(std::ostream &stream);
    Tensor operator[](uint32_t index);
    Tensor operator[](uint32_t index) const;

    // Autograd utilities
    Tensor transpose();
    Tensor sum_to_shape(std::vector<uint32_t> &target_shape);

    // Autograd methods
    void backward(Tensor &grad_output);
    void backward(); // For scalar tensors, starts with ones gradient
    bool is_leaf();  // True if tensor was created by user (not by operation)
};

Tensor matmul(Tensor &a, Tensor &b);
