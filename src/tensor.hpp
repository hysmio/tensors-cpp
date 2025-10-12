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
    Tensor* grad;

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
    
    // Autograd utilities
    Tensor transpose() const;
    Tensor sum_to_shape(const std::vector<uint32_t>& target_shape) const;
    
    // Autograd methods
    void backward(const Tensor& grad_output);
    void backward(); // For scalar tensors, starts with ones gradient
    bool is_leaf() const; // True if tensor was created by user (not by operation)
};
