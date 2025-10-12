#pragma once

#include "../pch.hpp"
#include <memory>
#include <vector>

// Forward declaration
class Tensor;

// Base class for all gradient computation nodes
struct GradNode {
    virtual ~GradNode() = default;
    
    // Compute gradients w.r.t. inputs given gradient w.r.t. output
    virtual std::vector<Tensor> backward(const Tensor& grad_output) = 0;
    
    // Edge information for connecting gradients to the right inputs
    struct Edge {
        std::shared_ptr<GradNode> function;
        uint32_t input_nr;  // Which input this edge corresponds to
    };
    std::vector<Edge> next_edges;
};

// Specific gradient functions for different operations
struct AddBackward : public GradNode {
    std::vector<uint32_t> lhs_shape, rhs_shape;
    bool lhs_needs_grad, rhs_needs_grad;
    
    AddBackward(const std::vector<uint32_t>& lhs_shape, 
                const std::vector<uint32_t>& rhs_shape,
                bool lhs_needs_grad, bool rhs_needs_grad)
        : lhs_shape(lhs_shape), rhs_shape(rhs_shape), 
          lhs_needs_grad(lhs_needs_grad), rhs_needs_grad(rhs_needs_grad) {}
    
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

struct MulBackward : public GradNode {
    std::vector<uint32_t> lhs_shape, rhs_shape;
    bool lhs_needs_grad, rhs_needs_grad;
    // Store original tensor values for multiplication gradients
    std::vector<float> lhs_data, rhs_data;
    
    MulBackward(const Tensor& lhs, const Tensor& rhs);
    
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

struct MatmulBackward : public GradNode {
    std::vector<uint32_t> lhs_shape, rhs_shape;
    bool lhs_needs_grad, rhs_needs_grad;
    // For matmul, we need the original tensors to compute gradients
    std::vector<float> lhs_data, rhs_data;
    
    MatmulBackward(const Tensor& lhs, const Tensor& rhs);
    
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};