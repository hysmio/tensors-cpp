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
    virtual void backward(Tensor &grad_output) = 0;

    // Edge information for connecting gradients to the right inputs
    struct Edge {
        std::shared_ptr<GradNode> function;
        uint32_t input_nr; // Which input this edge corresponds to
    };
    std::vector<Edge> next_edges;
};

// Specific gradient functions for different operations
struct AddBackward : public GradNode {
    std::vector<uint32_t> lhs_shape, rhs_shape;
    bool lhs_needs_grad, rhs_needs_grad;
    Tensor *lhs_ptr;
    Tensor *rhs_ptr;

    AddBackward(std::vector<uint32_t> &lhs_shape, std::vector<uint32_t> &rhs_shape,
                bool lhs_needs_grad, bool rhs_needs_grad, Tensor *lhs_ptr, Tensor *rhs_ptr)
        : lhs_shape(lhs_shape), rhs_shape(rhs_shape), lhs_needs_grad(lhs_needs_grad),
          rhs_needs_grad(rhs_needs_grad), lhs_ptr(lhs_ptr), rhs_ptr(rhs_ptr) {}

    void backward(Tensor &grad_output) override;
};

struct MulBackward : public GradNode {
    // Store pointers to original tensors for gradient accumulation
    Tensor *lhs_ptr;
    Tensor *rhs_ptr;

    MulBackward(Tensor *lhs, Tensor *rhs);

    void backward(Tensor &grad_output) override;
};

struct MatmulBackward : public GradNode {
    // Store pointers to original leaf tensors only
    Tensor *lhs_ptr;
    Tensor *rhs_ptr;

    MatmulBackward(Tensor *lhs, Tensor *rhs);

    void backward(Tensor &grad_output) override;
};
