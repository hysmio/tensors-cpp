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
    virtual void backward(Tensor *grad_output) = 0;

    // Edge information for connecting gradients to the right inputs
    struct Edge {
        std::shared_ptr<GradNode> function;
        uint32_t input_nr; // Which input this edge corresponds to
    };
    std::vector<Edge> next_edges;
};

// Specific gradient functions for different operations
struct AddBackward : public GradNode {
    Tensor *lhs_ptr;
    Tensor *rhs_ptr;

    AddBackward(Tensor *lhs_ptr, Tensor *rhs_ptr) : lhs_ptr(lhs_ptr), rhs_ptr(rhs_ptr) {}

    void backward(Tensor *grad_output) override;
};

// Specific gradient functions for different operations
struct SubBackward : public GradNode {
    Tensor *lhs_ptr;
    Tensor *rhs_ptr;

    SubBackward(Tensor *lhs_ptr, Tensor *rhs_ptr) : lhs_ptr(lhs_ptr), rhs_ptr(rhs_ptr) {}

    void backward(Tensor *grad_output) override;
};

struct MulBackward : public GradNode {
    // Store pointers to original tensors for gradient accumulation
    Tensor *lhs_ptr;
    Tensor *rhs_ptr;

    MulBackward(Tensor *lhs, Tensor *rhs);

    void backward(Tensor *grad_output) override;
};

struct MatmulBackward : public GradNode {
    // Store pointers to original leaf tensors only
    Tensor *lhs_ptr;
    Tensor *rhs_ptr;

    MatmulBackward(Tensor *lhs, Tensor *rhs);

    void backward(Tensor *grad_output) override;
};

struct LinearBackward : public GradNode {
    // Store pointers to input and weights (not transposed weights)
    Tensor *input_ptr;
    Tensor *weights_ptr;

    LinearBackward(Tensor *input, Tensor *weights);

    void backward(Tensor *grad_output) override;
};
