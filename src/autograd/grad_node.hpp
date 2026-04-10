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
    std::shared_ptr<Tensor> lhs_ptr;
    std::shared_ptr<Tensor> rhs_ptr;

    AddBackward(std::shared_ptr<Tensor> lhs_ptr, std::shared_ptr<Tensor> rhs_ptr)
        : lhs_ptr(lhs_ptr), rhs_ptr(rhs_ptr) {}

    void backward(Tensor &grad_output) override;
};

// Specific gradient functions for different operations
struct SubBackward : public GradNode {
    std::shared_ptr<Tensor> lhs_ptr;
    std::shared_ptr<Tensor> rhs_ptr;

    SubBackward(std::shared_ptr<Tensor> lhs_ptr, std::shared_ptr<Tensor> rhs_ptr)
        : lhs_ptr(lhs_ptr), rhs_ptr(rhs_ptr) {}

    void backward(Tensor &grad_output) override;
};

struct MulBackward : public GradNode {
    std::shared_ptr<Tensor> lhs_ptr;
    std::shared_ptr<Tensor> rhs_ptr;

    MulBackward(std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs);

    void backward(Tensor &grad_output) override;
};

struct DivBackward : public GradNode {
    std::shared_ptr<Tensor> lhs_ptr;
    std::shared_ptr<Tensor> rhs_ptr;

    DivBackward(std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs);

    void backward(Tensor &grad_output) override;
};

struct MatmulBackward : public GradNode {
    std::shared_ptr<Tensor> lhs_ptr;
    std::shared_ptr<Tensor> rhs_ptr;

    MatmulBackward(std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs);

    void backward(Tensor &grad_output) override;
};

struct LinearBackward : public GradNode {
    std::shared_ptr<Tensor> input_ptr;
    std::shared_ptr<Tensor> weights_ptr;

    LinearBackward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weights);

    void backward(Tensor &grad_output) override;
};

struct SumBackward : public GradNode {
    std::shared_ptr<Tensor> input_ptr;

    SumBackward(std::shared_ptr<Tensor> input);

    void backward(Tensor &grad_output) override;
};

struct DivScalarBackward : public GradNode {
    std::shared_ptr<Tensor> input_ptr;
    float scalar;

    DivScalarBackward(std::shared_ptr<Tensor> input, float scalar);

    void backward(Tensor &grad_output) override;
};

struct ReluBackward : public GradNode {
    std::shared_ptr<Tensor> input_ptr;

    ReluBackward(std::shared_ptr<Tensor> input);

    void backward(Tensor &grad_output) override;
};

struct TanhBackward : public GradNode {
    std::shared_ptr<Tensor> input_ptr;
    std::shared_ptr<Tensor> output_ptr;

    TanhBackward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output);

    void backward(Tensor &grad_output) override;
};
