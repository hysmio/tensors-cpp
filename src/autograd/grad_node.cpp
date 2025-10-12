#include "grad_node.hpp"
#include "../tensor.hpp"

// AddBackward implementation
std::vector<Tensor> AddBackward::backward(const Tensor& grad_output) {
    std::vector<Tensor> grads;

    if (lhs_needs_grad) {
        // For addition, gradient flows through unchanged, but may need shape adjustment
        Tensor lhs_grad = grad_output.sum_to_shape(lhs_shape);
        grads.push_back(lhs_grad);
    }

    if (rhs_needs_grad) {
        Tensor rhs_grad = grad_output.sum_to_shape(rhs_shape);
        grads.push_back(rhs_grad);
    }

    return grads;
}

// MulBackward implementation
MulBackward::MulBackward(const Tensor& lhs, const Tensor& rhs)
    : lhs_shape(lhs.shape), rhs_shape(rhs.shape),
      lhs_needs_grad(lhs.requires_grad), rhs_needs_grad(rhs.requires_grad) {

    if (rhs_needs_grad) {
        lhs_data.resize(lhs.size);
        std::copy(lhs.data, lhs.data + lhs.size, lhs_data.begin());
    }

    if (lhs_needs_grad) {
        rhs_data.resize(rhs.size);
        std::copy(rhs.data, rhs.data + rhs.size, rhs_data.begin());
    }
}

std::vector<Tensor> MulBackward::backward(const Tensor& grad_output) {
    std::vector<Tensor> grads;

    if (lhs_needs_grad) {
        // d/dlhs = grad_output * rhs
        Tensor rhs_tensor(rhs_shape, const_cast<float*>(&rhs_data[0]));
        Tensor lhs_grad = grad_output * rhs_tensor;
        grads.push_back(lhs_grad.sum_to_shape(lhs_shape));
    }

    if (rhs_needs_grad) {
        // d/drhs = grad_output * lhs
        Tensor lhs_tensor(lhs_shape, const_cast<float*>(&lhs_data[0]));
        Tensor rhs_grad = grad_output * lhs_tensor;
        grads.push_back(rhs_grad.sum_to_shape(rhs_shape));
    }

    return grads;
}

// MatmulBackward implementation
MatmulBackward::MatmulBackward(const Tensor& lhs, const Tensor& rhs)
    : lhs_shape(lhs.shape), rhs_shape(rhs.shape),
      lhs_needs_grad(lhs.requires_grad), rhs_needs_grad(rhs.requires_grad) {

    if (rhs_needs_grad) {
        lhs_data.resize(lhs.size);
        std::copy(lhs.data, lhs.data + lhs.size, lhs_data.begin());
    }

    if (lhs_needs_grad) {
        rhs_data.resize(rhs.size);
        std::copy(rhs.data, rhs.data + rhs.size, rhs_data.begin());
    }
}

std::vector<Tensor> MatmulBackward::backward(const Tensor& grad_output) {
    std::vector<Tensor> grads;

    if (lhs_needs_grad) {
        // d/dlhs = grad_output @ rhs^T
        Tensor rhs_tensor(rhs_shape, const_cast<float*>(&rhs_data[0]));
        Tensor rhs_transposed = rhs_tensor.transpose();
        Tensor lhs_grad = grad_output.matmul(rhs_transposed);
        grads.push_back(lhs_grad);
    }

    if (rhs_needs_grad) {
        // d/drhs = lhs^T @ grad_output
        Tensor lhs_tensor(lhs_shape, const_cast<float*>(&lhs_data[0]));
        Tensor lhs_transposed = lhs_tensor.transpose();
        Tensor rhs_grad = lhs_transposed.matmul(grad_output);
        grads.push_back(rhs_grad);
    }

    return grads;
}
