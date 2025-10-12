#include "grad_node.hpp"
#include "../tensor.hpp"

// AddBackward implementation
void AddBackward::backward(Tensor &grad_output) {
    if (lhs_needs_grad) {
        lhs_ptr->grad = new Tensor(lhs_ptr->shape, false);
        lhs_ptr->grad->zero();
    }
    if (rhs_needs_grad) {
        rhs_ptr->grad = new Tensor(rhs_ptr->shape, false);
        rhs_ptr->grad->zero();
    }
}

// MulBackward implementation
MulBackward::MulBackward(Tensor *lhs, Tensor *rhs) : lhs_ptr(lhs), rhs_ptr(rhs) {}

void MulBackward::backward(Tensor &grad_output) {
    if (lhs_ptr && lhs_ptr->requires_grad) {
        lhs_ptr->grad = new Tensor(lhs_ptr->shape, false);
        lhs_ptr->grad->zero();
        auto grad = grad_output * (*rhs_ptr);
        *(lhs_ptr->grad) += grad;
        if (lhs_ptr->grad_fn) {
            lhs_ptr->grad_fn->backward(grad_output);
        }
    }

    if (rhs_ptr && rhs_ptr->requires_grad) {
        rhs_ptr->grad = new Tensor(rhs_ptr->shape, false);
        rhs_ptr->grad->zero();
        auto grad = grad_output * (*lhs_ptr);
        *(rhs_ptr->grad) += grad;
        if (rhs_ptr->grad_fn) {
            rhs_ptr->grad_fn->backward(grad_output);
        }
    }
}

// MatmulBackward implementation
MatmulBackward::MatmulBackward(Tensor *lhs, Tensor *rhs) : lhs_ptr(lhs), rhs_ptr(rhs) {}

void MatmulBackward::backward(Tensor &grad_output) {
    if (lhs_ptr && lhs_ptr->requires_grad) {
        lhs_ptr->grad = new Tensor(lhs_ptr->shape, false);
        lhs_ptr->grad->zero();
        auto rhs_transposed = rhs_ptr->transpose();
        auto grad = matmul(grad_output, rhs_transposed);
        *(lhs_ptr->grad) += grad;
        if (lhs_ptr->grad_fn) {
            lhs_ptr->grad_fn->backward(*lhs_ptr->grad);
        }
    }

    if (rhs_ptr && rhs_ptr->requires_grad) {
        rhs_ptr->grad = new Tensor(rhs_ptr->shape, false);
        rhs_ptr->grad->zero();
        auto lhs_transposed = lhs_ptr->transpose();
        auto grad = matmul(lhs_transposed, grad_output);
        *(rhs_ptr->grad) += grad;
        if (rhs_ptr->grad_fn) {
            rhs_ptr->grad_fn->backward(*rhs_ptr->grad);
        }
    }
}

// LinearBackward implementation
LinearBackward::LinearBackward(Tensor *input, Tensor *weights) 
    : input_ptr(input), weights_ptr(weights) {}

void LinearBackward::backward(Tensor &grad_output) {
    // Gradient w.r.t. input: grad_output @ weights
    if (input_ptr && input_ptr->requires_grad) {
        if (!input_ptr->grad) {
            input_ptr->grad = new Tensor(input_ptr->shape, false);
            input_ptr->grad->zero();
        }
        auto grad = matmul(grad_output, *weights_ptr);
        *(input_ptr->grad) += grad;
        if (input_ptr->grad_fn) {
            input_ptr->grad_fn->backward(*input_ptr->grad);
        }
    }

    // Gradient w.r.t. weights: grad_output.T @ input
    if (weights_ptr && weights_ptr->requires_grad) {
        if (!weights_ptr->grad) {
            weights_ptr->grad = new Tensor(weights_ptr->shape, false);
            weights_ptr->grad->zero();
        }
        auto grad_transposed = grad_output.transpose();
        auto grad = matmul(grad_transposed, *input_ptr);
        *(weights_ptr->grad) += grad;
        if (weights_ptr->grad_fn) {
            weights_ptr->grad_fn->backward(*weights_ptr->grad);
        }
    }
}
