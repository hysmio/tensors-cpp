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
        Tensor lhs_grad(lhs_ptr->shape, false);
        lhs_grad.zero();
    }

    if (rhs_ptr && rhs_ptr->requires_grad) {
        Tensor rhs_grad(rhs_ptr->shape, false);
        rhs_grad.zero();
    }
}
