#include "grad_node.hpp"
#include "../backend/cuda/cuda_backend.cuh"
#include "../tensor.hpp"

// AddBackward implementation
void AddBackward::backward(Tensor &grad_output) {
    if (lhs_ptr->requires_grad) {
        if (!*lhs_ptr->grad) {
            *lhs_ptr->grad = std::make_shared<Tensor>(lhs_ptr->shape, false, lhs_ptr->device);
            (*lhs_ptr->grad)->zero();
        }
        **lhs_ptr->grad += grad_output;
        if (lhs_ptr->grad_fn) {
            lhs_ptr->grad_fn->backward(**lhs_ptr->grad);
        }
    }
    if (rhs_ptr->requires_grad) {
        if (!*rhs_ptr->grad) {
            *rhs_ptr->grad = std::make_shared<Tensor>(rhs_ptr->shape, false, rhs_ptr->device);
            (*rhs_ptr->grad)->zero();
        }
        **rhs_ptr->grad += grad_output;
        if (rhs_ptr->grad_fn) {
            rhs_ptr->grad_fn->backward(**rhs_ptr->grad);
        }
    }
}

// SubBackward implementation
void SubBackward::backward(Tensor &grad_output) {
    if (lhs_ptr->requires_grad) {
        if (!*lhs_ptr->grad) {
            *lhs_ptr->grad = std::make_shared<Tensor>(lhs_ptr->shape, false, lhs_ptr->device);
            (*lhs_ptr->grad)->zero();
        }
        **lhs_ptr->grad += grad_output;
        if (lhs_ptr->grad_fn) {
            lhs_ptr->grad_fn->backward(**lhs_ptr->grad);
        }
    }
    if (rhs_ptr->requires_grad) {
        if (!*rhs_ptr->grad) {
            *rhs_ptr->grad = std::make_shared<Tensor>(rhs_ptr->shape, false, rhs_ptr->device);
            (*rhs_ptr->grad)->zero();
        }

        switch (rhs_ptr->device) {
        case Device::CPU:
            // Negate gradient for rhs: d(a-b)/db = -1, accumulate the negated value
            for (uint32_t i = 0; i < rhs_ptr->size; i++) {
                (*rhs_ptr->grad)->data()[i] -= grad_output.data()[i];
            }
            break;
        case Device::CUDA:
            launch_vec_subtract((*rhs_ptr->grad)->data(), grad_output.data(),
                                (*rhs_ptr->grad)->data(), rhs_ptr->size);
            break;
        }
        if (rhs_ptr->grad_fn) {
            rhs_ptr->grad_fn->backward(**rhs_ptr->grad);
        }
    }
}

// MulBackward implementation
MulBackward::MulBackward(std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs)
    : lhs_ptr(lhs), rhs_ptr(rhs) {}

void MulBackward::backward(Tensor &grad_output) {
    if (lhs_ptr->requires_grad) {
        if (!*lhs_ptr->grad) {
            *lhs_ptr->grad = std::make_shared<Tensor>(lhs_ptr->shape, false, lhs_ptr->device);
            (*lhs_ptr->grad)->zero();
        }
        auto grad = grad_output * (*rhs_ptr);
        **lhs_ptr->grad += grad;
        if (lhs_ptr->grad_fn) {
            lhs_ptr->grad_fn->backward(**lhs_ptr->grad);
        }
    }

    if (rhs_ptr->requires_grad) {
        if (!*rhs_ptr->grad) {
            *rhs_ptr->grad = std::make_shared<Tensor>(rhs_ptr->shape, false, rhs_ptr->device);
            (*rhs_ptr->grad)->zero();
        }
        auto grad = grad_output * (*lhs_ptr);
        **rhs_ptr->grad += grad;
        if (rhs_ptr->grad_fn) {
            rhs_ptr->grad_fn->backward(**rhs_ptr->grad);
        }
    }
}

DivBackward::DivBackward(std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs)
    : lhs_ptr(lhs), rhs_ptr(rhs) {}

void DivBackward::backward(Tensor &grad_output) {
    if (lhs_ptr->requires_grad) {
        if (!*lhs_ptr->grad) {
            *lhs_ptr->grad = std::make_shared<Tensor>(lhs_ptr->shape, false, lhs_ptr->device);
            (*lhs_ptr->grad)->zero();
        }
        auto grad = grad_output * (*rhs_ptr);
        **lhs_ptr->grad += grad;
        if (lhs_ptr->grad_fn) {
            lhs_ptr->grad_fn->backward(**lhs_ptr->grad);
        }
    }

    if (rhs_ptr->requires_grad) {
        if (!*rhs_ptr->grad) {
            *rhs_ptr->grad = std::make_shared<Tensor>(rhs_ptr->shape, false, rhs_ptr->device);
            (*rhs_ptr->grad)->zero();
        }
        auto grad = grad_output * (*lhs_ptr);
        **rhs_ptr->grad += grad;
        if (rhs_ptr->grad_fn) {
            rhs_ptr->grad_fn->backward(**rhs_ptr->grad);
        }
    }
}

// MatmulBackward implementation
MatmulBackward::MatmulBackward(std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs)
    : lhs_ptr(lhs), rhs_ptr(rhs) {}

void MatmulBackward::backward(Tensor &grad_output) {
    if (lhs_ptr->requires_grad) {
        if (!*lhs_ptr->grad) {
            *lhs_ptr->grad = std::make_shared<Tensor>(lhs_ptr->shape, false, lhs_ptr->device);
            (*lhs_ptr->grad)->zero();
        }
        auto rhs_transposed = rhs_ptr->transpose();
        auto grad = matmul(grad_output, rhs_transposed);
        **lhs_ptr->grad += grad;
        if (lhs_ptr->grad_fn) {
            lhs_ptr->grad_fn->backward(**lhs_ptr->grad);
        }
    }

    if (rhs_ptr->requires_grad) {
        if (!*rhs_ptr->grad) {
            *rhs_ptr->grad = std::make_shared<Tensor>(rhs_ptr->shape, false, rhs_ptr->device);
            (*rhs_ptr->grad)->zero();
        }
        auto lhs_transposed = lhs_ptr->transpose();
        auto grad = matmul(lhs_transposed, grad_output);
        **rhs_ptr->grad += grad;
        if (rhs_ptr->grad_fn) {
            rhs_ptr->grad_fn->backward(**rhs_ptr->grad);
        }
    }
}

// LinearBackward implementation
LinearBackward::LinearBackward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weights)
    : input_ptr(input), weights_ptr(weights) {}

void LinearBackward::backward(Tensor &grad_output) {
    // Gradient w.r.t. input: grad_output @ weights
    if (input_ptr->requires_grad) {
        if (!*input_ptr->grad) {
            *input_ptr->grad = std::make_shared<Tensor>(input_ptr->shape, false, input_ptr->device);
            (*input_ptr->grad)->zero();
        }
        auto grad = matmul(grad_output, *weights_ptr);
        **input_ptr->grad += grad;
        if (input_ptr->grad_fn) {
            input_ptr->grad_fn->backward(**input_ptr->grad);
        }
    }

    // Gradient w.r.t. weights: grad_output.T @ input
    if (weights_ptr->requires_grad) {
        if (!*weights_ptr->grad) {
            *weights_ptr->grad =
                std::make_shared<Tensor>(weights_ptr->shape, false, weights_ptr->device);
            (*weights_ptr->grad)->zero();
        }
        auto grad_transposed = grad_output.transpose();
        auto grad = matmul(grad_transposed, *input_ptr);
        **weights_ptr->grad += grad;
        if (weights_ptr->grad_fn) {
            weights_ptr->grad_fn->backward(**weights_ptr->grad);
        }
    }
}

// SumBackward implementation
SumBackward::SumBackward(std::shared_ptr<Tensor> input) : input_ptr(input) {}

void SumBackward::backward(Tensor &grad_output) {
    if (input_ptr->requires_grad) {
        if (!*input_ptr->grad) {
            *input_ptr->grad = std::make_shared<Tensor>(input_ptr->shape, false, input_ptr->device);
            (*input_ptr->grad)->zero();
        }
        switch (input_ptr->device) {
        case Device::CPU: {
            float grad_val = grad_output.data()[0];
            for (uint32_t i = 0; i < input_ptr->size; i++) {
                (*input_ptr->grad)->data()[i] += grad_val;
            }
            break;
        }
        case Device::CUDA:
            launch_scalar_addp((*input_ptr->grad)->data(), grad_output.data(),
                               (*input_ptr->grad)->data(), input_ptr->size);
            break;
        }
        if (input_ptr->grad_fn) {
            input_ptr->grad_fn->backward(**input_ptr->grad);
        }
    }
}

// DivScalarBackward implementation
DivScalarBackward::DivScalarBackward(std::shared_ptr<Tensor> input, float scalar)
    : input_ptr(input), scalar(scalar) {}

void DivScalarBackward::backward(Tensor &grad_output) {
    if (input_ptr->requires_grad) {
        if (!*input_ptr->grad) {
            *input_ptr->grad = std::make_shared<Tensor>(input_ptr->shape, false, input_ptr->device);
            (*input_ptr->grad)->zero();
        }

        Tensor grad_output_scaled = grad_output / scalar;

        for (uint32_t i = 0; i < input_ptr->size; i++) {
            (**input_ptr->grad) += grad_output_scaled;
        }

        if (input_ptr->grad_fn) {
            input_ptr->grad_fn->backward(**input_ptr->grad);
        }
    }
}

ReluBackward::ReluBackward(std::shared_ptr<Tensor> input) : input_ptr(input) {}

void ReluBackward::backward(Tensor &grad_output) {
    constexpr float leak = 0.01f;
    if (input_ptr->requires_grad) {
        if (!*input_ptr->grad) {
            *input_ptr->grad = std::make_shared<Tensor>(input_ptr->shape, false, input_ptr->device);
            (*input_ptr->grad)->zero();
        }
        for (uint32_t i = 0; i < input_ptr->size; i++) {
            // this is just leaky relu, which i think is also identical to gelu idk
            float grad_mult = input_ptr->data()[i] > 0 ? 1.0f : leak;
            (*input_ptr->grad)->data()[i] += grad_output.data()[i] * grad_mult;
        }
        if (input_ptr->grad_fn) {
            input_ptr->grad_fn->backward(**input_ptr->grad);
        }
    }
}

// TanhBackward implementation: d/dx tanh(x) = 1 - tanh(x)^2
TanhBackward::TanhBackward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output)
    : input_ptr(input), output_ptr(output) {}

void TanhBackward::backward(Tensor &grad_output) {
    if (input_ptr->requires_grad) {
        if (!*input_ptr->grad) {
            *input_ptr->grad = std::make_shared<Tensor>(input_ptr->shape, false, input_ptr->device);
            (*input_ptr->grad)->zero();
        }

        auto squared = *output_ptr * *output_ptr;
        auto one_minus_squared = -squared + 1.0f;
        auto grad_mult = grad_output * one_minus_squared;
        *(*input_ptr->grad) += grad_mult;

        if (input_ptr->grad_fn) {
            input_ptr->grad_fn->backward(**input_ptr->grad);
        }
    }
}
