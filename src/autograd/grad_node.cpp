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
            lhs_ptr->grad_fn->backward(grad_output);
        }
    }
    if (rhs_ptr->requires_grad) {
        if (!*rhs_ptr->grad) {
            *rhs_ptr->grad = std::make_shared<Tensor>(rhs_ptr->shape, false, rhs_ptr->device);
            (*rhs_ptr->grad)->zero();
        }
        **rhs_ptr->grad += grad_output;
        if (rhs_ptr->grad_fn) {
            rhs_ptr->grad_fn->backward(grad_output);
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
            lhs_ptr->grad_fn->backward(grad_output);
        }
    }
    if (rhs_ptr->requires_grad) {
        if (!*rhs_ptr->grad) {
            *rhs_ptr->grad = std::make_shared<Tensor>(rhs_ptr->shape, false, rhs_ptr->device);
            (*rhs_ptr->grad)->zero();
        }

        // Negate gradient for rhs: d(a-b)/db = -1
        Tensor neg_grad(grad_output.shape, false, rhs_ptr->device);
        switch (rhs_ptr->device) {
        case Device::CPU:
            for (uint32_t i = 0; i < rhs_ptr->size; i++) {
                neg_grad.data()[i] = -grad_output.data()[i];
            }
            break;
        case Device::CUDA:
            launch_negate(grad_output.data(), neg_grad.data(), rhs_ptr->size);
            break;
        }
        **rhs_ptr->grad += neg_grad;
        if (rhs_ptr->grad_fn) {
            rhs_ptr->grad_fn->backward(neg_grad);
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
            lhs_ptr->grad_fn->backward(grad);
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
            rhs_ptr->grad_fn->backward(grad);
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
            lhs_ptr->grad_fn->backward(grad);
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
            rhs_ptr->grad_fn->backward(grad);
        }
    }
}

// MatmulBackward implementation
MatmulBackward::MatmulBackward(std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs,
                               bool transpose_a, bool transpose_b)
    : lhs_ptr(lhs), rhs_ptr(rhs), transpose_a(transpose_a), transpose_b(transpose_b) {}

void MatmulBackward::backward(Tensor &grad_output) {
    if (lhs_ptr->requires_grad) {
        if (!*lhs_ptr->grad) {
            *lhs_ptr->grad = std::make_shared<Tensor>(lhs_ptr->shape, false, lhs_ptr->device);
            (*lhs_ptr->grad)->zero();
        }
        Tensor grad(lhs_ptr->shape, false, lhs_ptr->device);
        if (!transpose_a && !transpose_b) {
            // C = A @ B -> grad_A = grad @ B^T
            grad = matmul(grad_output, *rhs_ptr, false, true);
        } else if (!transpose_a && transpose_b) {
            // C = A @ B^T -> grad_A = grad @ B
            grad = matmul(grad_output, *rhs_ptr, false, false);
        } else if (transpose_a && !transpose_b) {
            // C = A^T @ B -> grad_A = B @ grad^T
            grad = matmul(*rhs_ptr, grad_output, false, true);
        } else {
            // C = A^T @ B^T -> grad_A = B^T @ grad^T = (grad @ B)^T
            grad = matmul(*rhs_ptr, grad_output, true, true);
        }
        **lhs_ptr->grad += grad;
        if (lhs_ptr->grad_fn) {
            lhs_ptr->grad_fn->backward(grad);
        }
    }

    if (rhs_ptr->requires_grad) {
        if (!*rhs_ptr->grad) {
            *rhs_ptr->grad = std::make_shared<Tensor>(rhs_ptr->shape, false, rhs_ptr->device);
            (*rhs_ptr->grad)->zero();
        }
        Tensor grad(rhs_ptr->shape, false, rhs_ptr->device);
        if (!transpose_a && !transpose_b) {
            // C = A @ B -> grad_B = A^T @ grad
            grad = matmul(*lhs_ptr, grad_output, true, false);
        } else if (!transpose_a && transpose_b) {
            // C = A @ B^T -> grad_B = grad^T @ A
            grad = matmul(grad_output, *lhs_ptr, true, false);
        } else if (transpose_a && !transpose_b) {
            // C = A^T @ B -> grad_B = A @ grad
            grad = matmul(*lhs_ptr, grad_output, false, false);
        } else {
            // C = A^T @ B^T -> grad_B = grad^T @ A^T = (A @ grad)^T
            grad = matmul(grad_output, *lhs_ptr, true, true);
        }
        **rhs_ptr->grad += grad;
        if (rhs_ptr->grad_fn) {
            rhs_ptr->grad_fn->backward(grad);
        }
    }
}

// LinearBackward implementation
LinearBackward::LinearBackward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weights)
    : input_ptr(input), weights_ptr(weights) {}

void LinearBackward::backward(Tensor &grad_output) {
    // forward is y = x @ W^T (W stored as [out, in], so shape inference triggers transpose_b)
    // grad_x = grad_out @ W
    if (input_ptr->requires_grad) {
        if (!*input_ptr->grad) {
            *input_ptr->grad = std::make_shared<Tensor>(input_ptr->shape, false, input_ptr->device);
            (*input_ptr->grad)->zero();
        }
        // grad_out[batch,out] @ W[out,in] = grad_x[batch,in]
        auto grad = matmul(grad_output, *weights_ptr, false, false);
        **input_ptr->grad += grad;
        if (input_ptr->grad_fn) {
            input_ptr->grad_fn->backward(grad);
        }
    }

    // grad_W = grad_out^T @ x
    if (weights_ptr->requires_grad) {
        if (!*weights_ptr->grad) {
            *weights_ptr->grad =
                std::make_shared<Tensor>(weights_ptr->shape, false, weights_ptr->device);
            (*weights_ptr->grad)->zero();
        }
        // grad_out^T[out,batch] @ x[batch,in] = grad_W[out,in]
        auto grad = matmul(grad_output, *input_ptr, true, false);
        **weights_ptr->grad += grad;
        if (weights_ptr->grad_fn) {
            weights_ptr->grad_fn->backward(grad);
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
        // Broadcast scalar grad_output to all elements
        Tensor broadcast_grad(input_ptr->shape, false, input_ptr->device);
        switch (input_ptr->device) {
        case Device::CPU: {
            float grad_val = grad_output.data()[0];
            for (uint32_t i = 0; i < input_ptr->size; i++) {
                broadcast_grad.data()[i] = grad_val;
            }
            break;
        }
        case Device::CUDA:
            launch_fill_value(broadcast_grad.data(), 0.0f, input_ptr->size);
            launch_scalar_addp(broadcast_grad.data(), grad_output.data(), broadcast_grad.data(),
                               input_ptr->size);
            break;
        }
        **input_ptr->grad += broadcast_grad;
        if (input_ptr->grad_fn) {
            input_ptr->grad_fn->backward(broadcast_grad);
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
        **input_ptr->grad += grad_output_scaled;

        if (input_ptr->grad_fn) {
            input_ptr->grad_fn->backward(grad_output_scaled);
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
        Tensor local_grad(input_ptr->shape, false, input_ptr->device);
        for (uint32_t i = 0; i < input_ptr->size; i++) {
            float grad_mult = input_ptr->data()[i] > 0 ? 1.0f : leak;
            local_grad.data()[i] = grad_output.data()[i] * grad_mult;
            (*input_ptr->grad)->data()[i] += local_grad.data()[i];
        }
        if (input_ptr->grad_fn) {
            input_ptr->grad_fn->backward(local_grad);
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
        auto local_grad = grad_output * one_minus_squared;
        *(*input_ptr->grad) += local_grad;

        if (input_ptr->grad_fn) {
            input_ptr->grad_fn->backward(local_grad);
        }
    }
}
