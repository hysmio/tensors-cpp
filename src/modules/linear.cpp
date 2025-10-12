#include "./linear.hpp"
#include "../tensor.hpp"
#include "../autograd/grad_node.hpp"

Linear::Linear(uint32_t in_features, uint32_t out_features, bool bias)
    : in_features(in_features), out_features(out_features),
      weights({out_features, in_features}, true),
      biases(bias ? std::make_optional(Tensor({1, out_features}, true)) : std::nullopt) {
    assert(in_features > 0 && out_features > 0);
    this->weights.random();
}

Linear::~Linear() {
    // this->weights.free();
}

Tensor Linear::forward(Tensor &x) {
    if (x.shape.size() == 2)
        assert(x.shape[1] == this->in_features);

    // Create result tensor
    std::vector<uint32_t> result_shape = {x.shape[0], this->out_features};
    Tensor y(result_shape, x.requires_grad || this->weights.requires_grad);
    
    // Perform the computation: x @ weights.T
    for (uint32_t i = 0; i < x.shape[0]; i++) {
        for (uint32_t j = 0; j < this->out_features; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < this->in_features; k++) {
                sum += x.data[i * x.shape[1] + k] * this->weights.data[j * this->weights.shape[1] + k];
            }
            y.data[i * result_shape[1] + j] = sum;
        }
    }
    
    // Set up gradient function with original tensors
    if (x.requires_grad || this->weights.requires_grad) {
        y.grad_fn = std::make_shared<LinearBackward>(&x, &this->weights);
    }
    
    if (this->biases) {
        y += this->biases.value();
    }
    return y;
}
