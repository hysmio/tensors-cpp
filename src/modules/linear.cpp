#include "./linear.hpp"
#include "../autograd/grad_node.hpp"
#include "../tensor.hpp"

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

Tensor *Linear::forward(Tensor *x) {
    if (x->shape.size() == 2)
        assert(x->shape[1] == this->in_features);

    std::vector<uint32_t> result_shape = {x->shape[0], this->out_features};
    Tensor *y = new Tensor(result_shape, x->requires_grad || this->weights.requires_grad);

    // x @ weights.T
    y = matmul(x, this->weights.transpose());

    // Set up gradient function with original tensors
    if (x->requires_grad || this->weights.requires_grad) {
        y->grad_fn = std::make_shared<LinearBackward>(x, &this->weights);
    }

    // TODO: cbf dealing with this atm, not necessary for what I'm trying to do
    // if (this->biases) {
    //     (*y) += this->biases.value();
    // }
    return y;
}
