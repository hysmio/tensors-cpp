#include "./linear.hpp"
#include "../autograd/grad_node.hpp"
#include "../tensor.hpp"

Linear::Linear(uint32_t in_features, uint32_t out_features, bool bias, Device device)
    : in_features(in_features), out_features(out_features),
      weights(
          std::make_shared<Tensor>(std::vector<uint32_t>{out_features, in_features}, true, device)),
      biases(bias ? std::make_optional(std::make_shared<Tensor>(
                        std::vector<uint32_t>{1, out_features}, true, device))
                  : std::nullopt),
      device(device) {
    assert(in_features > 0 && out_features > 0);
    this->weights->xavier_uniform(in_features, out_features);
    if (this->biases) {
        this->biases.value()->zero();
    }
}

Linear::~Linear() {}

Tensor Linear::forward(Tensor &x) {
    if (x.shape.size() == 2)
        assert(x.shape[1] == this->in_features);

    std::vector<uint32_t> result_shape = {x.shape[0], this->out_features};

    // x @ weights.T
    Tensor transposed = this->weights->transpose();
    Tensor y = matmul(x, transposed);

    // Set up gradient function with original tensors
    if (x.requires_grad || this->weights->requires_grad) {
        y.grad_fn = std::make_shared<LinearBackward>(std::make_shared<Tensor>(x), this->weights);
    }

    // TODO: cbf dealing with this atm, not necessary for what I'm trying to do
    // if (this->biases) {
    //     (*y) += this->biases.value();
    // }
    return y;
}

std::map<std::string, std::shared_ptr<Tensor>> Linear::parameters() {
    std::map<std::string, std::shared_ptr<Tensor>> params;
    params["weights"] = this->weights;
    if (this->biases) {
        params["biases"] = this->biases.value();
    }
    return params;
}
