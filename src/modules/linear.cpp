#include "./linear.hpp"

Linear::Linear(uint32_t in_features, uint32_t out_features, bool bias) : in_features(in_features), out_features(out_features), weights({out_features, in_features}, true), biases(bias ? std::make_optional(Tensor({1, out_features}, true)) : std::nullopt) {
    assert(in_features > 0 && out_features > 0);
    this->weights.random();
}

Linear::~Linear() {
    // this->weights.free();
}

Tensor Linear::forward(Tensor x) {
    if (x.shape.size() == 2)
        assert(x.shape[1] == this->in_features);
    Tensor y = x * this->weights;
    if (this->biases) {
        y += this->biases.value();
    }
    return y;
}
