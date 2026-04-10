#include "sgd.hpp"

void SGD::step(Module &module) {
    float total_norm = 0.0f;
    for (auto &param : module.parameters()) {
        if (!*param.second->grad)
            continue;

        auto squared = **param.second->grad * **param.second->grad;
        auto sum = squared.sum().to(Device::CPU);
        total_norm += sum.data()[0];
    }
    total_norm = std::sqrt(total_norm);

    float clip_coef = (total_norm > max_grad_norm) ? (max_grad_norm / total_norm) : 1.0f;

    for (auto &param : module.parameters()) {
        if (!*param.second->grad) {
            std::cout << "Warning: No gradient for " << param.first << '\n';
            continue;
        }
        Tensor gradient_changes = **param.second->grad * (clip_coef * (float)this->learning_rate);
        *param.second -= gradient_changes;
    }
}
