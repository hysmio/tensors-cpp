#include "sgd.hpp"

void SGD::step(Module &module) {
    float total_norm = 0.0f;
    for (auto &param : module.parameters()) {
        if (!*param.second->grad)
            continue;
        for (uint32_t i = 0; i < (*param.second->grad)->size; i++) {
            total_norm += (*param.second->grad)->data()[i] * (*param.second->grad)->data()[i];
        }
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
