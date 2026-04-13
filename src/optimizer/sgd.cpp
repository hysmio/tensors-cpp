#include "sgd.hpp"
#include "../backend/cuda/cuda_backend.cuh"

void SGD::step(Module &module) {
    std::vector<Module *> modules = {&module};
    step(modules);
}

void SGD::step(std::vector<Module *> modules) {
    // Collect all (param, grad) pairs and detect device
    struct ParamGrad {
        Tensor *param;
        Tensor *grad;
    };
    std::vector<ParamGrad> all_params;
    Device device = Device::CPU;

    for (auto *mod : modules) {
        for (auto &[name, param] : mod->parameters()) {
            if (!*param->grad)
                continue;
            device = param->device;
            all_params.push_back({param.get(), (*param->grad).get()});
        }
    }

    if (all_params.empty())
        return;

    if (device == Device::CUDA) {
        Tensor sq_norm({1}, false, Device::CUDA);
        cudaMemsetAsync(sq_norm.data(), 0, sizeof(float));

        for (auto &pg : all_params) {
            launch_accumulate_sq_norm(pg.grad->data(), sq_norm.data(), pg.grad->size);
        }

        for (auto &pg : all_params) {
            launch_sgd_update(pg.param->data(), pg.grad->data(), sq_norm.data(),
                              (float)this->learning_rate, max_grad_norm, pg.param->size);
        }
    } else {
        float total_norm = 0.0f;
        for (auto &pg : all_params) {
            for (uint32_t i = 0; i < pg.grad->size; i++) {
                float v = pg.grad->data()[i];
                total_norm += v * v;
            }
        }
        total_norm = std::sqrt(total_norm);
        float clip_coef = (total_norm > max_grad_norm) ? (max_grad_norm / total_norm) : 1.0f;

        for (auto &pg : all_params) {
            for (uint32_t i = 0; i < pg.param->size; i++) {
                pg.param->data()[i] -= pg.grad->data()[i] * clip_coef * (float)this->learning_rate;
            }
        }
    }
}
