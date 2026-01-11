#include "../modules/module.hpp"

class Optimizer {
  public:
    Optimizer(double learning_rate) : learning_rate(learning_rate) {}

    virtual void step(Module &module) = 0;

    void zero_grad(Module &module) {
        for (auto &param : module.parameters()) {
            if (*param.second->grad) {
                std::fill_n((*param.second->grad)->data(), (*param.second->grad)->size, 0.0f);
            }
        }
    }

    double learning_rate;
};
