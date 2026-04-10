#include "../modules/module.hpp"

class Optimizer {
  public:
    Optimizer(double learning_rate) : learning_rate(learning_rate) {}

    virtual void step(Module &module) = 0;

    void zero_grad(Module &module) {
        for (auto &param : module.parameters()) {
            param.second->zero_grad();
        }
    }

    double learning_rate;
};
