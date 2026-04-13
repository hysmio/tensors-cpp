#include "./optimizer.hpp"
#include <cmath>

class SGD : public Optimizer {
  public:
    float max_grad_norm = 1.0f;

    SGD(double learning_rate, float max_grad_norm = 1.0f)
        : Optimizer(learning_rate), max_grad_norm(max_grad_norm) {}

    void step(Module &module) override;
    void step(std::vector<Module *> modules);
};
