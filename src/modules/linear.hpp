#include "../tensor.hpp"
#include "./module.hpp"

class Linear : public Module {

  public:
    uint32_t in_features = 0;
    uint32_t out_features = 0;
    Tensor weights;
    std::optional<Tensor> biases = std::nullopt;
    Linear(uint32_t in_features, uint32_t out_features, bool bias = true);
    ~Linear();
    virtual Tensor forward(Tensor &x) override;
};
