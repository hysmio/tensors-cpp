#include "../tensor.hpp"
#include "./module.hpp"
#include <memory>

class Linear : public Module {

  public:
    uint32_t in_features = 0;
    uint32_t out_features = 0;
    std::shared_ptr<Tensor> weights;
    std::optional<std::shared_ptr<Tensor>> biases = std::nullopt;
    Device device;

    Linear(uint32_t in_features, uint32_t out_features, bool bias = true,
           Device device = Device::CPU);
    ~Linear();
    virtual Tensor forward(Tensor &x) override;
    virtual std::map<std::string, std::shared_ptr<Tensor>> parameters() override;
};
