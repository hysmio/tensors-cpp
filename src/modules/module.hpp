#pragma once

#include "../pch.hpp"
#include "../tensor.hpp"
#include <memory>

class Module {
  public:
    virtual Tensor forward(Tensor &x) = 0;
    virtual std::map<std::string, std::shared_ptr<Tensor>> parameters() = 0;
};
