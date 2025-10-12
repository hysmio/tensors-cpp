#pragma once

#include "../pch.hpp"
#include "../tensor.hpp"

class Module {
public:
    virtual Tensor forward(Tensor x) = 0;
};
