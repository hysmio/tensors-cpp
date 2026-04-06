#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "device.hpp"

class TensorData {
  public:
    float *data;
    Device device;
    size_t size;

    explicit TensorData(size_t size, Device device);
    explicit TensorData(std::vector<float> d, Device device);
    TensorData(const float *ptr, size_t size, Device device);
    ~TensorData();

    float *ptr() { return data; }
    const float *ptr() const { return data; }
};

uint32_t compute_size(const std::vector<uint32_t> &shape);

std::vector<uint32_t> compute_strides(const std::vector<uint32_t> &shape);

size_t compute_linear_index(const std::vector<uint32_t> &indices,
                            const std::vector<uint32_t> &strides, size_t offset);
