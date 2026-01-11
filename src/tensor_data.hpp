#pragma once

#include <cstdint>
#include <memory>
#include <vector>

class TensorData {
  public:
    std::vector<float> data;

    explicit TensorData(size_t size) : data(size) {}
    explicit TensorData(std::vector<float> d) : data(std::move(d)) {}
    TensorData(const float *ptr, size_t size) : data(ptr, ptr + size) {}


    float *ptr() { return data.data(); }
    const float *ptr() const { return data.data(); }

    size_t size() const { return data.size(); }
};

uint32_t compute_size(const std::vector<uint32_t> &shape);

std::vector<uint32_t> compute_strides(const std::vector<uint32_t> &shape);

size_t compute_linear_index(const std::vector<uint32_t> &indices, const std::vector<uint32_t> &strides, size_t offset);