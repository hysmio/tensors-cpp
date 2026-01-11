#include "tensor_data.hpp"

uint32_t compute_size(const std::vector<uint32_t> &shape) {
    uint32_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

std::vector<uint32_t> compute_strides(const std::vector<uint32_t> &shape) {
    std::vector<uint32_t> strides(shape.size());
    if (shape.empty()) {
        return strides;
    }

    uint32_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

size_t compute_linear_index(const std::vector<uint32_t> &indices,
                                   const std::vector<uint32_t> &strides, size_t offset) {
    size_t idx = offset;
    for (size_t i = 0; i < indices.size(); ++i) {
        idx += indices[i] * strides[i];
    }
    return idx;
}