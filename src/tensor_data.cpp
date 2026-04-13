#include "tensor_data.hpp"
#include "backend/cuda/cuda_backend.cuh"

TensorData::TensorData(size_t size, Device device) : device(device), size(size) {
    if (device == Device::CPU) {
        data = new float[size];
    } else {
        NV_CHECK(cudaMalloc(&data, size * sizeof(float)));
    }
}

TensorData::TensorData(std::vector<float> d, Device device) : device(device), size(d.size()) {
    if (device == Device::CPU) {
        data = new float[d.size()];
        std::copy(d.begin(), d.end(), data);
    } else {
        NV_CHECK(cudaMalloc(&data, d.size() * sizeof(float)));
        NV_CHECK(cudaMemcpy(data, d.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    }
}

TensorData::TensorData(const float *ptr, size_t size, Device device) : device(device), size(size) {
    if (device == Device::CPU) {
        data = new float[size];
        std::copy(ptr, ptr + size, data);
    } else {
        NV_CHECK(cudaMalloc(&data, size * sizeof(float)));
        NV_CHECK(cudaMemcpy(data, ptr, size * sizeof(float), cudaMemcpyHostToDevice));
    }
}

TensorData::~TensorData() {
    if (device == Device::CPU) {
        delete[] data;
    } else {
        cudaFree(data);
    }
}

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
