#pragma once

#include "autograd/grad_node.hpp"
#include "pch.hpp"
#include "tensor.hpp"
#include <math.h>

#include "backend/cuda/cuda_backend.cuh"

// #define CEIL_DIV (x, y) (1 + ((x - 1) / y))

void sgemm(uint32_t m, uint32_t n, uint32_t k, float alpha, float *a, float *b, float beta,
           float *c) {
    /// a = float[m * n] = float[4, 6]
    /// b = float[n * k] = float[6, 5]
    /// c = float[m * k] = float[4, 5]
    ///
    /// loop cidx 0..19
    ///   loop i 0..5
    ///     ~> aidx = cidx / n + i = (0 / 6 + 0) = 0
    ///        bidx = cidx / k + i = (0 / 4 + 0) = 0
    ///     ~> aidx = 0 / 6 + 1 = 1
    ///        bidx = 0 / 4 + 1 = 1
    ///     ...
    ///   ~> cidx = 14 = 3rd row, 5th col
    ///     ~> aidx = 14 % 6 + 0 * 6 = (0 + 2)
    ///     ~> bidx = 14 % 5 + 0 = (4 + 0)
    ///     ...
    ///     ~> aidx = 14 % 6 + 3 * 6 = (2 + 18)

    // go through each row of a
    for (uint32_t cRow = 0; cRow < m; cRow++) {
        for (uint32_t cCol = 0; cCol < k; cCol++) {
            float tmp = 0.0;
            for (uint32_t i = 0; i < n; i++) {
                uint32_t aIdx = (cRow * n) + i; // cRow * aCols = aRow + i = aIdx
                uint32_t bIdx = (i * k) + cCol; // bCols * i = cRow + cCol = bIdx
                float result = a[aIdx] * b[bIdx];
                tmp += result;
            }

            // this allows for a single function to do `ab` & `ab + c` eg. `mx + b`
            c[(cRow * k) + cCol] = (alpha * tmp) + (beta * c[(cRow * k) + cCol]);
        }
    }
}

Tensor sin(Tensor &in) {
    Tensor out(in.shape, in.requires_grad, in.device);
    for (uint32_t i = 0; i < in.size; i++) {
        out.data()[i] = std::sin(in.data()[i]);
    }
    return out;
}

Tensor cos(Tensor &in) {
    Tensor out(in.shape, in.requires_grad, in.device);
    for (uint32_t i = 0; i < in.size; i++) {
        out.data()[i] = std::cos(in.data()[i]);
    }
    return out;
}

Tensor relu(Tensor &in) {
    constexpr float leak = 0.01f; // LeakyReLU
    Tensor out(in.shape, in.requires_grad, in.device);
    for (uint32_t i = 0; i < in.size; i++) {
        out.data()[i] = in.data()[i] > 0 ? in.data()[i] : leak * in.data()[i];
    }
    if (in.requires_grad) {
        out.grad_fn = std::make_shared<ReluBackward>(std::make_shared<Tensor>(in));
    }
    return out;
}

Tensor tanh(Tensor &in) {
    Tensor out(in.shape, in.requires_grad, in.device);
    switch (in.device) {
    case Device::CPU:
        for (uint32_t i = 0; i < in.size; i++) {
            out.data()[i] = std::tanh(in.data()[i]);
        }
        break;
    case Device::CUDA:
        launch_tanh_forward(in.data(), out.data(), in.size);
        break;
    }
    if (in.requires_grad) {
        out.grad_fn = std::make_shared<TanhBackward>(
            std::make_shared<Tensor>(in),
            std::make_shared<Tensor>(out) // Store output for backward
        );
    }
    return out;
}

// (1/n) * sum(y - y_pred)^2
Tensor mse(Tensor &y, Tensor &y_pred) {
    switch (y.device) {
    case Device::CPU: {
        Tensor error = y_pred - y;
        Tensor squared = error * error;
        return squared.mean();
    }
    case Device::CUDA: {
        Tensor error = y_pred - y;
        Tensor squared = error * error;
        return squared.mean();
    }
    }
}
