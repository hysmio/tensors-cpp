#pragma once

#include "pch.hpp"
#include "tensor_data.hpp"
#include <memory>

// Forward declarations
struct GradNode;

class Tensor {
    // Ref-counted storage - shared between tensor and its views
    std::shared_ptr<TensorData> storage;
    size_t offset; // Offset into storage where this tensor's data starts

  public:
    uint32_t size;
    std::vector<uint32_t> shape;
    std::vector<uint32_t> strides;
    bool requires_grad;

    // Autograd support
    std::shared_ptr<GradNode> grad_fn;
    // this feels super cursed, but don't think there's an elegant way to ensure it survives copies
    std::shared_ptr<std::shared_ptr<Tensor>> grad;

    // Factory methods
    static Tensor linspace(float start, float end, uint32_t num_points);
    static Tensor zeros(std::vector<uint32_t> shape, bool requires_grad = false);
    static Tensor ones_like(const Tensor &other);

    // Constructors
    Tensor(std::vector<uint32_t> shape, bool requires_grad = false);
    Tensor(const Tensor &other);                    // Deep copy
    Tensor(Tensor &&other) noexcept = default;      // Move constructor
    Tensor &operator=(const Tensor &other);         // Copy assignment
    Tensor &operator=(Tensor &&other) noexcept = default; // Move assignment
    ~Tensor() = default;                            // shared_ptr handles cleanup

    // view of tensor data
    Tensor(std::shared_ptr<TensorData> storage, size_t offset,
           std::vector<uint32_t> shape, std::vector<uint32_t> strides,
           bool requires_grad = false);

    float *data();
    const float *data() const;

    // Element access (handles strides correctly)
    float &at(const std::vector<uint32_t> &indices);
    float at(const std::vector<uint32_t> &indices) const;

    // Check if this tensor is a contiguous view
    bool is_contiguous() const;

    void zero();
    void ones();
    void random();
    void xavier_uniform(uint32_t fan_in, uint32_t fan_out);

    Tensor &operator+=(const Tensor &other);
    Tensor &operator-=(const Tensor &other);

    Tensor operator+(Tensor &other);
    Tensor operator-(Tensor &other);
    Tensor operator*(Tensor &other);
    Tensor operator/(Tensor &other);
    Tensor operator+(float other);
    Tensor operator-(float other);
    Tensor operator*(float other);
    Tensor operator/(float other);
    Tensor operator[](uint32_t index);
    Tensor operator[](uint32_t index) const;

    Tensor matmul(Tensor &other);

    Tensor transpose();
    Tensor sum_to_shape(std::vector<uint32_t> &target_shape);

    Tensor sum();
    Tensor mean();

    // Autograd methods
    void backward(Tensor &grad_output);
    void backward();
    bool is_leaf();
};

Tensor matmul(Tensor &a, Tensor &b);
