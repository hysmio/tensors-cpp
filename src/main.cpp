#include "backend/cuda/cuda_backend.cuh"
#include "linalg.hpp"
#include "modules/linear.hpp"
#include "optimizer/sgd.hpp"
#include "tensor.hpp"
#include <chrono>
#include <cmath>

using namespace std;
using namespace chrono;

#include <sys/resource.h>

void set_memory_limit(long limit_bytes) {
    struct rlimit limit;
    limit.rlim_cur = limit_bytes; // Soft limit
    limit.rlim_max = limit_bytes; // Hard limit

    // RLIMIT_AS: Max size of the process's virtual memory (address space)
    if (setrlimit(RLIMIT_AS, &limit) != 0) {
        perror("setrlimit failed");
    }
}

static ostream &printTensor(ostream &stream, const Tensor &tensor, const string &prefix = "") {
    if (tensor.device == Device::CUDA) {
        stream << prefix << "Tensor(shape=";
        for (uint32_t i = 0; i < tensor.shape.size(); i++) {
            stream << tensor.shape[i];
            if (i < tensor.shape.size() - 1) {
                stream << ", ";
            }
        }
        stream << ", device=CUDA)";
        return stream;
    }
    stream << prefix << "[";
    if (tensor.shape.size() == 1) {
        uint32_t const len = tensor.shape[0];
        for (uint32_t i = 0; i < len; i++) {
            stream << tensor.data()[i];
            if (i < len - 1) {
                stream << ", ";
            }
        }

        stream << "]";

        return stream;
    }
    for (uint32_t i = 0; i < tensor.shape[0]; i++) {
        stream << '\n';
        printTensor(stream, tensor[i], prefix + "  ");
        if (i < tensor.shape[0] - 1) {
            stream << ", ";
        } else {
            stream << '\n';
        }
    }
    stream << prefix << "]";
    return stream;
}

static ostream &operator<<(ostream &stream, const Tensor &tensor) {
    stream << "Tensor(";
    printTensor(stream, tensor);
    stream << ")";
    return stream;
}

int main() {
    const int size = 100;
    const float PI = 3.14159265358979f;
    set_memory_limit(1073741824);

    std::cout << "Starting, 1GB limit" << std::endl;
    Tensor x = Tensor::linspace(-1, 1, 100, Device::CPU);
    std::cout << "Created linspace tensor: " << x << std::endl;
    x.shape.push_back(1);
    Tensor y({size, 1}, false, Device::CPU);

    for (int i = 0; i < size; i++) {
        y.data()[i] = std::sin(x.data()[i]);
    }

    std::cout << "Created y " << y << std::endl;
    Device device = Device::CUDA;

    auto xCuda = x.to(device);
    auto yCuda = y.to(device);

    Linear lin(1, 32, false, device);
    Linear lin2(32, 1, false, device);

    SGD optimizer(0.0001, 5.f);

    const int n_iterations = 750000;
    const int print_every = 10000;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iterations; i++) {
        optimizer.zero_grad(lin);
        // cout << "lin grad: " << lin.weights << '\n';

        optimizer.zero_grad(lin2);
        // cout << "lin2 grad: " << lin2.weights << '\n';

        Tensor h = lin.forward(xCuda);
        cudaDeviceSynchronize();
        // cout << "h: " << h << '\n';
        h = tanh(h);
        cudaDeviceSynchronize();
        // cout << "h after tanh: " << h << '\n';
        Tensor y_hat = lin2.forward(h);
        cudaDeviceSynchronize();
        // cout << "y_hat: " << y_hat << '\n';

        Tensor loss = mse(y_hat, yCuda);
        cudaDeviceSynchronize();
        // cout << "loss: " << loss << '\n';
        loss.backward();
        cudaDeviceSynchronize();

        auto loss_cpu = loss.to(Device::CPU);

        if (i % print_every == 0 || i == n_iterations - 1) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            cout << "Iteration " << i << ": loss = " << loss_cpu.data()[0]
                 << " time = " << duration.count() << "ms\n";
            start = end;
        }

        optimizer.step(lin);
        optimizer.step(lin2);
    }

    cout << "Training completed!" << '\n';

    // Print sample predictions
    Tensor h = lin.forward(xCuda);
    h = tanh(h);
    Tensor y_pred = lin2.forward(h).to(Device::CPU);

    cout << "\nSample predictions:\n";
    cout << "x\t\tsin(x)\t\tpredicted\n";
    cout << "----------------------------------------\n";
    for (int i : {0, 25, 50, 75, 99}) {
        cout << x.data()[i] << "\t\t" << y.data()[i] << "\t\t" << y_pred.data()[i] << '\n';
    }

    return 0;
}
