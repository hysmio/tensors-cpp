#include "backend/cuda/cuda_backend.cuh"
#include "linalg.hpp"
#include "modules/linear.hpp"
#include "optimizer/sgd.hpp"
#include "tensor.hpp"
#include <chrono>
#include <cmath>
#include <cstring>

using namespace std;
using namespace chrono;

#include <sys/resource.h>

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

int main(int argc, char *argv[]) {
    const int size = 100;
    const float PI = 3.14159265358979f;

    std::cout << "Starting" << std::endl;
    Tensor x = Tensor::linspace(-1, 1, 100, Device::CPU);
    std::cout << "Created linspace tensor: " << x << std::endl;
    x.shape = {size, 1};
    x.strides = {1, 1};
    Tensor y({size, 1}, false, Device::CPU);

    for (int i = 0; i < size; i++) {
        y.data()[i] = std::sin(x.data()[i]);
    }

    std::cout << "Created y " << y << std::endl;

    Device device = Device::CPU;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--device") == 0) {
            if (strcmp(argv[i + 1], "cuda") == 0) {
                device = Device::CUDA;
            } else if (strcmp(argv[i + 1], "cpu") == 0) {
                device = Device::CPU;
            } else {
                std::cerr << "Invalid device: " << argv[i + 1] << std::endl;
                return 1;
            }
            i++;
        }
    }

    switch (device) {
    case Device::CUDA:
        std::cout << "Using CUDA" << std::endl;
        break;
    case Device::CPU:
        std::cout << "Using CPU" << std::endl;
        break;
    }

    auto xCuda = x.to(device);
    auto yCuda = y.to(device);

    int d_model = 256;
    Linear lin(1, d_model, false, device);
    Linear lin2(d_model, d_model, false, device);
    Linear lin3(d_model, 1, false, device);

    SGD optimizer(0.00001, 2.f);

    const int n_iterations = 750000;
    const int print_every = 5000;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iterations; i++) {
        optimizer.zero_grad(lin);
        // cout << "lin grad: " << lin.weights << '\n';

        optimizer.zero_grad(lin2);
        // cout << "lin2 grad: " << lin2.weights << '\n';

        Tensor h = lin.forward(xCuda);
        h = tanh(h);
        Tensor y_hat = lin2.forward(h);
        y_hat = tanh(y_hat);
        y_hat = lin3.forward(y_hat);

        Tensor loss = mse(y_hat, yCuda);
        loss.backward();

        if (i % print_every == 0 || i == n_iterations - 1) {
            auto loss_cpu = loss.to(Device::CPU);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            cout << "Iteration " << i << ": loss = " << loss_cpu.data()[0]
                 << " time = " << duration.count() << "ms\n";
            start = end;
        }

        optimizer.step({&lin, &lin2, &lin3});
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
