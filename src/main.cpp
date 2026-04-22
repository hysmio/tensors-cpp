#include "backend/cuda/cuda_backend.cuh"
#include "flags.hpp"
#include "linalg.hpp"
#include "modules/linear.hpp"
#include "optimizer/sgd.hpp"
#include "tensor.hpp"
#include <chrono>
#include <cmath>
#include <nvtx3/nvToolsExt.h>

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
    flags::Parser parser({
        {"device", 'd', flags::Type::String, std::string("cpu"), {}, {}, "Device (cpu or cuda)"},
        {"iterations", 'i', flags::Type::Int, 2, 1, {}, "Number of training iterations"},
        {"print", 'p', flags::Type::Int, 1, 1, {}, "Print every N iterations"},
        {"model", 'm', flags::Type::Int, 768, 1, {}, "Model dimension (number of hidden units)"},
    });

    if (!parser.parse(argc, argv)) {
        return 1;
    }

    std::string device_str = parser.get_string("device");
    const int n_iterations = parser.get_int("iterations");
    const int print_every = parser.get_int("print");
    const int d_model = parser.get_int("model");

    Device device;
    if (device_str == "cuda") {
        device = Device::CUDA;
        std::cout << "Using CUDA" << std::endl;
    } else if (device_str == "cpu") {
        device = Device::CPU;
        std::cout << "Using CPU" << std::endl;
    } else {
        std::cerr << "Invalid device: " << device_str << std::endl;
        return 1;
    }

    const int size = 1000;

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

    auto xCuda = x.to(device);
    auto yCuda = y.to(device);

    Linear lin(1, d_model, false, device);
    Linear lin2(d_model, d_model, false, device);
    Linear lin3(d_model, 1, false, device);

    SGD optimizer(0.00001, 2.f);

    auto start = std::chrono::high_resolution_clock::now();
    auto true_start = start;
    int lowered = 0;

    for (int i = 0; i < n_iterations; i++) {

        if (i == 5) {
            auto end = std::chrono::high_resolution_clock::now();
            auto micro = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            cout << "Warmup iterations completed in " << micro.count() / 1000 << "ms\n";
            start = end;
            true_start = end;
        }

        nvtxRangePush("zero");
        optimizer.zero_grad(lin);
        optimizer.zero_grad(lin2);
        optimizer.zero_grad(lin3);
        nvtxRangePop();

        nvtxRangePush("forward");

        Tensor h = lin.forward(xCuda);
        h.tanh();
        Tensor y_hat = lin2.forward(h);
        y_hat.tanh();
        y_hat = lin3.forward(y_hat);

        Tensor loss = mse(y_hat, yCuda);
        nvtxRangePop();
        
        nvtxRangePush("backward");
        loss.backward();
        nvtxRangePop();
        if ((i + 1) % print_every == 0 || i == n_iterations - 1) {
            auto end = std::chrono::high_resolution_clock::now();
            auto micro = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            auto total_micro =
                std::chrono::duration_cast<std::chrono::microseconds>(end - true_start);
            auto loss_cpu = loss.to(Device::CPU);
            cout << "Iteration " << i + 1 << ": loss = " << loss_cpu.data()[0]
                 << " time = " << micro.count() / 1000
                 << "ms | avg = " << (float)micro.count() / 1000.0 / (float)print_every
                 << "ms | global_avg = " << (float)total_micro.count() / 1000.0 / (float)(i + 1)
                 << "ms\n";
            start = end;
            if (loss_cpu.data()[0] < 1e-3 && lowered == 0) {
                lowered = 1;
                optimizer.learning_rate /= 2;
                cout << "LR decreased to " << optimizer.learning_rate
                     << "! Loss < 1e-3 at iteration " << i + 1 << '\n';
            } else if (loss_cpu.data()[0] < 1e-8) {
                cout << "Loss < 1e-8 at iteration " << i + 1 << "! Training completed!" << '\n';
                break;
            }
        }

        nvtxRangePush("optimizer step");
        optimizer.step({&lin, &lin2, &lin3});
        nvtxRangePop();
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
