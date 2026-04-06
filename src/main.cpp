#include "linalg.hpp"
#include "modules/linear.hpp"
#include "optimizer/sgd.hpp"
#include "tensor.hpp"
#include <chrono>
#include <cmath>

using namespace std;
using namespace chrono;

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

    std::cout << "Starting" << std::endl;
    Tensor x = Tensor::linspace(-1, 1, 100, Device::CPU);
    std::cout << "Created linspace tensor: " << x << std::endl;
    x.shape.push_back(1);
    Tensor y({size, 1}, false);

    std::cout << "Created y " << y << std::endl;

    for (int i = 0; i < size; i++) {
        y.data()[i] = std::sin(x.data()[i]);
    }

    Device device = Device::CUDA;

    x.to(device);
    y.to(device);

    Linear lin(1, 32, false, device);
    Linear lin2(32, 1, false, device);

    SGD optimizer(0.01, 5.f);

    const int n_iterations = 750000;
    const int print_every = 10000;

    for (int i = 0; i < n_iterations; i++) {
        optimizer.zero_grad(lin);
        optimizer.zero_grad(lin2);

        Tensor h = lin.forward(x);
        h = tanh(h);
        Tensor y_hat = lin2.forward(h);

        Tensor loss = mse(y_hat, y);
        loss.backward();

        if (i % print_every == 0 || i == n_iterations - 1) {
            cout << "Iteration " << i << ": loss = " << loss.data()[0] << '\n';
        }

        optimizer.step(lin);
        optimizer.step(lin2);
    }

    cout << "Training completed!" << '\n';

    // Print sample predictions
    Tensor h = lin.forward(x);
    h = tanh(h);
    Tensor y_pred = lin2.forward(h);

    cout << "\nSample predictions:\n";
    cout << "x\t\tsin(x)\t\tpredicted\n";
    cout << "----------------------------------------\n";
    for (int i : {0, 25, 50, 75, 99}) {
        cout << x.data()[i] << "\t\t" << y.data()[i] << "\t\t" << y_pred.data()[i] << '\n';
    }

    return 0;
}
