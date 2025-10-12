#include "tensor.hpp"
#include "modules/linear.hpp"

static std::ostream &printTensor(std::ostream &stream, const Tensor &tensor,
                                 const std::string &prefix = "") {
    stream << prefix << "[";
    if (tensor.shape.size() == 1) {
        uint32_t const len = tensor.shape[0];
        for (uint32_t i = 0; i < len; i++) {
            stream << tensor.data[i];
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

static std::ostream &operator<<(std::ostream &stream, const Tensor &tensor) {

    stream << "Tensor(";
    printTensor(stream, tensor);
    stream << ")";

    return stream;
}

int main() {
    Tensor x({5, 2}, false);
    x.random();

    for (uint32_t i = 0; i < x.shape[0]; i++) {
        for (uint32_t j = 0; j < x.shape[1]; j++) {
            x.data[i * x.shape[1] + j] = i + j;
        }
    }

    Linear lin(2, 2, false);
    Linear lin2(2, 1, false);

    Tensor y = lin.forward(x);
    Tensor y2 = lin2.forward(y);

    std::cout << "lin.weights: " << lin.weights << '\n';
    std::cout << "lin2.weights: " << lin.weights << '\n';
    std::cout << "y: " << y << '\n';
    std::cout << "y2: " << y2 << '\n';

    y2.backward();

    std::cout << "lin.weights.grad: " << lin.weights.grad << '\n';
    std::cout << "lin2.weights.grad: " << lin2.weights.grad << '\n';

    return 0;
}
