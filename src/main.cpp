#include "linalg.hpp"
#include "modules/linear.hpp"
#include "tensor.hpp"

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
        printTensor(stream, *tensor[i], prefix + "  ");
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
    auto a = Tensor::linspace(0.0f, 10.0f, 100);
    std::cout << *a << '\n';
    auto b = *((*sin(a)) * cos(a)) * 2.0f;
    std::cout << *b << '\n';
    // std::cout << "b.shape: " << b.shape << '\n';

    Tensor *x = new Tensor({100, 2}, true);
    x->random();

    Linear lin(2, 2, false);
    Linear lin2(2, 2, false);

    Tensor *y = lin.forward(x);

    Tensor *y2 = lin2.forward(y);

    // std::cout << "x: " << x << '\n';
    std::cout << "lin.weights: " << lin.weights << '\n';
    std::cout << "lin2.weights: " << lin2.weights << '\n';
    std::cout << "y: " << *y << '\n';
    std::cout << "y2: " << *y2 << '\n';

    Tensor *loss = mse(y2, y);
    loss->backward();

    std::cout << "lin.weights.grad: " << *lin.weights.grad << '\n';
    std::cout << "lin2.weights.grad: " << *lin2.weights.grad << '\n';

    return 0;
}
