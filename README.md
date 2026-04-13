<div align="center">
<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/tensors-cpp-light.png">
  <img alt="tiny corp logo" src="assets/tensors-cpp-dark.png" width="80%" height="80%">
</picture>
</div>

Just a toy recreation of PyTorch (CPU only *for now*) Tensor Library that I used for learning purposes and experimentation.

## Features

- **Basic Tensor Operations**: Core operations including matrix multiplication, element-wise operations, and more
- **Neural Net Modules**: Building blocks for neural networks, including linear layers, activation functions (*in progress*) and more.
- **Basic Automatic Differentiation**: Full backward pass support for gradient computation
- **Educational Focus**: Stripped back and extremely simplistic code that breaks down the fundamental concepts.
- **Supports CUDA**: CUDA support, very rough around the edges and hacked together, but working, on a 3 layer 256 hidden dim model, it averages 2ms per iter on my RTX 5080, compared to ~30ms on my 5950x. Roughly 15x speedup on my specific machine which is great!
- **No external deps**

## Getting Started

### Building the Project

```bash
# Clone the repository
git clone https://github.com/hysmio/tensors-cpp
cd tensors-cpp

# Build the project
make debug # this will automatically try and build with CUDA but it's optional so hopefully it doesn't die on Mac

# Run the executable
./build/bin/llm-cpp --device cuda # now with CUDA or CPU support!
```

#### Notes

Mostly just me & Nvidia docs, some help from Claude specifically diagnosing and profiling issues with CUDA outputs & performance.
