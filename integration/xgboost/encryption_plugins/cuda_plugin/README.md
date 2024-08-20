# CUDA plugin

This plugin use CUDA to do paillier encryption and addition.

# Build Instruction

## Install required dependencies
We need libgmp-dev, CUDA runtime >= 12.1, CUDA driver >= 12.1, NVIDIA GPU Driver >= 535
Compute compatibility >= 7.0, please refer to: https://developer.nvidia.com/cuda-gpus

## CMake

```
mkdir build
cd build
cmake ..
make
```
