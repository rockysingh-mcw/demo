# MXNET-MLIR
 
## Overview
MXNET-MLIR is a project designed to lower operations from a custom **MxNet Dialect to TOSA Dialect** and further to **LLVM IR**, enabling efficient execution on various hardware backends. It follows a structured transformation pipeline, converting **high-level tensor operations into optimized intermediate representations**. By leveraging MLIR’s extensibility, MXNET-MLIR ensures **portability, performance optimization, and compatibility** with diverse target architectures.
 
## Supported Operations
Currently, the following operations are supported for lowering in the MxNet dialect:
 
- **Basic Operations:**
  - `AddOp`: Element-wise addition of two tensors.
  - `AbsOp`: Element-wise Absolute of elements inside a tensor.
  - `RsqrtOp`: Computes element-wise reciprocal square root of each element inside a tensor.
 
- **Reduction Operations:**
  - `ReduceProdOp`: Computes the product along specified axes.
 
- ## **Specialized Operations:**
  - `Linspace`: generates a sequence of evenly spaced values between a specified start and end over a defined number of steps.

  ### Linspace operation

    The `linspace` function generates a sequence of evenly spaced numbers over a specified interval. The formula is given by:

    \[
    \text{linspace}(a, b, n) = a + \frac{(b - a)}{n - 1} \cdot i \quad \text{for} \quad i = 0, 1, 2, \dots, n-1
    \]

    Where:
    - \(a\) is the starting value.
    - \(b\) is the ending value.
    - \(n\) is the number of points (or the size of the output array).
    - \(i\) is the index of the points, ranging from \(0\) to \(n-1\).

    This formula ensures that the generated values are evenly spaced between \(a\) and \(b\).

    
    ### Softmax operation

    The `softmax` function is used to normalize a tensor into a probability distribution. The formula for the Softmax function is:

    \[
    \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}} \quad \text{for} \quad i = 1, 2, \dots, n
    \]

    Where:
    - \( z_i \) is the input value at index \( i \).
    - \( n \) is the number of elements in the input tensor.
    - \( e^{z_i} \) is the exponential of the input value at index \( i \).
    - The denominator is the sum of the exponentials of all the input values.

    The Softmax function converts the input values into probabilities by scaling them so that they sum to 1, which is useful for classification tasks.



 
## Project Structure
```
MXNET-MLIR/
├── include/        # Header files and function definitions
├── lib/            # Implementation of core functionality
├── mxnet-opt/      # Tool to run MLIR transformation passes
├── tests/          # MLIR test cases for checking lowering correctness
├── python_tests/   # Python-based tests using Torch to validate correctness
```
 
## Build Instructions

To build the project, follow these steps: 

mkdir build && cd build
cmake -G Ninja .. -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
                 -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
cmake --build . --target mxnet-opt
```
 
## Running Tests
To verify the correctness of the lowering passes, use the following commands:
 
# Running MxNet to LLVM lowering
./build/bin/MxNet-opt --mxnet-to-llvm ./test/Tosa/check.mlir
 
# Remove the existing build directory
rm -rf build
 
# Create and navigate into the build directory
mkdir build && cd build
 
# Run CMake with specified LLVM and MLIR directories
cmake -G Ninja .. -DLLVM_DIR=~/Documents/mlir/llvm-project/build/lib/cmake/llvm \
                 -DMLIR_DIR=~/Documents/mlir/llvm-project/build/lib/cmake/mlir
 
# Build the project
cmake --build . --target MxNet-opt
 
```
