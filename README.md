
# MXNET-MLIR

## Overview
MXNET-MLIR is a project designed to lower operations from a custom **MxNet Dialect** to **TOSA Dialect** and further to **LLVM IR**, enabling efficient execution on various hardware backends. It follows a structured transformation pipeline, converting **high-level tensor operations into optimized intermediate representations**. By leveraging MLIR’s extensibility, MXNET-MLIR ensures **portability, performance optimization, and compatibility** with diverse target architectures.

### Prerequisites

* [LLVM](https://llvm.org/)
* [MLIR](https://mlir.llvm.org/)
* [CMake](https://cmake.org/)
* [Ninja](https://ninja-build.org/)

We need to build our own MLIR in the local machine in advance. Please follow the build instruction for MLIR [here](https://mlir.llvm.org/getting_started/). 

### Project Structure
```
MXNET-MLIR/
├── include/        # Header files and function definitions
├── lib/            # Implementation of core functionality
├── MxNet-opt/      # Tool/custom shell to run MLIR transformation passes
├── test/          # MLIR test cases for checking lowering correctness
├── python/         # Python-based tests using Torch to validate correctness
```

### Build Instructions

Please make sure to build LLVM project first according to [the instruction](https://mlir.llvm.org/getting_started/).

To build the project, follow these steps:

```bash
mkdir build && cd build
cmake -G Ninja .. -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
```

### Build the Project:
```bash
mkdir build && cd build
cmake -G Ninja .. -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
cmake --build . --target MxNet-opt
```
### Running Tests
To verify the correctness of the lowering passes, use the following commands:
```bash
cmake --build . --target check-mxnet
```
To build the documentation from the TableGen description of the dialect operations, run
```bash
cmake --build . --target mlir-doc
```
**Running MxNet to Tosa Lowering**:
 - **mxnet-to-tosa** - Lowers from custom MxNet dialect to Lower Level Tosa dialect
```bash
./build/bin/MxNet-opt --mxnet-to-tosa ./test/MxNet/{filename}.mlir
```
**Running MxNet to LLVM Lowering**:
 - **mxnet-to-llvm** - Lowers from custom MxNet dialect to LLVM dialect
```bash
./build/bin/MxNet-opt --mxnet-to-llvm ./test/MxNet/{filename}.mlir
```

## Supported Operations
Currently, the following operations are supported for lowering in the MxNet dialect:

- **Basic Operations:**
  - `AddOp`: Element-wise addition of two tensors.
  - `AbsOp`: Element-wise Absolute of elements inside a tensor.
  - `RsqrtOp`: Computes element-wise reciprocal square root of each element inside a tensor.

- **Reduction Operations:**
  - `ReduceProdOp`: Computes the product along given axes.

- **Specialized Operations:**

	### Linspace Operation
	
	The `linspace` function generates a sequence of evenly spaced numbers over a specified interval. The formula is given by:

	![image](https://github.com/user-attachments/assets/a1c06f34-6e7b-4dc2-8ff5-311cd369a965)
	
	Where:
	- start - is the starting value.
	- end - is the ending value.
	- i - is the index of the points, ranging from \( 0 \) to \( n-1 \).
	- step - is the number of steps (or the size of the output tensor).
	
	This formula ensures that the generated values are evenly spaced between \( start \) and \( end \).
	
	### Softmax Operation
	
	The `softmax` function is used to normalize a tensor into a probability distribution. The formula for the Softmax function is:

	![image](https://github.com/user-attachments/assets/187112cf-d94e-4122-b93b-c0c8aeb2fdff)
	
	Where:
	- xi - is the input value at index \( i \).
	- n - is the number of elements in the input tensor.
	- exp(xi) - is the exponential of the input value at index \( i \).
	- The denominator is the sum of the exponentials of all the input values over specified dim.
	
	The Softmax function converts the input values into probabilities by scaling them so that they sum to 1, which is useful for classification tasks.
