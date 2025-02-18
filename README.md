
# MXNET-MLIR

## Overview
MXNET-MLIR is a project designed to lower operations from a custom **MxNet Dialect** to **TOSA Dialect** and further to **LLVM IR**, enabling efficient execution on various hardware backends. It follows a structured transformation pipeline, converting **high-level tensor operations into optimized intermediate representations**. By leveraging MLIR’s extensibility, MXNET-MLIR ensures **portability, performance optimization, and compatibility** with diverse target architectures.

## Project Structure
```
MXNET-MLIR/
├── include/        # Header files and function definitions
├── lib/            # Implementation of core functionality
├── MxNet-opt/      # Tool/custom shell to run MLIR transformation passes
├── tests/          # MLIR test cases for checking lowering correctness
├── python/   # Python-based tests using Torch to validate correctness
```

## Build Instructions

To build the project, follow these steps:

```bash
mkdir build && cd build
cmake -G Ninja .. -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
cmake --build . --target mxnet-opt
```

### Remove the Existing Build Directory:
```bash
rm -rf build
```

### Create and Navigate into the Build Directory:
```bash
mkdir build && cd build
```

### Run CMake with Specified LLVM and MLIR Directories:
```bash
cmake -G Ninja .. -DLLVM_DIR=~/Documents/mlir/llvm-project/build/lib/cmake/llvm -DMLIR_DIR=~/Documents/mlir/llvm-project/build/lib/cmake/mlir
```

### Build the Project:
```bash
cmake --build . --target MxNet-opt
```
## Running Tests

To verify the correctness of the lowering passes, use the following commands:

### Running MxNet to LLVM Lowering:
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
  - `ReduceProdOp`: Computes the product along specified axes.

- **Specialized Operations:**

	### Linspace Operation
	
	The `linspace` function generates a sequence of evenly spaced numbers over a specified interval. The formula is given by:

	![image](https://github.com/user-attachments/assets/a1c06f34-6e7b-4dc2-8ff5-311cd369a965)

	
	Where:
	- \( start \) is the starting value.
	- \( end \) is the ending value.
	- \( step \) is the number of steps (or the size of the output tensor).
	- \( i \) is the index of the points, ranging from \( 0 \) to \( n-1 \).
	
	This formula ensures that the generated values are evenly spaced between \( start \) and \( end \).
	
	### Softmax Operation
	
	The `softmax` function is used to normalize a tensor into a probability distribution. The formula for the Softmax function is:

	![image](https://github.com/user-attachments/assets/187112cf-d94e-4122-b93b-c0c8aeb2fdff)
	
	
	Where:
	- \( x_i \) is the input value at index \( i \).
	- \( n \) is the number of elements in the input tensor.
	- \( exp(x_i) \) is the exponential of the input value at index \( i \).
	- The denominator is the sum of the exponentials of all the input values over specified dim.
	
	The Softmax function converts the input values into probabilities by scaling them so that they sum to 1, which is useful for classification tasks.
