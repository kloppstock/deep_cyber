# deep_cyber
A minimalist deep learning framework for Python and C/C++. Because sometimes your cyber is just not deep enough. 

## Overview
This is the first version of the library which currently can only used for inference. 
This library was written as a learning experience, so it is not focused on optimal performance, maximum parallel execution or feature richness. The main focus was simplicity while not artificially hindering performance. 
When compiled with reasonable optimization flags (`-march=native -O3`), it can infere ~20 fps on a single core (AMD Ryzen 1700).

## Structure
The functions were first prototyped in Python and gradually ported to C until a pure C version was ready. Most stages can be found in the `prototype` folder. 

The following stages have been taken:
1. refefence net for CIFAR10 in pure keras (see`cifar10_reference.ipynb`)
2. the keras functions were extracted and executed seperately (see `cifar10_keras.ipynb`)
3. the extracted keras functions were reimplemented in pure python and checked against the reference (see `cifar10_custom_python.ipynb` and `function_tests.ipynb`)
4. than, the individual layers where ported to C and checked in Python (see `cifar10_native.ipynb`, `native_function_test.ipynb` and the `src` and `include` folder)
5. the network was ported completely to C (see `cifar10_native_interface.ipynb` and the `cifar10` folder)
6. a completely standalone verion was created (see `main.c`)

Additionally, googletest was used to test the low level functions and ensure, that the program didn't segfault (see `test` folder)

The C code itself is split into the `Tensor` structure and functions defined in the `tensor.h` and `tensor.c` files. The definitions for the actual framework can be found in `deep_cyber.h` while the implementation of the layers is split up into the `activation.c`, `conv2d.c`, `dense.c` and `pooling.c` files. 

For the predifined CIFAR10 network, the keras weights where exported and converted to C code (see `converter.py`). The actual definitions and the network are located in `difar10.h` and `cifar10.c` respectively. 

The standalone version only extends the CIFAR10 network with hardcoded input data. 

## Build

Other then CMake > 3.5, a standard compliant C compiler and (optinally) Python (with keras and tensorflow installed), nothing is required. 

```
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
./deep_cyber_test # optional
./deep_cyber_main # also optional
```
