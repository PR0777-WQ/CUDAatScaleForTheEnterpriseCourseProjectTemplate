## GPU Accelerated Edge Detection using CUDA (Sobel Filter)
## Overview
This project demonstrates the use of CUDA GPU acceleration to perform edge detection on images using the Sobel operator. The goal is to utilize the computational power of GPUs to process images faster than traditional CPU-based implementations.

Edge detection is an important image processing technique used in computer vision, robotics, medical imaging, and autonomous systems. In this project, the Sobel filter is implemented using CUDA kernels to process image pixels in parallel.

This project is part of the CUDA at Scale for the Enterprise course and helps illustrate how GPU parallel computing can significantly speed up image processing tasks.

## Code Organization

```bin/```
This folder should hold all binary/executable code that is built automatically or manually. Executable code should have use the .exe extension or programming language-specific extension.

```data/```
This folder should hold all example data in any format. If the original data is rather large or can be brought in via scripts, this can be left blank in the respository, so that it doesn't require major downloads when all that is desired is the code/structure.

```lib/```
Any libraries that are not installed via the Operating System-specific package manager should be placed here, so that it is easier for inclusion/linking.

```src/```
The source code should be placed here in a hierarchical fashion, as appropriate.

```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```INSTALL```
This file should hold the human-readable set of instructions for installing the code so that it can be executed. If possible it should be organized around different operating systems, so that it can be done by as many people as possible with different constraints.

```Makefile or CMAkeLists.txt or build.sh```
There should be some rudimentary scripts for building your project's code in an automatic fashion.

```run.sh```
An optional script used to run your executable code, either with or without command-line arguments.

### Key Concepts

This project demonstrates several important GPU computing and image processing concepts:

GPU Parallel Computing – Thousands of GPU threads process image pixels simultaneously.

CUDA Kernel Programming – Custom CUDA kernels are used to perform Sobel edge detection on the GPU.

Image Processing – The project applies convolution-based filters to detect edges in images.

Sobel Operator – A gradient-based method used to highlight edges in an image.

Host–Device Memory Transfer – Image data is transferred between CPU memory and GPU memory using CUDA APIs.

Performance Optimization – GPU acceleration significantly improves the speed of image processing tasks compared to CPU implementations.

### Supported SM Architectures

The program supports the following NVIDIA GPU Streaming Multiprocessor (SM) architectures:
```
SM 3.5
SM 3.7
SM 5.0
SM 5.2
SM 6.0
SM 6.1
SM 7.0
SM 7.2
SM 7.5
SM 8.0
SM 8.6
```
These architectures correspond to several generations of NVIDIA GPUs that support CUDA execution.

## Supported OSes

The project can run on the following operating systems:

Linux

Windows

Both platforms support CUDA development and GPU execution.

## Supported CPU Architecture

The program supports the following CPU architectures:

x86_64

ppc64le

armv7l

These architectures are commonly used in desktop systems, servers, and embedded systems that support CUDA.

## CUDA APIs Involved

The project uses several important CUDA APIs for GPU programming:

cudaMalloc() – Allocates memory on the GPU device.

cudaMemcpy() – Transfers data between host (CPU) and device (GPU) memory.

cudaFree() – Frees GPU memory after execution.

CUDA Kernel Launch – Executes the Sobel edge detection kernel on the GPU.

Example kernel launch:
```
sobelKernel<<<gridSize, blockSize>>>(d_input, d_output);
```
Each thread processes one pixel in the image.

### Dependencies Needed to Build/Run

The following libraries are required to build and run the project:

CUDA Toolkit

FreeImage Library

FreeImage is used for reading and writing image files such as PNG or JPG, while CUDA Toolkit provides GPU programming support.

### Prerequisites

Before building the project, ensure the following software is installed:

CUDA Toolkit 11.4 or later

NVIDIA GPU with CUDA support

FreeImage image processing library

C++ compiler (g++ for Linux or Visual Studio for Windows)

The CUDA Toolkit includes tools such as:

NVCC compiler

CUDA runtime libraries

GPU debugging tools

### Build and Run
### Windows

On Windows, the project can be built using Microsoft Visual Studio with CUDA support.

Steps:

Install Visual Studio and CUDA Toolkit.

Open the provided solution file:

```edgeDetection_vs2019.sln```

Build the project from Visual Studio.

The executable file will be generated in the bin/ directory.

Some systems may also require the DirectX SDK for certain CUDA samples.

### Linux

On Linux systems, the project can be built using Makefiles.

Navigate to the project directory and run:

```make```

Optional build parameters:

Build with debugging symbols:

```make dbg=1```

Specify target architecture:

```make TARGET_ARCH=x86_64```

Specify GPU architectures:

```make SMS="50 60"```

These options allow compilation for different systems and GPU configurations.

Running the Program

After the project is compiled successfully, the program can be executed using:

```make run```

This command runs the program and performs Sobel edge detection on the input image.

The default input image is:

```data/Lena.png```

The processed output image will be saved as:

```data/Lena_edges.png```

Alternatively, the executable can be run directly with custom input and output files:

```./bin/edgeDetectionCUDA --input data/Lena.png --output data/Lena_edges.png```

