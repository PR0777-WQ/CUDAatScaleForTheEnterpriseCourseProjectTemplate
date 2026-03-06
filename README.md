## GPU-Accelerated Image Denoising and Edge Detection using CUDA NPP
## Overview

This project demonstrates the use of NVIDIA Performance Primitives (NPP) together with CUDA to perform GPU-accelerated image denoising and edge detection on a large dataset of images.

The goal of this project is to process hundreds of grayscale images efficiently using GPU computation. Traditional CPU implementations of image filtering operations can be computationally expensive when applied to large datasets. By leveraging CUDA-enabled GPUs and optimized NPP library functions, this project significantly speeds up the processing pipeline.

The project performs the following GPU-accelerated operations:

Gaussian Smoothing for image denoising

Median Filtering to remove impulse noise

Sobel Edge Detection to detect object boundaries

The images are loaded from disk, transferred to GPU memory, processed using GPU kernels from the NPP library, and the results are written back to disk.

This project is designed as part of the CUDA at Scale for the Enterprise course and demonstrates how GPU libraries can be used for high-performance image processing


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


## Initialization and Setup
CUDA and NPP Information

The program begins by printing system configuration information including:

CUDA Runtime Version

CUDA Driver Version

NVIDIA GPU Device Name

NPP Library Version

This helps verify that the environment supports CUDA-based GPU computation.

The program then checks whether a compatible GPU device is available.

## Input File Handling

The program expects a directory containing hundreds of grayscale images.

If a directory is passed through command-line arguments, the program reads images from that directory. Otherwise, it defaults to a predefined directory such as:

`data/input_images/`

Each image file is validated before processing to ensure it can be opened correctly.

## Image Loading
Host Image Object

Each image is first loaded into host memory using the FreeImage library. The loaded image is stored in a host-side image object:

`npp::ImageCPU_8u_C1`

This object represents an 8-bit grayscale image stored in CPU memory.

Device Image Object

After loading the image into host memory, it is copied to GPU memory using:

npp::ImageNPP_8u_C1

This object represents an image stored in GPU device memory.

The transfer from host to device is performed using CUDA memory operations.

## Image Processing on GPU

Once the image data resides on the GPU, several processing operations are performed using NPP library functions.

## Gaussian Smoothing

Gaussian smoothing is applied to reduce noise and small variations in the image.

This is done using:

`nppiFilterGauss_8u_C1R`

The filter performs a convolution with a Gaussian kernel, producing a smoother image that suppresses noise.

## Median Filtering

Median filtering is applied to remove salt-and-pepper noise while preserving edges.

This is performed using:

`nppiFilterMedian_8u_C1R`

This filter replaces each pixel value with the median of its neighboring pixels.

## Sobel Edge Detection

To detect edges in the image, a Sobel filter is applied.

The Sobel operator calculates the gradient of the image intensity and highlights areas with strong spatial derivatives.

The NPP function used:

`nppiFilterSobel_8u_C1R`

This produces an edge-detected version of the image.

## Batch Processing of Large Image Dataset

The program processes hundreds of images sequentially.

For each image:

Load image into host memory

Copy image to GPU memory

Apply Gaussian smoothing

Apply Median filtering

Apply Sobel edge detection

Copy processed image back to host memory

Save processed image to disk

This pipeline ensures the GPU handles the heavy computation for every image in the dataset.

## Saving the Processed Images

After processing on the GPU is complete, the resulting image is copied back from device memory to host memory.

The processed images are saved to:

data/output_images/

 output filenames:

image1_gaussian.png
image1_median.png
image1_edges.png

This allows easy comparison between different filtering operations.

## Performance Benchmarking

The project includes a simple performance measurement system to compare GPU execution time.

The timing process includes:

GPU processing time per image

Total batch processing time

CUDA event timers are used to record the time spent executing GPU operations.

This demonstrates the advantage of GPU acceleration when processing large datasets.

## Cleanup

At the end of execution:

GPU memory allocations are freed

Host memory objects are released

File streams are closed

Proper memory management ensures that no memory leaks occur during execution.

The program includes try-catch error handling to detect and report runtime errors.

## Execution Flow

The complete execution pipeline is:

Initialize CUDA and NPP

Detect GPU device

Load images from dataset directory

Copy images from CPU memory to GPU memory

Apply GPU image processing filters

Copy results back to CPU memory

Save processed images

Print execution statistics

Clean up resources

## Code Organization

The repository follows the structure recommended by the CUDA course template.
```
project-root/

bin/
Executable binaries generated after compilation.

data/
Input and output image datasets.

data/input_images/
Original dataset images.

data/output_images/
Processed output images.

lib/
External libraries that are not installed through the operating system package manager.

src/
Source code files for CUDA image processing.

README.md
Project description and documentation explaining the purpose of the project.

INSTALL
Instructions for installing and running the project on different operating systems such as Linux and Windows.

Makefile
Build automation script used to compile the project.

run.sh
Shell script used to run the compiled executable with required arguments.
```

    Script to execute the compiled program
## Key Concepts

This project demonstrates the following CUDA concepts:

GPU Accelerated Image Processing

CUDA Memory Management

Host-to-Device Memory Transfers

Batch Processing of Large Datasets

NVIDIA NPP Library Functions

GPU Performance Optimization

## Supported SM Architectures

The program supports the following NVIDIA GPU architectures:
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
## Supported Operating Systems

The project can run on the following platforms:

Linux
Windows
```
Supported CPU Architectures
x86_64
ppc64le
armv7l
```
## CUDA APIs Involved
The following CUDA and NPP APIs are used in this project:

CUDA Runtime API

CUDA Memory Management

NPP Image Filtering APIs

CUDA Event Timing APIs

## Dependencies Needed to Build/Run

The project requires the following dependencies:

CUDA Toolkit 11.4 or newer

NVIDIA NPP Library

FreeImage Library

C++ Compiler (GCC / MSVC)

## Prerequisites

Before building the project, ensure the following are installed:

CUDA Toolkit 11.4 or later

Download from:

https://developer.nvidia.com/cuda-downloads

Install FreeImage Library

Linux:

```sudo apt-get install libfreeimage-dev```

Windows:

Download FreeImage from:

https://freeimage.sourceforge.io

Ensure the NVIDIA GPU drivers are installed and compatible with CUDA.

Build and Run
Windows

The Windows project can be built using Microsoft Visual Studio.

Solution files are provided in the format:

```*_vs<version>.sln```

Steps to build:

Open the solution file in Visual Studio

Select Release Mode

Click Build Solution

Some samples may require the Microsoft DirectX SDK.

## Linux

On Linux systems, the project is built using Makefiles.

Navigate to the project directory:

``cd <project_directory>``

Then build using:

`make`
Optional Build Parameters

Specify architecture:

make TARGET_ARCH=x86_64

Build with debug symbols:

make dbg=1

Specify SM architectures:

```make SMS="50 60"```

Specify host compiler:

```make HOST_COMPILER=g++```
Running the Program

After building the project, run the program using:

`make run`

This command will execute the compiled binary and process the images in the dataset directory.

Example execution:

`./bin/imageProcessingCUDA --input data/input_images --output data/output_images`

The program will process all images and store the results in the output folder.

Cleaning Up

To remove compiled binaries and generated files:

`make clean`

This deletes all files located inside the bin/ directory.
