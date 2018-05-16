# CUDA_Concurrent_Keyword-Density-Analyser
A concurrent solution, built in C++ using the NVIDIA Library CUDA 8.5 for performing keyword density analysis on large bodies of text.

Compared to a basic sequential implementation in C++, this application runs **17x faster** on a dataset of 70k words on an intel i5 3.3GHz CPU.

## About

This application was built in parallel with a GoLang implementation, also available on my Github, which takes advantage of CPU cores. Out of the two, this solution ran slower and scaled less.

## Getting Started

**Prerequisites**

1. Install Visual Studio 2015
2. Install Cuda 8.5 from NVIDIA's website: https://developer.nvidia.com/cuda-downloads 

----------

**Get Started:**

1. Clone the Repo.
2. Create a new CUDA project in Visual Studio.
3. Import the kernel.cu file into your project (Remember to remove the main function from the default kernel).
4. Copy any of the sample files which you wish to test into the root of the project.
5. Build and run!

## Files

 - kernel.cu - C++ Source Code for the application.
 - *.txt - These files are simply available for convenience so when you run the program you can select these to parse, these can be replaced with your own target files.
 - CalumBell_Text_processor.vxcproj - Example project settings

## Performance
The performance of this application has been measured against a sequential C++ solution and a shared memory solution in NVIDIA's CUDA.

![Performance Analysis](https://i.imgur.com/xddRvbN.png)

**Notice for particularly small datasets, this solution is slower, due to an increased overhead.**
