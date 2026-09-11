
# Standalone ORB Implementation

[![CI](https://github.com/kalwalt/orbFromCV/actions/workflows/ci.yml/badge.svg)](https://github.com/kalwalt/orbFromCV/actions/workflows/ci.yml)

This project is an experimental port of the **ORB (Oriented FAST and Rotated BRIEF)** feature detection and description algorithm, extracted and adapted from the OpenCV source code to run **without any OpenCV dependencies**.

## Project Overview

The goal of this repository is to provide a lightweight, header-and-source C++ implementation of ORB that is easy to integrate into resource-constrained environments or projects where a full OpenCV installation is not desirable (e.g., WebAssembly, embedded systems).

### Key Features
- **Zero OpenCV Dependencies**: All necessary structures (Image, KeyPoint, Point, Rect) and algorithms (FAST, Harris Response, Image Pyramid, Gaussian Blur, Bilinear Resize) are implemented from scratch using standard C++.
- **C++14 Standard**: Uses modern C++ features while maintaining broad compatibility.
- **Performance Oriented**: Includes optimized fixed-point Gaussian blurring and row-major memory access patterns designed for future SIMD/WASM optimizations.
- **Complete Pipeline**: Supports the full ORB pipeline including pyramid construction, FAST detection, Harris scoring, Intensity Centroid orientation, and BRIEF descriptor computation.

## Project Structure

- `orb.hpp/cpp`: High-level ORB class interface.
- `orb_core.hpp/cpp`: Core mathematical functions (Harris, ICAngles, Descriptors).
- `fast_detector.hpp/cpp`: FAST-9 corner detection implementation.
- `image_utils.hpp/cpp`: Image processing utilities (Bilinear resize, Gaussian blur).
- `orb_pattern.hpp`: The 256-bit pattern used for BRIEF descriptors.
- `stb_image.h`: Single-header library for image loading (used in `main.cpp`).
- `tests/`: Unit test suite covering the algorithm's critical parts (see [Testing](#testing)).

## Building the Project

The project uses CMake for a straightforward build process:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## Usage

The resulting executable `orb_test` accepts an image path as an argument:

```bash
./orb_test <path_to_image>
```

It will load the image, detect up to 500 keypoints, compute their descriptors, and print the details of the first detected feature to the console.

## Testing

The `tests/` directory contains a unit test suite covering the critical parts of
the pipeline: bilinear resize and Gaussian blur (`image_utils.cpp`), FAST-9
corner detection (`fast_detector.cpp`), Harris scoring / orientation / BRIEF
bit-packing (`orb_core.cpp`), and the end-to-end `ORB::detectAndCompute`
pipeline (`orb.cpp`). It uses a small header-only harness
(`tests/test_framework.hpp`) instead of an external framework, keeping the
project's zero-dependency policy intact.

Build and run the tests with CMake/CTest:

```bash
cmake --build build --target orb_tests
ctest --test-dir build --output-on-failure
```

Every push and pull request is built and tested automatically on Linux and
Windows via the [CI workflow](.github/workflows/ci.yml).
