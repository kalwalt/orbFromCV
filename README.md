
# Standalone ORB Implementation

[![CI](https://github.com/kalwalt/orbFromCV/actions/workflows/ci.yml/badge.svg)](https://github.com/kalwalt/orbFromCV/actions/workflows/ci.yml)

This project is an experimental port of the **ORB (Oriented FAST and Rotated BRIEF)** feature detection and description algorithm, extracted and adapted from the OpenCV source code to run **without any OpenCV dependencies**.

> **Status: v0.1.2, early/experimental.** The core pipeline is implemented, tested, and validated against OpenCV's own `cv::ORB` (see [Validation against OpenCV](#validation-against-opencv) below). The two largest hotspots have been fixed (about 4.8× faster end to end than v0.1.1), but it is still single-threaded and scalar — see [Project Status & Roadmap](#project-status--roadmap).

## Project Overview

The goal of this repository is to provide a lightweight, header-and-source C++ implementation of ORB that is easy to integrate into resource-constrained environments or projects where a full OpenCV installation is not desirable (e.g., WebAssembly, embedded systems).

### Key Features
- **Zero OpenCV Dependencies**: All necessary structures (Image, KeyPoint, Point, Rect) and algorithms (FAST, Harris Response, Image Pyramid, Gaussian Blur, Bilinear Resize) are implemented from scratch using standard C++.
- **C++14 Standard**: Uses modern C++ features while maintaining broad compatibility.
- **Performance Oriented**: Includes optimized fixed-point Gaussian blurring and row-major memory access patterns designed for future SIMD/WASM optimizations.
- **Complete Pipeline**: Supports the full ORB pipeline including pyramid construction, FAST detection, Harris scoring, Intensity Centroid orientation, and BRIEF descriptor computation.
- **Tested against OpenCV**: A diagnostic tool (`compare_orb`) validates keypoint, orientation, and descriptor agreement against `cv::ORB` on real images.

## Requirements

- A C++14 compiler (tested with MSVC on Windows and GCC on Linux via CI)
- CMake 3.10+
- Optional: an OpenCV installation, only if you want to build `compare_orb` (see [Diagnostic Tools](#diagnostic-tools))

## Project Structure

- `orb.hpp/cpp`: High-level `ORB` class interface (`detectAndCompute`, with optional per-stage timing output).
- `orb_core.hpp/cpp`: Core mathematical functions (Harris, ICAngles, Descriptors).
- `fast_detector.hpp/cpp`: FAST-9 corner detection implementation.
- `image_utils.hpp/cpp`: Image processing utilities (Bilinear resize, Gaussian blur).
- `orb_pattern.hpp`: The 256-bit pattern used for BRIEF descriptors.
- `stb_image.h`: Single-header library for image loading (used in `main.cpp` and the tools below).
- `tests/`: Unit test suite covering the algorithm's critical parts (see [Testing](#testing)).
- `tools/`: Standalone diagnostic executables (`profile_orb`, `compare_orb`) — see [Diagnostic Tools](#diagnostic-tools).
- `docs/design/`: Design documents written before implementing non-trivial features.

## Building the Project

The project uses CMake for a straightforward build process:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

This builds the `orb_test` CLI, the `profile_orb` diagnostic tool, and the `orb_tests` test suite — none of which require OpenCV.

## Usage

The resulting executable `orb_test` accepts an image path as an argument:

```bash
./orb_test <path_to_image>
```

It will load the image, detect up to 500 keypoints, compute their descriptors, and print the details of the first detected feature to the console.

## Diagnostic Tools

### `profile_orb` — per-stage timing breakdown

Always built, no external dependencies. Reports how much time `ORB::detectAndCompute` spends in each pipeline stage (pyramid construction, FAST detection, Harris scoring, orientation, Gaussian blur, descriptor computation), which is useful for guiding future optimization work:

```bash
cmake --build build --target profile_orb
./build/profile_orb <path_to_image> [repeats]
```

On a real photo, FAST detection is currently the largest stage, followed by Gaussian blur and pyramid construction. Two earlier hotspots have been fixed: the O(n²) non-maximal suppression in FAST ([issue #7](https://github.com/kalwalt/orbFromCV/issues/7)) and the column-strided Gaussian blur ([issue #13](https://github.com/kalwalt/orbFromCV/issues/13)).

### `compare_orb` — validation against OpenCV's `cv::ORB`

An optional tool that runs this implementation and OpenCV's `cv::ORB` on the same image (with matched parameters) and reports keypoint counts, response distribution, spatial match rate, descriptor Hamming distance, and timing. It requires an OpenCV installation but is gated behind a build option so the core library and every other target stay OpenCV-free by default:

```bash
cmake -S . -B build -DBUILD_OPENCV_COMPARISON=ON -DOpenCV_DIR="<path to your OpenCV package>/lib"
cmake --build build --target compare_orb --config Release
./build/Release/compare_orb <path_to_image>
```

See [`docs/design/compare-orb-design.md`](docs/design/compare-orb-design.md) for the full design, including Windows OpenCV-packaging notes.

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

## Validation against OpenCV

Beyond the unit test suite, this implementation has been validated end-to-end against OpenCV's own `cv::ORB` using the `compare_orb` tool described above. That process caught and fixed a real bug: an in-place `gaussianBlur7x7` call was silently zeroing every descriptor before it was sampled, which unit tests alone hadn't caught because none of them exercised the real end-to-end blur-then-descriptor code path with content-level assertions. After the fix, descriptors for cleanly-matched keypoints (same pyramid level, same pixel, same orientation) agree with OpenCV's to within a handful of bits out of 256 — see [issue #5](https://github.com/kalwalt/orbFromCV/issues/5) for the full investigation.

## Project Status & Roadmap

**Done:**
- Full ORB pipeline: pyramid construction, FAST-9 detection with non-maximal suppression, Harris/FAST scoring, orientation (intensity centroid), BRIEF descriptors (`WTA_K` 2/3/4).
- Unit test suite and CI (Linux + Windows).
- Validated against OpenCV's `cv::ORB` for keypoint/orientation/descriptor agreement.
- Performance hotspots fixed: grid-based FAST non-maximal suppression ([issue #7](https://github.com/kalwalt/orbFromCV/issues/7)) and a row-accumulated Gaussian blur ([issue #13](https://github.com/kalwalt/orbFromCV/issues/13)). On a 1637x2048 photo (`profile_orb`, MSVC Release, 8 levels) `detectAndCompute` went from ~918 ms to ~190 ms without changing its output.
- Gaussian blur handles images of any size, including pyramid levels of 1–3 px ([issue #15](https://github.com/kalwalt/orbFromCV/issues/15)).

**Known limitations / open work:**
- **Performance**: this is still a single-threaded, scalar (non-SIMD) implementation, and noticeably slower than OpenCV's SIMD/IPP/multi-threaded build as a result. The remaining time is spread across FAST detection (now the largest stage), Gaussian blur and pyramid construction. A SIMD variant is planned.
- **Not bit-identical to OpenCV**: this is an explicit non-goal (see [issue #3](https://github.com/kalwalt/orbFromCV/issues/3)) — the resize/blur implementations are close approximations, not exact ports, so small numerical differences from OpenCV are expected.

Track ongoing work via the [issue tracker](https://github.com/kalwalt/orbFromCV/issues). Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project's own code is licensed under the [GNU Lesser General Public
License v3.0](LICENSE). The LGPL builds on the GNU General Public License
v3.0, whose text is included in [COPYING](COPYING).

The ORB algorithm implementation (`orb_core.cpp`, `fast_detector.cpp`,
`orb.cpp`, `orb_pattern.hpp`) is adapted from OpenCV's
`modules/features2d/src/orb.cpp`, which carries a BSD-3-Clause header
(Copyright (c) 2009, Willow Garage, Inc.). `stb_image.h` is a separate
third-party single-header library (MIT / public domain, at the author's
option). The sample image `pinball.jpg` is a photograph by Robin van
Mourik, used under a Creative Commons license. See [NOTICE](NOTICE) for
full attribution.
