
# Standalone ORB Implementation

[![CI](https://github.com/kalwalt/orbFromCV/actions/workflows/ci.yml/badge.svg)](https://github.com/kalwalt/orbFromCV/actions/workflows/ci.yml)

This project is an experimental port of the **ORB (Oriented FAST and Rotated BRIEF)** feature detection and description algorithm, extracted and adapted from the OpenCV source code to run **without any OpenCV dependencies**.

> **Status: v0.1.3, early/experimental.** The core pipeline is implemented, tested, and matches OpenCV 4.10.0's `cv::ORB` on the committed reference scene: all 500 keypoints on all 8 octaves, 491 of them at the identical float position with a byte-identical descriptor (see [Validation against OpenCV](#validation-against-opencv) below). It is still single-threaded and scalar — see [Project Status & Roadmap](#project-status--roadmap).

## Project Overview

The goal of this repository is to provide a lightweight, header-and-source C++ implementation of ORB that is easy to integrate into resource-constrained environments or projects where a full OpenCV installation is not desirable (e.g., WebAssembly, embedded systems).

### Key Features
- **Zero OpenCV Dependencies**: All necessary structures (Image, KeyPoint, Point, Rect) and algorithms (FAST, Harris Response, Image Pyramid, Gaussian Blur, Bilinear Resize) are implemented from scratch using standard C++.
- **C++14 Standard**: Uses modern C++ features while maintaining broad compatibility.
- **Performance Oriented**: Separable Gaussian blur and fixed-point bilinear resize, both bit-identical to what `cv::ORB` computes, with row-major memory access patterns designed for future SIMD/WASM optimizations.
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

On a real photo, FAST detection is currently the largest stage, followed by Gaussian blur and pyramid construction. Two earlier hotspots have been fixed: the O(n²) non-maximal suppression in FAST ([issue #7](https://github.com/kalwalt/orbFromCV/issues/7)) and the column-strided Gaussian blur ([issue #13](https://github.com/kalwalt/orbFromCV/issues/13)). The blur got slower again in v0.1.3, when it was changed to compute exactly what `cv::ORB` computes ([issue #39](https://github.com/kalwalt/orbFromCV/issues/39)); the SIMD work in [issue #19](https://github.com/kalwalt/orbFromCV/issues/19) is meant to recover that.

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
Windows, plus a Clang ASan/UBSan job, via the [CI workflow](.github/workflows/ci.yml).

The suite also contains a parity test suite (`tests/test_opencv_ref.cpp`) that
compares every stage and the end-to-end pipeline against ground truth produced
by OpenCV **4.10.0** and committed under `tests/fixtures/opencv/`, so CI never
needs OpenCV. The fixtures are regenerated by hand with
`scripts/gen-opencv-fixtures.py` from a pinned `opencv-contrib-python` wheel;
see [`tests/fixtures/opencv/README.md`](tests/fixtures/opencv/README.md) for
the pin rules.

## Validation against OpenCV

The reference is OpenCV **4.10.0** with its SIMD/FMA dispatch off (what opencv.js
computes). On the committed 640x480 reference scene, `ORB::detectAndCompute`
with default parameters matches `cv::ORB` for all 500 keypoints on all 8
octaves; 491 of them sit at the identical float position with a byte-identical
descriptor, mean Hamming distance 0.0. Each stage is pinned on its own too:
bilinear resize (`INTER_LINEAR_EXACT`), the Gaussian blur, FAST with and
without non-maximal suppression (positions and scores), and the BRIEF
descriptors on given keypoints are all exact.

Getting there took a series of fixes, each with its own issue: descriptors on
pyramid levels >= 1 sampled at the wrong location (#34), out-of-bounds reads for
small `edgeThreshold` (#35), `cvRound` half-to-even rounding (#41), the FAST-9
early rejection (#20), OpenCV's corner score and Harris pre-filtering (#24),
pixel-centre resize sampling (#40, by jamicaw56), the float32 blur (#39), the
`nfeatures` cap for `FAST_SCORE` (#21), and stale outputs on early return (#22,
by jamicaw56).

The one known remaining divergence is the orientation angle: this port uses
`std::atan2` where `cv::ORB` uses the `fastAtan2` polynomial, which moves a few
descriptors by 1–3 bits ([issue #49](https://github.com/kalwalt/orbFromCV/issues/49)).

The optional `compare_orb` tool described above still exists for comparing
against an installed OpenCV on arbitrary images.

## Project Status & Roadmap

**Done:**
- Full ORB pipeline: pyramid construction, FAST-9 detection with non-maximal suppression, Harris/FAST scoring, orientation (intensity centroid), BRIEF descriptors (`WTA_K` 2/3/4).
- Unit test suite, OpenCV 4.10.0 parity fixtures, and CI (Linux + Windows + ASan/UBSan).
- Matches `cv::ORB` on the reference scene for all 500 keypoints (see above).
- Constructor validation: unsupported parameters (`firstLevel != 0`, `wta_k` outside 2–4, ...) throw `std::invalid_argument` ([issue #23](https://github.com/kalwalt/orbFromCV/issues/23)).
- Performance hotspots fixed: grid-based FAST non-maximal suppression ([issue #7](https://github.com/kalwalt/orbFromCV/issues/7)) and a row-accumulated Gaussian blur ([issue #13](https://github.com/kalwalt/orbFromCV/issues/13)).

**Known limitations / open work:**
- **Performance**: this is still a single-threaded, scalar (non-SIMD) implementation, and noticeably slower than OpenCV's SIMD/IPP/multi-threaded build as a result. A SIMD variant is planned ([issue #19](https://github.com/kalwalt/orbFromCV/issues/19)).
- **Orientation angle**: `std::atan2` instead of `fastAtan2` ([issue #49](https://github.com/kalwalt/orbFromCV/issues/49)); descriptors can differ from OpenCV's by a few bits at some keypoints.
- **`firstLevel`**: only 0 is supported; OpenCV's upscaled levels below the input are not implemented.
- WebAssembly build and CI job ([issue #27](https://github.com/kalwalt/orbFromCV/issues/27)) and a CMake package ([issue #28](https://github.com/kalwalt/orbFromCV/issues/28)) are planned for v0.2.0.

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
