# Agent Instructions

## Project
Standalone C++14 port of OpenCV's ORB (Oriented FAST and Rotated BRIEF) feature
detector/descriptor, with zero OpenCV dependency. See `README.md` for pipeline
and file-layout details.

## Build
```bash
mkdir build && cd build
cmake ..
cmake --build .
```
Run: `./orb_test <path_to_image>`

## Tests
`tests/` holds a dependency-free unit suite (see `tests/test_framework.hpp`)
covering `fast_detector.cpp`, `orb_core.cpp`, `image_utils.cpp`, and the
`ORB::detectAndCompute` pipeline.
```bash
cmake --build build --target orb_tests
ctest --test-dir build -C Debug --output-on-failure
```

## File-Scoped Commands
| Task | Command |
|------|---------|
| Compile single TU | `g++ -std=c++14 -Wall -Wextra -c path/to/file.cpp -o /tmp/out.o` |
| Full rebuild | `cmake --build build` |
| Run tests | `cmake --build build --target orb_tests && ctest --test-dir build -C Debug` |

## Key Conventions
- No OpenCV, no external deps beyond the bundled `stb_image.h` — keep it that way,
  including in `tests/` (no gtest/Catch2/doctest).
- Match existing header/source split: declarations in `.hpp`, implementation in `.cpp`.
- Preserve row-major memory layout and fixed-point paths in `image_utils.cpp` /
  `orb_core.cpp` — they're structured for future SIMD/WASM ports.
- `orb_pattern.hpp` (BRIEF sampling pattern) is generated data — don't hand-edit.
- Add a test in `tests/` alongside any change to core algorithm behavior.

## Commit Attribution
AI commits MUST include:
```
Co-Authored-By: (the agent model's name and attribution byline)
```
