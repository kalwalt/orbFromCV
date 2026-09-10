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

## File-Scoped Commands
| Task | Command |
|------|---------|
| Compile single TU | `g++ -std=c++14 -Wall -Wextra -c path/to/file.cpp -o /tmp/out.o` |
| Full rebuild | `cmake --build build` |

## Key Conventions
- No OpenCV, no external deps beyond the bundled `stb_image.h` — keep it that way.
- Match existing header/source split: declarations in `.hpp`, implementation in `.cpp`.
- Preserve row-major memory layout and fixed-point paths in `image_utils.cpp` /
  `orb_core.cpp` — they're structured for future SIMD/WASM ports.
- `orb_pattern.hpp` (BRIEF sampling pattern) is generated data — don't hand-edit.

## Commit Attribution
AI commits MUST include:
```
Co-Authored-By: (the agent model's name and attribution byline)
```
