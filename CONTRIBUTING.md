# Contributing

Thanks for your interest in improving this standalone ORB implementation. Bug reports, benchmarks on new hardware and pull requests are all welcome.

## Before you start

- Check the [issue tracker](https://github.com/kalwalt/orbFromCV/issues) for existing work. Issues labelled `good first issue` are small and self-contained.
- For anything larger than a bug fix, open or comment on an issue first so the approach can be agreed before you write code.

## Ground rules

- **No new dependencies.** The core library and the test suite depend only on the C++14 standard library and the bundled `stb_image.h`. OpenCV is used only by the optional `compare_orb` tool, behind `BUILD_OPENCV_COMPARISON` (off by default).
- **C++14**, declarations in `.hpp`, implementation in `.cpp`, matching the existing style.
- **Keep the row-major, fixed-point structure** of `image_utils.cpp` and `orb_core.cpp`; it is laid out for future SIMD/WASM work.
- `orb_pattern.hpp` is generated data from OpenCV. Do not edit it by hand.

## Building and testing

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build -C Debug --output-on-failure
```

CI runs the same build and tests on Ubuntu and Windows for every pull request.

## Pull requests

1. Branch from `dev` (not `main`) and open the pull request against `dev`. `main` is updated from `dev` at release time.
2. **Add or update a test for every behavior change.** For a bug fix, write the test first and check that it fails without your fix. The test framework is a small header in `tests/test_framework.hpp`; no external test libraries.
3. **Performance changes:** include before/after numbers from `profile_orb` (per-stage timings), with the image, build type and compiler. Say whether output is byte-identical; if it is not, explain why.
4. **Changes that affect keypoints or descriptors:** if you have OpenCV installed, include `compare_orb` results before and after.
5. Keep pull requests focused on one issue and reference it (`Fixes #N`).

## Reporting bugs

Please include:

- what you ran (code snippet or command) and on which image or image size,
- what you expected and what happened,
- compiler, OS and build type (Debug/Release).

Small, self-contained reproductions (for example a synthetic 7x7 image) are the most useful.

## AI-assisted contributions

AI-assisted contributions are welcome. Commits written with an AI coding agent must include a `Co-Authored-By:` line naming the model, as described in [AGENTS.md](AGENTS.md). You remain responsible for reviewing and testing what you submit.

## License

By contributing, you agree that your contributions are licensed under the project's [LGPL-3.0 license](LICENSE).
