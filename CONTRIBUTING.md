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
4. **Changes that affect keypoints or descriptors:** the OpenCV ground-truth tests in `tests/test_opencv_ref.cpp` (committed fixtures from OpenCV 4.10.0, see [tests/fixtures/opencv/README.md](tests/fixtures/opencv/README.md)) show the effect in CI; if your change makes a `CHECK_XFAIL` pass, turn it into a hard `CHECK` and raise the end-to-end match-rate floor. If you have OpenCV installed, `compare_orb` timings are welcome too.
5. Keep pull requests focused on one issue and reference it (`Fixes #N`).
6. Use the same Conventional Commits format for the PR title as for commits (see below).

## Commit messages

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): `<type>(<optional scope>): <imperative summary>`, for example `fix(blur): stay in bounds on images with a dimension <= 3 px` or `perf(fast): replace O(n²) non-maximal suppression with a grid lookup`. The allowed types (`feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `ci`, `chore`), the suggested scopes, the `!` / `BREAKING CHANGE:` rule and a full example are in the "Commit Messages" section of [AGENTS.md](AGENTS.md). Reference the issue in a footer (`Fixes #N`).

## OpenCV reference version

All parity values and links use **OpenCV 4.10.0**: `docs.opencv.org/4.10.0/...` for documentation and `github.com/opencv/opencv/blob/4.10.0/<path>#L<n>` for source. Unversioned (`4.x`, `master`) links are rejected by a test. Fixtures are generated with the wheel pinned in `scripts/requirements-fixtures.txt`; bumping it is a change of ground truth and needs every fixture regenerated and reviewed.

## Release merges

`main` is updated from `dev` with a pull request merged using **"Create a merge commit"**, never "Rebase and merge" or "Squash and merge". Rebasing copies `dev`'s commits onto `main` with new IDs, so the two branches diverge; this happened once (#18) and had to be repaired (#30).

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
