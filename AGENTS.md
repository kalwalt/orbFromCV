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

## Commit Messages
All commits follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):
```
<type>(<optional scope>): <imperative summary, lower case, no trailing period>

<optional body: why, and measurements for perf changes>

<optional footers: Fixes #N, BREAKING CHANGE: ..., Co-Authored-By: ...>
```

| type | use for |
|------|---------|
| `feat` | new capability or public API |
| `fix` | bug fix |
| `perf` | speedup with the same output |
| `refactor` | code change with no behavior change |
| `test` | tests only |
| `docs` | README, NOTICE, design docs, agent files |
| `build` | CMake, compiler flags, packaging |
| `ci` | GitHub Actions |
| `chore` | anything else (license files, housekeeping) |

Suggested scopes: `fast`, `blur`, `pyramid`, `harris`, `orientation`,
`descriptors`, `orb` (pipeline / `ORB` class), `core`, `tools`, `tests`,
`readme`, `agents`.

Rules:
- Breaking API changes: `!` after the type/scope (`feat(orb)!: …`) plus a
  `BREAKING CHANGE:` footer.
- Reference the issue in a footer: `Fixes #N` or `Refs #N`.
- Branch names use the same types: `<user>/<type>/<slug>`.
- PR titles follow the same format. GitHub merge commits
  (`Merge pull request #N …`) are exempt.
- Applies from now on; history is not rewritten.

Example:
```
fix(blur): stay in bounds on images with a dimension <= 3 px

A single BORDER_REFLECT_101 step maps index n+2 to n-4, which is negative
for n <= 3. Repeat the reflection until the index is inside [0, n), as
OpenCV's borderInterpolate does. Output for dimensions >= 4 px is unchanged.

Fixes #15
Co-Authored-By: (the agent model's name and attribution byline)
```

## Release Merges
`dev` → `main` release PRs MUST be merged with **"Create a merge commit"**.
"Rebase and merge" copies `dev`'s commits onto `main` with new IDs and makes
the two branches diverge (this happened on #18 and was repaired in #30).
