# Design: `compare_orb` — standalone ORB vs. OpenCV ORB comparison tool

Tracks: [#3](https://github.com/kalwalt/orbFromCV/issues/3)

## Understanding summary

- Validates this repo's from-scratch ORB port against the OpenCV implementation
  it was ported from: keypoint counts, spatial agreement, descriptor
  similarity, and timing.
- A manual dev-time diagnostic, not a CI gate and not a regression test.
- Must not add OpenCV as a dependency of `orb_lib`, `orb_test`, or `orb_tests`
  — those keep building with zero OpenCV, in CI and locally.
- A SIMD ORB variant is planned later; the tool's internals should make adding
  it a small addition, not a rewrite.

## Non-goals

- Bit-identical output between implementations.
- CI integration (OpenCV is not installed in the GitHub Actions runners).
- A multi-image benchmark/dataset runner (single image, path via CLI arg).
- Rigorous one-to-one (assignment-optimal) keypoint matching.

## Files

```
CMakeLists.txt        (+ BUILD_OPENCV_COMPARISON option, guarded target)
tools/
  compare_orb.cpp      (everything: main, runners, comparison, reporting)
docs/design/
  compare-orb-design.md (this document)
```

Kept as a single file: it's a diagnostic tool, not library code, and splitting
it up would add navigation cost with no reuse benefit at this size (~250-300
lines).

## Build integration

```cmake
option(BUILD_OPENCV_COMPARISON "Build the compare_orb tool (requires OpenCV)" OFF)

if(BUILD_OPENCV_COMPARISON)
    find_package(OpenCV REQUIRED)
    add_executable(compare_orb tools/compare_orb.cpp)
    target_link_libraries(compare_orb PRIVATE orb_lib ${OpenCV_LIBS})
    target_include_directories(compare_orb PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()
```

Default `OFF` — `orb_lib`/`orb_test`/`orb_tests`/CI are configured and built
exactly as before with no `find_package(OpenCV)` call ever made. Enabling it
locally, against an official OpenCV Windows package extracted to
`C:\tools\opencv`:

```bash
cmake -S . -B build -DBUILD_OPENCV_COMPARISON=ON -DOpenCV_DIR="C:/tools/opencv/build/x64/vc16/lib"
cmake --build build --target compare_orb --config Release
```

**Windows package notes** (verified against the official OpenCV 4.9 Windows
build):
- Point `OpenCV_DIR` at the per-toolset directory (`build/x64/vc16/lib`), not
  the top-level `build` — the generic top-level `OpenCVConfig.cmake` only
  probes for a fixed set of known toolset versions and may report "no
  binaries compatible" even when a usable one (e.g. vc16) exists.
- Build `compare_orb` in the **same configuration as the OpenCV package**
  (this package ships Release libs only) — mixing Debug/Release pulls in
  OpenCV's `debug_build_guard` ABI check and fails to link.
- The `opencv_world*.dll` needs to be next to `compare_orb.exe` (or on
  `PATH`) to run.

## Data flow

```
                 stb_image (existing loader)
                          │
                          ▼
                  Image8U grayImage
                    │            │
     (used as-is)   │            │  wrap: cv::Mat(rows, cols, CV_8UC1,
                     │            │        grayImage.data(), step).clone()
                     ▼            ▼
              "mine" pipeline   OpenCV pipeline
           ORB::detectAndCompute  cv::ORB::detectAndCompute
                     │            │
                     ▼            ▼
            ImplementationResult (common shape, see below)
                     │            │
                     └────┬───────┘
                          ▼
                 compareImplementations()
                          │
                          ▼
                   console report
```

Both implementations run on **the same pixel buffer** (one stb_image decode,
then a wrapped/cloned `cv::Mat` over it) rather than two independent image
loaders — this removes "different JPEG decoder / grayscale conversion" as a
confound, isolating the comparison to the ORB algorithm itself.

## Interfaces

```cpp
// Shared, framework-agnostic representation both sides get converted into.
struct CommonKeypoint {
    float x, y;
    float angle;
    float response;
    int   octave;
};

struct ImplementationResult {
    std::string name;                              // "mine" / "opencv" / (later) "simd"
    std::vector<CommonKeypoint> keypoints;
    std::vector<std::array<uint8_t, 32>> descriptors; // index-aligned with keypoints
    double meanMs;
    double stddevMs;
};

// One shared parameter set built once, used to construct both ORB instances,
// so any divergence reflects algorithm behavior, not configuration drift.
struct OrbParams {
    int nfeatures      = 500;
    float scaleFactor  = 1.2f;
    int nlevels        = 8;
    int edgeThreshold  = 31;
    int firstLevel     = 0;
    int wta_k          = 2;
    int patchSize      = 31;
    int fastThreshold  = 20;
};

ImplementationResult runMine(const Image8U& img, const OrbParams& p, int repeats);
ImplementationResult runOpenCv(const cv::Mat& img, const OrbParams& p, int repeats);

struct ComparisonReport {
    size_t countA, countB;
    size_t spatialMatches;          // count of A-keypoints with a B-neighbor within radius
    double meanHamming, medianHamming, maxHamming; // over matched pairs, bits (0-256)
    double meanAngleDiffDeg;        // over matched pairs
};

ComparisonReport compareImplementations(const ImplementationResult& a,
                                         const ImplementationResult& b,
                                         float spatialRadiusPx = 3.0f);
```

**Extension point for the future SIMD variant:** write
`runSimd(const Image8U&, const OrbParams&, int repeats) -> ImplementationResult`
and call `compareImplementations` again (e.g. mine-vs-simd, simd-vs-opencv).
No change to `ImplementationResult`, `OrbParams`, or `compareImplementations`
is needed — this is the concrete mechanism behind "extension, not rewrite."

## Algorithm details

1. **Load** the image once via stb_image into `Image8U` (same conversion loop
   as `main.cpp`); wrap+clone it into a `cv::Mat` for the OpenCV side.
2. **Timing.** For each implementation: one untimed warm-up call to
   `detectAndCompute` (avoids first-call effects), then `N = 10` timed
   repeats using `std::chrono::steady_clock`; report mean and stddev in ms.
3. **Convert** each implementation's native output into
   `ImplementationResult` (`cv::KeyPoint` fields map 1:1 onto `CommonKeypoint`;
   `cv::Mat` descriptor rows and `Image8U` descriptor rows both become
   32-byte arrays).
4. **Compare** (`compareImplementations`):
   - For each keypoint in `a`, greedy-nearest-neighbor search over `b`'s
     keypoints by Euclidean distance; a match exists if the nearest is within
     `spatialRadiusPx` (default 3 px). *Not* a mutual/one-to-one assignment —
     documented simplification, adequate for a diagnostic (see Trade-offs).
   - For each spatial match, compute descriptor Hamming distance
     (`popcount(desc_a[i] ^ desc_b[i])` summed over 32 bytes) and angle
     difference; aggregate mean/median/max Hamming and mean angle diff.
5. **Report** to stdout (plain text, no file output):

```
Image: pinball.jpg (640x480)
Params: nfeatures=500 scaleFactor=1.2 nlevels=8 edgeThreshold=31 wta_k=2 patchSize=31 fastThreshold=20

mine:   412 keypoints, mean 3.21 ms (stddev 0.18 ms, 10 runs)
opencv: 398 keypoints, mean 1.05 ms (stddev 0.06 ms, 10 runs)

Spatial match (mine -> opencv, radius <= 3px): 351/412 (85.2%)
Descriptor Hamming distance (matched pairs, out of 256 bits): mean 14.2, median 12.0, max 63
Angle difference (matched pairs): mean 2.1 deg
```

## Error handling

- Image fails to load (stb_image returns null) → print a clear error, exit 1.
- `BUILD_OPENCV_COMPARISON=ON` with OpenCV not found → surfaced by CMake's own
  `find_package(OpenCV REQUIRED)` failure; no extra handling needed (per
  brainstorming decision log).

## Trade-offs / what to revisit later

| Choice | Revisit if |
|---|---|
| Greedy nearest-neighbor matching (not mutual) | This tool starts driving real correctness claims rather than a diagnostic — switch to an optimal assignment (e.g. Hungarian) |
| Console-only output | Need to track these numbers over time/commits — add CSV/JSON output |
| Single image, manual run | Need broader coverage — add a directory-of-images loop and aggregated stats |
| Not run in CI | OpenCV becomes cheap/available in CI images — could add as a non-blocking informational job |
