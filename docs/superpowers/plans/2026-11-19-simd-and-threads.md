# SIMD and Multithreading (#19) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ORB::detectAndCompute` faster on x86-64, ARM64 and wasm32 with explicit SIMD in the three hot kernels and an opt-in thread pool, without changing a single output byte and without new dependencies.

**Architecture:** Each kernel keeps its scalar code as the reference and gains a vector path behind one internal header (`simd_utils.hpp`) that wraps SSE2, NEON and WASM SIMD128 with the same ten operations. Both paths are always compiled into the library, selected per call by a `useSimd` flag whose default is the build's capability, so the test suite compares them byte for byte in one binary. Threads come last: a small fixed `ThreadPool` splits rows into contiguous bands and merges per-band results in band order, so keypoint order is the scalar order. One task is one PR into `dev`, each with a measured before/after table.

**Tech Stack:** C++14, CMake >= 3.11, `<immintrin.h>` / `<arm_neon.h>` / `<wasm_simd128.h>`, `std::thread`, GitHub Actions (ubuntu x86-64, ubuntu arm64, windows, emsdk 4.0.19).

**Spec:** [issue #19](https://github.com/kalwalt/orbFromCV/issues/19), plus the measurements in the "Baseline" section below. Prerequisites #20 (FAST-9 early rejection), #25, #27 and #28 are merged on `dev` (`ed04c06`).

## Baseline

`profile_orb pinball.jpg 10` (1637x2048, 8 levels), ms per run. Native: WSL Ubuntu 24.04, g++ 13, Release. wasm32: emsdk 4.0.19, node 24, Release, flags of PR #57.

| stage | native | wasm32 | wasm32 `-msimd128` |
|---|---|---|---|
| pyramid | 18.7 | 36.5 | 35.9 |
| fast_detection | 90.1 | 107.2 | 89.6 |
| gaussian_blur | 24.6 | 132.7 | 47.1 |
| total | 135.7 | 280.1 | 176.1 |

The blur is float32 in OpenCV's evaluation order (#39), not the 16-bit integer kernel the issue text describes; `-ffp-contract=off` is what keeps it bit-identical. The native compiler already auto-vectorises the blur interior and part of the resize with SSE2; it vectorises nothing of FAST.

## Global Constraints

- No OpenCV, no external deps, no gtest/Catch2 (AGENTS.md). Threads are `std::thread` only.
- Identical output: same keypoints in the same order with the same fields, same descriptor bytes, for scalar vs vector and for 1 vs N threads, on every supported target (#19 "Requirements").
- Float arithmetic in the blur stays two roundings per tap: separate multiply and add in SIMD too, never `vfmaq_f32`/`vmlaq_f32`, `_mm_fmadd_ps` or `wasm_f32x4_relaxed_madd`; float-to-int converts truncate (`_mm_cvttps_epi32`, `vcvtq_s32_f32`, `wasm_i32x4_trunc_sat_f32x4`). Vector kernels live in the library `.cpp` files, where `-ffp-contract=off` applies; `simd_utils.hpp` holds only the wrappers.
- Vector loads never read past the row: a block loop over `cols - 16 + 1` (or `- 4 + 1` for float) and a scalar tail. Tested on tight buffers under ASan.
- `simd_utils.hpp` is internal: not installed, not included by any public header.
- Branches `kalwalt/<type>/<slug>` from `origin/dev`, Conventional Commits, `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`, PR body ends with the Claude Code line, one PR per task, squash-merged by the maintainer. Every PR body carries a before/after table from `profile_orb` on native and wasm32 (same machine, same run count) and marks its task done in this file.
- A kernel's vector path is kept only if it beats the scalar path by at least 15% on wasm32 or native; otherwise the PR documents the measurement and ships the scalar restructure alone.
- Out of scope, stated so nobody adds them: AVX2, runtime CPU dispatch, `-fwasm-exceptions` (an open decision for the maintainer, not SIMD), a wasm pthreads CI job (see Task 5 step 10), per-level parallelism, resizing every level from level 0.

## Review Focus

1. Images whose width is not a multiple of 16 (or of 4 for the float blur), including widths below one vector: the vector path must give the scalar bytes and read nothing past the row end (Tasks 2, 3, 4 tight-buffer tests).
2. A build with `ORB_ENABLE_SIMD=ON` on a target where no vector path is active (e.g. a wrong MSVC macro) silently tests scalar against scalar: `orbSimdActive()` must be true on every CI SIMD build (Task 1 test).
3. More threads than rows, 1-row images, images below the FAST margin: bands of zero rows must be skipped and the output unchanged (Task 5 tests).
4. An exception in a worker (e.g. `std::bad_alloc`) must reach the caller of `detectAndCompute`, not `std::terminate` (Task 5 test).
5. A wasm32 build without pthreads must not try to start a thread when `setNumThreads(4)` is called (Task 5 clamp and test).

---

### Task 1: `ORB_ENABLE_SIMD` option, `-msimd128`, arm64 and SIMD-off CI

**Files:**
- Modify: `CMakeLists.txt` (option, compile definition, Emscripten flag)
- Create: `simd_utils.hpp` (detection macros only for now)
- Modify: `orb_core.hpp`, `orb_core.cpp` (`orbSimdActive()`)
- Modify: `.github/workflows/ci.yml` (matrix)
- Modify: `tests/test_orb_core.cpp`
- Modify: `README.md` (build options), `AGENTS.md` (Build section: the option and the detection rule)

**Interfaces:**
- Produces: `bool orbSimdActive();` declared in `orb_core.hpp`: true when the library was built with `ORB_ENABLE_SIMD` and a vector target is compiled in. `simd_utils.hpp` defines `ORB_SIMD_SSE2`, `ORB_SIMD_NEON`, `ORB_SIMD_WASM` (each 0/1) and `ORB_SIMD` (1 if any is 1 and `ORB_ENABLE_SIMD` is 1). CMake defines `ORB_ENABLE_SIMD=0|1` PRIVATE on `orb_lib`; with the option ON under Emscripten `orb_lib` gets `-msimd128` PRIVATE.

- [x] **Step 1: Write the failing test** in `tests/test_orb_core.cpp`:

```cpp
TEST_CASE("orbSimdActive: every CI build with ORB_ENABLE_SIMD=ON has a vector path") {
    // The test binary gets ORB_TEST_EXPECT_SIMD from tests/CMakeLists.txt (= the option).
#if ORB_TEST_EXPECT_SIMD
    CHECK(orbSimdActive());
#else
    CHECK(!orbSimdActive());
#endif
}
```

- [x] **Step 2: Run the suite to verify it fails to compile** (`orbSimdActive` undeclared).

- [x] **Step 3: Implement.** `CMakeLists.txt`: `option(ORB_ENABLE_SIMD "Use SSE2/NEON/WASM SIMD128 kernels" ON)`; `target_compile_definitions(orb_lib PRIVATE ORB_ENABLE_SIMD=$<BOOL:${ORB_ENABLE_SIMD}>)`; inside `if(EMSCRIPTEN)`: `if(ORB_ENABLE_SIMD) target_compile_options(orb_lib PRIVATE -msimd128) endif()` (after `add_library`). `tests/CMakeLists.txt`: `ORB_TEST_EXPECT_SIMD=$<BOOL:${ORB_ENABLE_SIMD}>`. `simd_utils.hpp`:

```cpp
#define ORB_SIMD_SSE2 (defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
#define ORB_SIMD_NEON (defined(__ARM_NEON) || defined(_M_ARM64))
#define ORB_SIMD_WASM defined(__wasm_simd128__)
#define ORB_SIMD (ORB_ENABLE_SIMD && (ORB_SIMD_SSE2 || ORB_SIMD_NEON || ORB_SIMD_WASM))
```

(write the `defined()` tests as `#if` blocks setting each macro to 0/1, since `defined` inside a macro body is undefined behaviour; this is the rule, not the code). `orbSimdActive()` returns `ORB_SIMD != 0`. MSVC on x64 has no `__SSE2__`, which is why `_M_X64` is listed.

- [x] **Step 4: Run the suite** on the local build (SIMD on): PASS. Configure a second build with `-DORB_ENABLE_SIMD=OFF`: PASS, and the `.wasm` built with `-DORB_ENABLE_SIMD=OFF` must not contain SIMD opcodes (`wasm-objdump -d build-wasm-off/orb_test.wasm | grep -c v128` prints 0, with `wasm-objdump` from emsdk's binaryen or wabt).

- [x] **Step 5: CI.** `build-and-test` matrix becomes `os: [ubuntu-latest, ubuntu-24.04-arm, windows-latest]` × `simd: [ON, OFF]` with `-DORB_ENABLE_SIMD=${{ matrix.simd }}` on the configure line; the consumer steps run as today. `sanitizers` stays x86-64 with SIMD ON. `wasm` job: add a second configure/build/test with `-DORB_ENABLE_SIMD=OFF` into `build-wasm-off`. If the `ubuntu-24.04-arm` label is refused for this repository, drop NEON from the plan: remove `ORB_SIMD_NEON` and say so in the PR.

- [x] **Step 6: Measure** `profile_orb` native and wasm32 with the option ON and OFF; the wasm ON column must match the `-msimd128` baseline above within noise.

- [x] **Step 7: Docs and commit.** README "Building" gets the option; AGENTS.md Build section gets the option and "MSVC needs `_M_X64`, not `__SSE2__`". Mark Task 1 done here.

```bash
git commit -m "build: ORB_ENABLE_SIMD option, -msimd128 for wasm32, arm64 and SIMD-off CI matrix"
```

---

### Task 2: Gaussian blur vector path

**Files:**
- Modify: `simd_utils.hpp` (float wrappers)
- Modify: `image_utils.hpp`, `image_utils.cpp`
- Modify: `tests/test_image_utils.cpp`

**Interfaces:**
- Consumes: `ORB_SIMD` from Task 1.
- Produces: `void gaussianBlur7x7(const ImageView8U& src, Image8U& dst, bool useSimd = orbSimdActive());` (the third parameter is new; `false` forces the scalar path even in a SIMD build). `simd_utils.hpp` namespace `simd`: `struct f32x4` with `load(const float*)`, `store(float*, f32x4)`, `splat(float)`, `mul`, `add`, `load_u8x4_as_f32x4(const uint8_t*)` (4 bytes zero-extended to floats), `trunc_to_i32x4(f32x4)` and `store_u8x4_saturated(uint8_t*, i32x4)` (clamp 0..255, 4 bytes); each implemented for SSE2, NEON and WASM under the Task 1 macros.

- [x] **Step 1: Write the failing tests** in `tests/test_image_utils.cpp`:

```cpp
TEST_CASE("gaussianBlur7x7: vector path equals the scalar path byte for byte") {
    // Widths around every vector boundary, heights that give 1-row and multi-row column passes.
    const int widths[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 40, 63, 64, 65, 100};
    const int heights[] = {1, 2, 3, 4, 7, 8, 13};
    for (int w : widths) for (int h : heights) {
        Image8U src = pseudoRandomImage(w, h, 777u + w * 31 + h);
        Image8U scalar, vector;
        gaussianBlur7x7(src, scalar, false);
        gaussianBlur7x7(src, vector, true);
        CHECK_EQ(vector.data, scalar.data);
    }
}

TEST_CASE("gaussianBlur7x7: vector path reads nothing past a tight row") {
    // Heap buffer sized exactly cols*rows, step == cols: ASan sees any overread.
    for (int w = 7; w <= 40; ++w) {
        std::vector<uint8_t> buf(static_cast<size_t>(w) * 5, 90);
        ImageView8U view(buf.data(), w, 5);
        Image8U out;
        gaussianBlur7x7(view, out, true);
        CHECK_EQ(out.cols, w);
    }
    std::vector<uint8_t> big(static_cast<size_t>(16385) * 3, 90);
    Image8U out;
    gaussianBlur7x7(ImageView8U(big.data(), 16385, 3), out, true);
    CHECK_EQ(out.cols, 16385);
}
```

- [x] **Step 2: Run the suite to verify it fails** (3-argument call does not compile).

- [x] **Step 3: Implement.** Add the parameter and branch on `ORB_SIMD && useSimd` in two places of `gaussianBlur7x7`: the row-pass interior (`x` from `kRadius` while `x + 4 <= cols - kRadius`: seven `load_u8x4_as_f32x4(p + k)`, `mul` tap 0 then `add(mul(tap k, v), acc)` for k = 1..6 in that order, `store`) and the column pass (`x` while `x + 4 <= cols`: `mul(k3, r3)`, then for pairs (2,4), (1,5), (0,6): `add(mul(kernel, add(r_a, r_b)), acc)`; round with `add(acc, splat(12582912.0f))`, `trunc_to_i32x4`, subtract 12582912 in integer lanes, `store_u8x4_saturated`). The scalar loops run on the tail. Border strips stay scalar.

- [x] **Step 4: Run the suite** on the SIMD build, the SIMD-off build, the ASan build and the wasm build: all PASS, and the existing OpenCV blur fixtures (`test_opencv_ref.cpp`) still pass, which proves bit-identity with OpenCV, not only with the scalar path.

- [x] **Step 5: Measure** `profile_orb` before/after, native and wasm32. Apply the 15% rule; the wasm32 blur must at least match the 47.1 ms of auto-vectorisation.

- [x] **Step 6: Commit** `perf(blur): SSE2/NEON/WASM vector path for the float row and column passes`, body with the table and `Refs #19`. Mark Task 2 done.

---

### Task 3: Resize: cache the horizontal pass, vectorise the vertical pass

**Files:**
- Modify: `simd_utils.hpp` (integer wrappers)
- Modify: `image_utils.hpp`, `image_utils.cpp`
- Modify: `tests/test_image_utils.cpp`

**Interfaces:**
- Consumes: `ORB_SIMD`, `f32x4` layout conventions from Tasks 1-2.
- Produces: `void resizeBilinear(const ImageView8U& src, Image8U& dst, int new_width, int new_height, bool useSimd = orbSimdActive());`. `simd_utils.hpp` gains `struct i32x4` with `load`, `store`, `splat(int)`, `mul` (low 32 bits), `add`, `shr_arith(i32x4, 16)`, `load_u16x4_as_i32x4(const uint16_t*)`, and reuses `store_u8x4_saturated`.

- [ ] **Step 1: Write the failing tests:**

```cpp
TEST_CASE("resizeBilinear: vector path equals the scalar path byte for byte") {
    const int srcSizes[][2] = {{1, 1}, {2, 3}, {5, 4}, {17, 9}, {64, 48}, {97, 71}};
    const int dstSizes[][2] = {{1, 1}, {3, 2}, {4, 5}, {13, 7}, {53, 40}, {81, 59}, {120, 90}};
    for (const auto& s : srcSizes) for (const auto& d : dstSizes) {
        Image8U src = pseudoRandomImage(s[0], s[1], 4242u + s[0] + d[0]);
        Image8U scalar, vector;
        resizeBilinear(src, scalar, d[0], d[1], false);
        resizeBilinear(src, vector, d[0], d[1], true);
        CHECK_EQ(vector.data, scalar.data);
    }
}

TEST_CASE("resizeBilinear: vector path reads nothing past a tight row") {
    for (int w = 7; w <= 40; ++w) {
        std::vector<uint8_t> buf(static_cast<size_t>(w) * 6, 120);
        Image8U out;
        resizeBilinear(ImageView8U(buf.data(), w, 6), out, w * 5 / 6, 5, true);
        CHECK_EQ(out.rows, 5);
    }
}
```

- [ ] **Step 2: Run the suite to verify it fails** (5-argument call).

- [ ] **Step 3: Restructure the scalar code, no SIMD yet.** Keep a two-row cache of the horizontal pass: `std::vector<uint16_t> hrow[2]` of `new_width` entries, each holding `a0 * row[x0] + a1 * row[x1]` for one source row (max 65280 fits), and the index of the source row it holds; a destination row needs source rows `y0` and `min(y0 + 1, rows - 1)`, which is computed once per source row instead of once per destination row that touches it. The vertical pass reads the two cached rows: `(b0 * h0 + b1 * h1 + (1 << 15)) >> 16`, clamped. Run the suite: the existing `resizeBilinear` tests and the OpenCV pyramid fixtures must still pass; commit `perf(pyramid): compute the horizontal resize pass once per source row`.

- [ ] **Step 4: Vectorise the vertical pass.** Under `ORB_SIMD && useSimd`, `col` while `col + 4 <= new_width`: `load_u16x4_as_i32x4` of both cached rows, `add(add(mul(splat(b0), h0), mul(splat(b1), h1)), splat(1 << 15))`, `shr_arith(…, 16)`, `store_u8x4_saturated`. Scalar tail. The horizontal pass stays scalar: its source index `x0` is a gather, which SSE2 cannot do.

- [ ] **Step 5: Run the suite** on SIMD, SIMD-off, ASan and wasm builds: PASS, pyramid fixtures included.

- [ ] **Step 6: Measure** and apply the 15% rule separately to step 3 and step 4 (two tables in the PR body).

- [ ] **Step 7: Commit** `perf(pyramid): vector path for the vertical resize pass`, `Refs #19`. Mark Task 3 done.

---

### Task 4: FAST segment test on 16 pixels at a time

**Files:**
- Modify: `simd_utils.hpp` (byte wrappers)
- Modify: `fast_detector.hpp`, `fast_detector.cpp`
- Modify: `tests/test_fast_detector.cpp`

**Interfaces:**
- Consumes: `ORB_SIMD`.
- Produces: `void detectFAST(const ImageView8U& img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmaxSuppression = true, bool useSimd = orbSimdActive());`. `simd_utils.hpp` gains `struct u8x16` with `load(const uint8_t*)`, `splat(uint8_t)`, `adds` and `subs` (saturating), `cmpgt` (unsigned, all-ones lanes), `and_`, `or_`, `andnot`, `any(u8x16)` (true if any lane is non-zero) and `bitmask(u8x16)` (bit i set when lane i's top bit is set; SSE2 `_mm_movemask_epi8`, WASM `wasm_i8x16_bitmask`, NEON via the shift-and-narrow idiom).

- [ ] **Step 1: Write the failing tests:**

```cpp
TEST_CASE("detectFAST: vector path equals the scalar path, keypoints in the same order") {
    for (uint32_t seed = 1; seed <= 8; ++seed) {
        // Widths across the 16-lane boundary, plus the oracle images of the FAST-9 test.
        const int w = 7 + static_cast<int>(seed) * 11, h = 9 + static_cast<int>(seed) * 5;
        Image8U img = seededNoise(w, h, seed, 90, 60);
        for (bool nms : {false, true}) {
            std::vector<KeyPoint> scalar, vector;
            detectFAST(img, scalar, 20, nms, false);
            detectFAST(img, vector, 20, nms, true);
            CHECK_EQ(vector.size(), scalar.size());
            for (size_t i = 0; i < scalar.size() && i < vector.size(); ++i) {
                CHECK_EQ(vector[i].pt.x, scalar[i].pt.x);
                CHECK_EQ(vector[i].pt.y, scalar[i].pt.y);
                CHECK_EQ(vector[i].response, scalar[i].response);
            }
        }
    }
}

TEST_CASE("detectFAST: vector path reads nothing past a tight row") {
    for (int w = 7; w <= 40; ++w) {
        std::vector<uint8_t> buf(static_cast<size_t>(w) * 9, 50);
        buf[4 * w + w / 2] = 220; // one dot, so the candidate path runs
        std::vector<KeyPoint> kps;
        detectFAST(ImageView8U(buf.data(), w, 9), kps, 20, true, true);
        CHECK_EQ(kps.size(), static_cast<size_t>(w >= 7 ? 1 : 0));
    }
}
```

- [ ] **Step 2: Run the suite to verify it fails** (5-argument call).

- [ ] **Step 3: Implement** the OpenCV `FAST_t<16>` scheme in the row loop under `ORB_SIMD && useSimd`, for `x` while `x + 16 <= cols - margin`:
  1. `v = load(row + x)`, `lo = subs(v, splat(t))`, `hi = adds(v, splat(t))`.
  2. Coarse rejection with the pair rule the scalar code already uses (a 9-arc covers one pixel of every opposite pair i, i+8): for each of the 8 pairs, `bright = or_(cmpgt(px(i), hi), cmpgt(px(i + 8), hi))`, `dark = or_(cmpgt(lo, px(i)), cmpgt(lo, px(i + 8)))`, `maybeBright = and_(maybeBright, bright)`, `maybeDark = and_(maybeDark, dark)`; then `if (!any(or_(maybeBright, maybeDark))) continue;`.
  3. For surviving blocks, count consecutive brighter and darker pixels around the circle (k from 0 to 24, lanes with `cmpgt(px(k), hi)` / `cmpgt(lo, px(k))`), saturating byte counters via `adds` and `and_` with the lane mask, reset to 0 by `andnot`; a lane is a candidate when its counter reaches 9.
  4. `bitmask` of candidate lanes; for each set bit in ascending order run the existing scalar full test and `cornerScore` on that pixel (so the response and the raw order are the scalar ones).
  The scalar loop handles the remaining `x` up to `cols - margin`. NMS is unchanged.

- [ ] **Step 4: Run the suite** on SIMD, SIMD-off, ASan and wasm builds: PASS, including `detectFAST without NMS matches a brute-force FAST-9 oracle`.

- [ ] **Step 5: Measure**; FAST is the largest stage on every target and the compiler vectorises none of it, so this is the PR where the 15% rule should not bite.

- [ ] **Step 6: Commit** `perf(fast): 16-pixel SIMD segment test before the scalar full test`, `Refs #19`. Mark Task 4 done.

---

### Task 5: Thread pool and row-band parallelism

**Files:**
- Create: `thread_pool.hpp`, `thread_pool.cpp`
- Modify: `orb.hpp`, `orb.cpp`, `fast_detector.hpp/.cpp`, `image_utils.hpp/.cpp`
- Modify: `CMakeLists.txt`, `cmake/StandaloneORBConfig.cmake.in`
- Modify: `tests/CMakeLists.txt`, `tests/test_orb.cpp`, create `tests/test_thread_pool.cpp`
- Modify: `.github/workflows/ci.yml` (`tsan` job), `README.md`, `AGENTS.md`, `tools/profile_orb.cpp` (third argument: threads)

**Interfaces:**
- Consumes: the three kernels with their `useSimd` parameter.
- Produces, in `thread_pool.hpp`:

```cpp
class ThreadPool {
public:
    explicit ThreadPool(int workers);           // workers >= 1 std::threads, started here, joined in ~ThreadPool
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    int workers() const;
    // Splits [begin, end) into workers() + 1 contiguous bands of nearly equal
    // size (the caller runs one), skips empty bands, calls
    // body(bandIndex, bandBegin, bandEnd) and returns when every band is done.
    // An exception thrown by a band is rethrown here (first one wins).
    // One parallelFor at a time per pool: calls are serialised by a mutex.
    void parallelFor(int begin, int end, const std::function<void(int, int, int)>& body);
};
```

  In `orb.hpp`: `int setNumThreads(int n);` (stores `max(1, n)`, creates or drops the pool, returns the effective count: 1 on Emscripten without `__EMSCRIPTEN_PTHREADS__`) and `int numThreads() const;`. `ORB` holds `std::shared_ptr<ThreadPool>`, so copies share the pool and stay copyable; default is 1 thread and no pool. The kernels gain a trailing `ThreadPool* pool = nullptr`: `resizeBilinear(…, bool useSimd, ThreadPool* pool)`, `gaussianBlur7x7(…, bool useSimd, ThreadPool* pool)`, `detectFAST(…, bool useSimd, ThreadPool* pool)`.

- [ ] **Step 1: Write the failing tests** in `tests/test_thread_pool.cpp`:

```cpp
TEST_CASE("ThreadPool::parallelFor: covers the range once, in band order, with empty bands skipped") {
    ThreadPool pool(3);
    for (int n : {0, 1, 2, 3, 4, 5, 17, 100}) {
        std::vector<int> hits(n, 0);
        std::vector<int> bandOf(n, -1);
        pool.parallelFor(0, n, [&](int band, int b, int e) {
            CHECK(b < e);
            for (int i = b; i < e; ++i) { ++hits[i]; bandOf[i] = band; }
        });
        for (int i = 0; i < n; ++i) CHECK_EQ(hits[i], 1);
        for (int i = 1; i < n; ++i) CHECK(bandOf[i] >= bandOf[i - 1]);
    }
}

TEST_CASE("ThreadPool::parallelFor: an exception in a band reaches the caller") {
    ThreadPool pool(2);
    bool caught = false;
    try {
        pool.parallelFor(0, 8, [](int, int b, int) { if (b > 0) throw std::runtime_error("band"); });
    } catch (const std::runtime_error&) { caught = true; }
    CHECK(caught);
    int sum = 0;  // the pool is still usable afterwards
    pool.parallelFor(0, 8, [&](int, int b, int e) { sum += e - b; });
    CHECK_EQ(sum, 8);
}
```

  and in `tests/test_orb.cpp`:

```cpp
TEST_CASE("detectAndCompute: 1 and N threads give identical keypoints and descriptors") {
    // The 640x480 fixture scene plus odd-sized noise; thread counts above the row count too.
    // Scene: readPGM(fixturePath(opencv_ref::SCENE_FILE)) as test_opencv_ref.cpp's loadScene does
    // (tests/pgm.hpp, tests/fixture_scene.hpp). Noise: the LCG of test_fast_detector.cpp's seededNoise,
    // copied into this file's anonymous namespace (the test TUs share no helper header).
    Image8U scene = readPGM(fixturePath(opencv_ref::SCENE_FILE));
    std::vector<Image8U> images = {scene, noise(97, 71, 3u), noise(33, 1, 4u), noise(9, 9, 5u), noise(640, 2, 6u)};
    for (const Image8U& img : images) {
        ORB single;
        std::vector<KeyPoint> k1; Image8U d1;
        single.detectAndCompute(img, k1, d1);
        for (int n : {2, 3, 7, 16}) {
            ORB multi;
            CHECK_EQ(multi.setNumThreads(n), multi.numThreads());
            std::vector<KeyPoint> kn; Image8U dn;
            multi.detectAndCompute(img, kn, dn);
            CHECK(sameOutput(k1, d1, kn, dn));   // existing helper in this file: order, pt, angle, response, octave, bytes; add kp.size to it
        }
    }
}

TEST_CASE("setNumThreads: clamps to 1 for n <= 0 and reports the effective count") {
    ORB orb;
    CHECK_EQ(orb.setNumThreads(0), 1);
    CHECK_EQ(orb.setNumThreads(-3), 1);
    CHECK_EQ(orb.numThreads(), 1);
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    CHECK_EQ(orb.setNumThreads(4), 1);
#else
    CHECK_EQ(orb.setNumThreads(4), 4);
#endif
}
```

- [ ] **Step 2: Run the suite to verify it fails** (`thread_pool.hpp` missing; `setNumThreads` undeclared).

- [ ] **Step 3: Implement `ThreadPool`** in `thread_pool.cpp`: `workers` threads waiting on a condition variable for a job `{body, begin, end, bandCount}`, a counter of finished bands, an `std::exception_ptr` slot set by the first failing band, rethrown by `parallelFor` after every band returned. The caller thread runs band 0. Band bounds: `begin + (end - begin) * i / bands` to `begin + (end - begin) * (i + 1) / bands`, `bands = workers + 1`. `CMakeLists.txt`: `find_package(Threads REQUIRED)` and `target_link_libraries(orb_lib PUBLIC Threads::Threads)`; `cmake/StandaloneORBConfig.cmake.in`: `include(CMakeFindDependencyMacro)` and `find_dependency(Threads)` before the targets file. `tests/CMakeLists.txt` adds `test_thread_pool.cpp`.

- [ ] **Step 4: Run `test_thread_pool` cases: PASS.** Commit `feat(core): fixed ThreadPool with ordered row bands`.

- [ ] **Step 5: Parallelise the kernels** with the `pool` parameter, `nullptr` meaning the existing sequential loop:
  - `gaussianBlur7x7`: `parallelFor(0, rows, …)` over the row pass (writes its own rows of `temp`), then a second `parallelFor(0, rows, …)` over the column pass (reads up to 3 rows outside the band, all of `temp` is complete by then).
  - `resizeBilinear`: `parallelFor(0, new_height, …)`; each band owns its own horizontal-pass cache from Task 3 (a band's first row recomputes both source rows).
  - `detectFAST`: `parallelFor(margin, rows - margin, …)`, each band pushes its raw keypoints into `std::vector<std::vector<KeyPoint>> perBand[bandIndex]` and writes its own rows of `response_grid`; after the call the bands are concatenated in index order into `raw_keypoints`; NMS runs as today on the merged vector (its 3x3 window reads `response_grid`, which is complete).
  - `orb.cpp`: `setNumThreads` / `numThreads`; `detectAndCompute` passes `pool.get()` to the three kernels. Harris, ICAngles and descriptors stay sequential (together under 5% of the time).

- [ ] **Step 6: Run the suite** natively (SIMD on and off), under ASan, and in the wasm build (where `setNumThreads(4)` returns 1 and the pool is never created): PASS.

- [ ] **Step 7: Run the suite under ThreadSanitizer** locally: `cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++ "-DCMAKE_CXX_FLAGS=-fsanitize=thread" "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread"` then `ctest --test-dir build-tsan --output-on-failure`: no report.

- [ ] **Step 8: Package check.** Install to a prefix and build/run `tests/consumer` as CI does (`find_dependency(Threads)` must make `StandaloneORB::orb` link); on WSL also build the consumer via `add_subdirectory`.

- [ ] **Step 9: CI.** Add job `tsan` (ubuntu, clang, the flags of step 7, `ctest`). `profile_orb` takes `[image] [repeats] [threads]` and calls `setNumThreads`.

- [ ] **Step 10: Measure** `profile_orb pinball.jpg 10 1` vs `… 4` and `… 8` on native, with the per-stage table in the PR. wasm32 stays at 1 thread; a wasm pthreads build (`-pthread`, `PTHREAD_POOL_SIZE`, node flags) is a follow-up issue to open from this PR, not part of it.

- [ ] **Step 11: Docs and commit.** README: `setNumThreads`, the thread-count argument of `profile_orb`, the determinism guarantee; AGENTS.md Tests section: the `tsan` job and "kernels take `useSimd` and `pool`, tests compare paths in one binary". Commit `feat(orb): opt-in multithreading over row bands with identical output`, `Fixes #19`. Mark Task 5 done and Task 12 of the 2026-11-18 roadmap done.

---

## Decisions taken (defaults; override before executing)

| Point | Decision | Why |
|---|---|---|
| Path selection | Compile-time macros plus a per-call `useSimd` flag, both paths always built | The byte-for-byte test needs both in one binary; CI's SIMD-off matrix leg builds without intrinsics. |
| Thread API | `setNumThreads(int)` setter, default 1, pool shared by copies | Keeps the 9-argument constructor and existing timings unchanged. |
| Parallel unit | Row bands inside each kernel, pyramid chain kept sequential | Level 0 is a third of the pixels, so per-level parallelism caps at 3x; bands keep the output identical. |
| NEON | In scope, tested on `ubuntu-24.04-arm` | Without a runner no identical-output claim is possible; dropped if the label is unavailable (Task 1 step 5). |
| wasm threads | Out of scope, follow-up issue | Needs its own flags and a node-side pool size; the clamp keeps the wasm job green. |
| `-fwasm-exceptions` | Not touched | Not SIMD; 5% and a smaller binary, maintainer's call. |
