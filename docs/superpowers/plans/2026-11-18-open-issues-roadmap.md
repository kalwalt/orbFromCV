# Open Issues Roadmap (#19–#31) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 11 open GitHub issues (#19–#28, #31), one PR per issue into `dev`, in the order the issues' own dependencies dictate.

**Architecture:** Small, independent PRs. Correctness and API fixes first (`orb.cpp`, `fast_detector.cpp`), then CI hardening (ASan/UBSan, WASM), then packaging, with SIMD/threads (#19) last because it must be built on the corrected FAST test and the fixed-point resize. Every output-changing PR reports `compare_orb` / `profile_orb` before/after.

**Tech Stack:** C++14, CMake >= 3.10, GitHub Actions, no deps beyond `stb_image.h`; `compare_orb` needs a local OpenCV (`BUILD_OPENCV_COMPARISON=ON`).

**Spec:** the issue bodies on GitHub (`gh issue view N`). Verified on 2026-11-18 against `origin/dev` (`6433cb1`): none of the 11 issues is fixed, no open PRs, no unmerged remote branch touches them.

## Global Constraints

- No OpenCV, no external deps, no gtest/Catch2 (AGENTS.md "Key Conventions").
- Declarations in `.hpp`, implementation in `.cpp`; `orb_pattern.hpp` is generated, never hand-edited.
- A test in `tests/` alongside every core-algorithm change.
- Branch names `kalwalt/<type>/<slug>`; every branch starts from `origin/dev`, **not** from the stale local `kalwalt/perf/fast-gaussian-blur` (its two commits are already on `dev` under other SHAs).
- From Task 1 onward, commit subjects use Conventional Commits; AI commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- `dev` → `main` release PRs use "Create a merge commit".

## Review Focus

1. Reusing output variables across `detectAndCompute` calls (empty image, flat image, bad `wta_k`) must leave `keypoints` empty and `descriptors.rows == 0` (Task 3).
2. Any `firstLevel != 0` must be rejected at construction instead of silently duplicating level 0 (Task 4).
3. `FAST_SCORE` with `nfeatures = N` on a textured image must return `<= N` keypoints (Task 5).
4. A corner whose 9-arc touches only two compass pixels (e.g. circle pixels 1..9) must be detected, for all 16 rotations (Task 6).
5. `wta_k` 3/4 descriptor bytes must be byte-identical before and after the `ntuples` change (Task 8).

## Decisions taken (defaults; override before executing)

| Issue | Decision | Why |
|---|---|---|
| #23 | **Option 2: reject `firstLevel != 0`** in the constructor | Smaller change; option 1 only matters for detection above native resolution. |
| #31 | Docs only now; CI enforcement of commit subjects is a follow-up | Keeps the "no new dependencies" spirit; can be added later as a regex step. |
| #28 | Project version → `0.1.2` (matches latest tag); view type named `ImageView8U` | Issue asks to align with release tags. |
| #25 item 3 | Done in Task 9 as its own PR, after #24 | Changes output; must be measured in isolation. |
| #19 | Gets its **own plan** once Tasks 6, 9 and 11 are merged | Four sub-PRs (blur SIMD, resize SIMD, FAST SIMD, threads); too large to plan blind. |

---

### Task 1: Conventional Commits in AGENTS.md and CONTRIBUTING.md (#31)

**Files:**
- Modify: `AGENTS.md` (new section after "Commit Attribution")
- Modify: `CONTRIBUTING.md:27-33` ("Pull requests") and add a "Commit messages" section
- Branch: `kalwalt/docs/conventional-commits`

- [ ] **Step 1: Add "Commit Messages" to `AGENTS.md`**: format, the type table from the issue (feat, fix, perf, refactor, test, docs, build, ci, chore), suggested scopes (`fast`, `blur`, `pyramid`, `harris`, `orientation`, `descriptors`, `orb`, `core`, `tools`, `tests`, `readme`, `agents`), `!` + `BREAKING CHANGE:` rule, `Fixes #N` footer, one full example commit.
- [ ] **Step 2: Add "Commit messages" and "Release merges" to `CONTRIBUTING.md`**: link to conventionalcommits.org and AGENTS.md; state that `dev` → `main` PRs use "Create a merge commit" (rebase-merge on #18 diverged the branches, fixed by #30).
- [ ] **Step 3: Verify**: `grep -n "Commit Messages" AGENTS.md` and `grep -n "merge commit" CONTRIBUTING.md` both match.
- [ ] **Step 4: Commit** `docs(agents): adopt Conventional Commits and document the release merge method` with `Fixes #31`. Open PR into `dev`.

### Task 1b: Descriptors sampled at the wrong location on octaves >= 1 (#34) — PR #36

Found by the first ASan run of Task 2. `detectAndCompute` rescaled keypoints to level-0 coordinates only at the end, but `computeOrbDescriptors` expects them already rescaled (OpenCV semantics), so every octave >= 1 descriptor was sampled at `pt / scale`. Fix: move the rescale between the orientation and descriptor stages; test rebuilds level 1 and recomputes the descriptors (0/136 identical before, 136/136 after). Blocks Task 2.

### Task 1c: Small `edgeThreshold` reads outside the level (#35) — PR #37, stacked on #36

No level padding here, unlike OpenCV, so `edgeThreshold < 22` let the BRIEF pattern (19 px), `ICAngles` (15 px) and Harris (3 px) read out of bounds. Fix: border = `max(edgeThreshold, ceil(patchSize / 2 * sqrt(2)), HARRIS_BLOCK_SIZE / 2)`, documented in `orb.hpp`. Padding like OpenCV is a possible follow-up. Blocks Task 2.

### Task 2: ASan/UBSan CI job and `ORB_SANITIZE` option (#26) — PR #38, stacked on #37

**Files:**
- Modify: `CMakeLists.txt` (add `option(ORB_SANITIZE ...)`)
- Modify: `.github/workflows/ci.yml` (new job `sanitizers`)
- Branch: `kalwalt/ci/asan-ubsan`

**Interfaces:**
- Produces: CMake option `ORB_SANITIZE` (BOOL, default OFF) adding `-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all` to compile and link flags of all targets (GCC/Clang only; MSVC gets `/fsanitize=address`).

- [ ] **Step 1: Add the option** in `CMakeLists.txt` with `add_compile_options` / `add_link_options` guarded by `if(ORB_SANITIZE)`.
- [ ] **Step 2: Add job `sanitizers`** on `ubuntu-latest`, `CC=clang CXX=clang++`, `cmake -B build -DCMAKE_BUILD_TYPE=Debug -DORB_SANITIZE=ON`, build, `ctest --test-dir build --output-on-failure`. Triggers: same `push`/`pull_request` as `build-and-test`.
- [ ] **Step 3: Verify locally (or via the PR's CI)** that the job is green on `dev`.
- [ ] **Step 4: Verify the acceptance criterion**: revert commit `19718c3` (the #16 fix) in a scratch branch, run the sanitizer build, expect `heap-buffer-overflow` from the 1–3 px blur test; discard the scratch branch.
- [ ] **Step 5: Commit** `ci: add an ASan/UBSan job and an ORB_SANITIZE CMake option` with `Fixes #26`. PR into `dev`.

### Task 2b: OpenCV 4.10.0 ground-truth fixtures (new issue, replaces "measure with compare_orb" as the parity check)

Adopts the strategy of jsfeatNext (`tests/fixtures/opencv/`, `scripts/gen-opencv-fixtures.py`) and purecv (`scripts/opencv_ref/gen_orb_descriptors.js`, `src/features2d/orb_opencv_ref.rs`): reference values computed **once, by hand**, with a **pinned OpenCV 4.10.0** (`opencv-contrib-python==4.10.0.84`, IPP off), committed to the repo, and asserted by the normal test suite. CI never needs OpenCV. `compare_orb` stays as the manual timing/diagnostic tool; it is no longer the way parity is measured.

**Why 4.10.0:** it is the single reference both sister projects use (purecv `cpp_ref/` at commit `71d3237a`, jsfeatNext fixtures), so the whole WebARKit chain compares against one OpenCV build.

**Files:**
- Create: `scripts/gen-opencv-fixtures.py` (Python >= 3.11, stdlib + the pinned wheel; toolchain guard copied from jsfeatNext: refuse unless the installed wheel is exactly the pin and no other `cv2` distribution is installed; `cv2.ipp.setUseIPP(False)`)
- Create: `scripts/requirements-fixtures.txt` (`opencv-contrib-python==4.10.0.84`, `numpy==2.4.6`)
- Create: `tests/fixtures/opencv/scene-640x480.pgm` (binary P5; dumped once from a seeded C++ generator in `tests/`, same trust-chain argument as jsfeatNext: Python reads the bytes, never re-implements the RNG)
- Create: `tests/fixtures/opencv/opencv_ref.hpp` — **@generated** C++ header (no JSON parser in C++14 with zero deps; same choice as purecv's generated `.rs`). Contains `constexpr` meta (`OPENCV_REF_VERSION = "4.10.0"`, wheel, `Version control` lines of `getBuildInformation()`, `ipp = false`, scene sha256) and the payload arrays below.
- Create: `tests/fixtures/opencv/README.md` (why, which OpenCV, how to regenerate on Windows/Linux, "a fixture diff is a change of ground truth")
- Create: `tests/test_opencv_ref.cpp` (+ `tests/CMakeLists.txt` entry); tiny PGM P5 reader lives here
- Modify: `AGENTS.md` and `CONTRIBUTING.md`: "OpenCV reference version" section (pinned 4.10.0 links only: `docs.opencv.org/4.10.0/...`, `github.com/opencv/opencv/blob/4.10.0/...#L<n>`)
- Branch: `kalwalt/test/opencv-ground-truth-fixtures`

**Payloads (one generator function each, `--only <name>`):**

| name | OpenCV call | assertion in C++ | pins issue |
|---|---|---|---|
| `blur` | `cv2.GaussianBlur(scene, (7,7), 2, 2, BORDER_REFLECT_101)` on the 640x480 scene and on 1x1, 2x5, 3x3 crops | `gaussianBlur7x7` byte-identical | regression for #13/#15 |
| `fast` | `cv2.FAST(scene, 20, nonmaxSuppression=False, TYPE_9_16)` → sorted (x,y) | `detectFAST(nms=false)` exact set | #20 |
| `fast-nms` | same with `nonmaxSuppression=True` → (x,y,response) | exact set and `response == cornerScore` | #24 (score definition) |
| `orb-compute` | `cv2.ORB_create().compute(scene, fixed 40 keypoints with angles)` (purecv's recipe) | descriptor rows byte-identical | orientation + BRIEF stage |
| `orb-detect` | `cv2.ORB_create(500).detectAndCompute(scene)` → keypoints (x,y,size,angle,response,octave) + descriptors | **tolerance** check: spatial match rate within 2 px >= a threshold recorded in the test (raised as #20/#21/#24/#25 land); count == 500 | #20 #21 #24 #25 end-to-end |
| `resize` | `cv2.resize(scene, (533,400), INTER_LINEAR)` | `max |diff| <= 1` today; becomes the oracle for the fixed-point resize in Task 9 | #25 item 3 |

- [ ] **Step 1: Seeded scene generator** in `tests/`: `Image8U cornerScene(int w, int h, uint32_t seed)` (integer-only LCG + rectangles, like purecv's `orb_ref_image()`); dump `scene-640x480.pgm` once with a tiny `tests/dump_scene.cpp` target; test `scene: committed PGM equals cornerScene(640,480)` byte-for-byte.
- [ ] **Step 2: Generator script** with the toolchain guard, `--only`, deterministic output (re-run leaves `git status` clean), writes `opencv_ref.hpp` with a `// @generated by scripts/gen-opencv-fixtures.py -- DO NOT EDIT` banner.
- [ ] **Step 3: Pinned-version test**: `test_opencv_ref.cpp` asserts `OPENCV_REF_VERSION == "4.10.0"` and `OPENCV_REF_IPP == false`; a second test greps `AGENTS.md`, `CONTRIBUTING.md`, `README.md`, `*.cpp`, `*.hpp` for `docs.opencv.org/` or `github.com/opencv/opencv/blob/` URLs that are not followed by `4.10.0/` (same rule as purecv's `opencv_reference_links.rs`).
- [ ] **Step 4: Write the six comparison tests** from the table; run → `blur`, `orb-compute` expected PASS; `fast`, `fast-nms` expected FAIL today (document the counts in the PR, they are the #20/#24 evidence); `orb-detect` PASS with the measured baseline threshold; `resize` PASS or FAIL (record which).
- [ ] **Step 5:** tests that fail today are marked expected-to-fail with the issue number in `test_framework.hpp` style (or kept out of `ctest` behind a `ORB_RUN_PARITY_XFAIL` env var), so CI is green and Tasks 6/7 flip them to hard assertions.
- [ ] **Step 6: Commit** `test(parity): add OpenCV 4.10.0 ground-truth fixtures and a pinned-version guard` with `Fixes #<new issue>`. PR into `dev`.

**Effect on later tasks:** Tasks 6, 7 and 9 replace their "measure with `compare_orb`" steps with "flip the `fast` / `fast-nms` / `resize` fixture tests to hard assertions and raise the `orb-detect` match-rate threshold"; `profile_orb` timings stay.

**Done (issue #32).** Measured on `dev` with the fixtures: FAST without NMS finds 28,946 of OpenCV's 47,487 corners (#20); with NMS 9,960 corners in common and 0 identical scores (#24); end-to-end spatial match rate 8.8% (44/500), 0 matches on octaves 4–7. The fixtures also found three new parity issues, all in `v0.1.3`:
- **#39** `gaussianBlur7x7` differs from the blur `cv::ORB` applies (float separable filter, single rounding, because ORB blurs a submatrix and misses OpenCV's bit-exact 8U path) on 93% of pixels by 1–2 levels; descriptors on OpenCV's blurred level match 39/40, on ours 15/40.
- **#40** `resizeBilinear` is corner-aligned; `cv::resize INTER_LINEAR_EXACT` is pixel-centre aligned: max diff 215 levels, every level >= 1 shifted (0 matches on octaves 4–7; octaves 1–3 match at about octave 0's rate, so #20/#24 dominate there). It is Task 9 done against this oracle (supersedes "`max |diff| <= 1` vs the float version").
- **#41** `stdRound` = `lround` (half away from zero) vs `cvRound` (half to even): 1-bit descriptor differences at 30°, 60°, …; one-line fix (`std::lrint`).

Suggested order after this: #41 (trivial), then #20 and #24 as planned (they drive the octave-0 rate), with #39 and #40 alongside Task 9.

**Status after #41 (PR #43), #20 (#44), #24 (#45), #40 (#47 by jamicaw56), #22 (#46 by jamicaw56) and #39:** `ORB::detectAndCompute` matches `cv::ORB` on the #32 scene for all 500 keypoints on all 8 octaves, 491 of them at the identical float position with a byte-identical descriptor, mean Hamming 0.0; every stage fixture is a hard assertion, no `CHECK_XFAIL` left. #39 found that OpenCV's own blur output depends on its SIMD/FMA dispatch at rounding ties (1 pixel of the scene); the fixtures are now generated with the dispatch off (`cv2.setUseOptimized(False)`), which is what opencv.js computes, and `gaussianBlur7x7` is plain float32 in OpenCV's evaluation order, about 2.5x slower than the old integer kernel (12 → 25–33 ms on pinball.jpg) until the SIMD work in #19.

### Task 3: Fully define outputs on early return + constructor validation (#22)

**Files:**
- Modify: `orb.cpp:53-58` (constructor), `orb.cpp:64-68`, `orb.cpp:152`, `orb.cpp:214-216`
- Modify: `orb.hpp` (document thrown exception)
- Test: `tests/test_orb.cpp`
- Branch: `kalwalt/fix/stale-outputs-and-param-validation`

**Interfaces:**
- Produces: `ORB::ORB(...)` throws `std::invalid_argument` when `nfeatures < 0`, `scaleFactor <= 1`, `nlevels < 1`, `edgeThreshold <= 0`, `patchSize <= 0`, `wta_k` not in {2,3,4}. (Task 4 adds `firstLevel != 0` to the same check.)
- Produces: `detectAndCompute` calls `keypoints.clear()` and `descriptors = Image8U()` before any `return`.

- [ ] **Step 1: Write failing tests** in `tests/test_orb.cpp`:
  - `detectAndCompute: outputs are cleared on empty input after a previous call` — run on `pinball.jpg`-like textured 128x128 noise, then on `Image8U()`, assert `k.empty() && d.rows == 0`.
  - `detectAndCompute: outputs are cleared when no keypoints are found` — second call on a constant image, same assertions.
  - `ORB: constructor rejects unsupported wta_k` — `wta_k = 5` throws `std::invalid_argument`; also `nlevels = 0`, `scaleFactor = 1.0f`.
- [ ] **Step 2: Run** `cmake --build build --target orb_tests && ctest --test-dir build -C Debug -R orb` → expect the new tests to FAIL.
- [ ] **Step 3: Implement**: move the clears to the top of `detectAndCompute`; validate in the constructor; delete the `wta_k` fallback `return` at `orb.cpp:214-216` (now unreachable).
- [ ] **Step 4: Run tests** → PASS. Run the full suite to confirm no regressions.
- [ ] **Step 5: Commit** `fix(orb): clear outputs on every early return and validate constructor parameters` with `Fixes #22`. PR into `dev`.

### Task 4: Reject `firstLevel != 0` (#23)

**Files:**
- Modify: `orb.cpp` constructor validation (from Task 3), `orb.hpp` doc comment
- Test: `tests/test_orb.cpp`
- Branch: `kalwalt/fix/reject-first-level`

- [ ] **Step 1: Write failing test** `ORB: constructor rejects firstLevel != 0` — `ORB(500, 1.2f, 8, 31, 1)` throws `std::invalid_argument`; `firstLevel = 0` does not.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** the check; document in `orb.hpp` that only `firstLevel == 0` is supported.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `fix(orb): reject firstLevel != 0, which was never supported` with `Fixes #23`. PR into `dev`.

### Task 5: `retainBest` helper and cap for `FAST_SCORE` (#21)

**Files:**
- Modify: `orb.cpp:127-141`
- Create: `static void retainBest(std::vector<KeyPoint>& kps, int n)` in `orb.cpp` (sort by `response` descending, stable, resize to `n`)
- Test: `tests/test_orb.cpp`
- Branch: `kalwalt/fix/fast-score-nfeatures-cap`

**Interfaces:**
- Produces: `retainBest(std::vector<KeyPoint>&, int)` reused by Task 7.

- [ ] **Step 1: Write failing test** `detectAndCompute: FAST_SCORE returns at most nfeatures keypoints` — textured 256x256 image, `nfeatures = 100`, `FAST_SCORE`; assert `k.size() <= 100` and that every kept keypoint's response is >= the max response of the dropped ones at the same octave (compute via `detectFAST` directly).
- [ ] **Step 2: Run** → FAIL (currently returns thousands).
- [ ] **Step 3: Implement** `retainBest` and call it for both score types (Harris path keeps its current behavior for now).
- [ ] **Step 4: Run** → PASS; full suite green.
- [ ] **Step 5: Measure** `profile_orb pinball.jpg 10` with `FAST_SCORE` before/after; put the table in the PR.
- [ ] **Step 6: Commit** `fix(orb): cap FAST_SCORE keypoints at nfeatures per level` with `Fixes #21`. PR into `dev`.

### Task 6: FAST-9 early rejection (#20)

**Files:**
- Modify: `fast_detector.cpp:57-74`
- Test: `tests/test_fast_detector.cpp`
- Branch: `kalwalt/fix/fast9-early-rejection`

- [ ] **Step 1: Write a brute-force FAST-9 oracle** in the test file: `static bool isFast9Corner(const Image8U&, int x, int y, int t)` checking all 16 start positions for 9 contiguous brighter or darker pixels.
- [ ] **Step 2: Write failing tests**:
  - `detectFAST: detects a 9-arc covering only two compass pixels` — 7x7 image value 100, circle pixels `1..9` set to 200, threshold 20, `nms=false` → exactly 1 keypoint at (3,3); repeat for all 16 rotations.
  - `detectFAST(nms=false) matches a brute-force FAST-9 oracle` — 5 seeded random 64x64 images plus a checkerboard; compare the keypoint set to the oracle exactly.
- [ ] **Step 3: Run** → FAIL on the arc test (0 keypoints).
- [ ] **Step 4: Implement** OpenCV's opposite-pairs rejection: build `tab`-style bright/dark flags for pixels 0,8 then AND with (2,10), (4,12), (6,14); also the odd pairs; reject when neither bright nor dark survives.
- [ ] **Step 5: Run** → PASS; full suite green.
- [ ] **Step 6: Measure**: `compare_orb` spatial match rate and `profile_orb` before/after; expect more corners and higher `fast_detection` time; table in the PR.
- [ ] **Step 7: Commit** `fix(fast): use a FAST-9 early rejection instead of the FAST-12 compass test` with `Fixes #20`. PR into `dev`.

### Task 7: OpenCV keypoint selection: `cornerScore` and Harris pre-filter (#24)

**Files:**
- Modify: `fast_detector.cpp:14-27` (`cornerScore`), `orb.cpp` Harris block
- Test: `tests/test_fast_detector.cpp`, `tests/test_orb.cpp`
- Branch: `kalwalt/fix/opencv-keypoint-selection`
- Depends on Tasks 5 and 6.

- [ ] **Step 1: Write failing test** `cornerScore: equals the largest threshold at which the pixel is still a FAST-9 corner` — oracle: loop `t` from 0 to 255 using `isFast9Corner` from Task 6, on random images; expose `cornerScore` through `fast_detector.hpp` (or test via NMS ordering if the maintainer prefers it to stay internal).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** OpenCV's `cornerScore<16>` (binary search / min-max over 9-arcs) and fix the comment.
- [ ] **Step 4: Add** `retainBest(filteredKeypoints, 2 * featuresNum)` before `HarrisResponses` in `orb.cpp`; test `detectAndCompute: Harris runs only on the best 2*featuresNum FAST corners` by asserting no returned keypoint has a FAST response below the 2N-th best FAST response at its level.
- [ ] **Step 5: Run** → PASS; full suite green.
- [ ] **Step 6: Measure** `compare_orb` match rate before/after; table in the PR.
- [ ] **Step 7: Commit** `fix(fast): use OpenCV's max-threshold corner score and Harris pre-filtering` with `Fixes #24`. PR into `dev`.

### Task 8: Dead `initializeOrbPattern`, `ntuples`, blur on empty levels (#25) — done

Items 1 and 2 of #25 plus the blur skip from item 3, in one PR on
`kalwalt/refactor/orb-pattern-cleanup`; the fixed-point resize of item 3 landed
earlier through PR #47 (#40), so Task 9 is folded in here.

- [x] Pinned the whole output (keypoints + descriptor bytes, FNV-1a digest) for `wta_k` 2, 3, 4 on a seeded texture and for an 8-level dot grid whose top levels have no keypoints; generated with the code before the change.
- [x] Deleted the undeclared `initializeOrbPattern` copy and `<random>` from `orb_core.cpp`; `ntuples = 32 * 4` in `orb.cpp` (OpenCV's `descriptorSize() * 4`).
- [x] `detectAndCompute` skips `gaussianBlur7x7` on levels without keypoints; the pyramid is built from the unblurred levels, so nothing else reads the blurred image.
- [x] Suite 67/67 and ASan/UBSan clean with the digests unchanged. Commit `refactor(orb): drop the dead pattern copy, generate 128 tuples, skip the blur on empty levels`, `Fixes #25`.

### Task 10: WebAssembly build and CI job (#27) — done

Branch `kalwalt/ci/wasm-build`.

- [x] `if(EMSCRIPTEN)` block in `CMakeLists.txt`: `-fexceptions` (constructor throws, tests catch), link `-sNODERAWFS=1 -sALLOW_MEMORY_GROWTH=1 -sEXIT_RUNTIME=1`. No toolchain file in the repo: the Emscripten one registers node as the ctest emulator.
- [x] CI job `wasm` (`mymindstorm/setup-emsdk@v14`, emsdk 4.0.19 as on the maintainer's machine): `emcmake cmake` Release, build, `ctest`, `orb_test.js` on `pinball.jpg`.
- [x] `size_t` → `int` casts in `orb.cpp` for the two `Image8U` constructions (MSVC C4267).
- [x] README "Building for WebAssembly".

### Task 11: CMake install/export and `ImageView8U` (#28) — done

Branch `kalwalt/feat/cmake-package-and-image-view`. Version set to `0.2.0`
(the upcoming tag, not `0.1.2` as decided above: v0.1.3 shipped meanwhile).

- [x] `ImageView8U` (pointer, cols, rows, step) in `orb_core.hpp`; `Image8U` converts implicitly; every read-only image parameter (`detectFAST`, `HarrisResponses`, `ICAngles`, `computeOrbDescriptors`, `resizeBilinear`, `gaussianBlur7x7`) takes the view. `detectAndCompute` reads level 0 through the view and blurs it into an owned image, so the caller's buffer is never written; empty view clears outputs, malformed view throws `std::invalid_argument`. FAST's response grid is sized by `cols`, not `step`.
- [x] Tests: padded stride (`step` 160 for 128 px, garbage in the padding) gives the `Image8U` result and leaves the buffer untouched; a sub-image view equals the cropped `Image8U`; empty/malformed views.
- [x] Packaging: `StandaloneORB::orb` alias, `EXPORT_NAME orb`, `-ffp-contract=off` on the target, `ORB_BUILD_TOOLS`/`ORB_BUILD_TESTS`/`ORB_INSTALL` (top-level only), install of lib + headers to `include/StandaloneORB/`, `StandaloneORBConfig.cmake` + version file (`SameMinorVersion`, CMake >= 3.11).
- [x] `tests/consumer/` external project; CI installs to a prefix and builds/runs it on Linux and Windows.
- [x] `main.cpp` and `profile_orb` wrap the `stbi_load` buffer in a view.

### Task 12: SIMD and multithreading (#19) — done

Prerequisites (Tasks 6, 9, 10, 11) merged. The plan is `docs/superpowers/plans/2026-11-19-simd-and-threads.md`: five PRs (`ORB_ENABLE_SIMD` + `-msimd128` + CI matrix, blur SIMD, resize SIMD, FAST SIMD, `std::thread` row bands with ordered merge), byte-for-byte scalar-vs-vector and 1-vs-N-thread tests, a `tsan` job. Executed as PRs #59 (option + CI), #60 (blur), #61 (resize), #62 (FAST) and the threads PR (plan Task 5, `Fixes #19`). wasm32 pyramid+FAST+blur went from 280 ms to about 90 ms single-threaded on pinball.jpg; a wasm pthreads build is a follow-up issue.

---

## Open prerequisites

- Task 2b needs, once, a Python >= 3.11 venv with `opencv-contrib-python==4.10.0.84` on the maintainer's machine. After that no task needs OpenCV: parity is checked by the committed fixtures in CI. `compare_orb` (local OpenCV, `-DBUILD_OPENCV_COMPARISON=ON`) remains optional, for timing comparisons only.
- A GitHub issue for Task 2b still has to be opened (not done in this session).
