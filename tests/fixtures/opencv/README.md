# OpenCV ground-truth fixtures

Reference values computed by **OpenCV 4.10.0** once, by hand, and committed, so
the dependency-free test suite can assert this port against the implementation
it was ported from ([#32](https://github.com/kalwalt/orbFromCV/issues/32)). CI
runs against these files and never needs Python or OpenCV. The same approach is
used in [webarkit/jsfeatNext](https://github.com/webarkit/jsfeatNext)
(`tests/fixtures/opencv/`) and [webarkit/purecv](https://github.com/webarkit/purecv)
(`scripts/opencv_ref/`), against the same OpenCV build, so the whole WebARKit
chain compares against one reference.

## Files

| file | content | asserted by |
|---|---|---|
| `scene-640x480.pgm` | the input image, dumped once from `tests/fixture_scene.hpp` by the `dump_scene` target | `cornerScene(640, 480)` byte for byte |
| `opencv_ref.hpp` | **@generated** header: toolchain meta (version, wheel, build lines, IPP off, scene sha256) and the small payloads below | `tests/test_opencv_ref.cpp` |
| `blur-scene.pgm` | the blur `cv::ORB` applies to each level (see below) | `gaussianBlur7x7`, byte for byte (#39) |
| `fast-t20.bin` | `cv2.FAST(scene, 20, nonmaxSuppression=False, TYPE_9_16)`: uint16 LE `x, y` per corner, sorted | `detectFAST(nms=false)` exact set (#20) |
| `fast-t20-nms.bin` | same with NMS: uint16 LE `x, y` + float32 LE `response` | `detectFAST(nms=true)` exact set and scores (#20, #24) |
| in header | `cv2.ORB_create(500).compute()` on 40 fixed level-0 keypoints with angles | `computeOrbDescriptors` byte for byte |
| in header | `cv2.ORB_create(500).detectAndCompute(scene)`: 500 keypoints + descriptors | `ORB::detectAndCompute` spatial match rate (floor raised as issues close) |
| in header | `cv2.ORB_create(500, scoreType=ORB_FAST_SCORE).detectAndCompute(scene)`: 527 keypoints (ties at the cut kept) + descriptors | `ORB::detectAndCompute` with `FAST_SCORE`: count, positions, responses exact; descriptors (#49) |
| `resize-533x400.pgm` | `cv2.resize(scene, (533, 400), INTER_LINEAR_EXACT)` (level 1 at 1.2) | `resizeBilinear` within 1 gray level (#40) |

Checks that fail on purpose until an issue is fixed use `CHECK_XFAIL(cond, "#N")`
from `tests/test_framework.hpp`: reported, not counted, and failing as `XPASS`
once the condition holds, so the fixing PR turns them into hard assertions.

## Which OpenCV

`scripts/requirements-fixtures.txt` pins `opencv-contrib-python==4.10.0.84` and
`numpy==2.4.6`. The generator refuses to run unless the installed wheel is
exactly that version (all four parts, not only `cv2.__version__`) and no other
`cv2`-providing distribution is installed, and it switches Intel IPP and the
SIMD dispatch off (the reference is OpenCV's generic scalar path). `opencv_ref.hpp` records the wheel and
the `Version control` lines of `cv2.getBuildInformation()`; a test checks them
against the pin, and another test rejects OpenCV links in docs and sources that
are not pinned to `4.10.0`.

Bumping the pin means regenerating every fixture and reviewing the diffs as a
change of ground truth.

## What the blur fixture is

`cv::ORB` blurs every pyramid level in place as a submatrix of its padded buffer
([orb.cpp#L1230](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L1230)).
`cv::GaussianBlur` only takes its 8-bit bit-exact path when
`(borderType & BORDER_ISOLATED) || !_src.isSubmatrix()`
([smooth.dispatch.cpp#L657](https://github.com/opencv/opencv/blob/4.10.0/modules/imgproc/src/smooth.dispatch.cpp#L657)),
so ORB actually gets the generic float separable filter with a single final
rounding. `blur-scene.pgm` is that blur (`sepFilter2D` with
`getGaussianKernel(7, 2, CV_32F)`), which reproduces `cv::ORB::compute()`
descriptors bit for bit; the standalone `cv2.GaussianBlur` on a plain array
differs from it on 30,769 pixels by ±1 and does not.

The generator also switches the SIMD dispatch off (`cv2.setUseOptimized(False)`).
With it on, the x86 wheel uses fused multiply-adds, which move pixels whose exact
value sits on a rounding tie by one gray level (1 pixel of the scene, at
(556, 409)); with it off the wheel computes plain float32 multiply-adds, which is
exactly what opencv.js 4.10.0 (WebAssembly, no FMA) computes and what
`gaussianBlur7x7` reproduces bit for bit on every compiler. ORB's keypoints and
descriptors on the scene are identical either way.

## Regenerating

Python 3.11 or newer. The virtualenv lives in `.venv/` (gitignored).

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r scripts\requirements-fixtures.txt
.venv\Scripts\python scripts\gen-opencv-fixtures.py              # all fixtures
.venv\Scripts\python scripts\gen-opencv-fixtures.py --only fast  # one of blur|fast|orb-compute|orb-detect|resize
```

Linux / macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/requirements-fixtures.txt
.venv/bin/python scripts/gen-opencv-fixtures.py
```

Behind a TLS-intercepting proxy add `--use-feature=truststore` to the `pip
install`. Generation is deterministic: with the pinned toolchain and the
committed scene a re-run leaves `git status` clean (verified with the Windows
wheel; the Linux wheel of the same version is expected to agree, since IPP is
off and the fixtures record the build lines to tell otherwise), so a fixture
diff always means the ground truth changed.

The scene itself is regenerated only if `tests/fixture_scene.hpp` changes:

```bash
cmake --build build --target dump_scene
build/tests/dump_scene tests/fixtures/opencv/scene-640x480.pgm
```

then re-run the generator for every fixture.
