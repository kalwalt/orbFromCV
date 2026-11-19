#!/usr/bin/env python3
"""Regenerate the OpenCV ground-truth fixtures in tests/fixtures/opencv/ (issue #32).

    python scripts/gen-opencv-fixtures.py [--only blur|fast|orb-compute|orb-detect|orb-detect-fast|resize]

Run by hand from the gitignored .venv; never by CI or CMake. CI asserts against
the committed output. See tests/fixtures/opencv/README.md.

Output (all deterministic: the same pinned toolchain and the same committed
scene produce the same bytes, so a re-run leaves `git status` clean):

  opencv_ref.hpp         @generated header: toolchain meta, small payloads
  blur-scene.pgm         the blur cv::ORB applies per level: sepFilter2D with getGaussianKernel(7, 2, CV_32F),
                         BORDER_REFLECT_101 (float path, single rounding, no FMA; see orb_blur())
  resize-533x400.pgm     resize(scene, (533, 400), INTER_LINEAR_EXACT), as ORB builds its pyramid
  fast-t20.bin           FAST(scene, 20, nonmaxSuppression=False, TYPE_9_16): uint16 LE (x, y) pairs
  fast-t20-nms.bin       FAST(scene, 20, nonmaxSuppression=True):  uint16 LE x, y + float32 LE response

The input image scene-640x480.pgm is dumped once by the C++ `dump_scene`
target from tests/fixture_scene.hpp and committed; this script only reads it,
so both sides see the same pixels without re-implementing the generator.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import struct
import sys
from collections import namedtuple
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REQUIREMENTS = REPO / "scripts" / "requirements-fixtures.txt"
FIXTURE_DIR = REPO / "tests" / "fixtures" / "opencv"
SCENE_NAME = "scene-640x480.pgm"
HEADER_NAME = "opencv_ref.hpp"

OPENCV_DIST = "opencv-contrib-python"
NUMPY_DIST = "numpy"
# Other distributions that also install a top-level ``cv2`` package. Next to
# the pinned wheel, which ``cv2`` is imported is undefined, so refuse to run.
OTHER_OPENCV_DISTS = ("opencv-python", "opencv-python-headless", "opencv-contrib-python-headless")

Toolchain = namedtuple("Toolchain", "cv2 np wheel build contrib_build")
Scene = namedtuple("Scene", "image sha256")


# ---------------------------------------------------------------------------
# Toolchain guard (same rules as jsfeatNext's scripts/gen-opencv-fixtures.py)
# ---------------------------------------------------------------------------


def parse_pin(requirements_text: str, dist: str) -> str:
    match = re.search(
        r"^\s*" + re.escape(dist) + r"\s*==\s*([0-9]+(?:\.[0-9]+)*)\s*(?:#.*)?$",
        requirements_text,
        re.MULTILINE,
    )
    if not match:
        sys.exit(f"error: {REQUIREMENTS.name} has no exact '{dist}==X.Y.Z' pin")
    return match.group(1)


def installed_version(dist: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def build_version(cv2, label: str) -> str:
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith(label):
            return line.strip()[len(label):].strip()
    sys.exit(f"error: cv2.getBuildInformation() has no {label!r} line")


def require_pinned_toolchain() -> Toolchain:
    """Import cv2 and numpy, exiting non-zero unless they match the pins exactly."""
    requirements = REQUIREMENTS.read_text(encoding="utf-8")
    cv2_pin = parse_pin(requirements, OPENCV_DIST)
    np_pin = parse_pin(requirements, NUMPY_DIST)

    # The wheel itself, to all four parts: cv2.__version__ is only A.B.C, so
    # another build of the same release would pass the module check alone.
    wheel = installed_version(OPENCV_DIST)
    if wheel != cv2_pin:
        sys.exit(
            f"error: OpenCV wheel mismatch: {REQUIREMENTS.name} pins {OPENCV_DIST}=={cv2_pin}, "
            f"but the installed {OPENCV_DIST} is {wheel or 'not installed'}.\n"
            f"Refusing to generate fixtures with the wrong ground truth. "
            f"Reinstall: python -m pip install -r scripts/{REQUIREMENTS.name}"
        )
    others = [f"{d}=={installed_version(d)}" for d in OTHER_OPENCV_DISTS if installed_version(d)]
    if others:
        sys.exit(
            f"error: another OpenCV distribution is installed next to {OPENCV_DIST}=={cv2_pin}: "
            f"{', '.join(others)}. Each provides cv2 and which one is imported is undefined. "
            f"Uninstall it: python -m pip uninstall {' '.join(d.split('==')[0] for d in others)}"
        )

    try:
        import cv2
        import numpy as np
    except ImportError as err:
        sys.exit(f"error: cannot import {err.name}: {err}\nInstall: python -m pip install -r scripts/{REQUIREMENTS.name}")

    wanted = ".".join(cv2_pin.split(".")[:3])
    if cv2.__version__ != wanted:
        sys.exit(f"error: cv2.__version__ is {cv2.__version__!r}, the pin {cv2_pin} needs {wanted!r}")
    if np.__version__ != np_pin:
        sys.exit(f"error: numpy is {np.__version__}, {REQUIREMENTS.name} pins {np_pin}")

    # Generic CPU path only: IPP can take a different code path for some
    # functions, and the SIMD dispatch uses fused multiply-add on x86 CPUs that
    # have it, which moves pixels sitting on a rounding tie by one gray level
    # (1 pixel of the 640x480 scene). With the dispatch off the wheel computes
    # exactly what opencv.js (WebAssembly, no FMA) computes, which is the
    # reference this port reproduces bit for bit (#39).
    cv2.ipp.setUseIPP(False)
    if cv2.ipp.useIPP():
        sys.exit("error: could not disable Intel IPP")
    cv2.setUseOptimized(False)
    if cv2.useOptimized():
        sys.exit("error: could not disable the SIMD dispatch (cv2.setUseOptimized(False))")

    return Toolchain(cv2, np, f"{OPENCV_DIST}=={wheel}",
                     build_version(cv2, "Version control:"),
                     build_version(cv2, "Version control (extra):"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_scene(tc: Toolchain) -> Scene:
    path = FIXTURE_DIR / SCENE_NAME
    if not path.is_file():
        sys.exit(f"error: {path} is missing. Build and run the dump_scene target first:\n"
                 f"  cmake --build build --target dump_scene && build/tests/dump_scene {path}")
    image = tc.cv2.imread(str(path), tc.cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (480, 640):
        sys.exit(f"error: cv2.imread could not read a 640x480 image from {path}")
    return Scene(image, hashlib.sha256(path.read_bytes()).hexdigest())


def write_pgm(path: Path, image) -> None:
    h, w = image.shape
    path.write_bytes(f"P5\n{w} {h}\n255\n".encode("ascii") + image.tobytes())


def write_bytes_if_changed(path: Path, data: bytes) -> None:
    if path.is_file() and path.read_bytes() == data:
        return
    path.write_bytes(data)


def c_float(v: float) -> str:
    """Shortest repr that round-trips a float32, as a C float literal."""
    import numpy as np

    s = repr(float(np.float32(v)))
    return s + "f"


def cpp_u8_array(name: str, rows, per_line: int = 16) -> list[str]:
    out = [f"static const uint8_t {name}[] = {{"]
    flat = [int(v) for row in rows for v in row]
    for i in range(0, len(flat), per_line):
        out.append("    " + ", ".join(f"0x{v:02x}" for v in flat[i:i + per_line]) + ",")
    out.append("};")
    return out


# ---------------------------------------------------------------------------
# Fixtures. Each returns header lines and may write side files.
# ---------------------------------------------------------------------------

# Crops whose dimensions are <= 3 px exercise the repeated REFLECT_101 rule (#15).
BLUR_CROPS = [(1, 1), (2, 5), (5, 3), (3, 3), (3, 8), (8, 3), (40, 3)]


def orb_blur(tc: Toolchain, image):
    """The blur cv::ORB applies to each pyramid level.

    ORB_Impl blurs every level in place as a *submatrix* of the padded pyramid
    buffer (orb.cpp#L1230, 4.10.0). cv::GaussianBlur takes its 8-bit bit-exact
    fixed-point path only `if (sdepth == CV_8U && ((borderType & BORDER_ISOLATED)
    || !_src.isSubmatrix()))` (smooth.dispatch.cpp#L657), so ORB gets the generic
    sepFilter2D path: float32 kernel getGaussianKernel(7, 2), float intermediate,
    one saturate_cast<uchar> (round half to even) at the end.
    Verified: this reproduces cv::ORB::compute() descriptors bit for bit, the
    standalone cv2.GaussianBlur on a plain array does not (17/40 rows).

    Computed with the SIMD dispatch off (see require_pinned_toolchain), i.e.
    plain float32 multiply-adds, which is also what opencv.js computes.
    """
    cv2 = tc.cv2
    k = cv2.getGaussianKernel(7, 2, cv2.CV_32F)
    return cv2.sepFilter2D(image, -1, k, k, borderType=cv2.BORDER_REFLECT_101)


def gen_blur(tc: Toolchain, scene: Scene) -> list[str]:
    cv2 = tc.cv2
    blurred = orb_blur(tc, scene.image)
    write_pgm(FIXTURE_DIR / "blur-scene.pgm", blurred)
    bitexact = cv2.GaussianBlur(scene.image, (7, 7), 2, sigmaY=2, borderType=cv2.BORDER_REFLECT_101)
    differing = int((bitexact != blurred).sum())
    out = ["// --- blur: the blur cv::ORB applies per level = sepFilter2D(scene, getGaussianKernel(7, 2, CV_32F)",
           "//     x2, BORDER_REFLECT_101), i.e. the float separable path with a single final rounding ---",
           "// Full image: blur-scene.pgm. Small crops of the scene's top-left corner, blurred",
           "// as standalone images, exercise dimensions <= 3 px (REFLECT_101 repeated, #15).",
           "// For the record: the 8-bit bit-exact cv2.GaussianBlur(scene, (7, 7), 2, 2, REFLECT_101)",
           f"// differs from this on {differing} of {blurred.size} pixels (by at most 1).",
           f"static const int BLUR_BITEXACT_DIFFERING_PIXELS = {differing};",
           f"static const int BLUR_CROP_COUNT = {len(BLUR_CROPS)};",
           "static const int BLUR_CROP_SIZE[][2] = {  // {cols, rows}",
           *[f"    {{{w}, {h}}}," for w, h in BLUR_CROPS],
           "};"]
    for w, h in BLUR_CROPS:
        crop = scene.image[0:h, 0:w].copy()
        b = orb_blur(tc, crop)
        out += cpp_u8_array(f"BLUR_CROP_{w}x{h}", b)
    out.append("static const uint8_t* const BLUR_CROP[] = {")
    out += [f"    BLUR_CROP_{w}x{h}," for w, h in BLUR_CROPS]
    out.append("};")
    return out


FAST_THRESHOLD = 20


def gen_fast(tc: Toolchain, scene: Scene) -> list[str]:
    cv2 = tc.cv2
    out = [f"// --- fast: cv2.FastFeatureDetector(threshold={FAST_THRESHOLD}, TYPE_9_16) on the scene ---",
           "// Keypoints in fast-t20.bin (uint16 LE x, y) and fast-t20-nms.bin (uint16 LE x, y, float32 LE response).",
           f"static const int FAST_THRESHOLD = {FAST_THRESHOLD};"]
    for nms, name in ((False, "fast-t20.bin"), (True, "fast-t20-nms.bin")):
        det = cv2.FastFeatureDetector_create(threshold=FAST_THRESHOLD, nonmaxSuppression=nms,
                                             type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
        kps = det.detect(scene.image, None)
        pts = sorted((int(round(k.pt[0])), int(round(k.pt[1])), float(k.response)) for k in kps)
        buf = bytearray()
        for x, y, r in pts:
            buf += struct.pack("<HH", x, y)
            if nms:
                buf += struct.pack("<f", r)
        write_bytes_if_changed(FIXTURE_DIR / name, bytes(buf))
        out.append(f"static const int FAST_COUNT{'_NMS' if nms else ''} = {len(pts)};")
    return out


ORB_COMPUTE_N = 40
ORB_COMPUTE_ANGLES = [0, 17.5, 90, 133.3, 270.9, 45, 180, 359.5, 222.2, 300.1,
                      11.25, 60, 120, 243.7, 315, 5.5, 72.8, 199.9, 30, 150]


def orb_compute_keypoints():
    # >= 40 px from every border, so no reader (pattern 19 px, ICAngles 15 px) leaves the image.
    return [(40 + (i * 37 + 11) % 560, 40 + (i * 59 + 23) % 400, ORB_COMPUTE_ANGLES[i % len(ORB_COMPUTE_ANGLES)])
            for i in range(ORB_COMPUTE_N)]


def gen_orb_compute(tc: Toolchain, scene: Scene) -> list[str]:
    cv2 = tc.cv2
    kps_in = orb_compute_keypoints()
    kps = [cv2.KeyPoint(float(x), float(y), 31.0, float(a), 0.0, 0, -1) for x, y, a in kps_in]
    orb = cv2.ORB_create(500)  # defaults: 1.2, 8 levels, edgeThreshold 31, WTA_K 2, HARRIS, patch 31, fast 20
    kps_out, desc = orb.compute(scene.image, kps)
    if len(kps_out) != len(kps) or desc.shape != (len(kps), 32):
        sys.exit(f"error: OpenCV changed the keypoint set: {len(kps_out)} kps, desc {desc.shape}")
    for k_in, k_out in zip(kps, kps_out):
        if k_in.pt != k_out.pt or tc.np.float32(k_in.angle) != tc.np.float32(k_out.angle) or k_out.octave != 0:
            sys.exit("error: OpenCV modified a keypoint passed to compute()")
    out = ["// --- orb-compute: cv2.ORB_create(500).compute(scene, keypoints) with fixed level-0 keypoints ---",
           "// (x, y, angle in degrees); size 31, octave 0. One descriptor row per keypoint.",
           f"static const int ORB_COMPUTE_COUNT = {len(kps_in)};",
           "static const float ORB_COMPUTE_KEYPOINT[][3] = {",
           *[f"    {{{x}.0f, {y}.0f, {c_float(a)}}}," for x, y, a in kps_in],
           "};"]
    out += cpp_u8_array("ORB_COMPUTE_DESCRIPTOR", desc)
    return out


def gen_orb_detect(tc: Toolchain, scene: Scene) -> list[str]:
    cv2 = tc.cv2
    orb = cv2.ORB_create(500)
    kps, desc = orb.detectAndCompute(scene.image, None)
    out = ["// --- orb-detect: cv2.ORB_create(500).detectAndCompute(scene) ---",
           "// (x, y, size, angle, response, octave) in OpenCV's output order; descriptors row-aligned.",
           f"static const int ORB_DETECT_COUNT = {len(kps)};",
           "static const float ORB_DETECT_KEYPOINT[][6] = {"]
    for k in kps:
        out.append(f"    {{{c_float(k.pt[0])}, {c_float(k.pt[1])}, {c_float(k.size)}, "
                   f"{c_float(k.angle)}, {c_float(k.response)}, {k.octave}.0f}},")
    out.append("};")
    out += cpp_u8_array("ORB_DETECT_DESCRIPTOR", desc)
    return out


def gen_orb_detect_fast(tc: Toolchain, scene: Scene) -> list[str]:
    cv2 = tc.cv2
    orb = cv2.ORB_create(500, scoreType=cv2.ORB_FAST_SCORE)
    kps, desc = orb.detectAndCompute(scene.image, None)
    out = ["// --- orb-detect-fast: cv2.ORB_create(500, scoreType=ORB_FAST_SCORE).detectAndCompute(scene) ---",
           "// Same layout as orb-detect; responses are FAST scores and each level keeps its best featuresNum (#21).",
           f"static const int ORB_DETECT_FAST_COUNT = {len(kps)};",
           "static const float ORB_DETECT_FAST_KEYPOINT[][6] = {"]
    for k in kps:
        out.append(f"    {{{c_float(k.pt[0])}, {c_float(k.pt[1])}, {c_float(k.size)}, "
                   f"{c_float(k.angle)}, {c_float(k.response)}, {k.octave}.0f}},")
    out.append("};")
    out += cpp_u8_array("ORB_DETECT_FAST_DESCRIPTOR", desc)
    return out


RESIZE_SIZE = (533, 400)


def gen_resize(tc: Toolchain, scene: Scene) -> list[str]:
    cv2 = tc.cv2
    # INTER_LINEAR_EXACT is what ORB_Impl uses to build its pyramid (orb.cpp#L1128, 4.10.0).
    r = cv2.resize(scene.image, RESIZE_SIZE, interpolation=cv2.INTER_LINEAR_EXACT)
    write_pgm(FIXTURE_DIR / "resize-533x400.pgm", r)
    return ["// --- resize: cv2.resize(scene, (533, 400), INTER_LINEAR_EXACT) -> resize-533x400.pgm ---",
            "//     (533x400 is pyramid level 1 of 640x480 at scaleFactor 1.2; ORB resizes with INTER_LINEAR_EXACT)",
            f"static const int RESIZE_COLS = {RESIZE_SIZE[0]};",
            f"static const int RESIZE_ROWS = {RESIZE_SIZE[1]};"]


GENERATORS = {
    "blur": gen_blur,
    "fast": gen_fast,
    "orb-compute": gen_orb_compute,
    "orb-detect": gen_orb_detect,
    "orb-detect-fast": gen_orb_detect_fast,
    "resize": gen_resize,
}


# ---------------------------------------------------------------------------
# Header assembly
# ---------------------------------------------------------------------------

SECTION_BEGIN = "// @section {name}"
SECTION_END = "// @endsection {name}"


def header_meta(tc: Toolchain, scene: Scene) -> list[str]:
    return [
        "// @generated by scripts/gen-opencv-fixtures.py -- DO NOT EDIT.",
        f"// Reference: OpenCV {tc.cv2.__version__} ({tc.wheel}), Intel IPP off, on {SCENE_NAME}.",
        "// See tests/fixtures/opencv/README.md. Regenerate from the pinned .venv:",
        "//   python scripts/gen-opencv-fixtures.py",
        "#ifndef OPENCV_REF_HPP",
        "#define OPENCV_REF_HPP",
        "",
        "#include <cstdint>",
        "",
        "namespace opencv_ref {",
        "",
        f'static const char* const OPENCV_VERSION = "{tc.cv2.__version__}";',
        f'static const char* const OPENCV_WHEEL = "{tc.wheel}";',
        f'static const char* const OPENCV_BUILD = "{tc.build}";          // Version control: line of getBuildInformation()',
        f'static const char* const OPENCV_CONTRIB_BUILD = "{tc.contrib_build}";  // Version control (extra):',
        "static const bool OPENCV_IPP = false;",
        "static const bool OPENCV_OPTIMIZED = false;  // cv2.setUseOptimized(False): generic scalar path, no FMA",
        f'static const char* const SCENE_FILE = "{SCENE_NAME}";',
        f'static const char* const SCENE_SHA256 = "{scene.sha256}";',
        "static const int SCENE_COLS = 640;",
        "static const int SCENE_ROWS = 480;",
        "",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=sorted(GENERATORS), help="regenerate one fixture (others are kept)")
    args = parser.parse_args()

    tc = require_pinned_toolchain()
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    scene = load_scene(tc)
    header_path = FIXTURE_DIR / HEADER_NAME

    # Keep the other sections verbatim when regenerating only one.
    existing: dict[str, list[str]] = {}
    if header_path.is_file():
        lines = header_path.read_text(encoding="utf-8").splitlines()
        for name in GENERATORS:
            b, e = SECTION_BEGIN.format(name=name), SECTION_END.format(name=name)
            if b in lines and e in lines:
                existing[name] = lines[lines.index(b) + 1:lines.index(e)]

    out = header_meta(tc, scene)
    for name, gen in GENERATORS.items():
        out.append(SECTION_BEGIN.format(name=name))
        if args.only and args.only != name and name in existing:
            out += existing[name]
        else:
            out += gen(tc, scene)
            print(f"generated {name}")
        out.append(SECTION_END.format(name=name))
        out.append("")
    out += ["} // namespace opencv_ref", "", "#endif // OPENCV_REF_HPP", ""]
    write_bytes_if_changed(header_path, "\n".join(out).encode("utf-8"))
    print(f"wrote {header_path.relative_to(REPO)} (OpenCV {tc.cv2.__version__}, {tc.wheel}, scene sha256 {scene.sha256[:12]}...)")


if __name__ == "__main__":
    main()
