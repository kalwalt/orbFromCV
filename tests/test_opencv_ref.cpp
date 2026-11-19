// Parity against OpenCV 4.10.0 ground-truth fixtures (issue #32).
//
// The fixtures in tests/fixtures/opencv/ were produced once by
// scripts/gen-opencv-fixtures.py with the pinned opencv-contrib-python wheel
// and are committed; nothing here needs OpenCV. Checks that are known to fail
// until an open issue is fixed use CHECK_XFAIL(cond, "#N"): they are reported,
// not counted, and start failing ("XPASS") once the fix lands so the fixing PR
// turns them into hard assertions.
#include "fast_detector.hpp"
#include "fixture_scene.hpp"
#include "fixtures/opencv/opencv_ref.hpp"
#include "image_utils.hpp"
#include "orb.hpp"
#include "orb_core.hpp"
#include "orb_pattern.hpp"
#include "pgm.hpp"
#include "test_framework.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace {

std::string fixturePath(const char* name) {
    return std::string(ORB_FIXTURE_DIR) + "/" + name;
}

Image8U loadScene() {
    Image8U scene = readPGM(fixturePath(opencv_ref::SCENE_FILE));
    CHECK(scene.cols == opencv_ref::SCENE_COLS && scene.rows == opencv_ref::SCENE_ROWS);
    return scene;
}

std::vector<uint8_t> readBinary(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

uint16_t u16le(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
float f32le(const uint8_t* p) {
    uint32_t u = static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
                 (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

bool sameBytes(const Image8U& a, const Image8U& b, int* firstDiff = nullptr) {
    if (a.cols != b.cols || a.rows != b.rows) return false;
    for (int y = 0; y < a.rows; ++y)
        for (int x = 0; x < a.cols; ++x)
            if (a.at(y, x) != b.at(y, x)) {
                if (firstDiff) { firstDiff[0] = x; firstDiff[1] = y; }
                return false;
            }
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// The pin: one OpenCV build for every fixture, and the committed scene
// ---------------------------------------------------------------------------

TEST_CASE("opencv_ref: fixtures were generated with the pinned OpenCV 4.10.0, IPP off") {
    CHECK(std::string(opencv_ref::OPENCV_VERSION) == "4.10.0");
    CHECK(std::string(opencv_ref::OPENCV_WHEEL) == "opencv-contrib-python==4.10.0.84");
    CHECK(std::string(opencv_ref::OPENCV_BUILD) == "4.10.0");
    CHECK(std::string(opencv_ref::OPENCV_CONTRIB_BUILD) == "4.10.0");
    CHECK(!opencv_ref::OPENCV_IPP);
    CHECK(!opencv_ref::OPENCV_OPTIMIZED);
}

TEST_CASE("opencv_ref: the committed scene PGM is cornerScene(640, 480) byte for byte") {
    Image8U scene = loadScene();
    Image8U generated = cornerScene(opencv_ref::SCENE_COLS, opencv_ref::SCENE_ROWS);
    CHECK(sameBytes(scene, generated));
}

TEST_CASE("opencv_ref: every OpenCV link in docs and sources is pinned to 4.10.0") {
    // Same rule as purecv's opencv_reference_links test: a docs.opencv.org or
    // github.com/opencv/opencv{,_contrib}/{blob,tree,raw,blame}/ URL must name
    // the reference version, never 4.x, master or another release.
    const char* files[] = {
        "README.md", "AGENTS.md", "CONTRIBUTING.md", "NOTICE",
        "docs/design/compare-orb-design.md", "tests/fixtures/opencv/README.md",
        "orb.cpp", "orb.hpp", "orb_core.cpp", "orb_core.hpp", "fast_detector.cpp",
        "fast_detector.hpp", "image_utils.cpp", "image_utils.hpp",
        "tests/test_orb.cpp", "tests/test_orb_core.cpp", "tests/test_fast_detector.cpp",
        "tests/test_image_utils.cpp",
        "tools/compare_orb.cpp", "tools/profile_orb.cpp", "scripts/gen-opencv-fixtures.py",
    }; // not this file: it spells the bare prefixes below
    const char* prefixes[] = {
        "docs.opencv.org/",
        "github.com/opencv/opencv/blob/", "github.com/opencv/opencv/tree/",
        "github.com/opencv/opencv/raw/", "github.com/opencv/opencv/blame/",
        "github.com/opencv/opencv_contrib/blob/", "github.com/opencv/opencv_contrib/tree/",
        "raw.githubusercontent.com/opencv/opencv/",
    };
    int unpinned = 0;
    for (const char* rel : files) {
        std::ifstream in(std::string(ORB_SOURCE_DIR) + "/" + rel);
        if (!in) std::fprintf(stderr, "    cannot open %s\n", rel);
        CHECK(static_cast<bool>(in)); // a file that cannot be read must not pass silently
        std::string line;
        int lineNo = 0;
        while (std::getline(in, line)) {
            ++lineNo;
            for (const char* prefix : prefixes) {
                size_t pos = 0;
                while ((pos = line.find(prefix, pos)) != std::string::npos) {
                    pos += std::strlen(prefix);
                    if (line.compare(pos, 7, "4.10.0/") != 0) {
                        std::fprintf(stderr, "    unpinned OpenCV link: %s:%d: %s\n", rel, lineNo, line.c_str());
                        ++unpinned;
                    }
                }
            }
        }
    }
    CHECK_EQ(unpinned, 0);
}

// ---------------------------------------------------------------------------
// gaussianBlur7x7 vs cv2.GaussianBlur((7, 7), 2, 2, BORDER_REFLECT_101)
// ---------------------------------------------------------------------------

// blur-scene.pgm is the blur cv::ORB applies to each level: the float
// separable path with one final rounding, generic scalar arithmetic (no FMA,
// as opencv.js; see scripts/gen-opencv-fixtures.py, orb_blur()).
// gaussianBlur7x7 reproduces it bit for bit (#39).
TEST_CASE("opencv_ref: gaussianBlur7x7 equals the blur cv::ORB applies, byte for byte") {
    Image8U scene = loadScene();
    Image8U expected = readPGM(fixturePath("blur-scene.pgm"));
    Image8U actual;
    gaussianBlur7x7(scene, actual);
    CHECK(actual.cols == expected.cols && actual.rows == expected.rows);
    long differing = 0;
    int maxDiff = 0;
    for (int y = 0; y < expected.rows && y < actual.rows; ++y)
        for (int x = 0; x < expected.cols && x < actual.cols; ++x) {
            int d = std::abs(static_cast<int>(actual.at(y, x)) - static_cast<int>(expected.at(y, x)));
            if (d && differing < 5)
                std::fprintf(stderr, "    differs at (%d, %d): ours %d, OpenCV %d\n", x, y, actual.at(y, x), expected.at(y, x));
            differing += d != 0;
            maxDiff = std::max(maxDiff, d);
        }
    std::fprintf(stderr, "    differing pixels: %ld of %d, max |diff| = %d (OpenCV's own bit-exact 8U path differs on %d)\n",
                 differing, expected.cols * expected.rows, maxDiff, opencv_ref::BLUR_BITEXACT_DIFFERING_PIXELS);
    CHECK_EQ(maxDiff, 0);
    CHECK_EQ(differing, 0L);
}

TEST_CASE("opencv_ref: gaussianBlur7x7 equals the blur cv::ORB applies on 1-40 px crops") {
    Image8U scene = loadScene();
    bool allSame = true;
    for (int i = 0; i < opencv_ref::BLUR_CROP_COUNT; ++i) {
        const int w = opencv_ref::BLUR_CROP_SIZE[i][0], h = opencv_ref::BLUR_CROP_SIZE[i][1];
        Image8U crop(w, h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) crop.at(y, x) = scene.at(y, x);
        Image8U actual;
        gaussianBlur7x7(crop, actual);
        CHECK(actual.cols == w && actual.rows == h);
        int maxDiff = 0;
        for (int y = 0; y < h && y < actual.rows; ++y)
            for (int x = 0; x < w && x < actual.cols; ++x)
                maxDiff = std::max(maxDiff, std::abs(static_cast<int>(actual.at(y, x)) -
                                                     static_cast<int>(opencv_ref::BLUR_CROP[i][y * w + x])));
        if (maxDiff) std::fprintf(stderr, "    crop %dx%d differs, max |diff| = %d\n", w, h, maxDiff);
        CHECK_EQ(maxDiff, 0);
        allSame = allSame && maxDiff == 0;
    }
    CHECK(allSame);
}

// ---------------------------------------------------------------------------
// detectFAST vs cv2.FastFeatureDetector(20, TYPE_9_16)
// ---------------------------------------------------------------------------

namespace {

using XY = std::pair<int, int>;

std::set<XY> fastSet(const std::vector<KeyPoint>& kps) {
    std::set<XY> s;
    for (const auto& k : kps) s.insert({static_cast<int>(std::lround(k.pt.x)), static_cast<int>(std::lround(k.pt.y))});
    return s;
}

void reportSetDiff(const std::set<XY>& ours, const std::set<XY>& ref) {
    int missing = 0, extra = 0;
    for (const auto& p : ref) missing += ours.count(p) == 0;
    for (const auto& p : ours) extra += ref.count(p) == 0;
    std::fprintf(stderr, "    OpenCV: %zu corners, ours: %zu, missing here: %d, extra here: %d\n",
                 ref.size(), ours.size(), missing, extra);
}

} // namespace

TEST_CASE("opencv_ref: detectFAST without NMS finds the same corners as cv2.FAST") {
    Image8U scene = loadScene();
    std::vector<uint8_t> bin = readBinary(fixturePath("fast-t20.bin"));
    CHECK_EQ(static_cast<int>(bin.size() / 4), opencv_ref::FAST_COUNT);
    std::set<XY> ref;
    for (size_t i = 0; i + 4 <= bin.size(); i += 4) ref.insert({u16le(&bin[i]), u16le(&bin[i + 2])});

    std::vector<KeyPoint> kps;
    detectFAST(scene, kps, opencv_ref::FAST_THRESHOLD, false);
    std::set<XY> ours = fastSet(kps);
    reportSetDiff(ours, ref);
    CHECK(ours == ref);
}

TEST_CASE("opencv_ref: detectFAST with NMS matches cv2.FAST corners and scores") {
    Image8U scene = loadScene();
    std::vector<uint8_t> bin = readBinary(fixturePath("fast-t20-nms.bin"));
    CHECK_EQ(static_cast<int>(bin.size() / 8), opencv_ref::FAST_COUNT_NMS);
    std::set<XY> ref;
    std::vector<std::tuple<int, int, float>> refScored;
    for (size_t i = 0; i + 8 <= bin.size(); i += 8) {
        ref.insert({u16le(&bin[i]), u16le(&bin[i + 2])});
        refScored.emplace_back(u16le(&bin[i]), u16le(&bin[i + 2]), f32le(&bin[i + 4]));
    }

    std::vector<KeyPoint> kps;
    detectFAST(scene, kps, opencv_ref::FAST_THRESHOLD, true);
    std::set<XY> ours = fastSet(kps);
    reportSetDiff(ours, ref);
    CHECK(ours == ref);

    // Where both agree on a corner, the score should be OpenCV's max-threshold
    // score; this port computes a SAD-style sum instead (#24).
    int common = 0, sameScore = 0;
    std::set<std::tuple<int, int, float>> oursScored;
    for (const auto& k : kps)
        oursScored.insert(std::make_tuple(static_cast<int>(std::lround(k.pt.x)),
                                          static_cast<int>(std::lround(k.pt.y)), k.response));
    for (const auto& r : refScored) {
        if (ours.count({std::get<0>(r), std::get<1>(r)}) == 0) continue;
        ++common;
        sameScore += oursScored.count(r) > 0;
    }
    std::fprintf(stderr, "    common corners: %d, identical scores: %d\n", common, sameScore);
    CHECK(common > 0 && sameScore == common);
}

// ---------------------------------------------------------------------------
// computeOrbDescriptors vs cv2.ORB.compute() on fixed keypoints
// ---------------------------------------------------------------------------

// Two checks: described on OpenCV's own blurred level (blur-scene.pgm) and
// on our own blur, both must be exact.
TEST_CASE("opencv_ref: computeOrbDescriptors equals cv2.ORB.compute() on fixed level-0 keypoints") {
    Image8U scene = loadScene();
    Image8U blurred;
    gaussianBlur7x7(scene, blurred); // ORB blurs the level before describing it

    std::vector<KeyPoint> kps(opencv_ref::ORB_COMPUTE_COUNT);
    for (int i = 0; i < opencv_ref::ORB_COMPUTE_COUNT; ++i) {
        kps[i].pt = {opencv_ref::ORB_COMPUTE_KEYPOINT[i][0], opencv_ref::ORB_COMPUTE_KEYPOINT[i][1]};
        kps[i].angle = opencv_ref::ORB_COMPUTE_KEYPOINT[i][2];
        kps[i].size = 31.f;
        kps[i].octave = 0;
    }
    std::vector<Rect> layerInfo = {{0, 0, scene.cols, scene.rows}};
    std::vector<float> layerScale = {1.0f};
    const Point2i* pattern0 = reinterpret_cast<const Point2i*>(bit_pattern_31_);
    std::vector<Point2i> pattern(pattern0, pattern0 + 512);
    Image8U desc(32, opencv_ref::ORB_COMPUTE_COUNT);
    computeOrbDescriptors(blurred, layerInfo, layerScale, kps, desc, pattern, 32, 2);

    int identicalRows = 0;
    for (int i = 0; i < opencv_ref::ORB_COMPUTE_COUNT; ++i) {
        bool same = true;
        for (int k = 0; k < 32; ++k)
            same = same && desc.at(i, k) == opencv_ref::ORB_COMPUTE_DESCRIPTOR[i * 32 + k];
        identicalRows += same;
    }
    std::fprintf(stderr, "    identical descriptor rows on our blur: %d / %d\n", identicalRows, opencv_ref::ORB_COMPUTE_COUNT);
    CHECK_EQ(identicalRows, opencv_ref::ORB_COMPUTE_COUNT);

    // The same keypoints described on OpenCV's own blurred level isolate the
    // descriptor code from the blur: this must be exact.
    Image8U refBlurred = readPGM(fixturePath("blur-scene.pgm"));
    Image8U desc2(32, opencv_ref::ORB_COMPUTE_COUNT);
    computeOrbDescriptors(refBlurred, layerInfo, layerScale, kps, desc2, pattern, 32, 2);
    int identicalOnRefBlur = 0;
    for (int i = 0; i < opencv_ref::ORB_COMPUTE_COUNT; ++i) {
        int hamming = 0;
        for (int k = 0; k < 32; ++k) {
            uint8_t x = desc2.at(i, k) ^ opencv_ref::ORB_COMPUTE_DESCRIPTOR[i * 32 + k];
            while (x) { hamming += x & 1; x >>= 1; }
        }
        identicalOnRefBlur += hamming == 0;
        if (hamming) std::fprintf(stderr, "      row %2d angle %6.2f: %d bits differ\n", i, kps[i].angle, hamming);
    }
    std::fprintf(stderr, "    identical rows when described on OpenCV's blurred level: %d / %d\n",
                 identicalOnRefBlur, opencv_ref::ORB_COMPUTE_COUNT);
    CHECK_EQ(identicalOnRefBlur, opencv_ref::ORB_COMPUTE_COUNT);
}

// ---------------------------------------------------------------------------
// ORB::detectAndCompute vs cv2.ORB.detectAndCompute (end to end, tolerance)
// ---------------------------------------------------------------------------

TEST_CASE("opencv_ref: ORB::detectAndCompute keypoints spatially match cv2.ORB (end to end)") {
    Image8U scene = loadScene();
    ORB orb; // defaults, as cv2.ORB_create(500)
    std::vector<KeyPoint> kps;
    Image8U desc;
    orb.detectAndCompute(scene, kps, desc);
    CHECK_EQ(static_cast<int>(kps.size()), opencv_ref::ORB_DETECT_COUNT);

    // For every OpenCV keypoint, is there one of ours on the same octave within
    // 2 px? The rate is the end-to-end parity figure; it rises as #20, #21,
    // #24 and #25 land, and the floor below is raised with them.
    int matched = 0;
    long hammingSum = 0;
    for (int i = 0; i < opencv_ref::ORB_DETECT_COUNT; ++i) {
        const float* r = opencv_ref::ORB_DETECT_KEYPOINT[i];
        int best = -1;
        float bestD2 = 4.0f + 1e-3f;
        for (size_t j = 0; j < kps.size(); ++j) {
            if (kps[j].octave != static_cast<int>(r[5])) continue;
            float dx = kps[j].pt.x - r[0], dy = kps[j].pt.y - r[1];
            float d2 = dx * dx + dy * dy;
            if (d2 < bestD2) { bestD2 = d2; best = static_cast<int>(j); }
        }
        if (best < 0) continue;
        ++matched;
        for (int k = 0; k < 32; ++k) {
            uint8_t x = desc.at(best, k) ^ opencv_ref::ORB_DETECT_DESCRIPTOR[i * 32 + k];
            while (x) { hammingSum += x & 1; x >>= 1; }
        }
    }
    int identicalDesc = 0;
    for (int i = 0; i < opencv_ref::ORB_DETECT_COUNT; ++i) {
        const float* r = opencv_ref::ORB_DETECT_KEYPOINT[i];
        for (size_t j = 0; j < kps.size(); ++j) {
            if (kps[j].octave != static_cast<int>(r[5]) || kps[j].pt.x != r[0] || kps[j].pt.y != r[1]) continue;
            bool same = true;
            for (int k = 0; k < 32; ++k) same = same && desc.at(static_cast<int>(j), k) == opencv_ref::ORB_DETECT_DESCRIPTOR[i * 32 + k];
            identicalDesc += same;
            break;
        }
    }
    std::fprintf(stderr, "    keypoints at the exact position with a byte-identical descriptor: %d / %d\n",
                 identicalDesc, opencv_ref::ORB_DETECT_COUNT);
    const double rate = static_cast<double>(matched) / opencv_ref::ORB_DETECT_COUNT;
    std::fprintf(stderr, "    spatial match rate (same octave, <= 2 px): %.1f%% (%d / %d), mean Hamming of matches: %.1f bits\n",
                 100.0 * rate, matched, opencv_ref::ORB_DETECT_COUNT,
                 matched ? static_cast<double>(hammingSum) / matched : 0.0);

    // Diagnostics: per-octave counts and matches, and the rate ignoring octaves.
    int refPerOctave[8] = {0}, oursPerOctave[8] = {0}, matchedPerOctave[8] = {0}, anyOctave = 0;
    for (int i = 0; i < opencv_ref::ORB_DETECT_COUNT; ++i) {
        const float* r = opencv_ref::ORB_DETECT_KEYPOINT[i];
        int oct = std::min(7, std::max(0, static_cast<int>(r[5])));
        ++refPerOctave[oct];
        bool sameOct = false, any = false;
        for (const auto& k : kps) {
            float dx = k.pt.x - r[0], dy = k.pt.y - r[1];
            if (dx * dx + dy * dy > 4.0f) continue;
            any = true;
            if (k.octave == static_cast<int>(r[5])) sameOct = true;
        }
        matchedPerOctave[oct] += sameOct;
        anyOctave += any;
    }
    for (const auto& k : kps) ++oursPerOctave[std::min(7, std::max(0, k.octave))];
    std::fprintf(stderr, "    within 2 px on any octave: %d / %d\n", anyOctave, opencv_ref::ORB_DETECT_COUNT);
    for (int o = 0; o < 8; ++o)
        std::fprintf(stderr, "    octave %d: OpenCV %3d, ours %3d, matched %3d\n", o, refPerOctave[o], oursPerOctave[o], matchedPerOctave[o]);
    // Floor history: 8.8% on dev at #32; 9.2% after #20; 33.6% (168 / 500)
    // after #24 (all 109 octave-0 keypoints matched); 100% (500 / 500, mean
    // Hamming 0.6 bits) after #40 put every pyramid level on OpenCV's own
    // pixels. What is left is the +-1 blur difference (#39), which only flips
    // descriptor bits. Keep this floor at 95%: a few keypoints may legitimately
    // move by a pixel if the blur fix changes a Harris response at the cut.
    CHECK(rate >= 0.95);
}

// ---------------------------------------------------------------------------
// resizeBilinear vs cv2.resize(INTER_LINEAR)
// ---------------------------------------------------------------------------

// ORB builds its pyramid with INTER_LINEAR_EXACT. resizeBilinear reproduces
// that bit for bit: pixel-centre alignment, 8-bit fixed-point coefficients per
// pass, one rounding over 16 bits (#40), so this is a hard assertion and the
// CHECK_XFAIL that guarded it is gone.
TEST_CASE("opencv_ref: resizeBilinear is byte-identical to cv2.resize INTER_LINEAR_EXACT (#40)") {
    Image8U scene = loadScene();
    Image8U expected = readPGM(fixturePath("resize-533x400.pgm"));
    CHECK(expected.cols == opencv_ref::RESIZE_COLS && expected.rows == opencv_ref::RESIZE_ROWS);
    Image8U actual;
    resizeBilinear(scene, actual, opencv_ref::RESIZE_COLS, opencv_ref::RESIZE_ROWS);
    CHECK(actual.cols == expected.cols && actual.rows == expected.rows);

    int maxDiff = 0;
    long differing = 0;
    for (int y = 0; y < expected.rows && y < actual.rows; ++y)
        for (int x = 0; x < expected.cols && x < actual.cols; ++x) {
            int d = std::abs(static_cast<int>(actual.at(y, x)) - static_cast<int>(expected.at(y, x)));
            maxDiff = std::max(maxDiff, d);
            differing += d != 0;
        }
    std::fprintf(stderr, "    differing pixels: %ld of %d, max |diff| = %d\n",
                 differing, expected.cols * expected.rows, maxDiff);
    CHECK_EQ(maxDiff, 0);
    CHECK_EQ(differing, 0L);
}
