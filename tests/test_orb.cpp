#include "orb.hpp"
#include "orb_core.hpp"
#include "image_utils.hpp"
#include "orb_pattern.hpp"
#include "test_framework.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

// A grid of isolated bright dots on a dark background. Each dot's full 16-point
// FAST ring differs from the background, making it an unambiguous corner —
// unlike a checkerboard, whose grid intersections are saddle points that FAST
// (correctly) does not fire on.
Image8U makeDotGrid(int size, uint8_t low, uint8_t high) {
    Image8U img(size, size);
    for (auto& v : img.data) v = low;
    for (int y = 30; y < size - 30; y += 20)
        for (int x = 30; x < size - 30; x += 20)
            img.at(y, x) = high;
    return img;
}

} // namespace

TEST_CASE("ORB::detectAndCompute: empty image produces no keypoints") {
    Image8U image; // 0x0
    ORB orb;
    std::vector<KeyPoint> keypoints;
    Image8U descriptors;

    orb.detectAndCompute(image, keypoints, descriptors);

    CHECK(keypoints.empty());
}

TEST_CASE("ORB::detectAndCompute: flat image produces no keypoints") {
    Image8U image(64, 64);
    for (auto& v : image.data) v = 128;

    ORB orb(50, 1.2f, 3, 16, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    std::vector<KeyPoint> keypoints;
    Image8U descriptors;

    orb.detectAndCompute(image, keypoints, descriptors);

    CHECK(keypoints.empty());
}

TEST_CASE("ORB::detectAndCompute: dot grid yields keypoints with matching descriptors") {
    Image8U image = makeDotGrid(128, 30, 220);

    ORB orb(50, 1.2f, 1, 16, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    std::vector<KeyPoint> keypoints;
    Image8U descriptors;

    orb.detectAndCompute(image, keypoints, descriptors);

    CHECK(!keypoints.empty());
    CHECK_EQ(descriptors.rows, static_cast<int>(keypoints.size()));
    CHECK_EQ(descriptors.cols, 32); // dsize: 256-bit descriptor = 32 bytes

    for (const auto& kp : keypoints) {
        CHECK(kp.pt.x >= 0.0f && kp.pt.x < static_cast<float>(image.cols));
        CHECK(kp.pt.y >= 0.0f && kp.pt.y < static_cast<float>(image.rows));
    }
}

TEST_CASE("ORB::detectAndCompute: descriptors are not degenerate (all-zero)") {
    // Regression test: an aliased-call bug in gaussianBlur7x7 (fixed
    // alongside this test - see image_utils.cpp) silently zeroed the
    // pyramid image before descriptor sampling, producing an all-zero
    // descriptor for every single keypoint. None of the other
    // detectAndCompute tests inspect descriptor *content*, only shape, so
    // this slipped through undetected until compared against OpenCV
    // (issue #5).
    Image8U image = makeDotGrid(128, 30, 220);

    ORB orb(50, 1.2f, 1, 16, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    std::vector<KeyPoint> keypoints;
    Image8U descriptors;

    orb.detectAndCompute(image, keypoints, descriptors);

    CHECK(!keypoints.empty());
    for (int row = 0; row < descriptors.rows; ++row) {
        bool allZero = true;
        const uint8_t* desc = descriptors.ptr(row);
        for (int col = 0; col < descriptors.cols; ++col) {
            if (desc[col] != 0) {
                allZero = false;
                break;
            }
        }
        CHECK(!allZero);
    }
}

TEST_CASE("ORB::detectAndCompute: results are deterministic across runs") {
    Image8U image = makeDotGrid(128, 30, 220);

    ORB orbA(50, 1.2f, 1, 16, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    std::vector<KeyPoint> keypointsA;
    Image8U descriptorsA;
    orbA.detectAndCompute(image, keypointsA, descriptorsA);

    ORB orbB(50, 1.2f, 1, 16, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    std::vector<KeyPoint> keypointsB;
    Image8U descriptorsB;
    orbB.detectAndCompute(image, keypointsB, descriptorsB);

    CHECK_EQ(keypointsA.size(), keypointsB.size());
    CHECK_EQ(descriptorsA.data.size(), descriptorsB.data.size());
    for (size_t i = 0; i < descriptorsA.data.size(); ++i) {
        CHECK_EQ(descriptorsA.data[i], descriptorsB.data[i]);
    }
    for (size_t i = 0; i < keypointsA.size(); ++i) {
        CHECK_EQ(keypointsA[i].pt.x, keypointsB[i].pt.x);
        CHECK_EQ(keypointsA[i].pt.y, keypointsB[i].pt.y);
    }
}

TEST_CASE("ORB::detectAndCompute: nfeatures caps the number of keypoints returned") {
    Image8U image = makeDotGrid(128, 30, 220);

    ORB orb(5, 1.2f, 1, 16, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    std::vector<KeyPoint> keypoints;
    Image8U descriptors;

    orb.detectAndCompute(image, keypoints, descriptors);

    CHECK(keypoints.size() <= 5);
}

TEST_CASE("ORB::detectAndCompute: multi-level pyramid runs without crashing") {
    Image8U image = makeDotGrid(128, 30, 220);

    ORB orb(50, 1.2f, 4, 16, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    std::vector<KeyPoint> keypoints;
    Image8U descriptors;

    orb.detectAndCompute(image, keypoints, descriptors);

    CHECK_EQ(descriptors.rows, static_cast<int>(keypoints.size()));
    for (const auto& kp : keypoints) {
        CHECK(kp.octave >= 0 && kp.octave < 4);
    }
}

TEST_CASE("ORB::detectAndCompute: stage timings do not change keypoints or descriptors") {
    Image8U image = makeDotGrid(128, 30, 220);

    ORB orbA(50, 1.2f, 2, 16, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    std::vector<KeyPoint> keypointsA;
    Image8U descriptorsA;
    orbA.detectAndCompute(image, keypointsA, descriptorsA); // no profiling requested

    ORB orbB(50, 1.2f, 2, 16, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    std::vector<KeyPoint> keypointsB;
    Image8U descriptorsB;
    std::vector<StageTiming> stageTimings;
    orbB.detectAndCompute(image, keypointsB, descriptorsB, &stageTimings); // profiling requested

    CHECK_EQ(keypointsA.size(), keypointsB.size());
    CHECK_EQ(descriptorsA.data.size(), descriptorsB.data.size());
    for (size_t i = 0; i < descriptorsA.data.size(); ++i) {
        CHECK_EQ(descriptorsA.data[i], descriptorsB.data[i]);
    }
    for (size_t i = 0; i < keypointsA.size(); ++i) {
        CHECK_EQ(keypointsA[i].pt.x, keypointsB[i].pt.x);
        CHECK_EQ(keypointsA[i].pt.y, keypointsB[i].pt.y);
        CHECK_EQ(keypointsA[i].angle, keypointsB[i].angle);
    }
}

TEST_CASE("ORB::detectAndCompute: stage timings cover every pipeline stage with non-negative durations") {
    Image8U image = makeDotGrid(128, 30, 220);

    ORB orb(50, 1.2f, 2, 16, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    std::vector<KeyPoint> keypoints;
    Image8U descriptors;
    std::vector<StageTiming> stageTimings;

    orb.detectAndCompute(image, keypoints, descriptors, &stageTimings);

    CHECK(!keypoints.empty()); // otherwise only the first 3 stages would be recorded
    const std::vector<std::string> expectedStages = {
        "pyramid", "fast_detection", "harris_scoring", "orientation",
        "pattern_init", "gaussian_blur", "descriptors",
    };
    CHECK_EQ(stageTimings.size(), expectedStages.size());
    for (size_t i = 0; i < stageTimings.size() && i < expectedStages.size(); ++i) {
        CHECK_EQ(stageTimings[i].name, expectedStages[i]);
        CHECK(stageTimings[i].ms >= 0.0);
    }
}

TEST_CASE("ORB::detectAndCompute: stage timings on an empty image record nothing") {
    Image8U image; // 0x0
    ORB orb;
    std::vector<KeyPoint> keypoints;
    Image8U descriptors;
    std::vector<StageTiming> stageTimings;

    orb.detectAndCompute(image, keypoints, descriptors, &stageTimings);

    CHECK(stageTimings.empty());
}

// ---------------------------------------------------------------------------
// Descriptor location on pyramid levels >= 1 (issue #34)
// ---------------------------------------------------------------------------

namespace {

// Seeded texture with enough corners on every level.
Image8U makeTexture(int w, int h, uint32_t seed) {
    Image8U img(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            seed = seed * 1664525u + 1013904223u;
            img.at(y, x) = static_cast<uint8_t>((seed >> 24) ^ (((x / 16 + y / 16) & 1) ? 0x80 : 0));
        }
    return img;
}

} // namespace

TEST_CASE("ORB::detectAndCompute: octave-1 descriptors are sampled at the keypoint location") {
    const int W = 256, H = 256;
    Image8U image = makeTexture(W, H, 7);

    const float scaleFactor = 1.2f;
    ORB orb(300, scaleFactor, 2, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    std::vector<KeyPoint> keypoints;
    Image8U descriptors;
    orb.detectAndCompute(image, keypoints, descriptors);

    // Rebuild level 1 exactly as the pipeline does: resize from level 0, then blur.
    Image8U level1;
    resizeBilinear(image, level1,
                   static_cast<int>(std::round(W / scaleFactor)),
                   static_cast<int>(std::round(H / scaleFactor)));
    gaussianBlur7x7(level1, level1);

    std::vector<Rect> layerInfo = {{0, 0, W, H}, {0, 0, level1.cols, level1.rows}};
    std::vector<float> layerScale = {1.0f, scaleFactor};
    const Point2i* pattern0 = reinterpret_cast<const Point2i*>(bit_pattern_31_);
    std::vector<Point2i> pattern(pattern0, pattern0 + 512);

    // computeOrbDescriptors takes keypoints in level-0 coordinates, which is
    // what detectAndCompute returns: recomputing the descriptor of every
    // octave-1 keypoint from them must reproduce the pipeline's bytes.
    int octave1 = 0, identical = 0;
    for (size_t i = 0; i < keypoints.size(); ++i) {
        if (keypoints[i].octave != 1) continue;
        ++octave1;
        std::vector<KeyPoint> one = {keypoints[i]};
        Image8U expected(32, 1);
        computeOrbDescriptors(level1, layerInfo, layerScale, one, expected, pattern, 32, 2);
        bool same = true;
        for (int k = 0; k < 32; ++k) same = same && expected.at(0, k) == descriptors.at(static_cast<int>(i), k);
        identical += same;
    }

    CHECK(octave1 > 50);
    CHECK_EQ(identical, octave1);
}

// ---------------------------------------------------------------------------
// Small edgeThreshold values must not read outside the level (issue #35)
// ---------------------------------------------------------------------------

TEST_CASE("ORB::detectAndCompute: small edgeThreshold keeps every reader inside the level") {
    // Corners right at the image border on every level: dots start at 1 px.
    const int size = 96;
    Image8U image(size, size);
    for (auto& v : image.data) v = 30;
    for (int y = 1; y < size; y += 6)
        for (int x = 1; x < size; x += 6)
            image.at(y, x) = 220;

    // The fixed 31 px BRIEF pattern reaches 19 px from the keypoint whatever
    // patchSize is; ICAngles reads patchSize / 2, Harris 3 px. Below that the
    // sampling must still stay inside the level (ASan catches any violation),
    // and no returned keypoint may be closer to its level's edge than the
    // readers need: max(ceil(patchSize / 2 * sqrt(2)), 19).
    const int patternReach = 19;
    const float scaleFactor = 1.2f;
    const int nlevels = 4;

    for (int patchSize : {31, 15, 7})
    for (int edgeThreshold : {0, 3, 5, 10, 16, 21}) {
        const int minBorder = std::max(
            static_cast<int>(std::ceil((patchSize / 2) * std::sqrt(2.0))), patternReach);
        ORB orb(500, scaleFactor, nlevels, edgeThreshold, 0, 2, ORB::HARRIS_SCORE, patchSize, 20);
        std::vector<KeyPoint> keypoints;
        Image8U descriptors;
        orb.detectAndCompute(image, keypoints, descriptors);

        CHECK(!keypoints.empty());
        for (const auto& kp : keypoints) {
            const float scale = std::pow(scaleFactor, static_cast<float>(kp.octave));
            const int cols = static_cast<int>(std::round(size / scale));
            const int rows = static_cast<int>(std::round(size / scale));
            const float lx = kp.pt.x / scale, ly = kp.pt.y / scale;
            CHECK(lx >= minBorder && lx < cols - minBorder);
            CHECK(ly >= minBorder && ly < rows - minBorder);
        }
    }
}
