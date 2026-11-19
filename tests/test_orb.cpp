#include "orb.hpp"
#include "orb_core.hpp"
#include "fast_detector.hpp"
#include "image_utils.hpp"
#include "orb_pattern.hpp"
#include "test_framework.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <cstdint>
#include <stdexcept>

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

TEST_CASE("ORB::detectAndCompute: empty image produces no keypoints") {
    Image8U image; // 0x0
    ORB orb;
    std::vector<KeyPoint> keypoints;
    Image8U descriptors;

    orb.detectAndCompute(image, keypoints, descriptors);

    CHECK(keypoints.empty());
}

// Every early return must leave both outputs empty, not holding whatever the
// previous call produced (issue #22). Each test below runs one successful
// detection first, so the outputs are populated when the second call starts.
TEST_CASE("ORB::detectAndCompute: an empty image clears outputs from a previous call") {
    Image8U image = makeDotGrid(128, 30, 220);

    ORB orb(50, 1.2f, 2, 16, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    std::vector<KeyPoint> keypoints;
    Image8U descriptors;
    orb.detectAndCompute(image, keypoints, descriptors);
    CHECK(!keypoints.empty());
    CHECK_EQ(descriptors.rows, static_cast<int>(keypoints.size()));

    Image8U emptyImage; // 0x0
    orb.detectAndCompute(emptyImage, keypoints, descriptors);

    CHECK(keypoints.empty());
    CHECK_EQ(descriptors.rows, 0);
    CHECK_EQ(descriptors.cols, 0);
    CHECK(descriptors.data.empty());
}

TEST_CASE("ORB::detectAndCompute: a featureless image clears outputs from a previous call") {
    Image8U image = makeDotGrid(128, 30, 220);

    ORB orb(50, 1.2f, 2, 16, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    std::vector<KeyPoint> keypoints;
    Image8U descriptors;
    orb.detectAndCompute(image, keypoints, descriptors);
    CHECK(!keypoints.empty());

    Image8U flat(64, 64);
    for (auto& v : flat.data) v = 128;
    orb.detectAndCompute(flat, keypoints, descriptors);

    CHECK(keypoints.empty());
    CHECK_EQ(descriptors.rows, 0);
    CHECK(descriptors.data.empty());
}

// An unsupported wta_k is rejected by the constructor (see the constructor
// validation tests at the end of this file), so detectAndCompute never sees
// one; the #22 "clear outputs on every early return" rule is covered by the
// empty-image and featureless-image tests above.

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
    // A texture, not the dot grid: identical dots have identical Harris
    // responses, and OpenCV's retainBest keeps every point tied with the
    // n-th one (see the next test).
    Image8U image = makeTexture(128, 128, 5);

    ORB orb(5, 1.2f, 1, 16, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    std::vector<KeyPoint> keypoints;
    Image8U descriptors;

    orb.detectAndCompute(image, keypoints, descriptors);

    CHECK(keypoints.size() <= 5);
}

TEST_CASE("ORB::detectAndCompute: points tied with the nfeatures-th response are kept, as in OpenCV") {
    Image8U image = makeDotGrid(128, 30, 220); // every dot scores the same

    ORB orb(5, 1.2f, 1, 16, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    std::vector<KeyPoint> keypoints;
    Image8U descriptors;
    orb.detectAndCompute(image, keypoints, descriptors);

    CHECK(keypoints.size() >= 5);
    for (const auto& kp : keypoints) CHECK_EQ(kp.response, keypoints[0].response);
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

// ---------------------------------------------------------------------------
// Harris pre-filtering (#24): OpenCV keeps the best 2 * featuresNum corners by
// FAST score before computing Harris, so a weak FAST corner can never be
// selected however strong its Harris response.
// ---------------------------------------------------------------------------

TEST_CASE("ORB::detectAndCompute: Harris only runs on the best 2*featuresNum FAST corners") {
    Image8U image = makeTexture(256, 256, 3);
    const int nfeatures = 40, threshold = 20, edge = 31;

    // Reference: FAST corners of level 0 inside the border, ranked by FAST score.
    std::vector<KeyPoint> fast;
    detectFAST(image, fast, threshold, true);
    std::vector<float> responses;
    std::map<std::pair<int, int>, float> fastResponse;
    for (const auto& k : fast) {
        if (k.pt.x < edge || k.pt.x >= image.cols - edge || k.pt.y < edge || k.pt.y >= image.rows - edge) continue;
        responses.push_back(k.response);
        fastResponse[{static_cast<int>(k.pt.x), static_cast<int>(k.pt.y)}] = k.response;
    }
    std::sort(responses.begin(), responses.end(), std::greater<float>());
    CHECK(responses.size() > 4 * nfeatures); // the pre-filter must actually cut
    const float cutoff = responses[2 * nfeatures - 1];

    ORB orb(nfeatures, 1.2f, 1, edge, 0, 2, ORB::HARRIS_SCORE, 31, threshold);
    std::vector<KeyPoint> kps;
    Image8U desc;
    orb.detectAndCompute(image, kps, desc);
    CHECK(!kps.empty());

    int belowCutoff = 0;
    for (const auto& k : kps) {
        auto it = fastResponse.find({static_cast<int>(k.pt.x), static_cast<int>(k.pt.y)});
        CHECK(it != fastResponse.end());
        if (it != fastResponse.end() && it->second < cutoff) ++belowCutoff;
    }
    std::fprintf(stderr, "    selected keypoints whose FAST score is below the 2N cutoff: %d / %zu\n", belowCutoff, kps.size());
    CHECK_EQ(belowCutoff, 0);
}

// ---------------------------------------------------------------------------
// FAST_SCORE must honour nfeatures like HARRIS_SCORE does (#21)
// ---------------------------------------------------------------------------

TEST_CASE("ORB::detectAndCompute: FAST_SCORE keeps at most nfeatures keypoints, the best by FAST score") {
    Image8U image = makeTexture(256, 256, 9);
    const int nfeatures = 60;

    ORB orb(nfeatures, 1.2f, 1, 31, 0, 2, ORB::FAST_SCORE, 31, 20);
    std::vector<KeyPoint> kps;
    Image8U desc;
    orb.detectAndCompute(image, kps, desc);
    std::fprintf(stderr, "    FAST_SCORE keypoints returned: %zu (nfeatures %d)\n", kps.size(), nfeatures);
    // At most nfeatures, except for points tied with the nfeatures-th response,
    // which OpenCV's retainBest keeps too (FAST scores are small integers, so
    // ties at the cut are common).
    std::vector<float> responses;
    for (const auto& k : kps) responses.push_back(k.response);
    std::sort(responses.begin(), responses.end(), std::greater<float>());
    CHECK(responses.size() >= static_cast<size_t>(nfeatures));
    for (size_t i = nfeatures; i < responses.size(); ++i) CHECK_EQ(responses[i], responses[nfeatures - 1]);
    CHECK_EQ(desc.rows, static_cast<int>(kps.size()));

    // The kept ones are the strongest: no dropped corner inside the border may
    // outscore the weakest kept one.
    std::vector<KeyPoint> all;
    detectFAST(image, all, 20, true);
    float weakestKept = 1e30f;
    for (const auto& k : kps) weakestKept = std::min(weakestKept, k.response);
    int strongerDropped = 0;
    for (const auto& k : all) {
        if (k.pt.x < 31 || k.pt.x >= image.cols - 31 || k.pt.y < 31 || k.pt.y >= image.rows - 31) continue;
        bool kept = false;
        for (const auto& s : kps) kept = kept || (s.pt.x == k.pt.x && s.pt.y == k.pt.y);
        if (!kept && k.response > weakestKept) ++strongerDropped;
    }
    CHECK_EQ(strongerDropped, 0);
}

// ---------------------------------------------------------------------------
// Constructor validation (#23, #22): unsupported parameters are rejected up
// front with std::invalid_argument instead of being discovered after detection.
// ---------------------------------------------------------------------------

namespace {

bool constructorThrows(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel,
                       int wta_k, ORB::ScoreType scoreType, int patchSize, int fastThreshold) {
    try {
        ORB orb(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, wta_k, scoreType, patchSize, fastThreshold);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

} // namespace

TEST_CASE("ORB: constructor rejects firstLevel != 0, which was never supported") {
    // Level 0 is always the original image at scale 1 (orb.cpp hard-codes
    // layerScale[0] = 1), so any other firstLevel would duplicate level 0 and
    // shift every scale by one step.
    CHECK(constructorThrows(500, 1.2f, 8, 31, 1, 2, ORB::HARRIS_SCORE, 31, 20));
    CHECK(constructorThrows(500, 1.2f, 8, 31, -1, 2, ORB::HARRIS_SCORE, 31, 20));
    CHECK(!constructorThrows(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20));
}

TEST_CASE("ORB: constructor rejects the other unsupported parameters") {
    CHECK(constructorThrows(500, 1.2f, 8, 31, 0, 5, ORB::HARRIS_SCORE, 31, 20)); // wta_k not in {2, 3, 4}
    CHECK(constructorThrows(500, 1.2f, 8, 31, 0, 1, ORB::HARRIS_SCORE, 31, 20));
    CHECK(constructorThrows(500, 1.2f, 0, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20)); // nlevels < 1
    CHECK(constructorThrows(500, 1.0f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20)); // scaleFactor <= 1
    CHECK(constructorThrows(500, 0.9f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20));
    CHECK(constructorThrows(-1, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20));  // nfeatures < 0
    CHECK(constructorThrows(500, 1.2f, 8, -1, 0, 2, ORB::HARRIS_SCORE, 31, 20)); // edgeThreshold < 0
    CHECK(constructorThrows(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 0, 20));  // patchSize <= 0
    CHECK(constructorThrows(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 0));  // fastThreshold <= 0
    CHECK(constructorThrows(500, 1.2f, 8, 31, 0, 2, static_cast<ORB::ScoreType>(2), 31, 20)); // scoreType
    // Valid combinations, including wta_k 3 and 4 and FAST_SCORE, construct fine.
    CHECK(!constructorThrows(500, 1.2f, 8, 31, 0, 3, ORB::HARRIS_SCORE, 31, 20));
    CHECK(!constructorThrows(500, 1.2f, 8, 31, 0, 4, ORB::FAST_SCORE, 31, 20));
    CHECK(!constructorThrows(0, 1.01f, 1, 0, 0, 2, ORB::HARRIS_SCORE, 1, 1));
}
