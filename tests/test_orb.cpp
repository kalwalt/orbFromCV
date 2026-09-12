#include "orb.hpp"
#include "test_framework.hpp"

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
        "pattern_init", "gaussian_blur", "descriptors", "finalize",
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
