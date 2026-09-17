#include "fast_detector.hpp"
#include "test_framework.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace {

// Reference 3x3 non-maximal suppression: a point survives unless some other
// point within the 3x3 window has a strictly greater response (ties kept).
std::vector<KeyPoint> bruteForceNms(const std::vector<KeyPoint>& raw) {
    std::vector<KeyPoint> kept;
    for (size_t i = 0; i < raw.size(); ++i) {
        bool isMax = true;
        for (size_t j = 0; j < raw.size() && isMax; ++j) {
            if (i == j) continue;
            if (std::abs(raw[i].pt.x - raw[j].pt.x) <= 1.0f &&
                std::abs(raw[i].pt.y - raw[j].pt.y) <= 1.0f &&
                raw[j].response > raw[i].response) {
                isMax = false;
            }
        }
        if (isMax) kept.push_back(raw[i]);
    }
    return kept;
}

bool sameKeypointSet(std::vector<KeyPoint> a, std::vector<KeyPoint> b) {
    auto byPos = [](const KeyPoint& l, const KeyPoint& r) {
        return l.pt.y != r.pt.y ? l.pt.y < r.pt.y : l.pt.x < r.pt.x;
    };
    std::sort(a.begin(), a.end(), byPos);
    std::sort(b.begin(), b.end(), byPos);
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].pt.x != b[i].pt.x || a[i].pt.y != b[i].pt.y ||
            a[i].response != b[i].response) return false;
    }
    return true;
}

// Dense grid of isolated bright dots: every dot is a corner, so the raw
// corner count scales with the image area. No dot has a 3x3 neighbour, which
// is the worst case for an all-pairs NMS (no early exit).
Image8U denseDotImage(int size) {
    Image8U img(size, size);
    for (auto& v : img.data) v = 50;
    for (int y = 4; y < size - 4; y += 4)
        for (int x = 4; x < size - 4; x += 4)
            img.at(y, x) = 220;
    return img;
}

double detectMillis(const Image8U& img) {
    std::vector<KeyPoint> kpts;
    auto t0 = std::chrono::steady_clock::now();
    detectFAST(img, kpts, 20, true);
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

} // namespace

TEST_CASE("detectFAST: NMS matches a brute-force 3x3 reference, including ties") {
    Image8U img(64, 64);
    for (auto& v : img.data) v = 50;
    // Isolated dots, uneven blobs and mirror-symmetric pairs (equal responses).
    img.at(10, 10) = 220;
    img.at(20, 20) = 220; img.at(20, 21) = 210; img.at(21, 20) = 200; img.at(21, 21) = 190;
    img.at(30, 30) = 220; img.at(30, 31) = 220;
    img.at(40, 40) = 220; img.at(41, 40) = 220;
    img.at(50, 10) = 220; img.at(50, 11) = 220; img.at(50, 12) = 220;
    img.at(10, 50) = 200; img.at(11, 51) = 200;

    std::vector<KeyPoint> raw, suppressed;
    detectFAST(img, raw, 20, false);
    detectFAST(img, suppressed, 20, true);

    CHECK(raw.size() > 6);
    CHECK(sameKeypointSet(suppressed, bruteForceNms(raw)));

    // Make sure the tie path is really exercised: some neighbouring raw
    // corners must share a response and both survive.
    int tiedSurvivors = 0;
    for (size_t i = 0; i < suppressed.size(); ++i)
        for (size_t j = i + 1; j < suppressed.size(); ++j)
            if (std::abs(suppressed[i].pt.x - suppressed[j].pt.x) <= 1.0f &&
                std::abs(suppressed[i].pt.y - suppressed[j].pt.y) <= 1.0f)
                ++tiedSurvivors;
    CHECK(tiedSurvivors > 0);
}

TEST_CASE("detectFAST with NMS scales ~linearly with image area, not quadratically") {
    Image8U small = denseDotImage(256);
    Image8U large = denseDotImage(512); // ~4x the corners

    std::vector<KeyPoint> check;
    detectFAST(large, check, 20, false);
    CHECK(check.size() > 10000);

    // Interleave the samples so a transient CPU stall on a shared CI runner
    // hits both sizes rather than inflating only one of them.
    double smallMs = 1e300, largeMs = 1e300;
    for (int rep = 0; rep < 5; ++rep) {
        smallMs = std::min(smallMs, detectMillis(small));
        largeMs = std::min(largeMs, detectMillis(large));
    }

    // Linear: ratio ~4. The old all-pairs NMS: ratio ~16 (Debug and Release
    // alike, since the pair loop dwarfs the scan). 12x keeps a wide gap on
    // both sides so only the growth rate matters, not machine speed.
    CHECK(largeMs < 12.0 * std::max(smallMs, 0.5));
}

TEST_CASE("detectFAST: flat image yields no keypoints") {
    Image8U img(20, 20);
    for (auto& v : img.data) v = 100;

    std::vector<KeyPoint> kpts;
    detectFAST(img, kpts, 20, true);

    CHECK(kpts.empty());
}

TEST_CASE("detectFAST: image smaller than the sampling margin yields no keypoints") {
    Image8U img(5, 5); // margin requires at least 7x7
    for (auto& v : img.data) v = 200;

    std::vector<KeyPoint> kpts;
    detectFAST(img, kpts, 10, true);

    CHECK(kpts.empty());
}

TEST_CASE("detectFAST: isolated bright dot is detected as a corner") {
    Image8U img(20, 20);
    for (auto& v : img.data) v = 50;
    img.at(10, 10) = 220; // diff of 170, well above threshold

    std::vector<KeyPoint> kpts;
    detectFAST(img, kpts, 20, true);

    CHECK(!kpts.empty());
    bool foundDot = false;
    for (const auto& kp : kpts) {
        if (kp.pt.x == 10.0f && kp.pt.y == 10.0f) foundDot = true;
    }
    CHECK(foundDot);
}

TEST_CASE("detectFAST: raising threshold above the intensity delta removes the corner") {
    Image8U img(20, 20);
    for (auto& v : img.data) v = 50;
    img.at(10, 10) = 220; // diff of 170

    std::vector<KeyPoint> kpts;
    detectFAST(img, kpts, 200, true); // threshold exceeds the delta

    CHECK(kpts.empty());
}

TEST_CASE("detectFAST: a straight edge alone is not a corner") {
    Image8U img(20, 20);
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            img.at(y, x) = (x < 10) ? 50 : 200;
        }
    }

    std::vector<KeyPoint> kpts;
    detectFAST(img, kpts, 20, true);

    CHECK(kpts.empty());
}

TEST_CASE("detectFAST: non-maximal suppression collapses a cluster to its strongest point") {
    Image8U img(20, 20);
    for (auto& v : img.data) v = 50;
    // A small 2x2 bright blob: several neighboring pixels will individually
    // qualify as raw corners, but NMS should keep only the local maximum.
    img.at(10, 10) = 220;
    img.at(10, 11) = 210;
    img.at(11, 10) = 200;
    img.at(11, 11) = 190;

    std::vector<KeyPoint> suppressed;
    detectFAST(img, suppressed, 20, true);

    std::vector<KeyPoint> raw;
    detectFAST(img, raw, 20, false);

    CHECK(suppressed.size() <= raw.size());
    CHECK(!suppressed.empty());
}
