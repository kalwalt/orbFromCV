#include "fast_detector.hpp"
#include "test_framework.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <chrono>
#include <cmath>
#include <vector>

namespace {

// Reference 3x3 non-maximal suppression, OpenCV's rule: a point survives only
// if its response is strictly greater than every other point's in the 3x3
// window, so tied neighbours are both dropped.
std::vector<KeyPoint> bruteForceNms(const std::vector<KeyPoint>& raw) {
    std::vector<KeyPoint> kept;
    for (size_t i = 0; i < raw.size(); ++i) {
        bool isMax = true;
        for (size_t j = 0; j < raw.size() && isMax; ++j) {
            if (i == j) continue;
            if (std::abs(raw[i].pt.x - raw[j].pt.x) <= 1.0f &&
                std::abs(raw[i].pt.y - raw[j].pt.y) <= 1.0f &&
                raw[j].response >= raw[i].response) {
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

TEST_CASE("detectFAST: NMS matches a brute-force 3x3 reference, ties are both dropped") {
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
    // corners share a response, and no two survivors are 3x3 neighbours.
    int tiedRawPairs = 0, adjacentSurvivors = 0;
    for (size_t i = 0; i < raw.size(); ++i)
        for (size_t j = i + 1; j < raw.size(); ++j)
            if (std::abs(raw[i].pt.x - raw[j].pt.x) <= 1.0f && std::abs(raw[i].pt.y - raw[j].pt.y) <= 1.0f &&
                raw[i].response == raw[j].response)
                ++tiedRawPairs;
    for (size_t i = 0; i < suppressed.size(); ++i)
        for (size_t j = i + 1; j < suppressed.size(); ++j)
            if (std::abs(suppressed[i].pt.x - suppressed[j].pt.x) <= 1.0f &&
                std::abs(suppressed[i].pt.y - suppressed[j].pt.y) <= 1.0f)
                ++adjacentSurvivors;
    CHECK(tiedRawPairs > 0);
    CHECK_EQ(adjacentSurvivors, 0);
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

// ---------------------------------------------------------------------------
// FAST-9 completeness: the early rejection must never drop a real corner (#20)
// ---------------------------------------------------------------------------

namespace {

// Same Bresenham circle as fast_detector.cpp, (x, y) offsets, index 0 at the top.
const int kCircle[16][2] = {
    {0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0}, {3, 1}, {2, 2}, {1, 3},
    {0, 3}, {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3},
};

// Brute-force FAST-9 segment test: 9 contiguous circle pixels all brighter than
// p + t, or all darker than p - t, for some of the 16 start positions.
bool isFast9Corner(const Image8U& img, int x, int y, int t) {
    const int p = img.at(y, x);
    int ring[32];
    for (int i = 0; i < 16; ++i) ring[i] = ring[i + 16] = img.at(y + kCircle[i][1], x + kCircle[i][0]);
    for (int start = 0; start < 16; ++start) {
        bool bright = true, dark = true;
        for (int j = 0; j < 9; ++j) {
            bright = bright && ring[start + j] > p + t;
            dark = dark && ring[start + j] < p - t;
        }
        if (bright || dark) return true;
    }
    return false;
}

std::vector<std::pair<int, int>> sortedXY(const std::vector<KeyPoint>& kps) {
    std::vector<std::pair<int, int>> v;
    for (const auto& k : kps) v.emplace_back(static_cast<int>(k.pt.x), static_cast<int>(k.pt.y));
    std::sort(v.begin(), v.end());
    return v;
}

Image8U seededNoise(int w, int h, uint32_t seed, uint8_t low, uint8_t span) {
    Image8U img(w, h);
    for (auto& v : img.data) {
        seed = seed * 1664525u + 1013904223u;
        v = static_cast<uint8_t>(low + (seed >> 24) % span);
    }
    return img;
}

} // namespace

TEST_CASE("detectFAST: a 9-arc covering only two compass pixels is a corner, all 16 rotations") {
    // Circle pixels start..start+8 brighter than the centre. For start = 1 the
    // arc 1..9 touches only compass pixels 4 and 8, which the old 3-of-4
    // early rejection (a FAST-12 criterion) threw away.
    int detected = 0;
    for (int start = 0; start < 16; ++start) {
        Image8U img(7, 7);
        for (auto& v : img.data) v = 100;
        for (int j = 0; j < 9; ++j) {
            const int i = (start + j) % 16;
            img.at(3 + kCircle[i][1], 3 + kCircle[i][0]) = 200;
        }
        CHECK(isFast9Corner(img, 3, 3, 20));
        std::vector<KeyPoint> kps;
        detectFAST(img, kps, 20, false);
        const bool found = kps.size() == 1 && kps[0].pt.x == 3.0f && kps[0].pt.y == 3.0f;
        if (!found) std::fprintf(stderr, "    arc starting at circle pixel %d not detected (%zu keypoints)\n", start, kps.size());
        detected += found;
    }
    CHECK_EQ(detected, 16);
}

TEST_CASE("detectFAST without NMS matches a brute-force FAST-9 oracle on noise and structure") {
    const int t = 20;
    int totalRef = 0, totalMissing = 0, totalExtra = 0;
    for (uint32_t seed = 1; seed <= 5; ++seed) {
        // Low-contrast noise with a few hard-edged rectangles: corners of every shape.
        Image8U img = seededNoise(96, 72, seed, 90, 60);
        for (int y = 10; y < 40; ++y) for (int x = 20; x < 60; ++x) img.at(y, x) = static_cast<uint8_t>(255 - img.at(y, x));
        for (int y = 30; y < 65; ++y) for (int x = 50; x < 90; ++x) img.at(y, x) = static_cast<uint8_t>(img.at(y, x) / 3);

        std::vector<std::pair<int, int>> ref;
        for (int y = 3; y < img.rows - 3; ++y)
            for (int x = 3; x < img.cols - 3; ++x)
                if (isFast9Corner(img, x, y, t)) ref.emplace_back(x, y);
        std::sort(ref.begin(), ref.end()); // set_difference needs (x, y) order on both sides

        std::vector<KeyPoint> kps;
        detectFAST(img, kps, t, false);
        std::vector<std::pair<int, int>> ours = sortedXY(kps);

        std::vector<std::pair<int, int>> missing, extra;
        std::set_difference(ref.begin(), ref.end(), ours.begin(), ours.end(), std::back_inserter(missing));
        std::set_difference(ours.begin(), ours.end(), ref.begin(), ref.end(), std::back_inserter(extra));
        totalRef += static_cast<int>(ref.size());
        totalMissing += static_cast<int>(missing.size());
        totalExtra += static_cast<int>(extra.size());
    }
    std::fprintf(stderr, "    oracle corners: %d, missing: %d, extra: %d\n", totalRef, totalMissing, totalExtra);
    CHECK(totalRef > 500); // the images must actually have corners
    CHECK_EQ(totalMissing, 0);
    CHECK_EQ(totalExtra, 0);
}

// ---------------------------------------------------------------------------
// Corner score definition (#24): OpenCV's cornerScore<16> is the largest
// threshold at which the pixel is still a FAST-9 corner.
// ---------------------------------------------------------------------------

TEST_CASE("detectFAST: response is the largest threshold at which the pixel is still a corner") {
    const int t = 20;
    int checked = 0, mismatched = 0;
    for (uint32_t seed = 11; seed <= 13; ++seed) {
        Image8U img = seededNoise(80, 60, seed, 40, 170);
        std::vector<KeyPoint> kps;
        detectFAST(img, kps, t, false);
        for (const auto& k : kps) {
            const int x = static_cast<int>(k.pt.x), y = static_cast<int>(k.pt.y);
            int maxT = t;
            while (maxT + 1 <= 255 && isFast9Corner(img, x, y, maxT + 1)) ++maxT;
            ++checked;
            if (k.response != static_cast<float>(maxT)) {
                if (mismatched < 3)
                    std::fprintf(stderr, "    (%d,%d): response %.0f, largest corner threshold %d\n", x, y, k.response, maxT);
                ++mismatched;
            }
        }
    }
    std::fprintf(stderr, "    corners checked: %d, score mismatches: %d\n", checked, mismatched);
    CHECK(checked > 200);
    CHECK_EQ(mismatched, 0);
}
