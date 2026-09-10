#include "orb_core.hpp"
#include "test_framework.hpp"

// ---------------------------------------------------------------------------
// HarrisResponses
// ---------------------------------------------------------------------------

TEST_CASE("HarrisResponses: flat region has zero response") {
    Image8U img(20, 20);
    for (auto& v : img.data) v = 128;

    std::vector<Rect> layerinfo = {{0, 0, 20, 20}};
    std::vector<KeyPoint> pts(1);
    pts[0].pt = {10.0f, 10.0f};
    pts[0].octave = 0;

    HarrisResponses(img, layerinfo, pts, 7, 0.04f);

    CHECK_EQ(pts[0].response, 0.0f);
}

TEST_CASE("HarrisResponses: a straight edge gives a negative response") {
    Image8U img(20, 20);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            img.at(y, x) = (x < 10) ? 0 : 255;

    std::vector<Rect> layerinfo = {{0, 0, 20, 20}};
    std::vector<KeyPoint> pts(1);
    pts[0].pt = {10.0f, 10.0f};
    pts[0].octave = 0;

    HarrisResponses(img, layerinfo, pts, 7, 0.04f);

    CHECK(pts[0].response < 0.0f);
}

TEST_CASE("HarrisResponses: a two-sided corner scores higher than a straight edge") {
    Image8U edgeImg(20, 20);
    for (int y = 0; y < edgeImg.rows; ++y)
        for (int x = 0; x < edgeImg.cols; ++x)
            edgeImg.at(y, x) = (x < 10) ? 0 : 255;

    Image8U cornerImg(20, 20);
    for (int y = 0; y < cornerImg.rows; ++y)
        for (int x = 0; x < cornerImg.cols; ++x)
            cornerImg.at(y, x) = (x < 10 && y < 10) ? 0 : 255;

    std::vector<Rect> layerinfo = {{0, 0, 20, 20}};

    std::vector<KeyPoint> edgePts(1);
    edgePts[0].pt = {10.0f, 10.0f};
    edgePts[0].octave = 0;
    HarrisResponses(edgeImg, layerinfo, edgePts, 7, 0.04f);

    std::vector<KeyPoint> cornerPts(1);
    cornerPts[0].pt = {10.0f, 10.0f};
    cornerPts[0].octave = 0;
    HarrisResponses(cornerImg, layerinfo, cornerPts, 7, 0.04f);

    CHECK(cornerPts[0].response > edgePts[0].response);
}

// ---------------------------------------------------------------------------
// ICAngles
// ---------------------------------------------------------------------------

// umax for a patch with halfPatchSize = 3, derived the same way ORB::detectAndCompute
// derives it for patchSize = 7 (kept small so the neighborhood fits an easy-to-build image).
static const std::vector<int> kUMax3 = {3, 3, 2, 1};

TEST_CASE("ICAngles: a flat patch has angle 0") {
    Image8U img(20, 20);
    for (auto& v : img.data) v = 100;

    std::vector<Rect> layerinfo = {{0, 0, 20, 20}};
    std::vector<KeyPoint> pts(1);
    pts[0].pt = {10.0f, 10.0f};
    pts[0].octave = 0;

    ICAngles(img, layerinfo, pts, kUMax3, 3);

    CHECK_EQ(pts[0].angle, 0.0f);
}

TEST_CASE("ICAngles: intensity increasing along +x gives angle 0") {
    Image8U img(20, 20);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            img.at(y, x) = static_cast<uint8_t>(100 + (x - 10) * 5);

    std::vector<Rect> layerinfo = {{0, 0, 20, 20}};
    std::vector<KeyPoint> pts(1);
    pts[0].pt = {10.0f, 10.0f};
    pts[0].octave = 0;

    ICAngles(img, layerinfo, pts, kUMax3, 3);

    CHECK_NEAR(pts[0].angle, 0.0, 1e-3);
}

TEST_CASE("ICAngles: intensity increasing along +y gives angle 90") {
    Image8U img(20, 20);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            img.at(y, x) = static_cast<uint8_t>(100 + (y - 10) * 5);

    std::vector<Rect> layerinfo = {{0, 0, 20, 20}};
    std::vector<KeyPoint> pts(1);
    pts[0].pt = {10.0f, 10.0f};
    pts[0].octave = 0;

    ICAngles(img, layerinfo, pts, kUMax3, 3);

    CHECK_NEAR(pts[0].angle, 90.0, 1e-3);
}

// ---------------------------------------------------------------------------
// computeOrbDescriptors (wta_k == 2 bit-packing)
// ---------------------------------------------------------------------------

TEST_CASE("computeOrbDescriptors: pins the exact byte for a known pattern and image") {
    // 9x9 image where each pixel's value uniquely encodes its coordinates,
    // so every sampled comparison has a hand-computable outcome.
    Image8U img(9, 9);
    for (int y = 0; y < 9; ++y)
        for (int x = 0; x < 9; ++x)
            img.at(y, x) = static_cast<uint8_t>(y * 9 + x);

    std::vector<Rect> layerInfo = {{0, 0, 9, 9}};
    std::vector<float> layerScale = {1.0f};

    std::vector<KeyPoint> kpts(1);
    kpts[0].pt = {3.0f, 3.0f};
    kpts[0].angle = 0.0f; // zero rotation: sampled (dx, dy) map directly onto the image
    kpts[0].octave = 0;

    // 16 offsets (8 comparison pairs) relative to the keypoint, matching the
    // GET_VALUE(idx) consumption order used by the wta_k == 2 branch.
    std::vector<Point2i> pattern = {
        {1, 0}, {-1, 0},
        {0, 1}, {0, -1},
        {-1, 0}, {1, 0},
        {0, -1}, {0, 1},
        {2, 0}, {-2, 0},
        {-2, 0}, {2, 0},
        {0, 2}, {0, -2},
        {0, -2}, {0, 2},
    };

    Image8U descriptors(1, 1); // dsize = 1 byte
    computeOrbDescriptors(img, layerInfo, layerScale, kpts, descriptors, pattern, 1, 2);

    CHECK_EQ(static_cast<int>(descriptors.ptr(0)[0]), 172);
}

TEST_CASE("computeOrbDescriptors: wta_k 3 and 4 run without crashing") {
    Image8U img(9, 9);
    for (int y = 0; y < 9; ++y)
        for (int x = 0; x < 9; ++x)
            img.at(y, x) = static_cast<uint8_t>(y * 9 + x);

    std::vector<Rect> layerInfo = {{0, 0, 9, 9}};
    std::vector<float> layerScale = {1.0f};

    std::vector<Point2i> pattern12 = {
        {1, 0}, {-1, 0}, {0, 1},
        {0, -1}, {-1, 0}, {1, 0},
        {2, 0}, {-2, 0}, {0, 2},
        {0, -2}, {1, 1}, {-1, -1},
    };

    std::vector<KeyPoint> kpts3(1);
    kpts3[0].pt = {3.0f, 3.0f};
    kpts3[0].angle = 0.0f;
    kpts3[0].octave = 0;
    Image8U desc3(1, 1);
    computeOrbDescriptors(img, layerInfo, layerScale, kpts3, desc3, pattern12, 1, 3);

    std::vector<Point2i> pattern16 = {
        {1, 0}, {-1, 0}, {0, 1}, {0, -1},
        {-1, 0}, {1, 0}, {0, -1}, {0, 1},
        {2, 0}, {-2, 0}, {0, 2}, {0, -2},
        {1, 1}, {-1, -1}, {1, -1}, {-1, 1},
    };

    std::vector<KeyPoint> kpts4(1);
    kpts4[0].pt = {3.0f, 3.0f};
    kpts4[0].angle = 0.0f;
    kpts4[0].octave = 0;
    Image8U desc4(1, 1);
    computeOrbDescriptors(img, layerInfo, layerScale, kpts4, desc4, pattern16, 1, 4);

    // No specific bit pattern asserted here — this only guards against crashes
    // / out-of-bounds access in the less commonly used wta_k branches.
    CHECK_EQ(desc3.cols, 1);
    CHECK_EQ(desc4.cols, 1);
}
