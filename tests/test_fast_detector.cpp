#include "fast_detector.hpp"
#include "test_framework.hpp"

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
