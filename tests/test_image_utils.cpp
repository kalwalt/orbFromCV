#include "image_utils.hpp"
#include "test_framework.hpp"

TEST_CASE("resizeBilinear: upscale produces requested dimensions") {
    Image8U src(2, 2);
    src.at(0, 0) = 0;   src.at(0, 1) = 255;
    src.at(1, 0) = 255; src.at(1, 1) = 0;

    Image8U dst;
    resizeBilinear(src, dst, 4, 4);

    CHECK_EQ(dst.cols, 4);
    CHECK_EQ(dst.rows, 4);
}

TEST_CASE("resizeBilinear: constant image stays constant") {
    Image8U src(4, 4);
    for (auto& v : src.data) v = 100;

    Image8U up, down;
    resizeBilinear(src, up, 8, 8);
    resizeBilinear(src, down, 2, 2);

    for (auto v : up.data) CHECK(std::abs(static_cast<int>(v) - 100) <= 1);
    for (auto v : down.data) CHECK(std::abs(static_cast<int>(v) - 100) <= 1);
}

TEST_CASE("resizeBilinear: top-left source pixel maps exactly to top-left dest pixel") {
    // src_x/src_y are 0 at dst (0,0) regardless of the scale ratio, so this
    // corner is the only one exactly predictable without reproducing the
    // implementation's src/dst ratio formula.
    Image8U src(2, 2);
    src.at(0, 0) = 7;   src.at(0, 1) = 250;
    src.at(1, 0) = 10;  src.at(1, 1) = 200;

    Image8U dst;
    resizeBilinear(src, dst, 10, 10);

    CHECK_EQ(static_cast<int>(dst.at(0, 0)), 7);
}

TEST_CASE("resizeBilinear: output values stay within the source's value range") {
    Image8U src(2, 2);
    src.at(0, 0) = 0;   src.at(0, 1) = 250;
    src.at(1, 0) = 10;  src.at(1, 1) = 200;

    Image8U dst;
    resizeBilinear(src, dst, 10, 10);

    for (auto v : dst.data) {
        CHECK(v >= 0 && v <= 250);
    }
}

TEST_CASE("resizeBilinear: intensity increases monotonically along a linear ramp") {
    Image8U src(2, 8);
    for (int y = 0; y < 8; ++y) {
        src.at(y, 0) = 0;
        src.at(y, 1) = 250;
    }

    Image8U dst;
    resizeBilinear(src, dst, 20, 8);

    for (int y = 0; y < 8; ++y)
        for (int x = 1; x < 20; ++x)
            CHECK(dst.at(y, x) >= dst.at(y, x - 1));
}

TEST_CASE("resizeBilinear: zero target dimension leaves dst untouched") {
    Image8U dst(3, 3);
    Image8U src(4, 4);

    resizeBilinear(src, dst, 0, 5);
    CHECK_EQ(dst.cols, 3);
    CHECK_EQ(dst.rows, 3);

    resizeBilinear(src, dst, 5, 0);
    CHECK_EQ(dst.cols, 3);
    CHECK_EQ(dst.rows, 3);
}

TEST_CASE("gaussianBlur7x7: constant image is unchanged") {
    Image8U src(10, 10);
    for (auto& v : src.data) v = 128;

    Image8U dst;
    gaussianBlur7x7(src, dst);

    for (auto v : dst.data) CHECK_EQ(static_cast<int>(v), 128);
}

TEST_CASE("gaussianBlur7x7: spreads an isolated bright impulse") {
    Image8U src(15, 15);
    for (auto& v : src.data) v = 0;
    src.at(7, 7) = 255;

    Image8U dst;
    gaussianBlur7x7(src, dst);

    // Energy should spread: center drops, immediate neighbors rise above 0.
    CHECK(dst.at(7, 7) < 255);
    CHECK(dst.at(7, 7) > 0);
    CHECK(dst.at(7, 6) > 0);
    CHECK(dst.at(6, 7) > 0);
    // Far corners remain unaffected by a single central impulse.
    CHECK_EQ(static_cast<int>(dst.at(0, 0)), 0);
}

TEST_CASE("gaussianBlur7x7: horizontally symmetric input stays symmetric") {
    const int size = 15;
    Image8U src(size, size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            int dx = std::abs(x - size / 2);
            src.at(y, x) = static_cast<uint8_t>(255 - dx * 10);
        }
    }

    Image8U dst;
    gaussianBlur7x7(src, dst);

    for (int y = 0; y < size; ++y) {
        for (int dx = 1; dx <= size / 2; ++dx) {
            CHECK_EQ(dst.at(y, size / 2 - dx), dst.at(y, size / 2 + dx));
        }
    }
}

TEST_CASE("gaussianBlur7x7: handles minimal-size image without crashing") {
    Image8U src(7, 7);
    for (auto& v : src.data) v = 42;

    Image8U dst;
    gaussianBlur7x7(src, dst);

    CHECK_EQ(dst.cols, 7);
    CHECK_EQ(dst.rows, 7);
}

TEST_CASE("gaussianBlur7x7: safe to call in-place (dst aliasing src)") {
    // ORB::detectAndCompute calls gaussianBlur7x7(imagePyramid[level],
    // imagePyramid[level]) - src and dst are the same object. An earlier
    // version reassigned dst (= reset src's data to zero) before the
    // horizontal pass had read from it, silently blurring garbage/zeros
    // into every descriptor. Regression test for that bug.
    Image8U separateResult;
    {
        Image8U src(15, 15);
        for (auto& v : src.data) v = 0;
        src.at(7, 7) = 255;
        gaussianBlur7x7(src, separateResult);
    }

    Image8U aliased(15, 15);
    for (auto& v : aliased.data) v = 0;
    aliased.at(7, 7) = 255;
    gaussianBlur7x7(aliased, aliased);

    CHECK_EQ(aliased.data.size(), separateResult.data.size());
    for (size_t i = 0; i < aliased.data.size(); ++i) {
        CHECK_EQ(aliased.data[i], separateResult.data[i]);
    }
    // Would have failed before the fix: the aliased call zeroed everything.
    CHECK(aliased.at(7, 7) > 0);
}
