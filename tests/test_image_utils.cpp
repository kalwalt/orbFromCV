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

// Pixel-centre alignment as cv::resize defines it: srcpos = scale * (d + 0.5)
// - 0.5, with scale = srcsize / dstsize. An exact horizontal ramp reads the
// mapping straight back out of the resized image, so this pins the geometry
// independently of the committed fixture (#40).
TEST_CASE("resizeBilinear: maps destination centres to source centres, as cv::resize does") {
    const int srcW = 100, srcH = 80, dstW = 83, dstH = 66;
    Image8U ramp(srcW, srcH);
    for (int y = 0; y < srcH; ++y)
        for (int x = 0; x < srcW; ++x) ramp.at(y, x) = static_cast<uint8_t>(x);

    Image8U dst;
    resizeBilinear(ramp, dst, dstW, dstH);

    // Expected value: dst(x) = round_half_even(srcpos), srcpos =
    // (srcW/dstW) * (x + 0.5) - 0.5, in exact rationals.
    int mismatches = 0, worst = 0;
    for (int y = 0; y < dstH; ++y) {
        for (int x = 0; x < dstW; ++x) {
            long long num = static_cast<long long>(srcW) * (2LL * x + 1) - dstW;
            long long den = 2LL * dstW;
            long long base = num / den;
            long long frac = num - base * den;
            if (frac < 0) { frac += den; --base; }

            int expected;
            if (base < 0 || base >= srcW - 1) {
                expected = static_cast<int>(base < 0 ? 0 : srcW - 1); // edge replicated
            } else {
                // Fixed-point weights first (the implementation's own steps),
                // then one rounding over the two passes, as OpenCV's bit-exact
                // path does.
                int w1 = static_cast<int>((256 * frac + den / 2) / den);
                int w0 = 256 - w1;
                expected = (w0 * static_cast<int>(base) + w1 * static_cast<int>(base + 1) + 128) >> 8;
            }
            int got = static_cast<int>(dst.at(y, x));
            int d = std::abs(got - expected);
            worst = std::max(worst, d);
            mismatches += d != 0;
        }
    }
    std::fprintf(stderr, "    mapping mismatches: %d of %d, worst |diff| = %d\n",
                 mismatches, dstW * dstH, worst);
    CHECK_EQ(mismatches, 0);
}

TEST_CASE("resizeBilinear: nearest-edge replication outside the source") {
    // The first destination pixel of an upscale maps before the first source
    // pixel; OpenCV clamps to the edge with a full weight rather than reading
    // out of bounds or blending with a neighbour.
    Image8U src(2, 2);
    src.at(0, 0) = 0;   src.at(0, 1) = 200;
    src.at(1, 0) = 0;   src.at(1, 1) = 200;

    Image8U dst;
    resizeBilinear(src, dst, 5, 5);

    CHECK_EQ(static_cast<int>(dst.at(0, 0)), 0);
    CHECK_EQ(static_cast<int>(dst.at(4, 4)), 200);
}

TEST_CASE("resizeBilinear: downscale averages whole pixels, not corners") {
    // A 2x checkerboard averaged down must come out mid-grey everywhere; the
    // old corner-aligned mapping sampled the corners instead and returned the
    // checkerboard at the destination resolution.
    Image8U src(8, 8);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x)
            src.at(y, x) = ((x + y) & 1) ? 255 : 0;

    Image8U dst;
    resizeBilinear(src, dst, 2, 2);

    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x)
            CHECK(std::abs(static_cast<int>(dst.at(y, x)) - 128) <= 2);
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

namespace {

// BORDER_REFLECT_101 as OpenCV's borderInterpolate defines it: reflect
// repeatedly until the index lands inside [0, n), and a 1-pixel dimension
// always maps to 0. For n >= 4 a single reflection is enough for the 7-tap
// kernel; smaller dimensions need the repetition.
int reflect101(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - i - 2;
    return i;
}

// Per-pixel reference of the blur cv::ORB applies: float32 kernel
// getGaussianKernel(7, 2, CV_32F), row pass accumulating taps 0..6, column pass
// centre tap first then symmetric pairs, plain float32 multiply-adds (two
// roundings, no FMA), one round-half-to-even at the end. Written per pixel with
// index reflection, independently of the optimised loops.
float fma32(float a, float b, float c) {
    return static_cast<float>(static_cast<float>(a * b) + c);
}

Image8U referenceBlur7x7(const Image8U& src) {
    const float kernel[7] = {0.0701593235f, 0.131074876f, 0.190712824f, 0.216105938f,
                             0.190712824f, 0.131074876f, 0.0701593235f};
    std::vector<float> temp(static_cast<size_t>(src.rows) * src.cols);
    Image8U out(src.cols, src.rows);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            float acc = kernel[0] * static_cast<float>(src.at(y, reflect101(x - 3, src.cols)));
            for (int k = 1; k < 7; ++k)
                acc = fma32(kernel[k], static_cast<float>(src.at(y, reflect101(x - 3 + k, src.cols))), acc);
            temp[static_cast<size_t>(y) * src.cols + x] = acc;
        }
    auto T = [&](int y, int x) { return temp[static_cast<size_t>(reflect101(y, src.rows)) * src.cols + x]; };
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            float acc = kernel[3] * T(y, x);
            for (int k = 1; k <= 3; ++k) acc = fma32(kernel[3 - k], static_cast<float>(T(y - k, x) + T(y + k, x)), acc);
            long v = std::lrint(acc);
            out.at(y, x) = static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    return out;
}

Image8U pseudoRandomImage(int cols, int rows, uint32_t seed) {
    Image8U img(cols, rows);
    for (auto& v : img.data) {
        seed = seed * 1664525u + 1013904223u;
        v = static_cast<uint8_t>(seed >> 24);
    }
    return img;
}

} // namespace

TEST_CASE("gaussianBlur7x7: matches a naive reference byte-for-byte on noise") {
    // Mixed small sizes so the border strips, the interior and the
    // "no interior at all" (cols <= 6) paths are all exercised. Dimensions
    // below 4 px are covered by the dedicated test further down.
    const int sizes[][2] = {{41, 29}, {7, 7}, {8, 13}, {6, 20}, {5, 9}, {20, 5}, {4, 4}};
    for (const auto& s : sizes) {
        Image8U src = pseudoRandomImage(s[0], s[1], 12345u + s[0] * 7 + s[1]);
        Image8U expected = referenceBlur7x7(src);
        Image8U actual;
        gaussianBlur7x7(src, actual);
        CHECK_EQ(actual.cols, expected.cols);
        CHECK_EQ(actual.rows, expected.rows);
        int mismatches = 0;
        for (size_t i = 0; i < expected.data.size(); ++i)
            if (actual.data[i] != expected.data[i]) ++mismatches;
        if (mismatches)
            std::cerr << "    " << s[0] << "x" << s[1] << ": " << mismatches << " pixels differ\n";
        CHECK_EQ(mismatches, 0);
    }
}

TEST_CASE("gaussianBlur7x7: images with a dimension of 1-3 px stay in bounds") {
    // Issue #15: a single REFLECT_101 step maps index n+2 to n-4, which is
    // negative when n <= 3, so the blur read before the row / before the
    // buffer. ORB::detectAndCompute reaches such sizes on the top pyramid
    // levels of any input with a side of ~2-12 px.
    const int sizes[][2] = {{1, 1}, {2, 2}, {3, 3}, {2, 5}, {5, 3}, {3, 8},
                            {40, 3}, {3, 40}, {1, 12}, {12, 1}, {2, 1}};
    for (const auto& s : sizes) {
        // A constant image must come back unchanged (the kernel sums to 1),
        // whatever the reflection rule - any out-of-bounds tap breaks this.
        Image8U flat(s[0], s[1]);
        for (auto& v : flat.data) v = 200;
        Image8U flatOut;
        gaussianBlur7x7(flat, flatOut);
        int changed = 0;
        for (uint8_t v : flatOut.data)
            if (v != 200) ++changed;
        if (changed)
            std::cerr << "    " << s[0] << "x" << s[1] << ": " << changed << " constant pixels changed\n";
        CHECK_EQ(changed, 0);

        // Noise must match the reference, in place and out of place.
        Image8U src = pseudoRandomImage(s[0], s[1], 777u + s[0] * 31 + s[1]);
        Image8U expected = referenceBlur7x7(src);
        Image8U actual;
        gaussianBlur7x7(src, actual);
        gaussianBlur7x7(src, src);
        CHECK_EQ(actual.cols, s[0]);
        CHECK_EQ(actual.rows, s[1]);
        int mismatches = 0, inPlaceMismatches = 0;
        for (size_t i = 0; i < expected.data.size(); ++i) {
            if (actual.data[i] != expected.data[i]) ++mismatches;
            if (src.data[i] != expected.data[i]) ++inPlaceMismatches;
        }
        if (mismatches || inPlaceMismatches)
            std::cerr << "    " << s[0] << "x" << s[1] << ": " << mismatches << " / "
                      << inPlaceMismatches << " (in place) pixels differ\n";
        CHECK_EQ(mismatches, 0);
        CHECK_EQ(inPlaceMismatches, 0);
    }
}

TEST_CASE("resizeBilinear: vector path equals the scalar path byte for byte") {
    const int srcSizes[][2] = {{1, 1}, {2, 3}, {5, 4}, {17, 9}, {64, 48}, {97, 71}};
    const int dstSizes[][2] = {{1, 1}, {3, 2}, {4, 5}, {13, 7}, {53, 40}, {81, 59}, {120, 90}};
    for (const auto& s : srcSizes) for (const auto& d : dstSizes) {
        Image8U src = pseudoRandomImage(s[0], s[1], 4242u + s[0] + d[0]);
        Image8U scalar, vector;
        resizeBilinear(src, scalar, d[0], d[1], false);
        resizeBilinear(src, vector, d[0], d[1], true);
        CHECK_EQ(vector.data, scalar.data);
    }
}

TEST_CASE("resizeBilinear: vector path reads nothing past a tight row") {
    for (int w = 7; w <= 40; ++w) {
        std::vector<uint8_t> buf(static_cast<size_t>(w) * 6, 120);
        Image8U out;
        resizeBilinear(ImageView8U(buf.data(), w, 6), out, w * 5 / 6, 5, true);
        CHECK_EQ(out.rows, 5);
    }
}

TEST_CASE("gaussianBlur7x7: vector path equals the scalar path byte for byte") {
    // Widths around every vector boundary, heights that give 1-row and multi-row column passes.
    const int widths[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 40, 63, 64, 65, 100};
    const int heights[] = {1, 2, 3, 4, 7, 8, 13};
    for (int w : widths) for (int h : heights) {
        Image8U src = pseudoRandomImage(w, h, 777u + w * 31 + h);
        Image8U scalar, vector;
        gaussianBlur7x7(src, scalar, false);
        gaussianBlur7x7(src, vector, true);
        CHECK_EQ(vector.data, scalar.data);
    }
}

TEST_CASE("gaussianBlur7x7: vector path reads nothing past a tight row") {
    // Heap buffer sized exactly cols*rows, step == cols: ASan sees any overread.
    for (int w = 7; w <= 40; ++w) {
        std::vector<uint8_t> buf(static_cast<size_t>(w) * 5, 90);
        ImageView8U view(buf.data(), w, 5);
        Image8U out;
        gaussianBlur7x7(view, out, true);
        CHECK_EQ(out.cols, w);
    }
    std::vector<uint8_t> big(static_cast<size_t>(16385) * 3, 90);
    Image8U out;
    gaussianBlur7x7(ImageView8U(big.data(), 16385, 3), out, true);
    CHECK_EQ(out.cols, 16385);
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
