#include "fast_detector.hpp"
#include "simd_utils.hpp"
#include "thread_pool.hpp"
#include <cmath>
#include <algorithm>

// Circle pixel offsets for a radius of 3
// Coordinates are (x, y) relative to the center pixel
const int circle_offsets[16][2] = {
    {0, -3}, {1, -3}, {2, -2}, {3, -1},
    {3, 0}, {3, 1}, {2, 2}, {1, 3},
    {0, 3}, {-1, 3}, {-2, 2}, {-3, 1},
    {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}
};

// FAST corner score, as OpenCV's cornerScore<16>: the largest threshold at
// which the pixel is still a FAST-9 corner, i.e. (max over the 16 arcs of 9
// of the arc's minimum contrast) - 1, with both bright and dark arcs
// considered. Used to rank neighbours in non-maximal suppression (#24).
inline float cornerScore(const ImageView8U& img, int x, int y, const int* pixel_offsets, int threshold) {
    const uint8_t* ptr = img.ptr(y) + x;
    const int v = ptr[0];
    int d[25]; // center minus circle pixel, circle repeated so arcs can wrap
    for (int k = 0; k < 25; ++k) d[k] = v - ptr[pixel_offsets[k & 15]];

    // Dark arcs (circle darker than center): d > 0. a0 = best arc minimum.
    int a0 = threshold;
    for (int k = 0; k < 16; k += 2) {
        int a = std::min(d[k + 1], d[k + 2]);
        a = std::min(a, d[k + 3]);
        if (a <= a0) continue;
        a = std::min(a, d[k + 4]);
        a = std::min(a, d[k + 5]);
        a = std::min(a, d[k + 6]);
        a = std::min(a, d[k + 7]);
        a = std::min(a, d[k + 8]);
        a0 = std::max(a0, std::min(a, d[k]));
        a0 = std::max(a0, std::min(a, d[k + 9]));
    }

    // Bright arcs (circle brighter than center): d < 0. b0 = best arc maximum.
    int b0 = -a0;
    for (int k = 0; k < 16; k += 2) {
        int b = std::max(d[k + 1], d[k + 2]);
        b = std::max(b, d[k + 3]);
        b = std::max(b, d[k + 4]);
        b = std::max(b, d[k + 5]);
        if (b >= b0) continue;
        b = std::max(b, d[k + 6]);
        b = std::max(b, d[k + 7]);
        b = std::max(b, d[k + 8]);
        b0 = std::min(b0, std::max(b, d[k]));
        b0 = std::min(b0, std::max(b, d[k + 9]));
    }

    return static_cast<float>(-b0 - 1);
}

void detectFAST(const ImageView8U& img, std::vector<KeyPoint>& keypoints, int threshold,
                bool nonmaxSuppression, bool useSimd, ThreadPool* pool) {
    keypoints.clear();
#if ORB_SIMD
    const bool vec = useSimd;   // read only by the vector block below
#else
    (void)useSimd;
#endif
    
    // We need a margin of 3 pixels to access the full circle
    const int margin = 3;
    if (img.rows < 2 * margin + 1 || img.cols < 2 * margin + 1) return;

    // Precompute 1D offsets for fast memory access
    int offsets[16];
    for (int i = 0; i < 16; ++i) {
        offsets[i] = circle_offsets[i][1] * img.step + circle_offsets[i][0];
    }

    std::vector<KeyPoint> raw_keypoints;

    // Per-pixel corner response, 0 where there is no corner. Responses are
    // sums of non-negative terms, so 0 can never outrank a real corner.
    std::vector<float> response_grid;
    if (nonmaxSuppression) response_grid.assign(static_cast<size_t>(img.rows) * img.cols, 0.0f);

    // Full FAST-9 test of one pixel, plus the score when it is a corner. The
    // scalar loop reaches it after the pair-rule rejection below; the vector
    // block hands it the lanes its 16-wide segment test accepted, so the
    // response and the raw order are the scalar ones on both paths.
    auto fullTest = [&](std::vector<KeyPoint>& out, const uint8_t* row_ptr, int x, int y) {
        const int p = row_ptr[x];
        const int lower_threshold = p - threshold;
        const int upper_threshold = p + threshold;

        // We duplicate the circle array to simulate circularity without modulo operations
        int circle[32];
        for (int i = 0; i < 16; ++i) {
            circle[i] = row_ptr[x + offsets[i]];
            circle[i + 16] = circle[i];
        }

        bool is_corner = false;
        // Check for 9 contiguous pixels
        for (int i = 0; i < 16; ++i) {
            int c_bright = 0, c_dark = 0;
            for (int j = 0; j < 9; ++j) {
                if (circle[i + j] > upper_threshold) c_bright++;
                if (circle[i + j] < lower_threshold) c_dark++;
            }
            if (c_bright == 9 || c_dark == 9) {
                is_corner = true;
                break;
            }
        }

        if (is_corner) {
            KeyPoint kp;
            kp.pt.x = static_cast<float>(x);
            kp.pt.y = static_cast<float>(y);
            kp.response = cornerScore(img, x, y, offsets, threshold);
            kp.octave = 0;
            out.push_back(kp);
            if (nonmaxSuppression) response_grid[static_cast<size_t>(y) * img.cols + x] = kp.response;
        }
    };

    // Rows [yBegin, yEnd), skipping the borders. Each band appends its raw
    // corners to its own vector (merged in band order below) and writes only
    // its own rows of response_grid, so bands share nothing.
    std::vector<std::vector<KeyPoint>> perBand(pool ? pool->workers() + 1 : 1);
    auto band = [&](int bandIndex, int yBegin, int yEnd) {
        std::vector<KeyPoint>& out = perBand[static_cast<size_t>(bandIndex)];
        for (int y = yBegin; y < yEnd; ++y) {
            const uint8_t* row_ptr = img.ptr(y);
            int x = margin;

    #if ORB_SIMD
            if (vec) {
                // OpenCV's FAST_t<16> scheme: 16 centre pixels per step. The
                // segment test runs lane-wise on bytes (saturating v +- t, unsigned
                // compares); only the lanes it accepts reach fullTest, in
                // ascending order, so nothing about the output changes.
                const uint8_t tByte = static_cast<uint8_t>(std::max(0, std::min(threshold, 255)));
                const simd::u8x16 t = simd::splat_u8(tByte);
                const simd::u8x16 one = simd::splat_u8(1), eight = simd::splat_u8(8);
                const simd::u8x16 ones = simd::cmpgt(one, simd::splat_u8(0)); // all lanes set
                for (; x + 16 <= img.cols - margin; x += 16) {
                    const uint8_t* c = row_ptr + x;
                    const simd::u8x16 v = simd::load(c);
                    // v - t and v + t saturated: a circle pixel is "brighter" when
                    // > hi and "darker" when < lo, exactly as the int compares.
                    const simd::u8x16 lo = simd::subs(v, t), hi = simd::adds(v, t);

                    // Coarse rejection by the pair rule: a 9-arc covers one pixel
                    // of every opposite pair (i, i + 8).
                    simd::u8x16 maybeBright = ones, maybeDark = ones;
                    for (int i = 0; i < 8; ++i) {
                        const simd::u8x16 a = simd::load(c + offsets[i]);
                        const simd::u8x16 b = simd::load(c + offsets[i + 8]);
                        maybeBright = simd::and_(maybeBright, simd::or_(simd::cmpgt(a, hi), simd::cmpgt(b, hi)));
                        maybeDark = simd::and_(maybeDark, simd::or_(simd::cmpgt(lo, a), simd::cmpgt(lo, b)));
                    }
                    if (!simd::any(simd::or_(maybeBright, maybeDark))) continue;

                    // Segment test: run lengths of brighter / darker pixels around
                    // the circle (25 steps so every wrapping arc of 9 is seen); a
                    // lane whose counter reaches 9 is a candidate.
                    simd::u8x16 countBright = simd::splat_u8(0), countDark = simd::splat_u8(0);
                    simd::u8x16 candidate = simd::splat_u8(0);
                    for (int k = 0; k < 25; ++k) {
                        const simd::u8x16 px = simd::load(c + offsets[k & 15]);
                        countBright = simd::and_(simd::adds(countBright, one), simd::cmpgt(px, hi));
                        countDark = simd::and_(simd::adds(countDark, one), simd::cmpgt(lo, px));
                        candidate = simd::or_(candidate,
                                              simd::or_(simd::cmpgt(countBright, eight), simd::cmpgt(countDark, eight)));
                    }
                    int mask = simd::bitmask(candidate);
                    for (int lane = 0; mask != 0; ++lane, mask >>= 1)
                        if (mask & 1) fullTest(out, row_ptr, x + lane, y);
                }
            }
    #endif

            for (; x < img.cols - margin; ++x) {
                int p = row_ptr[x];

                int lower_threshold = p - threshold;
                int upper_threshold = p + threshold;

                // Early rejection, as OpenCV's FAST_t<16>: a 9-pixel arc covers at
                // least one pixel of every pair of opposite circle pixels (i, i+8),
                // so if any pair has neither pixel brighter (or neither darker)
                // there is no bright (dark) arc. Checked pair by pair, cheapest
                // first. (The previous "3 of the 4 compass pixels" test is the
                // FAST-12 criterion and dropped about a third of FAST-9 corners, #20.)
                bool maybe_bright = true, maybe_dark = true;
                for (int i = 0; i < 8 && (maybe_bright || maybe_dark); ++i) {
                    // order 0, 4, 2, 6, 1, 5, 3, 7: compass pairs first
                    static const int pair_order[8] = {0, 4, 2, 6, 1, 5, 3, 7};
                    const int a = row_ptr[x + offsets[pair_order[i]]];
                    const int b = row_ptr[x + offsets[pair_order[i] + 8]];
                    maybe_bright = maybe_bright && (a > upper_threshold || b > upper_threshold);
                    maybe_dark = maybe_dark && (a < lower_threshold || b < lower_threshold);
                }
                if (!maybe_bright && !maybe_dark) {
                    continue; // Reject early
                }

                fullTest(out, row_ptr, x, y);
            }
        }

    };
    if (pool) pool->parallelFor(margin, img.rows - margin, band);
    else band(0, margin, img.rows - margin);

    size_t total = 0;
    for (const auto& v : perBand) total += v.size();
    raw_keypoints.reserve(total);
    for (const auto& v : perBand) raw_keypoints.insert(raw_keypoints.end(), v.begin(), v.end());
    // Non-Maximal Suppression (NMS), as OpenCV's FAST_t: a corner survives only
    // if its response is strictly greater than that of every pixel in its 3x3
    // window, so two tied neighbours are both dropped (#24). Corners lie inside
    // the margin, so the window never leaves the grid.
    if (nonmaxSuppression) {
        keypoints.reserve(raw_keypoints.size());
        for (const KeyPoint& kp : raw_keypoints) {
            const int x = static_cast<int>(kp.pt.x);
            const int y = static_cast<int>(kp.pt.y);
            bool is_max = true;
            for (int dy = -1; dy <= 1 && is_max; ++dy) {
                const float* row = response_grid.data() + static_cast<size_t>(y + dy) * img.cols + x;
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    if (row[dx] >= kp.response) { is_max = false; break; }
                }
            }
            if (is_max) keypoints.push_back(kp);
        }
    } else {
        keypoints = raw_keypoints;
    }
}