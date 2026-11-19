#include "fast_detector.hpp"
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
inline float cornerScore(const Image8U& img, int x, int y, const int* pixel_offsets, int threshold) {
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

void detectFAST(const Image8U& img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmaxSuppression) {
    keypoints.clear();
    
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
    if (nonmaxSuppression) response_grid.assign(static_cast<size_t>(img.rows) * img.step, 0.0f);

    // Iterate over the image, skipping the borders
    // This row-by-row structure is ideal for future WASM SIMD (v128) optimizations
    for (int y = margin; y < img.rows - margin; ++y) {
        const uint8_t* row_ptr = img.ptr(y);
        
        for (int x = margin; x < img.cols - margin; ++x) {
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

            // Full test: Check all 16 pixels
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
                raw_keypoints.push_back(kp);
                if (nonmaxSuppression) response_grid[static_cast<size_t>(y) * img.step + x] = kp.response;
            }
        }
    }

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
                const float* row = response_grid.data() + static_cast<size_t>(y + dy) * img.step + x;
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