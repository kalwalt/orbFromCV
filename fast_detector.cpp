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

// Calculates the FAST corner score (used for Non-Maximal Suppression)
// The score is defined as the maximum threshold at which the point would still be a corner
inline float cornerScore(const Image8U& img, int x, int y, const int* pixel_offsets, int threshold) {
    int center_val = img.at(y, x);
    int score = 0;

    for (int i = 0; i < 16; ++i) {
        int diff = std::abs(img.data[(y + circle_offsets[i][1]) * img.step + (x + circle_offsets[i][0])] - center_val);
        if (diff > threshold) {
            score += diff - threshold;
        }
    }
    return static_cast<float>(score);
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

    // Non-Maximal Suppression (NMS)
    // Keeps a corner unless a neighbour in its 3x3 window has a strictly
    // greater response (ties survive). Corners lie inside the margin, so the
    // 3x3 window never leaves the grid.
    if (nonmaxSuppression) {
        keypoints.reserve(raw_keypoints.size());
        for (const KeyPoint& kp : raw_keypoints) {
            const int x = static_cast<int>(kp.pt.x);
            const int y = static_cast<int>(kp.pt.y);
            bool is_max = true;
            for (int dy = -1; dy <= 1 && is_max; ++dy) {
                const float* row = response_grid.data() + static_cast<size_t>(y + dy) * img.step + x;
                if (row[-1] > kp.response || row[0] > kp.response || row[1] > kp.response) is_max = false;
            }
            if (is_max) keypoints.push_back(kp);
        }
    } else {
        keypoints = raw_keypoints;
    }
}