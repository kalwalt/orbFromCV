#include "image_utils.hpp"
#include <algorithm>
#include <utility>
#include <vector>

void resizeBilinear(const Image8U& src, Image8U& dst, int new_width, int new_height) {
    if (new_width == 0 || new_height == 0) return;

    dst = Image8U(new_width, new_height);
    
    float x_ratio = static_cast<float>(src.cols - 1) / new_width;
    float y_ratio = static_cast<float>(src.rows - 1) / new_height;

    for (int y = 0; y < new_height; ++y) {
        float src_y = y * y_ratio;
        int y1 = static_cast<int>(src_y);
        int y2 = std::min(y1 + 1, src.rows - 1);
        float y_diff = src_y - y1;

        uint8_t* dst_row = dst.ptr(y);
        const uint8_t* src_row1 = src.ptr(y1);
        const uint8_t* src_row2 = src.ptr(y2);

        for (int x = 0; x < new_width; ++x) {
            float src_x = x * x_ratio;
            int x1 = static_cast<int>(src_x);
            int x2 = std::min(x1 + 1, src.cols - 1);
            float x_diff = src_x - x1;

            // Bilinear interpolation formula
            float a = src_row1[x1];
            float b = src_row1[x2];
            float c = src_row2[x1];
            float d = src_row2[x2];

            float pixel = a * (1.0f - x_diff) * (1.0f - y_diff) +
                          b * (x_diff) * (1.0f - y_diff) +
                          c * (1.0f - x_diff) * (y_diff) +
                          d * (x_diff) * (y_diff);

            dst_row[x] = static_cast<uint8_t>(pixel);
        }
    }
}

void gaussianBlur7x7(const Image8U& src, Image8U& dst) {
    if (src.cols == 0 || src.rows == 0) return;

    const int cols = src.cols, rows = src.rows;
    Image8U temp(cols, rows);   // horizontal-pass intermediate
    Image8U result(cols, rows); // vertical-pass output

    // 1D Gaussian kernel for sigma = 2.0 (approx integer weights, sum = 256),
    // symmetric so each tap pair shares one multiply. Integer math only.
    const int kernel[7] = {18, 33, 49, 56, 49, 33, 18};
    const int kRadius = 3;
    const int k0 = kernel[0], k1 = kernel[1], k2 = kernel[2], k3 = kernel[3];

    // BORDER_REFLECT_101: index -1 -> 1, n -> n-2.
    auto reflect = [](int i, int n) { return i < 0 ? -i : (i >= n ? 2 * n - i - 2 : i); };

    // Horizontal pass: reflection only on the two 3-pixel border strips, the
    // interior runs branch-free on the symmetric kernel.
    const int leftEnd = std::min(kRadius, cols);
    const int rightStart = std::max(kRadius, cols - kRadius);
    for (int y = 0; y < rows; ++y) {
        const uint8_t* s = src.ptr(y);
        uint8_t* t = temp.ptr(y);

        auto borderPixel = [&](int x) {
            int sum = 0;
            for (int k = -kRadius; k <= kRadius; ++k) sum += s[reflect(x + k, cols)] * kernel[k + kRadius];
            t[x] = static_cast<uint8_t>(sum >> 8);
        };
        for (int x = 0; x < leftEnd; ++x) borderPixel(x);
        for (int x = kRadius; x < cols - kRadius; ++x) {
            int sum = k0 * (s[x - 3] + s[x + 3]) + k1 * (s[x - 2] + s[x + 2]) +
                      k2 * (s[x - 1] + s[x + 1]) + k3 * s[x];
            t[x] = static_cast<uint8_t>(sum >> 8);
        }
        for (int x = rightStart; x < cols; ++x) borderPixel(x);
    }

    // Vertical pass as row accumulation: the seven source rows are resolved
    // (with reflection) once per output row, then read contiguously.
    for (int y = 0; y < rows; ++y) {
        const uint8_t* r[7];
        for (int k = 0; k < 7; ++k) r[k] = temp.ptr(reflect(y + k - kRadius, rows));
        uint8_t* out = result.ptr(y);
        for (int x = 0; x < cols; ++x) {
            int sum = k0 * (r[0][x] + r[6][x]) + k1 * (r[1][x] + r[5][x]) +
                      k2 * (r[2][x] + r[4][x]) + k3 * r[3][x];
            out[x] = static_cast<uint8_t>(sum >> 8);
        }
    }

    // Deferred until every read of `src` is done, so gaussianBlur7x7(x, x)
    // (dst aliasing src - exactly how ORB::detectAndCompute calls this) is
    // safe: reassigning dst any earlier would overwrite src's data out from
    // under the horizontal pass before it could read it.
    dst = std::move(result);
}