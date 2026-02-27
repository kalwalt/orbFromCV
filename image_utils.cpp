#include "image_utils.hpp"
#include <algorithm>
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
    
    dst = Image8U(src.cols, src.rows);
    Image8U temp(src.cols, src.rows); // Intermediate buffer

    // 1D Gaussian kernel for sigma = 2.0 (approx integer weights, sum = 256)
    // Avoids floating point math entirely! 
    // weights = [18, 33, 49, 56, 49, 33, 18]
    const int kernel[7] = {18, 33, 49, 56, 49, 33, 18};
    const int kRadius = 3;

    // Horizontal pass
    for (int y = 0; y < src.rows; ++y) {
        const uint8_t* src_row = src.ptr(y);
        uint8_t* temp_row = temp.ptr(y);

        for (int x = 0; x < src.cols; ++x) {
            int sum = 0;
            for (int k = -kRadius; k <= kRadius; ++k) {
                // Border reflection (BORDER_REFLECT_101 equivalent)
                int px = x + k;
                if (px < 0) px = -px;
                else if (px >= src.cols) px = 2 * src.cols - px - 2;

                sum += src_row[px] * kernel[k + kRadius];
            }
            // Shift right by 8 is equivalent to division by 256
            temp_row[x] = static_cast<uint8_t>(sum >> 8); 
        }
    }

    // Vertical pass
    for (int y = 0; y < src.rows; ++y) {
        uint8_t* dst_row = dst.ptr(y);

        for (int x = 0; x < src.cols; ++x) {
            int sum = 0;
            for (int k = -kRadius; k <= kRadius; ++k) {
                // Border reflection
                int py = y + k;
                if (py < 0) py = -py;
                else if (py >= src.rows) py = 2 * src.rows - py - 2;

                sum += temp.at(py, x) * kernel[k + kRadius];
            }
            dst_row[x] = static_cast<uint8_t>(sum >> 8);
        }
    }
}