#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP

#include "orb_core.hpp"

// Resizes an image using bilinear interpolation
void resizeBilinear(const ImageView8U& src, Image8U& dst, int new_width, int new_height);

// Applies a 7x7 Gaussian blur with sigma=2.0, bit-identical to the blur
// cv::ORB applies to its pyramid levels (float separable filter with a
// single final rounding, see image_utils.cpp)
// Uses an integer approximation of a separable 1D kernel for high performance
// Borders use BORDER_REFLECT_101; any size >= 1x1 is supported. src may be a
// view of dst's own buffer (in-place blur).
void gaussianBlur7x7(const ImageView8U& src, Image8U& dst);

#endif // IMAGE_UTILS_HPP