#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP

#include "orb_core.hpp"

// Resizes an image with bilinear interpolation, byte-identical to
// cv2.resize(..., INTER_LINEAR_EXACT), the interpolation cv::ORB builds its
// pyramid with (#40). useSimd selects the SSE2/NEON/WASM SIMD128 path of the
// vertical pass when the library was built with one (orbSimdActive()); false
// forces the scalar path. Both produce the same bytes.
void resizeBilinear(const ImageView8U& src, Image8U& dst, int new_width, int new_height,
                    bool useSimd = orbSimdActive());

// Applies a 7x7 Gaussian blur with sigma=2.0, bit-identical to the blur
// cv::ORB applies to its pyramid levels (float separable filter with a
// single final rounding, see image_utils.cpp).
// Borders use BORDER_REFLECT_101; any size >= 1x1 is supported. src may be a
// view of dst's own buffer (in-place blur).
// useSimd selects the SSE2/NEON/WASM SIMD128 path when the library was built
// with one (orbSimdActive()); false forces the scalar path. Both produce the
// same bytes; the parameter exists so tests can compare them in one binary.
void gaussianBlur7x7(const ImageView8U& src, Image8U& dst, bool useSimd = orbSimdActive());

#endif // IMAGE_UTILS_HPP