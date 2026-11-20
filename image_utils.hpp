#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP

#include "orb_core.hpp"

class ThreadPool;

// Resizes an image with bilinear interpolation, byte-identical to
// cv2.resize(..., INTER_LINEAR_EXACT), the interpolation cv::ORB builds its
// pyramid with (#40). useSimd selects the SSE2/NEON/WASM SIMD128 path of the
// vertical pass when the library was built with one (orbSimdActive()); false
// forces the scalar path. Both produce the same bytes.
// pool: when non-null, the destination rows are split into bands and run on
// its threads; the output is identical either way.
void resizeBilinear(const ImageView8U& src, Image8U& dst, int new_width, int new_height,
                    bool useSimd = orbSimdActive(), ThreadPool* pool = nullptr);

// Applies a 7x7 Gaussian blur with sigma=2.0, bit-identical to the blur
// cv::ORB applies to its pyramid levels (float separable filter with a
// single final rounding, see image_utils.cpp).
// Borders use BORDER_REFLECT_101; any size >= 1x1 is supported. src may be a
// view of dst's own buffer (in-place blur).
// useSimd selects the SSE2/NEON/WASM SIMD128 path when the library was built
// with one (orbSimdActive()); false forces the scalar path. Both produce the
// same bytes; the parameter exists so tests can compare them in one binary.
// pool: when non-null, each pass is split into row bands run on its threads.
void gaussianBlur7x7(const ImageView8U& src, Image8U& dst, bool useSimd = orbSimdActive(),
                     ThreadPool* pool = nullptr);

#endif // IMAGE_UTILS_HPP