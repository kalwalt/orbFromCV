#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP

#include "orb_core.hpp"

// Resizes an image using bilinear interpolation
void resizeBilinear(const Image8U& src, Image8U& dst, int new_width, int new_height);

// Applies a 7x7 Gaussian blur with sigma=2.0
// Uses an integer approximation of a separable 1D kernel for high performance
void gaussianBlur7x7(const Image8U& src, Image8U& dst);

#endif // IMAGE_UTILS_HPP