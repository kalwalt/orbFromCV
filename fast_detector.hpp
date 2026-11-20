// Adapted from OpenCV's FAST feature detector (BSD-3-Clause, Willow Garage) -
// see NOTICE at the repo root for full attribution.
#ifndef FAST_DETECTOR_HPP
#define FAST_DETECTOR_HPP

#include "orb_core.hpp"
#include <vector>

// Fast feature detection implementation (FAST-9)
// nonmaxSuppression: if true, applies non-maximal suppression to cluster corners
// useSimd: run the 16-pixel SSE2/NEON/WASM SIMD128 segment test when the
// library was built with one (orbSimdActive()); false forces the scalar path.
// Both return the same keypoints, responses and order.
void detectFAST(const ImageView8U& img, std::vector<KeyPoint>& keypoints, int threshold,
                bool nonmaxSuppression = true, bool useSimd = orbSimdActive());

#endif // FAST_DETECTOR_HPP